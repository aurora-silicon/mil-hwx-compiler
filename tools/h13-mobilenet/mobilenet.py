"""MobileNetV1 224 (alpha 1.0) as one H13 linked conv chain.

usage: mobilenet.py OUTDIR [--weights model.tflite] [--seed N] [--head ane|cpu]

The ANE runs the network in a mirrored frame: the host flips the image on
both spatial axes and every 3x3 kernel likewise. TensorFlow's SAME padding at
stride 2 samples the odd input positions; in the mirrored frame those are the
even positions, which is what a symmetric-pad stride-1 layer followed by a
stride-2 1x1 layer samples. So the chain below is exactly the TF network:

  - the stem (3x3 s2, 3 -> 32) is a space-to-depth of the mirrored image
    (12 channels, zero-padded to 16) followed by a 3x3 s1 conv whose kernel
    holds the stem taps in a 2x2 window;
  - each depthwise s2 layer runs at stride 1 and the stride moves onto the
    following 1x1 layer (128 output channels per task above 128 outputs);
  - the classifier (1x1 conv, 1024 -> 1001 logits, zero-padded to 1024) runs
    on the ANE at 7x7 before the average pool; pooling commutes with it, so
    the host only averages 49 positions per logit (or --head cpu leaves the
    pool and classifier to the host).

Without --weights the weights are random (He-scaled), for timing.
"""
import argparse, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain import conv_mil, compile_chain, save_params   # noqa: E402

CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
       (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]

def tf_layers():
    """The 28 TF layers: (kind, cin, cout, stride)."""
    out = [('conv', 3, 32, 2)]
    c = 32
    for co, s in CFG:
        out.append(('dw', c, c, s))
        out.append(('pw', c, co, 1))
        c = co
    out.append(('fc', 1024, 1001, 1))
    return out

def ane_layers(head='ane'):
    layers = [dict(cout=32, k=3, s=1, g=1, bias=True, relu=True)]
    c = 32
    for co, s in CFG:
        layers.append(dict(cout=c, k=3, s=1, g=c, bias=True, relu=True))
        layers.append(dict(cout=co, k=1, s=s, g=1, bias=True, relu=True))
        c = co
    if head == 'ane':
        layers.append(dict(cout=1024, k=1, s=1, g=1, bias=True, relu=False))
    return layers

def random_tf_weights(seed=0):
    rng = np.random.default_rng(seed)
    ws = []
    for kind, ci, co, s in tf_layers():
        k = 1 if kind in ('pw', 'fc') else 3
        fan = (1 if kind == 'dw' else ci) * k * k
        shape = (co, 1, k, k) if kind == 'dw' else (co, ci, k, k)
        w = rng.standard_normal(shape) * np.sqrt(2.0 / fan)
        b = rng.standard_normal(co) * 0.05
        ws.append(dict(w=w.astype(np.float32), b=b.astype(np.float32)))
    return ws

def to_ane_params(tfw, head='ane'):
    """TF-frame float weights -> the mirrored-frame fp16 chain parameters."""
    params = []
    # Stem: out[o, k, l] = sum_{c,dy,dx} w[o,c,dy,dx] x[c, 2k+dy-?, ...] in the
    # mirrored frame the taps are rows 2k-1, 2k, 2k+1 (pad 1, even centres).
    # Space-to-depth channel (c, py, px) at (k, l) is x[c, 2k+py, 2l+px], so
    # row 2k-1 is (k-1, py=1), 2k is (k, 0), 2k+1 is (k, 1).
    w0 = tfw[0]['w'][:, :, ::-1, ::-1]        # mirror the kernel
    stem = np.zeros((32, 16, 3, 3), np.float32)
    tap = {0: (0, 1), 1: (1, 0), 2: (1, 1)}   # kernel row -> (window row, phase)
    for dy in range(3):
        wy, py = tap[dy]
        for dx in range(3):
            wx, px = tap[dx]
            for c in range(3):
                stem[:, c * 4 + py * 2 + px, wy, wx] = w0[:, c, dy, dx]
    params.append(dict(w=stem.astype(np.float16), b=tfw[0]['b'].astype(np.float16)))
    for i in range(1, 27):
        w = tfw[i]['w']
        if w.shape[2] == 3:
            w = w[:, :, ::-1, ::-1]
        params.append(dict(w=np.ascontiguousarray(w).astype(np.float16), b=tfw[i]['b'].astype(np.float16)))
    if head == 'ane':
        w = np.zeros((1024, 1024, 1, 1), np.float32); b = np.zeros(1024, np.float32)
        w[:1001] = tfw[27]['w']; b[:1001] = tfw[27]['b']
        params.append(dict(w=w.astype(np.float16), b=b.astype(np.float16)))
    return params

def preprocess(img):
    """[3,224,224] float image in [-1, 1] -> mirrored space-to-depth [16,112,112] fp16."""
    m = img[:, ::-1, ::-1]
    s = np.zeros((16, 112, 112), np.float32)
    for c in range(3):
        for py in range(2):
            for px in range(2):
                s[c * 4 + py * 2 + px] = m[c, py::2, px::2]
    return s.astype(np.float16)

def tf_reference(img, tfw, relu6=True):
    """float64 TF-frame MobileNetV1 (SAME padding) -> 1001 logits."""
    x = img.astype(np.float64)
    act = (lambda v: np.clip(v, 0, 6)) if relu6 else (lambda v: np.maximum(v, 0))
    for (kind, ci, co, s), P in zip(tf_layers(), tfw):
        w = P['w'].astype(np.float64); b = P['b'].astype(np.float64)
        if kind == 'fc':
            return w[:, :, 0, 0] @ x.mean(axis=(1, 2)) + b
        k = w.shape[2]
        C, H, W = x.shape
        Ho, Wo = -(-H // s), -(-W // s)
        ph = max((Ho - 1) * s + k - H, 0); pw = max((Wo - 1) * s + k - W, 0)
        xp = np.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        y = np.zeros((co, Ho, Wo))
        for dy in range(k):
            for dx in range(k):
                patch = xp[:, dy:dy + s * Ho:s, dx:dx + s * Wo:s]
                if kind == 'dw':
                    y += w[:, 0, dy, dx][:, None, None] * patch
                else:
                    y += np.einsum('oc,chw->ohw', w[:, :, dy, dx], patch, optimize=True)
        x = act(y + b[:, None, None])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--weights')
    ap.add_argument('--target', choices=['H13', 'H14'], default='H13')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--head', choices=['ane', 'cpu'], default='ane')
    ap.add_argument('--no-perop', action='store_true')
    a = ap.parse_args()
    os.environ['MILHWXC_TARGET'] = a.target
    if a.weights:
        from tflite_weights import mobilenet_v1_weights
        tfw = mobilenet_v1_weights(a.weights)
    else:
        tfw = random_tf_weights(a.seed)
    layers = ane_layers(a.head)
    params = to_ane_params(tfw, a.head)
    milc = os.environ.get('MILC', 'build/mil-hwxc')
    shutil.rmtree(a.out, ignore_errors=True)
    mil, params, outshape = conv_mil(layers, (16, 112, 112), os.path.join(a.out, 'src'), weights=params)
    save_params(os.path.join(a.out, 'params.npz'), layers, params, (16, 112, 112))
    np.savez(os.path.join(a.out, 'tfweights.npz'), **{f'w{i}': P['w'] for i, P in enumerate(tfw)},
             **{f'b{i}': P['b'] for i, P in enumerate(tfw)})
    m = compile_chain(milc, mil, os.path.join(a.out, 'src'), os.path.join(a.out, 'linked'), os.environ.get('LINKED_SCHEDULE', 'linked'))
    msg = (f"{a.out}: {len(layers)} layers, linked {m['programs'][0]['taskDescriptors']} tasks, "
           f"{m['programs'][0]['bytes']} bytes, scratch {m['scratch']['bytes']}")
    if not a.no_perop:
        p = compile_chain(milc, mil, os.path.join(a.out, 'src'), os.path.join(a.out, 'perop'), None)
        msg += f"; per-op {len(p['programs'])} programs"
    print(msg + f'; output {outshape}')
