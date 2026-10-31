"""Host side: synthetic test images and float64 CPU references for a
MobileNet package from mobilenet.py.

usage: mn_refs.py NETDIR COUNT

Writes NETDIR/refs.npz: images [N,3,224,224] in [-1, 1], the ANE-frame inputs
[N,16,112,112] fp16, `logits_relu` from the float64 reference of the network
the ANE runs (ReLU, fp16 weights, fp16 between layers) and, when the TF
weights are present, `logits_tf_relu6` from the original TF network (ReLU6,
float32 weights) for information.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mobilenet as M   # noqa: E402
from chain import chain_ref, load_params   # noqa: E402

def smooth_image(seed):
    """Multi-scale smooth random field: not a natural image, but with the
    low-frequency structure that keeps activations in a realistic range."""
    r = np.random.default_rng(seed)
    f = np.zeros((3, 224, 224))
    for s in (2, 4, 8, 16, 32, 56):
        g = r.standard_normal((3, s, s))
        f += np.kron(g, np.ones((224 // s + 1, 224 // s + 1)))[:, :224, :224] / s ** 0.5
    return np.clip(f / np.abs(f).max(), -1, 1)

def pattern_image(seed):
    """Procedural patterns (stripes, checkers, rings, dots, plaid) in random
    colours: they land in many different ImageNet classes, which makes top-1
    agreement a stronger check than one class for every input."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:224, 0:224] / 224.0
    kind = seed % 5
    f = r.uniform(4, 40)
    a = r.uniform(0, np.pi)
    u = xx * np.cos(a) + yy * np.sin(a)
    v = -xx * np.sin(a) + yy * np.cos(a)
    if kind == 0: g = np.sin(2 * np.pi * f * u)
    elif kind == 1: g = np.sign(np.sin(2 * np.pi * f * u) * np.sin(2 * np.pi * f * v))
    elif kind == 2: g = np.sin(2 * np.pi * f * np.hypot(xx - r.uniform(), yy - r.uniform()))
    elif kind == 3: g = (np.sin(2 * np.pi * f * u) * np.sin(2 * np.pi * f * v) > 0.6) * 2.0 - 1
    else: g = np.sin(2 * np.pi * f * u) + np.sin(2 * np.pi * r.uniform(4, 40) * v)
    c0, c1 = r.uniform(-1, 1, 3), r.uniform(-1, 1, 3)
    t = (g - g.min()) / (np.ptp(g) + 1e-9)
    img = c0[:, None, None] * (1 - t) + c1[:, None, None] * t
    return np.clip(img + 0.3 * smooth_image(seed) , -1, 1)

net, count = sys.argv[1], int(sys.argv[2])
layers, params, inshape = load_params(os.path.join(net, 'params.npz'))
tz = np.load(os.path.join(net, 'tfweights.npz'))
tfw = [dict(w=tz[f'w{i}'], b=tz[f'b{i}']) for i in range(28)]
imgs, inputs, lr, ltf = [], [], [], []
for i in range(count):
    img = (pattern_image(2000 + i) if i % 4 else smooth_image(1000 + i)).astype(np.float32)
    x = M.preprocess(img)
    out = chain_ref(x, layers, params)[-1]
    logits = out.astype(np.float64).mean(axis=(1, 2))[:1001] if len(layers) == 28 else None
    imgs.append(img); inputs.append(x); lr.append(logits)
    ltf.append(M.tf_reference(img, tfw, relu6=True))
    print(i, 'ref top1', int(np.argmax(logits)), 'tf relu6 top1', int(np.argmax(ltf[-1])), flush=True)
np.savez(os.path.join(net, 'refs.npz'), images=np.array(imgs), inputs=np.array(inputs),
         logits_relu=np.array(lr), logits_tf_relu6=np.array(ltf))
