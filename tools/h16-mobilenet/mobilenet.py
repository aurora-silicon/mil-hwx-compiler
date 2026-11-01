#!/usr/bin/env python3
"""MobileNetV1 224 (alpha 1.0) as ONE linked H16G/H17 program.

usage: mobilenet.py OUTDIR [--target H17|H16G] [--weights model.tflite | --tfweights tfweights.npz]
                          [--seed N] [--images images.npz]

The network is the TensorFlow frame: the 3->32 stem and the four depthwise
layers run at stride 2 (sparse kernels), SAME padding, ReLU where the original
uses ReLU6. The classifier is a 1x1 conv (1024 -> 1001, zero-padded to 1024)
at 7x7 before the average pool, so the host only averages 49 positions per
logit. `--schedule linked` chains all 28 convolutions through DRAM scratch.

Without weights the weights are random (He-scaled). The float64 reference
rounds every layer to fp16, as the engine stores its surfaces; `--images`
(an npz holding `images` [N,3,224,224] in [-1, 1]) adds per-image references
and the input surfaces.
"""
import argparse, json, os, shutil, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'h13-mobilenet'))

CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
       (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]


def layers():
    out = [dict(kind='conv', cin=3, cout=32, k=3, s=2, g=1, relu=True)]
    c = 32
    for co, s in CFG:
        out.append(dict(kind='dw', cin=c, cout=c, k=3, s=s, g=c, relu=True))
        out.append(dict(kind='pw', cin=c, cout=co, k=1, s=1, g=1, relu=True))
        c = co
    out.append(dict(kind='fc', cin=1024, cout=1024, k=1, s=1, g=1, relu=False))
    return out


def random_weights(seed):
    rng = np.random.default_rng(seed)
    ws = []
    for L in layers():
        cout = 1001 if L['kind'] == 'fc' else L['cout']
        fan = L['cin'] // L['g'] * L['k'] * L['k']
        w = rng.standard_normal((cout, L['cin'] // L['g'], L['k'], L['k'])) * np.sqrt(2.0 / fan)
        ws.append(dict(w=w.astype(np.float32), b=(rng.standard_normal(cout) * 0.05).astype(np.float32)))
    return ws


def ane_params(tfw):
    """TF-frame float weights -> fp16 parameters, classifier padded to 1024 outputs."""
    params = []
    for L, P in zip(layers(), tfw):
        w, b = P['w'], P['b']
        if L['kind'] == 'fc':
            w2 = np.zeros((1024, 1024, 1, 1), np.float32); w2[:w.shape[0]] = w.reshape(w.shape[0], 1024, 1, 1)
            b2 = np.zeros(1024, np.float32); b2[:b.shape[0]] = b
            w, b = w2, b2
        params.append(dict(w=np.ascontiguousarray(w).astype(np.float16), b=b.astype(np.float16)))
    return params


def blob_write(path, arrays):
    out = bytearray(struct.pack('<II', 1, 2) + bytes(56))
    offs = []
    for a in arrays:
        data = np.ascontiguousarray(a, dtype=np.float16).tobytes()
        meta = len(out)
        out += struct.pack('<IIQQ', 0xDEADBEEF, 1, len(data), meta + 64) + bytes(40) + data
        out += bytes((-len(out)) % 64)
        offs.append(meta)
    with open(path, 'wb') as f:
        f.write(out)
    return offs


def write_mil(outdir, params):
    os.makedirs(os.path.join(outdir, 'weights'), exist_ok=True)
    arrays = []
    for P in params:
        arrays += [P['w'], P['b']]
    offs = blob_write(os.path.join(outdir, 'weights', 'weight.bin'), arrays)
    body, cur, c, h = [], 'x', 3, 224
    for i, (L, P) in enumerate(zip(layers(), params)):
        co, k, s, g = L['cout'], L['k'], L['s'], L['g']
        ho = -(-h // s)
        cig = c // g
        blob = 'BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({}))'
        body += [
            f'    string pt{i} = const()[name = string("pt{i}"), val = string("same")];',
            f'    tensor<int32, [2]> st{i} = const()[name = string("st{i}"), val = tensor<int32, [2]>([{s}, {s}])];',
            f'    tensor<int32, [4]> pd{i} = const()[name = string("pd{i}"), val = tensor<int32, [4]>([0, 0, 0, 0])];',
            f'    tensor<int32, [2]> dl{i} = const()[name = string("dl{i}"), val = tensor<int32, [2]>([1, 1])];',
            f'    int32 gp{i} = const()[name = string("gp{i}"), val = int32({g})];',
            f'    tensor<fp16, [{co}, {cig}, {k}, {k}]> W{i} = const()[name = string("W{i}"), val = tensor<fp16, [{co}, {cig}, {k}, {k}]>({blob.format(offs[2 * i])})];',
            f'    tensor<fp16, [{co}]> B{i} = const()[name = string("B{i}"), val = tensor<fp16, [{co}]>({blob.format(offs[2 * i + 1])})];',
            f'    tensor<fp16, [1, {co}, {ho}, {ho}]> c{i} = conv(bias = B{i}, dilations = dl{i}, groups = gp{i}, pad = pd{i}, pad_type = pt{i}, strides = st{i}, weight = W{i}, x = {cur})[name = string("conv{i}")];',
        ]
        cur = f'c{i}'
        if L['relu']:
            body.append(f'    tensor<fp16, [1, {co}, {ho}, {ho}]> r{i} = relu(x = {cur})[name = string("relu{i}")];')
            cur = f'r{i}'
        c, h = co, ho
    mil = ('program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
           '  func main<ios18>(tensor<fp16, [1, 3, 224, 224]> x) {\n' + '\n'.join(body) +
           f'\n  }} -> ({cur});\n}}\n')
    path = os.path.join(outdir, 'model.mil')
    with open(path, 'w') as f:
        f.write(mil)
    return path


def conv_ref(x16, P, L):
    """float64 TF-SAME conv of one fp16 CHW tensor, then ReLU, rounded to fp16."""
    x = x16.astype(np.float64); w = P['w'].astype(np.float64); b = P['b'].astype(np.float64)
    C, H, W = x.shape; k, s = L['k'], L['s']
    Ho = -(-H // s); ph = max((Ho - 1) * s + k - H, 0)
    xp = np.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (ph // 2, ph - ph // 2)))
    y = np.zeros((w.shape[0], Ho, Ho))
    for dy in range(k):
        for dx in range(k):
            patch = xp[:, dy:dy + s * (Ho - 1) + 1:s, dx:dx + s * (Ho - 1) + 1:s]
            if L['g'] == 1:
                y += np.einsum('oc,chw->ohw', w[:, :, dy, dx], patch, optimize=True)
            else:
                y += w[:, 0, dy, dx][:, None, None] * patch
    y += b[:, None, None]
    return (np.maximum(y, 0) if L['relu'] else y).astype(np.float16)


def surface(a16):
    """[C,H,W] fp16 -> conv surface bytes (rows padded to 64 bytes)."""
    C, H, W = a16.shape; row = (2 * W + 63) // 64 * 64
    buf = np.zeros((C, H, row // 2), np.float16); buf[:, :, :W] = a16
    return buf.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out'); ap.add_argument('--target', choices=['H17', 'H16G'], default='H17')
    ap.add_argument('--weights'); ap.add_argument('--tfweights'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--images')
    a = ap.parse_args()
    if a.weights:
        from tflite_weights import mobilenet_v1_weights
        tfw = mobilenet_v1_weights(a.weights)
    elif a.tfweights:
        z = np.load(a.tfweights)
        tfw = [dict(w=z[f'w{i}'], b=z[f'b{i}']) for i in range(28)]
    else:
        tfw = random_weights(a.seed)
    params = ane_params(tfw)
    shutil.rmtree(a.out, ignore_errors=True)
    src = os.path.join(a.out, 'src')
    mil = write_mil(src, params)
    milc = os.environ.get('MILC', os.path.join(HERE, '..', '..', 'build', 'mil-hwxc'))
    r = subprocess.run([milc, '--target', a.target, '--format', 'hwx', '--schedule', 'linked', '--mil', mil,
                        '--model-root', src, '--output', os.path.join(a.out, 'linked')], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f'linked compile failed: {r.stderr.strip()[-800:]}')
    m = json.load(open(os.path.join(a.out, 'linked', 'manifest.json')))
    np.savez(os.path.join(a.out, 'params.npz'), **{f'w{i}': P['w'] for i, P in enumerate(params)},
             **{f'b{i}': P['b'] for i, P in enumerate(params)})
    msg = (f"{a.out}: {a.target} linked {m['programs'][0]['taskDescriptors']} tasks, "
           f"{m['programs'][0]['bytes']} bytes, scratch {m['scratch']['bytes']}")
    if a.images:
        imgs = np.load(a.images)['images']
        refs = os.path.join(a.out, 'refs'); os.makedirs(refs)
        tops = []
        for i, img in enumerate(imgs):
            x = img.astype(np.float16)
            with open(os.path.join(refs, f'input_{i:02d}.bin'), 'wb') as f:
                f.write(surface(x))
            for L, P in zip(layers(), params):
                x = conv_ref(x, P, L)
            with open(os.path.join(refs, f'output_{i:02d}.bin'), 'wb') as f:
                f.write(surface(x))
            tops.append(int(x[:1001].astype(np.float64).mean(axis=(1, 2)).argmax()))
        json.dump({'top1': tops}, open(os.path.join(refs, 'top1.json'), 'w'))
        msg += f'; references for {len(imgs)} images'
    print(msg)


if __name__ == '__main__':
    main()
