"""Shared helpers for H13 linked-chain cases: layer specs, MIL, references."""
import json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'h13-conv-bench'))
from milgen import conv_mil            # noqa: E402
from surf import pack, unpack, f16_ties_away, stats   # noqa: E402

def parse_layers(text):
    """'cout,k,s,g,bias,relu;...' -> list of layer dicts."""
    layers = []
    for item in text.split(';'):
        co, k, s, g, b, r = map(int, item.split(','))
        layers.append(dict(cout=co, k=k, s=s, g=g, bias=bool(b), relu=bool(r)))
    return layers

def conv_ref(x, w, b, s, g, pad, relu):
    """float64 convolution of one CHW image (groups 1 or depthwise)."""
    C, H, W = x.shape
    Co, cig, k, _ = w.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - k) // s + 1
    Wo = (W + 2 * pad - k) // s + 1
    out = np.zeros((Co, Ho, Wo))
    wd = w.astype(np.float64)
    for dy in range(k):
        for dx in range(k):
            patch = xp[:, dy:dy + s * Ho:s, dx:dx + s * Wo:s]
            if g == 1:
                out += np.einsum('oc,chw->ohw', wd[:, :, dy, dx], patch, optimize=True)
            else:
                out += wd[:, 0, dy, dx][:, None, None] * patch
    if b is not None:
        out += b.astype(np.float64)[:, None, None]
    return np.maximum(out, 0) if relu else out

def chain_ref(x, layers, params, per_layer_round=True):
    """Every layer in float64; the surface between layers is fp16 rounded
    ties-away, as the engine stores it. Returns the list of layer outputs."""
    outs = []
    cur = x
    for L, P in zip(layers, params):
        y = conv_ref(cur, P['w'], P['b'], L['s'], L['g'], (L['k'] - 1) // 2, L['relu'])
        cur = f16_ties_away(y) if per_layer_round else y
        outs.append(cur)
    return outs

def compile_chain(milc, mil, root, out, schedule, generic=True):
    env = dict(os.environ)
    if generic:
        env['MILHWXC_H13_CONV_GENERIC'] = '1'
    target = env.get('MILHWXC_TARGET', 'H13')
    if target == 'H14':
        env['MILHWXC_H14_CONV_GENERIC'] = '1'
    cmd = [milc, '--target', target, '--mil', mil, '--model-root', root, '--output', out]
    if schedule:
        cmd += ['--schedule', schedule]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f'{schedule or "per-op"} compile failed: {r.stderr.strip()[-600:]}')
    return json.load(open(os.path.join(out, 'manifest.json')))

def save_params(path, layers, params, inshape):
    arrays = {'inshape': np.array(inshape), 'layers': json.dumps(layers)}
    for i, P in enumerate(params):
        arrays[f'w{i}'] = P['w']
        if P['b'] is not None:
            arrays[f'b{i}'] = P['b']
    np.savez(path, **arrays)

def load_params(path):
    z = np.load(path, allow_pickle=False)
    layers = json.loads(str(z['layers']))
    params = [dict(w=z[f'w{i}'], b=z[f'b{i}'] if f'b{i}' in z else None) for i in range(len(layers))]
    return layers, params, tuple(int(v) for v in z['inshape'])
