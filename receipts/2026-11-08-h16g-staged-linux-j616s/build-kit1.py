#!/usr/bin/env python3
"""ANE2 probe kit: compile H16G staged transformer primitives and write
inputs plus float64 references. Usage: gen.py COMPILER KITDIR"""
import json, os, pathlib, shutil, subprocess, sys
import numpy as np

comp, kit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
repo = comp.parent.parent
rng = np.random.default_rng(1108)
HDR = 'program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
cases = []


def h(a):
    return np.asarray(a, dtype=np.float16)


def mil(name, sig, body, out):
    return f'{HDR}  func main<ios18>({sig}) {{\n{body}  }} -> ({out});\n}}\n'


def compile_case(name, text):
    d = kit / name
    shutil.rmtree(d, ignore_errors=True)
    src = kit / 'mil' / f'{name}.mil'
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(text)
    r = subprocess.run([str(comp), '--mil', str(src), '--model-root', str(repo / 'tests/models'),
                        '--output', str(d)], capture_output=True, text=True)
    if r.returncode:
        print(name, 'REFUSED', r.stderr.strip().splitlines()[-1])
        return None
    return json.loads((d / 'manifest.json').read_text())


def save(d, fname, arr):
    (kit / d / fname).write_bytes(h(arr).tobytes())


# square runtime matmul
for n in (128, 256, 512, 768, 1536, 3072):
    t = f'tensor<fp16, [1, {n}, {n}]>'
    m = mil('', f'{t} x, {t} y',
            f'    bool f = const()[name = string("f"), val = bool(false)];\n'
            f'    {t} z = matmul(transpose_x = f, transpose_y = f, x = x, y = y)[name = string("z")];\n', 'z')
    name = f'mm{n}'
    if compile_case(name, m) is None:
        continue
    a = h(rng.standard_normal((n, n)))
    b = h(rng.standard_normal((n, n)) / np.sqrt(n))
    save(name, 'in0.bin', a); save(name, 'in1.bin', b)
    save(name, 'ref_xy.bin', a.astype(np.float64) @ b.astype(np.float64))
    save(name, 'ref_yx.bin', b.astype(np.float64) @ a.astype(np.float64))
    cases.append({'name': name, 'programs': ['program-0.hwx'], 'inputs': [['in0.bin', 'in1.bin']],
                  'refs': ['ref_xy.bin', 'ref_yx.bin']})

# binary ALU and LUT unary, square [1,1,N,N]
for n in (128, 512, 2048):
    t = f'tensor<fp16, [1, 1, {n}, {n}]>'
    a = h(rng.standard_normal((n, n)) * 2)
    b = h(rng.standard_normal((n, n)) * 2)
    for op, fn in (('add', np.add), ('mul', np.multiply)):
        name = f'{op}{n}'
        if compile_case(name, mil('', f'{t} x, {t} y',
                                  f'    {t} z = {op}(x = x, y = y)[name = string("z")];\n', 'z')) is None:
            continue
        save(name, 'in0.bin', a); save(name, 'in1.bin', b)
        save(name, 'ref.bin', fn(a.astype(np.float64), b.astype(np.float64)))
        cases.append({'name': name, 'programs': ['program-0.hwx'], 'inputs': [['in0.bin', 'in1.bin']],
                      'refs': ['ref.bin']})
    import math
    for op in ('gelu', 'exp'):
        name = f'{op}{n}'
        extra = ', mode = string("EXACT")' if op == 'gelu' else ''
        if compile_case(name, mil('', f'{t} x',
                                  f'    {t} z = {op}(x = x{extra})[name = string("z")];\n', 'z')) is None:
            continue
        x = h(rng.standard_normal((n, n)) * (2 if op == 'gelu' else 1))
        xd = x.astype(np.float64)
        if op == 'gelu':
            erf = np.vectorize(math.erf)
            ref = 0.5 * xd * (1 + erf(xd / np.sqrt(2)))
        else:
            ref = np.exp(xd)
        save(name, 'in0.bin', x); save(name, 'ref.bin', ref)
        cases.append({'name': name, 'programs': ['program-0.hwx'], 'inputs': [['in0.bin']],
                      'refs': ['ref.bin']})

# one BERT-base attention head (S=128, d_head=64 zero-padded to 128, scale 1/8)
t = 'tensor<fp16, [1, 1, 128, 128]>'
body = ('    bool f = const()[name = string("f"), val = bool(false)];\n'
        f'    {t} scores = matmul(transpose_x = f, transpose_y = f, x = q, y = k)[name = string("scores")];\n'
        '    fp16 scale = const()[name = string("scale"), val = fp16(0.125)];\n'
        f'    {t} scaled = mul(x = scores, y = scale)[name = string("scaled")];\n'
        '    int32 axis = const()[name = string("axis"), val = int32(-1)];\n'
        f'    {t} probabilities = softmax(axis = axis, x = scaled)[name = string("probabilities")];\n'
        f'    {t} y = matmul(transpose_x = f, transpose_y = f, x = probabilities, y = v)[name = string("y")];\n')
man = compile_case('bert_head', mil('', f'{t} q, {t} k, {t} v', body, 'y'))
if man:
    S, D = 128, 64
    q = np.zeros((S, 128)); k = np.zeros((128, S)); v = np.zeros((S, 128))
    q[:, :D] = rng.standard_normal((S, D)); k[:D, :] = rng.standard_normal((D, S))
    v[:, :D] = rng.standard_normal((S, D))
    q, k, v = h(q), h(k), h(v)
    sc = q.astype(np.float64) @ k.astype(np.float64) * 0.125
    sc16 = h(sc)
    e = np.exp(sc16.astype(np.float64) - sc16.astype(np.float64).max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    y = h(p).astype(np.float64) @ v.astype(np.float64)
    for nm, arr in (('q.bin', q), ('k.bin', k), ('v.bin', v), ('ref_scaled.bin', sc),
                    ('ref_probs.bin', p), ('ref_y.bin', y)):
        save('bert_head', nm, arr)
    cases.append({'name': 'bert_head', 'chain': True,
                  'programs': [a['file'] for a in man['artifacts']],
                  'operations': [a['operations'] for a in man['artifacts']]})

# decoded one-program 4-head attention (S=64, d_head=64), upstream fixture
man = compile_case('attn4h', (repo / 'tests/fixtures/attention.mil').read_text())
if man:
    x = h(rng.standard_normal((64, 4, 192)))
    xd = x.astype(np.float64)
    Q0, K0, V0 = xd[..., 0:64], xd[..., 64:128], xd[..., 128:192]   # [64(c),4(h),64(w)]
    Q = Q0.transpose(1, 2, 0)          # perm [0,2,3,1] -> [1,4,64,64]
    K = K0.transpose(1, 0, 2)          # perm [0,2,1,3]
    V = V0.transpose(1, 2, 0)
    sc = Q @ K * 0.125
    e = np.exp(sc - sc.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True)
    ctx = p @ V                         # [4,64,64]
    out = ctx.transpose(2, 0, 1)        # perm [0,3,1,2] -> [1,64,4,64]
    save('attn4h', 'in0.bin', x); save('attn4h', 'ref.bin', out)
    cases.append({'name': 'attn4h', 'programs': ['program-0.hwx'], 'inputs': [['in0.bin']],
                  'refs': ['ref.bin']})

(kit / 'cases.json').write_text(json.dumps(cases, indent=1))
print('cases:', [c['name'] for c in cases])
