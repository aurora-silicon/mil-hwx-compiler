#!/usr/bin/env python3
"""ANE2: compare device outputs (pulled to OUT) against float64 references.
usage: analyze.py KITROOT OUTROOT"""
import json, pathlib, re, sys
import numpy as np

K, O = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])


def ld(p, n=None):
    a = np.frombuffer(pathlib.Path(p).read_bytes(), np.float16).astype(np.float64)
    return a if n is None else a[:n]


def err(got, ref):
    got = got[:ref.size]
    bad = ~np.isfinite(got)
    d = got - ref
    rl2 = np.linalg.norm(d[~bad]) / max(np.linalg.norm(ref), 1e-30)
    ex = np.mean(np.float16(got) == np.float16(ref))
    return f'relL2 {rl2:.2e} maxabs {np.nanmax(np.abs(d)):.3g} exact {ex*100:.1f}% nonfinite {bad.sum()}'


def p50(log):
    m = re.search(r'p50 ([0-9.]+) us', pathlib.Path(log).read_text()) if pathlib.Path(log).exists() else None
    return f'p50 {m.group(1)} us' if m else 'no timing'


rows = []
for kit, out in (('kit1', 'o1'), ('kit3', 'o3')):
    if not (K / kit / 'cases.json').exists():
        continue
    for c in json.loads((K / kit / 'cases.json').read_text()):
        n = c['name']
        if c.get('chain'):
            continue
        sub = 'a'
        got_p = O / out / n / sub / 'output_00.bin'
        if kit == 'kit3':
            got_p = O / out / n / 'output_00.bin'
        if not got_p.exists():
            rows.append(f'{kit} {n}: NO OUTPUT'); continue
        refs = [K / kit / n / r for r in c['refs']]
        size = ld(refs[0]).size
        got = ld(got_p, size)
        res = ' | '.join(f'{r.stem}: {err(got, ld(r))}' for r in refs)
        log = (O / out / f'{n}.{sub}.log') if kit == 'kit1' else (O / out / f'{n}.log')
        rows.append(f'{kit} {n}: {res} | {p50(log)}')
# BERT head chain
b = K / 'kit1' / 'bert_head'; o = O / 'o1' / 'bert_head'
if (o / 'qk' / 'output_00.bin').exists():
    rows.append('bert_head p0 scaled: ' + err(ld(o / 'qk' / 'output_00.bin'), ld(b / 'ref_scaled.bin')) + ' | ' + p50(O / 'o1' / 'bert_head.qk.log'))
    sc = ld(o / 'qk' / 'output_00.bin').reshape(128, 128)
    e = np.exp(sc - sc.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True)
    if (o / 'sm' / 'output_00.bin').exists():
        rows.append('bert_head p1 softmax(dev scaled): ' + err(ld(o / 'sm' / 'output_00.bin'), p.ravel()) + ' | ' + p50(O / 'o1' / 'bert_head.sm.log'))
    for sub in ('pv', 'vp'):
        if (o / sub / 'output_00.bin').exists():
            pr = ld(o / 'sm' / 'output_00.bin').reshape(128, 128)
            v = ld(b / 'v.bin').reshape(128, 128)
            rows.append(f'bert_head p2 ({sub}) vs devprobs@v: ' + err(ld(o / sub / 'output_00.bin'), (pr @ v).ravel())
                        + ' | end-to-end vs ref_y: ' + err(ld(o / sub / 'output_00.bin'), ld(b / 'ref_y.bin')) + ' | ' + p50(O / 'o1' / f'bert_head.{sub}.log'))
# kit2
o2 = O / 'o2'
if (o2 / 'oproj' / 'output_00.bin').exists():
    ref = np.load(K / 'kit2' / 'ref.npy', allow_pickle=True).item()
    x = ld(K / 'kit2' / 'x.bin').reshape(128, 768)
    for i, n in enumerate('qkv'):
        got = ld(o2 / 'proj' / f'output_{i:02d}.bin').reshape(896, 896)[:128, :768]
        rows.append(f'kit2 proj {n}: ' + err(got.ravel(), ref['proj'][n].astype(np.float64).ravel()))
    out = ld(o2 / 'oproj' / 'output_00.bin').reshape(896, 896)[:128, :768]
    res = np.float16(x + out).astype(np.float64)
    rows.append('kit2 out-proj: ' + err(out.ravel(), ref['out'].astype(np.float64).ravel()))
    rows.append('kit2 BLOCK OUTPUT (x + attn): ' + err(res.ravel(), ref['res'].astype(np.float64).ravel()))
    for f in ('proj', 'qk', 'sm', 'pv', 'oproj'):
        rows.append(f'kit2 {f}: ' + p50(o2 / f'{f}.log'))
o4 = O / 'o4'
if (o4 / 'res' / 'output_00.bin').exists():
    ref = np.load(K / 'kit4' / 'ref.npy', allow_pickle=True).item()
    rows.append('kit4 ffn1 h: ' + err(ld(o4 / 'h.bin'), ref['h'].astype(np.float64).ravel()))
    rows.append('kit4 gelu: ' + err(ld(o4 / 'gelu' / 'output_00.bin', 128 * 3072), ref['gelu'].astype(np.float64).ravel()))
    got = ld(o4 / 'res' / 'output_00.bin', 128 * 768)
    rows.append('kit4 FFN BLOCK OUTPUT vs fp16-boundary ref: ' + err(got, ref['out'].astype(np.float64).ravel()))
    rows.append('kit4 FFN BLOCK OUTPUT vs exact float64: ' + err(got, ref['exact'].astype(np.float64).ravel()))
    for f in ('ffn1', 'gelu', 'ffn2', 's1', 's2', 's3', 'res'):
        rows.append(f'kit4 {f}: ' + p50(o4 / f'{f}.log'))
print('\n'.join(rows))
