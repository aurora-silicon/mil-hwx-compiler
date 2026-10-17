#!/usr/bin/env python3
"""Detail behind the compare-unary.py verdict: lanes outside the bound, error by input range, gelu against both forms, and call times.

Usage: analyze-unary.py SETDIR OUTDIR RESULTS_TXT
Read-only analysis of saved device output. It judges nothing: compare-unary.py holds the verdict."""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cu", here / "compare-unary.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)

set_dir, out_dir, results = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

print("== lanes outside the bound")
details = {}
for case in sorted(d for d in set_dir.glob("unary_*") if d.is_dir()):
    op, count = re.fullmatch(r"unary_([a-z_]+?)_c(\d+)", case.name).groups()
    count = int(count)
    x = cu.lanes_of(np.fromfile(case / "x.f16", dtype=np.float16), count).astype(np.float64)
    got = cu.lanes_of(np.fromfile(out_dir / f"{case.name}.cal.y.f16", dtype=np.float16), count).astype(np.float64)
    params = cu.parse_mil((case / "op.mil").read_text(), op)
    ref = cu.reference(op, x, params)
    kind, bound = cu.BOUNDS[op]
    err = np.abs(got - ref)
    tol = {"abs": np.full(count, bound), "bits": np.full(count, 0.0), "rel": bound * np.abs(ref) + cu.FLOOR,
           "ulp": bound * cu.ulp(ref.astype(np.float16).astype(np.float64))}[kind]
    details[case.name] = (op, x, got, ref, err, tol)
    bad = np.flatnonzero(err > tol) if kind != "bits" else np.flatnonzero(err > 0)
    if len(bad):
        print(f"{case.name}: {len(bad)} lane(s) outside the bound ({kind} {bound:g})")
        for j in bad[np.argsort(-err[bad])][:12]:
            print(f"   x={x[j]:<10g} reference={ref[j]:<12.6g} device={got[j]:<12.6g} error={err[j]:.5g} relative={err[j] / abs(ref[j]):.5f}")

print("\n== error by input range (c512 programs), relative error for exp, sqrt, rsqrt and absolute error for the rest")
for case, (op, x, got, ref, err, _) in details.items():
    if not case.endswith("_c512") or op in ("abs", "leaky_relu"):
        continue
    rel = op in ("exp", "sqrt", "rsqrt")
    metric = err / np.where(ref == 0, 1.0, np.abs(ref)) if rel else err  # a zero reference has zero error here, not 0/0
    edges = np.linspace(x.min(), x.max(), 5)
    pieces = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= lo) & (x <= hi)
        pieces.append(f"[{lo:.2f},{hi:.2f}] max {metric[m].max():.5f}" if m.any() else f"[{lo:.2f},{hi:.2f}] none")
    print(f"{case:22} {'relative' if rel else 'absolute'}: " + "; ".join(pieces))

print("\n== gelu against the erf form and the tanh form (the device is judged against the erf form by the gate)")
for case, (op, x, got, ref, err, _) in details.items():
    if op != "gelu":
        continue
    tanh_form = cu.gelu_tanh(x)
    print(f"{case}: erf form  mean abs {np.mean(err):.5f}  rms {np.sqrt(np.mean(err ** 2)):.5f}  max {np.max(err):.5f}")
    e2 = np.abs(got - tanh_form)
    print(f"{' ' * len(case)}  tanh form mean abs {np.mean(e2):.5f}  rms {np.sqrt(np.mean(e2 ** 2)):.5f}  max {np.max(e2):.5f}")
    print(f"{' ' * len(case)}  lanes closer to the erf form: {int((err < e2).sum())}, closer to the tanh form: {int((e2 < err).sum())}, equal: {int((err == e2).sum())}")

print("\n== call times and duty (from results.txt; this is not part of the numerical verdict)")
shape = None
rows = []
for ln in results.read_text().splitlines():
    m = re.match(r"== (unary_\w+) ", ln)
    if m:
        shape = m.group(1)
    m = re.match(r"exec ms over (\d+) calls: min ([\d.]+) p10 [\d.]+ p25 [\d.]+ median ([\d.]+)", ln)
    if m and int(m.group(1)) > 1000:
        rows.append([shape, int(m.group(1)), m.group(3)])
    m = re.match(r"rc_long\((\w+)\)=(\d+) repeat=(\d+) wall_s=([\d.]+) loop_duty_pct=([\d.]+)", ln)
    if m:
        rows[-1] += [m.group(4), m.group(5)]
for r in rows:
    print(f"{r[0]:24} {r[1]:>6} calls  median {r[2]} ms  loop {r[3]} s  duty {r[4]} %")
meds = [float(r[2]) for r in rows]
print(f"medians span {min(meds)} to {max(meds)} ms over {len(rows)} loops; loop duty {min(float(r[4]) for r in rows)} to {max(float(r[4]) for r in rows)} %")
