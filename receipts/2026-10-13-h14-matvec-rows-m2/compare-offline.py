#!/usr/bin/env python3
"""Compare saved device output of H14 matvec runs with a float64 reference, off the device, using ane-run's own matvec rules.

Usage: compare-offline.py SETDIR OUTDIR [results.txt]
SETDIR holds the shape directories from build-rows.py (x.f16, w.raw.f16). OUTDIR holds mMkKnN.cal.y.f16, the output file that
ane-run wrote. Per lane: want = float64 sum of the fp16 products rounded to fp16, ties away from zero; a lane is in band when it is within
2 ulp of want or within 4 condition units (4 * 2^-11 * the sum of the absolute products). Padding lanes after the M * N lanes must be zero.
With results.txt, every shape that ane-run itself checked must give the same counts, which validates this comparator.
Exit status 0 only when every shape in SETDIR has an output, every shape passes, no padding lane is nonzero, and every printed count
is reproduced. Anything else exits 1."""
import re
import sys
from pathlib import Path

import numpy as np

set_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
printed = {}
if len(sys.argv) > 3:
    for line in Path(sys.argv[3]).read_text().splitlines():
        m = re.match(r"matvec M=(\d+) K=(\d+) N=(\d+): (\d+)/(\d+) lanes within 2 ulp or 4 cond-units \((\d+) bit-exact\), "
                     r"max ([\d.]+) ulp, max ([\d.]+) cond-units; padding lanes zero: PASS", line)
        if m:
            g = m.groups()
            printed[tuple(map(int, g[:3]))] = (int(g[3]), int(g[5]), float(g[6]), float(g[7]))


def round_half_away(acc):
    h = acc.astype(np.float16)
    hd = h.astype(np.float64)
    lo = np.where(hd < acc, h, np.nextafter(h, np.float16(-np.inf)))
    hi = np.where(hd < acc, np.nextafter(h, np.float16(np.inf)), h)
    tie = (hd != acc) & ((lo.astype(np.float64) + hi.astype(np.float64)) / 2 == acc)
    return np.where(tie, np.where(acc > 0, hi, lo), h).astype(np.float16)


def ulp(v):
    a = np.abs(v.astype(np.float64))
    _, e = np.frexp(a)
    return np.where((a == 0) | (e < -13), 2.0 ** -24, np.ldexp(1.0, e - 11))


rows = []
for d in sorted(set_dir.glob("m*k*n*")):
    m, k, n = map(int, re.match(r"m(\d+)k(\d+)n(\d+)", d.name).groups())
    y_file = out_dir / f"{d.name}.cal.y.f16"
    if not y_file.exists():
        continue
    x = np.fromfile(d / "x.f16", dtype=np.float16)[:m * k].astype(np.float64).reshape(m, k)
    w = np.fromfile(d / "w.raw.f16", dtype=np.float16).astype(np.float64).reshape(n, k)
    y = np.fromfile(y_file, dtype=np.float16)
    acc = x @ w.T
    sumabs = np.abs(x) @ np.abs(w).T
    want = round_half_away(acc.reshape(-1))
    got = y[:m * n]
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64)) / ulp(want)
    nerr = np.where(sumabs.reshape(-1) > 0, np.abs(got.astype(np.float64) - acc.reshape(-1)) / (sumabs.reshape(-1) * 2.0 ** -11), 0.0)
    in_band = int(((diff <= 2.0) | (nerr <= 4.0)).sum())
    exact = int((got.view(np.uint16) == want.view(np.uint16)).sum())
    pad = int(np.count_nonzero(y[m * n:].view(np.uint16)))
    worst_ulp = float(np.max(np.nan_to_num(diff, nan=np.inf)))
    worst_nerr = float(np.max(np.nan_to_num(nerr, nan=np.inf)))
    rows.append(((m, k, n), in_band, m * n, exact, round(worst_ulp, 3) if np.isfinite(worst_ulp) else worst_ulp,
                 round(worst_nerr, 3) if np.isfinite(worst_nerr) else worst_nerr, pad))

bad = 0
failed = 0
compared = {shape for shape, *_ in rows}
for (m, k, n), in_band, lanes, exact, max_ulp, max_nerr, pad in rows:
    status = "PASS" if in_band == lanes and pad == 0 else "FAIL"
    failed += status == "FAIL"
    note = ""
    if (m, k, n) in printed:
        p = printed[(m, k, n)]
        same = (p[0], p[1]) == (lanes, exact) and abs(p[2] - max_ulp) < 0.5 and abs(p[3] - max_nerr) < 0.0015
        note = " | ane-run's counts " + ("reproduced" if same else f"DIFFER (printed {p})")
        bad += not same
    print(f"M={m:<2} K={k:<4} N={n:<4} {in_band}/{lanes} lanes in band, {exact} bit-exact, max {max_ulp:g} ulp, "
          f"max {max_nerr} cond-units, padding nonzero {pad}: {status}{note}")
missing = [d.name for d in sorted(set_dir.glob("m*k*n*")) if not (out_dir / f"{d.name}.cal.y.f16").exists()]
for name in missing:
    print(f"{name}: MISSING output file {name}.cal.y.f16")
unmatched = [shape for shape in printed if shape not in compared]
for m, k, n in unmatched:
    print(f"M={m} K={k} N={n}: ane-run printed counts but there is no output to compare")
print(f"{len(rows)} shapes compared; {len(printed)} had ane-run counts to reproduce; {bad} mismatches; "
      f"{failed} failed; {len(missing)} missing outputs; {len(unmatched)} printed shapes without output")
sys.exit(1 if (bad or failed or missing or unmatched or not rows) else 0)
