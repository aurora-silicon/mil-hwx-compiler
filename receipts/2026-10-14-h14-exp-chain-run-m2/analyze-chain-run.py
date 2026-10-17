#!/usr/bin/env python3
"""Read-only analysis of the saved exp chain run. It judges nothing: compare-expchain.py holds the verdict (verdict.txt).

Usage: analyze-chain-run.py [ARCHIVE_DIR]
For every lane where a chain stage differs from the ties-to-even result it says which of three classes the lane is in: an exact midpoint where the
device gave the half-away result, a lane with a subnormal fp16 result, or other. Then it tests one model of the mul stage on the lanes with a
subnormal result: round the product to 11 significant bits as if the exponent range were unbounded, then round that to the subnormal grid with
ties away from zero. The model is fitted to these lanes after the run; it is a description of the saved output, not a prediction that was registered."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ce", HERE.parent / "2026-10-13-h14-exp-chain-m2/compare-expchain.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)
MIN_NORMAL, SUB_STEP = 2.0 ** -14, 2.0 ** -24


def lanes(path, c):
    return np.fromfile(path, dtype=np.float16)[::32][:c].astype(np.float64)


def away_grid(v, step):
    return np.sign(v) * np.floor(np.abs(v) / step + 0.5) * step


def sig11(v, tie_away):
    m, e = np.frexp(v)
    q = np.abs(m) * 2048
    r = np.floor(q + 0.5) if tie_away else np.round(q)
    return np.ldexp(np.sign(m) * r / 2048, e)


total, match, single_rne, single_away = 0, 0, 0, 0
for c in (64, 512):
    x, k = lanes(HERE / f"inputs/x_c{c}.f16", c), lanes(HERE / f"inputs/k_c{c}.f16", c)
    e, cc, d, y = (lanes(HERE / f"device-output/exp_c{c}.{s}.cal.y.f16", c) for s in ("s1_e", "s2_c", "s3_d", "s4_y"))
    for name, got, a, b, op in (("s2", cc, x, k, "*"), ("s3", d, e, cc, "*"), ("s4", y, e, d, "+")):
        want = a * b if op == "*" else a + b
        rne = want.astype(np.float16).astype(np.float64)
        away = ce.half_away(want)
        differ = got != rne
        mid = differ & (away != rne) & (got == away)
        sub = differ & ~mid & (np.abs(want) < MIN_NORMAL)
        print(f"c{c} {name}: {int(differ.sum())} lanes differ from ties-to-even: {int(mid.sum())} exact midpoints at the half-away result, "
              f"{int(sub.sum())} with a subnormal result (device larger in magnitude on {int((np.abs(got[sub]) > np.abs(rne[sub])).sum())}), "
              f"{int((differ & ~mid & ~sub).sum())} other")
        if op == "*":
            s = (np.abs(want) < MIN_NORMAL) & (want != 0)
            w, g = want[s], got[s]
            total += len(w)
            match += int((g == away_grid(sig11(w, True), SUB_STEP)).sum())
            single_rne += int((g == np.round(w / SUB_STEP) * SUB_STEP).sum())
            single_away += int((g == away_grid(w, SUB_STEP)).sum())
print(f"mul stages, lanes with a subnormal exact result: {total}")
print(f"  single rounding to the subnormal grid, ties to even: {single_rne} match; ties away: {single_away} match")
print(f"  11-significant-bit rounding then the subnormal grid with ties away: {match} match")
