#!/usr/bin/env python3
"""Compare saved device output of H14 unary programs with a float64 reference, off the device.

Usage: compare-unary.py SETDIR OUTDIR
SETDIR holds one directory per program, named unary_<op>_c<C> (from build-unary.py), each with x.f16: the input surface, where lane j is
one fp16 at half index j * 32 (one value per 64-byte channel). OUTDIR holds <case>.cal.y.f16 and <case>.long.y.f16, the files ane-run wrote
after the 20-call calibration and after the long loop. A case passes only when both files exist, are bit-identical to each other, hold no
NaN or infinity where the reference is finite, have zero padding, and every lane is within the per-operation bound in BOUNDS.
Exit status 0 only when every case in SETDIR passes and there is at least one case. Anything else, including a missing output, exits 1."""
import math
import re
import sys
from pathlib import Path

import numpy as np

LANE_STRIDE = 32  # half-index distance between lanes: one fp16 per 64-byte channel
FLOOR = 2.0 ** -24  # smallest fp16 subnormal, the absolute floor of a relative bound

# Per-operation bounds, fixed before the first device run. kind "abs": |error| <= bound. kind "rel": |error| <= bound * |reference| + FLOOR.
# kind "ulp": |error| <= bound * ulp(reference) of fp16.
BOUNDS = {
    "abs": ("abs", 0.0),
    "leaky_relu": ("ulp", 1.0),
    "sqrt": ("rel", 2.0 ** -9),
    "rsqrt": ("rel", 2.0 ** -9),
    "exp": ("rel", 2.0 ** -8),
    "sigmoid": ("abs", 2.0 ** -8),
    "tanh": ("abs", 2.0 ** -8),
    "silu": ("abs", 2.0 ** -6),
    "gelu": ("abs", 2.0 ** -6),
}
ALPHA = 0.125  # leaky_relu slope in the oracle MIL
ERF = np.vectorize(math.erf)


def reference(op, x):
    if op == "abs":
        return np.abs(x)
    if op == "leaky_relu":
        return np.where(x >= 0, x, ALPHA * x)
    if op == "sqrt":
        return np.sqrt(x)
    if op == "rsqrt":
        return 1.0 / np.sqrt(x)
    if op == "exp":
        return np.exp(x)
    if op == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if op == "tanh":
        return np.tanh(x)
    if op == "silu":
        return x / (1.0 + np.exp(-x))
    if op == "gelu":
        return 0.5 * x * (1.0 + ERF(x / math.sqrt(2.0)))
    raise KeyError(op)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def ulp(v):
    a = np.abs(v)
    _, e = np.frexp(a)
    return np.where((a == 0) | (e < -13), 2.0 ** -24, np.ldexp(1.0, e - 11))


def lanes_of(surface, count):
    return surface[::LANE_STRIDE][:count]


def padding_nonzero(surface, count):
    mask = np.ones(len(surface), dtype=bool)
    mask[::LANE_STRIDE][:count] = False
    return int(np.count_nonzero(surface[mask].view(np.uint16)))


def check_case(case_dir, out_dir):
    """Return (ok, line) for one program directory."""
    m = re.fullmatch(r"unary_([a-z_]+?)_c(\d+)", case_dir.name)
    if not m or m.group(1) not in BOUNDS:
        return False, f"{case_dir.name}: UNKNOWN operation (no bound registered)"
    op, count = m.group(1), int(m.group(2))
    cal, long = out_dir / f"{case_dir.name}.cal.y.f16", out_dir / f"{case_dir.name}.long.y.f16"
    gone = [p.name for p in (cal, long) if not p.exists()]
    if gone:
        return False, f"{case_dir.name}: MISSING output file(s) {', '.join(gone)}"
    x_surface = np.fromfile(case_dir / "x.f16", dtype=np.float16)
    x = lanes_of(x_surface, count).astype(np.float64)
    y_cal, y_long = np.fromfile(cal, dtype=np.float16), np.fromfile(long, dtype=np.float16)
    problems = []
    if len(y_cal) < count * LANE_STRIDE:
        return False, f"{case_dir.name}: output too short ({len(y_cal)} halves for {count} lanes)"
    if y_cal.tobytes() != y_long.tobytes():
        problems.append("calibration and long-loop outputs differ")
    ref = reference(op, x)
    want = ref.astype(np.float16).astype(np.float64)
    got = lanes_of(y_cal, count).astype(np.float64)
    finite = np.isfinite(got)
    if not finite.all() and np.isfinite(ref).all():
        problems.append(f"{int((~finite).sum())} non-finite lane(s)")
    err = np.abs(np.where(finite, got, np.inf) - ref)
    kind, bound = BOUNDS[op]
    tol = {"abs": np.full(count, bound), "rel": bound * np.abs(ref) + FLOOR, "ulp": bound * ulp(want)}[kind]
    out_of_band = int((err > tol).sum())
    pad = padding_nonzero(y_cal, count)
    if out_of_band:
        problems.append(f"{out_of_band} lane(s) outside the bound")
    if pad:
        problems.append(f"{pad} nonzero padding value(s)")
    exact = int((lanes_of(y_cal, count).view(np.uint16) == ref.astype(np.float16).view(np.uint16)).sum())
    worst = int(np.argmax(np.where(np.isfinite(err), err, np.inf)))
    rel = np.where(np.abs(ref) > 2.0 ** -6, err / np.where(ref == 0, 1, np.abs(ref)), 0.0)
    extra = ""
    if op == "gelu":
        extra = f", max abs error against the tanh form {float(np.max(np.abs(got - gelu_tanh(x)))):.5f}"
    status = "FAIL: " + "; ".join(problems) if problems else "PASS"
    return not problems, (f"{case_dir.name}: {count} lanes, {exact} bit-exact, max abs error {float(err[worst]):.5f} at x={x[worst]:g}, "
                          f"max relative error {float(np.max(rel)):.5f}, bound {kind} {bound:g}{extra}: {status}")


def main():
    set_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    cases = sorted(d for d in set_dir.glob("unary_*") if d.is_dir())
    failed = 0
    for case in cases:
        ok, line = check_case(case, out_dir)
        print(line)
        failed += not ok
    print(f"{len(cases)} cases, {failed} failed")
    return 1 if (failed or not cases) else 0


if __name__ == "__main__":
    sys.exit(main())
