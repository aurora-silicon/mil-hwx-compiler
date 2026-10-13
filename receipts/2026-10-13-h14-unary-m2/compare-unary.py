#!/usr/bin/env python3
"""Compare saved device output of H14 unary programs with a float64 reference, off the device.

Usage: compare-unary.py SETDIR OUTDIR [--oracles DIR]
The cases required are exactly those of unary_cases.expected_cases(DIR), the list build-unary.py builds (default DIR:
research/oracles/h14 of this repository). SETDIR must hold exactly one directory per case, each with x.f16 (the input surface: lane j
is one fp16 at half index j * 32, one value per 64-byte channel) and op.mil. OUTDIR holds <case>.cal.y.f16 and <case>.long.y.f16, the files
ane-run wrote after the 20-call calibration and after the long loop.
The reference is read from each case's own MIL: the operation, and its parameters (leaky_relu alpha, rsqrt epsilon, both fp32). The MIL in the
case directory must equal the oracle record's MIL, the text the program was built from. A parameter the gate does not know, or a missing one, fails the case.
A case passes only when both output files exist, are bit-identical to each other, hold no NaN or infinity, have zero padding, and every lane
meets the bound in BOUNDS (abs: the same bits as the reference, so -0.0 for +0.0 fails).
Exit status 0 only when the case set is exactly the expected one and every case passes. A missing, extra or failing case, or an empty
expected set, exits 1."""
import hashlib
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unary_cases import CASE_NAME, expected_cases, oracle_mil  # noqa: E402

LANE_STRIDE = 32  # half-index distance between lanes: one fp16 per 64-byte channel
FLOOR = 2.0 ** -24  # smallest fp16 subnormal, the absolute floor of a relative bound

# Per-operation bounds, fixed before the first device run. kind "bits": the same fp16 bits as the correctly rounded reference.
# kind "abs": |error| <= bound. kind "rel": |error| <= bound * |reference| + FLOOR. kind "ulp": |error| <= bound * ulp(reference) of fp16.
BOUNDS = {
    "abs": ("bits", 0.0),
    "leaky_relu": ("ulp", 1.0),
    "sqrt": ("rel", 2.0 ** -9),
    "rsqrt": ("rel", 2.0 ** -9),
    "exp": ("rel", 2.0 ** -8),
    "sigmoid": ("abs", 2.0 ** -8),
    "tanh": ("abs", 2.0 ** -8),
    "silu": ("abs", 2.0 ** -6),
    "gelu": ("abs", 2.0 ** -6),
}
REQUIRED_PARAMS = {"leaky_relu": {"alpha"}, "rsqrt": {"epsilon"}}  # every other operation takes none
ALLOWED_STRINGS = {"gelu": {"mode": "EXACT"}}  # the reference is the erf form; no other string argument is accepted anywhere
ERF = np.vectorize(math.erf)
CALL = re.compile(r"=\s*([a-z_]+)\(([^\n]*?)\)\s*\[name")
PARAM = re.compile(r"(\w+)\s*=\s*fp32\(\s*([-+0-9.eE]+)\s*\)")
STRING_PARAM = re.compile(r'(\w+)\s*=\s*string\("([^"]*)"\)')


def parse_mil(text, op):
    """Return the parameters of the operation call in the MIL, as float32 values. Raise ValueError for anything unexpected."""
    m = CALL.search(text)
    if not m or m.group(1) != op:
        raise ValueError(f"the MIL does not call {op}")
    args = m.group(2)
    params = {name: float(np.float32(value)) for name, value in PARAM.findall(args)}
    strings = dict(STRING_PARAM.findall(args))
    rest = STRING_PARAM.sub("", PARAM.sub("", args))
    rest = re.sub(r"\bx\s*=\s*x\b", "", rest)
    if re.sub(r"[\s,]", "", rest):
        raise ValueError(f"unrecognised argument(s) in the MIL call: {rest.strip()!r}")
    if strings != ALLOWED_STRINGS.get(op, {}):
        raise ValueError(f"string arguments {strings} are not the accepted {ALLOWED_STRINGS.get(op, {})}")
    if set(params) != REQUIRED_PARAMS.get(op, set()):
        raise ValueError(f"parameters {sorted(params)} are not the expected {sorted(REQUIRED_PARAMS.get(op, set()))}")
    return params


def reference(op, x, params):
    if op == "abs":
        return np.abs(x)
    if op == "leaky_relu":
        return np.where(x >= 0, x, params["alpha"] * x)
    if op == "sqrt":
        return np.sqrt(x)
    if op == "rsqrt":
        return 1.0 / np.sqrt(x + params["epsilon"])
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


def check_case(case_dir, out_dir, oracle_dir):
    """Return (ok, line) for one program directory."""
    m = CASE_NAME.fullmatch(case_dir.name)
    if not m or m.group(1) not in BOUNDS:
        return False, f"{case_dir.name}: UNKNOWN operation (no bound registered)"
    op, count = m.group(1), int(m.group(2))
    cal, long = out_dir / f"{case_dir.name}.cal.y.f16", out_dir / f"{case_dir.name}.long.y.f16"
    gone = [p.name for p in (cal, long) if not p.exists()]
    if gone:
        return False, f"{case_dir.name}: MISSING output file(s) {', '.join(gone)}"
    oracle_mil_text = oracle_mil(oracle_dir, case_dir.name)
    if oracle_mil_text != (case_dir / "op.mil").read_text():
        return False, f"{case_dir.name}: FAIL: op.mil differs from the oracle record's MIL (the reference would not describe the program that ran)"
    try:
        params = parse_mil((case_dir / "op.mil").read_text(), op)
    except (OSError, ValueError) as error:
        return False, f"{case_dir.name}: FAIL: cannot read the operation from op.mil: {error}"
    x_bytes = (case_dir / "x.f16").read_bytes()
    x_surface = np.frombuffer(x_bytes, dtype=np.float16)
    x = lanes_of(x_surface, count).astype(np.float64)
    y_cal, y_long = np.fromfile(cal, dtype=np.float16), np.fromfile(long, dtype=np.float16)
    problems = []
    if len(y_cal) < count * LANE_STRIDE:
        return False, f"{case_dir.name}: output too short ({len(y_cal)} halves for {count} lanes)"
    if y_cal.tobytes() != y_long.tobytes():
        problems.append("calibration and long-loop outputs differ")
    ref = reference(op, x, params)
    got_half = lanes_of(y_cal, count)
    got = got_half.astype(np.float64)
    finite = np.isfinite(got)
    if not finite.all() and np.isfinite(ref).all():
        problems.append(f"{int((~finite).sum())} non-finite lane(s)")
    err = np.abs(np.where(finite, got, np.inf) - ref)
    kind, bound = BOUNDS[op]
    want_bits = ref.astype(np.float16).view(np.uint16)
    if kind == "bits":
        out_of_band = int((got_half.view(np.uint16) != want_bits).sum())
    else:
        tol = {"abs": np.full(count, bound), "rel": bound * np.abs(ref) + FLOOR, "ulp": bound * ulp(ref.astype(np.float16).astype(np.float64))}[kind]
        out_of_band = int((err > tol).sum())
    pad = padding_nonzero(y_cal, count)
    if out_of_band:
        problems.append(f"{out_of_band} lane(s) outside the bound")
    if pad:
        problems.append(f"{pad} nonzero padding value(s)")
    exact = int((got_half.view(np.uint16) == want_bits).sum())
    worst = int(np.argmax(np.where(np.isfinite(err), err, np.inf)))
    rel = np.where(np.abs(ref) > 2.0 ** -6, err / np.where(ref == 0, 1, np.abs(ref)), 0.0)
    extra = ""
    if op == "gelu":
        extra = f", max abs error against the tanh form {float(np.max(np.abs(got - gelu_tanh(x)))):.5f}"
    parameters = "".join(f", {k} {v:g}" for k, v in params.items())
    status = "FAIL: " + "; ".join(problems) if problems else "PASS"
    return not problems, (f"{case_dir.name}: {count} lanes, {exact} bit-exact, max abs error {float(err[worst]):.5f} at x={x[worst]:g}, "
                          f"max relative error {float(np.max(rel)):.5f}, bound {kind} {bound:g}{parameters}{extra}, "
                          f"input sha256 {hashlib.sha256(x_bytes).hexdigest()[:12]}: {status}")


def main():
    args = sys.argv[1:]
    oracles = Path(__file__).resolve().parents[2] / "research/oracles/h14"
    if "--oracles" in args:
        i = args.index("--oracles")
        oracles = Path(args[i + 1])
        del args[i:i + 2]
    set_dir, out_dir = Path(args[0]), Path(args[1])
    expected = expected_cases(oracles)
    present = sorted(d.name for d in set_dir.glob("unary_*") if d.is_dir())
    failed = 0
    for name in sorted(set(expected) - set(present)):
        print(f"{name}: MISSING case directory in SETDIR")
        failed += 1
    for name in sorted(set(present) - set(expected)):
        print(f"{name}: UNEXPECTED case directory (not in the expected case list)")
        failed += 1
    judged = 0
    for name in sorted(set(expected) & set(present)):
        ok, line = check_case(set_dir / name, out_dir, oracles)
        print(line)
        judged += 1
        failed += not ok
    print(f"{len(expected)} expected cases, {judged} judged, {failed} failed")
    return 1 if (failed or not expected) else 0


if __name__ == "__main__":
    sys.exit(main())
