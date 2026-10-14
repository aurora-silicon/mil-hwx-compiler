#!/usr/bin/env python3
"""Judge the saved output of the exp chain cell off the device.

Usage: compare-expchain.py INDIR RUNDIR [--unary-dir DIR] [--sums FILE]
INDIR holds x_c<C>.f16 and k_c<C>.f16 (build-expchain.py writes them). RUNDIR holds results.txt and, for C in 64 and 512 and each stage
s1_e, s2_c, s3_d, s4_y, the files exp_c<C>.<stage>.cal.y.f16 and exp_c<C>.<stage>.long.y.f16 that run-expchain.sh kept. DIR is the unary
receipt directory whose compare-unary.py holds the exp reference and the preregistered bound; it is imported, not copied.
FILE (default SHA256SUMS-inputs beside this script) lists the sha256 of the six programs. The gate checks them against results.txt.
Numerical criteria, for each size (a missing file, size or size mismatch fails):
  0. The inputs are the registered ones: x is the saved input of the failed run, byte for byte; K is fp16 0x0b18 in every lane, zero padding.
  1. Each output has zero padding and no NaN or infinity, and its calibration and loop files are bit-identical.
  2. Stages s2, s3 and s4 equal the correctly rounded fp16 result of their own saved inputs, lane by lane: s2 = x * K, s3 = e * c, s4 = e + d,
     where e, c and d are the saved outputs of s1, s2 and s3 (the host only copied them). Zero compares equal to negative zero.
  3. The final output s4 meets the exp bound (relative 2^-8 plus the absolute floor of one fp16 subnormal, 2^-24) against exp(x) in float64, on
     every lane. The gate asserts that compare-unary.py is the pinned file (sha256) and that its exp bound and floor are those values. The
     reference is the correctly rounded fp16 result with ties to even, which is also what the stage checks of criterion 2 use.
Run identity, duty and time criteria, reported apart from the numerical ones and also needed for acceptance: results.txt names target_s=30 and the T6021 chip, no other
ane-run process before or after, the program hash of each stage as in SHA256SUMS-inputs, and the sha256 of every output file (they must match the files); results.txt says SUMMARY with fail=0 and no skipped
loop, each stage's calibration median is at most 0.60 ms, each loop's duty is at least 90 percent, the ticket duty is at least 90 percent
and the total wall time is at most 900 s. Exit status 0 only when the numerical and the timing criteria both hold."""
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANE_STRIDE = 32
SIZES = (64, 512)
STAGES = ("s1_e", "s2_c", "s3_d", "s4_y")
MAX_MEDIAN_MS, MIN_DUTY_PCT, MAX_WALL_S = 0.60, 90.0, 900.0
TARGET_S = 30  # the registered loop length; the run script prints the value it used
PIN_UNARY_SHA256 = "1c827f76cb07217cc2ec24350b8aac00917f1443ecf82c4108ba818b82a1adfd"  # receipts/2026-10-13-h14-unary-m2/compare-unary.py
PIN_X_SHA256 = {64: "6b29533c770b00d1a04a6bb5aeb6cd80f4fe9053b3ffd4f6aff37cd9f80cd230", 512: "f03fecef7615ca1417dbbeb8595c3602f9194553360ed8d308857fec5b8c7d19"}
CHIP = "t6021"
PROGRAM_OF_STAGE = {"s1_e": "exp", "s2_c": "mul", "s3_d": "mul", "s4_y": "add"}
K_BITS = 0x0B18  # fp16 0.00021648..., log2(e) / fp16(log2(e)) - 1, the relative error of the scale word 0x3dc5


def load_unary(unary_dir):
    spec = importlib.util.spec_from_file_location("cu", Path(unary_dir) / "compare-unary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def half(path):
    return np.fromfile(path, dtype=np.float16)


def lanes(surface, count):
    return surface[::LANE_STRIDE][:count]


def padding_nonzero(surface, count):
    mask = np.ones(len(surface), dtype=bool)
    mask[::LANE_STRIDE][:count] = False
    return int(np.count_nonzero(surface[mask].view(np.uint16)))


def rounded(values):
    return values.astype(np.float16)


def half_away(values):
    """Nearest fp16, ties away from zero. For an exact fp16 product or sum it differs from `rounded` only at exact midpoints."""
    h = values.astype(np.float16)
    lo = np.where(h.astype(np.float64) <= values, h, np.nextafter(h, np.float16(-np.inf))).astype(np.float64)
    hi = np.where(h.astype(np.float64) >= values, h, np.nextafter(h, np.float16(np.inf))).astype(np.float64)
    return np.where(values - lo < hi - values, lo, np.where(hi - values < values - lo, hi, np.where(np.abs(lo) > np.abs(hi), lo, hi)))


def check_size(c, indir, rundir, cu):
    problems, notes = [], []
    surfaces = {}
    for name in (f"x_c{c}.f16", f"k_c{c}.f16"):
        if not (indir / name).exists():
            return [f"missing input {name}"], notes
    x_s, k_s = half(indir / f"x_c{c}.f16"), half(indir / f"k_c{c}.f16")
    tile_halves = -(-c * 64 // 16384) * 16384 // 2
    if len(x_s) != tile_halves or len(k_s) != tile_halves:
        return [f"input surface is not {tile_halves} halves"], notes
    if hashlib.sha256(x_s.tobytes()).hexdigest() != PIN_X_SHA256[c]:
        problems.append("x differs from the pinned sha256")
    saved_x = Path(cu.__file__).resolve().parent / "inputs" / f"unary_exp_c{c}" / "x.f16"
    if not saved_x.exists() or saved_x.read_bytes() != x_s.tobytes():
        problems.append("x is not the saved input of the failed run")
    if not np.array_equal(lanes(k_s, c).view(np.uint16), np.full(c, K_BITS, dtype=np.uint16)) or padding_nonzero(k_s, c):
        problems.append(f"K is not fp16 0x{K_BITS:04x} in every lane with zero padding")
    for stage in STAGES:
        cal, long = rundir / f"exp_c{c}.{stage}.cal.y.f16", rundir / f"exp_c{c}.{stage}.long.y.f16"
        gone = [p.name for p in (cal, long) if not p.exists()]
        if gone:
            problems.append(f"{stage}: missing {', '.join(gone)}")
            continue
        a, b = half(cal), half(long)
        if len(a) != tile_halves:
            problems.append(f"{stage}: output has {len(a)} halves, not {tile_halves}")
            continue
        if a.tobytes() != b.tobytes():
            problems.append(f"{stage}: calibration and loop outputs differ")
        if not np.isfinite(lanes(a, c).astype(np.float64)).all():
            problems.append(f"{stage}: non-finite lane(s)")
        pad = padding_nonzero(a, c)
        if pad:
            problems.append(f"{stage}: {pad} nonzero padding value(s)")
        surfaces[stage] = a
    if len(surfaces) < len(STAGES):
        return problems, notes
    x, k = lanes(x_s, c).astype(np.float64), lanes(k_s, c).astype(np.float64)
    e, cc, d, y = (lanes(surfaces[s], c).astype(np.float64) for s in STAGES)
    for stage, got, want in (("s2_c", cc, x * k), ("s3_d", d, e * cc), ("s4_y", y, e + d)):
        bad = int(np.count_nonzero(got != rounded(want).astype(np.float64)))
        notes.append(f"{stage}: {c - bad} of {c} lanes equal the correctly rounded result")
        if bad:
            problems.append(f"{stage}: {bad} lane(s) differ from the correctly rounded result of the saved inputs")
            differ = got != rounded(want).astype(np.float64)
            midpoint = (half_away(want) != rounded(want).astype(np.float64)) & (got == half_away(want))
            notes.append(f"{stage}: of the {bad} differing lane(s), {int(np.count_nonzero(differ & midpoint))} are exact midpoints where the device gave "
                         "the half-away result (a rounding-rule difference, still a criterion 3 failure)")
    ref = cu.reference("exp", x, {})
    kind, bound = cu.BOUNDS["exp"]
    err = np.abs(y - ref)
    outside = int(np.count_nonzero(err > bound * np.abs(ref) + cu.FLOOR))
    rel = err / np.abs(ref)
    worst = int(np.argmax(rel))
    notes.append(f"final: max relative error {rel[worst]:.5f} at x={x[worst]:g}, bound {kind} {bound:g}, {outside} lane(s) outside")
    if outside:
        problems.append(f"final: {outside} lane(s) outside the exp bound")
    return problems, notes


def check_timing(results_path, rundir, sums_path):
    problems = []
    if not results_path.exists():
        return ["results.txt missing"]
    text = results_path.read_text()
    if len(re.findall(r"^SUMMARY ", text, re.M)) != 1:
        return ["results.txt must hold exactly one SUMMARY line"]
    m = re.search(r"^SUMMARY pass=(\d+) fail=(\d+) skipped_loops=(\d+) total_wall_s=([0-9.]+) ticket_duty_pct=(-?[0-9.]+)", text, re.M)
    if not m:
        return ["no SUMMARY line in results.txt"]
    passed, failed, skipped, wall, duty = int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5])
    stages = len(SIZES) * len(STAGES)
    if failed or skipped or passed != stages:
        problems.append(f"SUMMARY pass={passed} fail={failed} skipped_loops={skipped}, expected pass={stages} fail=0 skipped_loops=0")
    if wall > MAX_WALL_S:
        problems.append(f"total wall time {wall} s exceeds {MAX_WALL_S:g} s")
    if duty < MIN_DUTY_PCT:
        problems.append(f"ticket duty {duty}% is below {MIN_DUTY_PCT:g}%")
    medians = re.findall(r"median_ms=([0-9.]+)", text)
    if len(medians) != stages:
        problems.append(f"{len(medians)} calibration medians, expected {stages}")
    problems += [f"median {v} ms exceeds {MAX_MEDIAN_MS:g} ms" for v in medians if float(v) > MAX_MEDIAN_MS]
    loops = re.findall(r"loop_duty_pct=(-?[0-9.]+)", text)
    if len(loops) != stages:
        problems.append(f"{len(loops)} loop duty values, expected {stages}")
    problems += [f"loop duty {v}% is below {MIN_DUTY_PCT:g}%" for v in loops if float(v) < MIN_DUTY_PCT]
    target = re.search(r"target_s=(\d+)", text)
    if not target or int(target.group(1)) != TARGET_S:
        problems.append(f"target_s is {target.group(1) if target else 'missing'}, registered {TARGET_S}")
    # which run this is
    chip = re.search(r"^chip=(\S*)", text, re.M)
    if not chip or CHIP not in chip.group(1).lower():
        problems.append(f"chip line does not name {CHIP}")
    others = dict(re.findall(r"^other_ane_run_(before|after)=(\S+)", text, re.M))
    if others.get("before") != "0" or others.get("after") != "0":
        problems.append(f"other ane-run processes before/after: {others.get('before')}/{others.get('after')}, expected 0/0")
    sums = {}
    if Path(sums_path).exists():
        for line in Path(sums_path).read_text().splitlines():
            digest, _, name = line.partition("  ")
            sums[name] = digest
    seen = re.findall(r"^== exp_c(\d+)\.(s\d_[a-z]) program_sha256=([0-9a-f]+) ", text, re.M)
    if sorted((int(c), st) for c, st, _ in seen) != sorted((c, st) for c in SIZES for st in STAGES):
        problems.append("results.txt does not name the program hash of each of the eight stages once")
    for c, st, digest in seen:
        want = sums.get(f"prog/{PROGRAM_OF_STAGE.get(st, '?')}_c{c}.anec", "")
        if not want or not digest or not want.startswith(digest) or len(digest) < 16:
            problems.append(f"c{c} {st}: program hash {digest} does not match SHA256SUMS-inputs")
    hashes = dict(re.findall(r"^OUTPUT_SHA256 (\S+) ([0-9a-f]{64})", text, re.M))
    for c in SIZES:
        for st in STAGES:
            for run in ("cal", "long"):
                name = f"exp_c{c}.{st}.{run}.y.f16"
                path = rundir / name
                if name not in hashes:
                    problems.append(f"{name}: no OUTPUT_SHA256 line in results.txt")
                elif path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
                    problems.append(f"{name}: sha256 differs from results.txt")
    return problems


def main():
    args = sys.argv[1:]
    unary_dir = HERE.parent / "2026-10-13-h14-unary-m2"
    if "--unary-dir" in args:
        i = args.index("--unary-dir")
        unary_dir = Path(args[i + 1])
        del args[i:i + 2]
    sums_path = HERE / "SHA256SUMS-inputs"
    if "--sums" in args:
        i = args.index("--sums")
        sums_path = Path(args[i + 1])
        del args[i:i + 2]
    indir, rundir = Path(args[0]), Path(args[1])
    numeric = []
    unary_file = Path(unary_dir) / "compare-unary.py"
    if not unary_file.exists() or hashlib.sha256(unary_file.read_bytes()).hexdigest() != PIN_UNARY_SHA256:
        numeric.append("compare-unary.py is not the pinned file")
    cu = load_unary(unary_dir)
    if cu.BOUNDS.get("exp") != ("rel", 2.0 ** -8) or cu.FLOOR != 2.0 ** -24:
        numeric.append(f"the exp bound or floor in compare-unary.py is not rel 2^-8 plus 2^-24: {cu.BOUNDS.get('exp')}, {cu.FLOOR}")
    for line in numeric:
        print(f"FAIL: {line}")
    for c in SIZES:
        problems, notes = check_size(c, indir, rundir, cu)
        for line in notes:
            print(f"c{c} {line}")
        for line in problems:
            print(f"c{c} FAIL: {line}")
        numeric += problems
    timing = check_timing(rundir / "results.txt", rundir, sums_path)
    for line in timing:
        print(f"timing FAIL: {line}")
    print(f"numerical criteria: {'PASS' if not numeric else 'FAIL'}; run identity, duty and time criteria: {'PASS' if not timing else 'FAIL'}")
    return 1 if (numeric or timing) else 0


if __name__ == "__main__":
    sys.exit(main())
