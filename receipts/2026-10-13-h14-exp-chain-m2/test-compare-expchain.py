#!/usr/bin/env python3
"""Tests of compare-expchain.py's exit status on made-up run directories. No device.

Usage: test-compare-expchain.py [path-to-compare-expchain.py]
Part 1 runs the checks against the script. Part 2 builds copies of the script with one known fault each and requires the checks to catch
every copy. A gate whose tests cannot fail is not a gate.
The made-up run uses the saved device output of the failed exp program as stage s1 (real device lanes) and computes stages s2 to s4 in
numpy. That is a simulated device: a pass on it shows the gate accepts a well-formed run, not that a device run would pass."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "compare-expchain.py"
UNARY = HERE.parent / "2026-10-13-h14-unary-m2"
K_BITS = 0x0B18
STRIDE = 32
GOOD_RESULTS = "".join(
    [f"rc_cal(s{i})=0 median_ms=0.5\nrc_long(s{i})=0 repeat=60000 wall_s=30.1 loop_duty_pct=99.6\n" for i in range(8)]
    + ["\nSUMMARY pass=8 fail=0 skipped_loops=1".replace("skipped_loops=1", "skipped_loops=0") + " total_wall_s=300.0 ticket_duty_pct=99.0\n"])


def fixture(root, corrected=True):
    """Write INDIR and RUNDIR for a clean simulated run; return their paths."""
    indir, rundir = root / "in", root / "run"
    indir.mkdir()
    rundir.mkdir()
    for c in (64, 512):
        x_bytes = (UNARY / "inputs" / f"unary_exp_c{c}" / "x.f16").read_bytes()
        (indir / f"x_c{c}.f16").write_bytes(x_bytes)
        halves = len(x_bytes) // 2
        k = np.zeros(halves, dtype=np.float16)
        k[::STRIDE][:c] = np.array([K_BITS], dtype=np.uint16).view(np.float16)[0] if corrected else 0
        (indir / f"k_c{c}.f16").write_bytes(k.tobytes())
        x = np.frombuffer(x_bytes, dtype=np.float16)
        e = np.fromfile(UNARY / "device-output" / f"unary_exp_c{c}.cal.y.f16", dtype=np.float16)
        cc = (x.astype(np.float64) * k.astype(np.float64)).astype(np.float16)
        d = (e.astype(np.float64) * cc.astype(np.float64)).astype(np.float16)
        y = (e.astype(np.float64) + d.astype(np.float64)).astype(np.float16)
        for stage, data in (("s1_e", e), ("s2_c", cc), ("s3_d", d), ("s4_y", y)):
            for run in ("cal", "long"):
                (rundir / f"exp_c{c}.{stage}.{run}.y.f16").write_bytes(data.tobytes())
    (rundir / "results.txt").write_text(GOOD_RESULTS)
    return indir, rundir


def rechain(indir, rundir, c, e_edit):
    """Change s1 (the exp output) and recompute s2 to s4 from it, so only the final bound can object."""
    x = np.fromfile(indir / f"x_c{c}.f16", dtype=np.float16).astype(np.float64)
    k = np.fromfile(indir / f"k_c{c}.f16", dtype=np.float16).astype(np.float64)
    e = np.fromfile(rundir / f"exp_c{c}.s1_e.cal.y.f16", dtype=np.float16)
    e_edit(e)
    cc = (x * k).astype(np.float16)
    d = (e.astype(np.float64) * cc.astype(np.float64)).astype(np.float16)
    y = (e.astype(np.float64) + d.astype(np.float64)).astype(np.float16)
    for stage, data in (("s1_e", e), ("s2_c", cc), ("s3_d", d), ("s4_y", y)):
        for run in ("cal", "long"):
            (rundir / f"exp_c{c}.{stage}.{run}.y.f16").write_bytes(data.tobytes())


def gate(script, indir, rundir):
    return subprocess.run([sys.executable, str(script), str(indir), str(rundir), "--unary-dir", str(UNARY)], capture_output=True, text=True, check=False)


def edit(path, fn):
    data = np.fromfile(path, dtype=np.float16)
    fn(data)
    path.write_bytes(data.tobytes())


def checks(script):
    """Return a list of (label, passed)."""
    out = []

    def case(label, expect_ok, setup=None, corrected=True, needle=None):
        with tempfile.TemporaryDirectory() as tmp:
            indir, rundir = fixture(Path(tmp), corrected)
            if setup:
                setup(indir, rundir)
            r = gate(script, indir, rundir)
        ok = (r.returncode == 0) == expect_ok and (needle is None or needle in r.stdout)
        out.append((label, ok))

    case("a well-formed corrected run passes", True)
    case("the same run without the correction (K = 0) fails on the final bound", False, corrected=False)
    case("K = 0 is refused as an input, not only by the bound", False, corrected=False, needle="K is not fp16")
    case("a different x file fails", False, lambda i, r: edit(i / "x_c512.f16", lambda d: d.__setitem__(STRIDE * 5, np.float16(1.5))), needle="saved input")
    case("an x file that differs only in padding fails", False, lambda i, r: edit(i / "x_c64.f16", lambda d: d.__setitem__(7, np.float16(1.0))), needle="saved input")
    case("an exp output off by 1 percent, every later stage consistent, fails on the final bound only", False,
         lambda i, r: rechain(i, r, 512, lambda e: e.__setitem__(STRIDE * 40, e[STRIDE * 40] * np.float16(1.01))), needle="outside the exp bound")
    case("an exp output off by 0.3 percent, every later stage consistent, passes (inside the bound)", True,
         lambda i, r: rechain(i, r, 512, lambda e: e.__setitem__(STRIDE * 40, np.float16(float(e[STRIDE * 40]) * 1.002))))
    case("a missing output file fails", False, lambda i, r: (r / "exp_c512.s3_d.long.y.f16").unlink(), needle="missing")
    case("a wrong-sized output fails", False, lambda i, r: (r / "exp_c64.s1_e.cal.y.f16").write_bytes(b"\0" * 100), needle="halves")
    case("calibration and loop outputs that differ fail", False,
         lambda i, r: edit(r / "exp_c512.s4_y.long.y.f16", lambda d: d.__setitem__(STRIDE * 3, d[STRIDE * 3] + np.float16(0.25))), needle="differ")
    case("nonzero padding fails", False,
         lambda i, r: [edit(r / f"exp_c64.s2_c.{k}.y.f16", lambda d: d.__setitem__(7, np.float16(1.0))) for k in ("cal", "long")], needle="padding")
    case("a NaN in s1 fails", False,
         lambda i, r: [edit(r / f"exp_c512.s1_e.{k}.y.f16", lambda d: d.__setitem__(STRIDE * 9, np.float16(np.nan))) for k in ("cal", "long")], needle="non-finite")
    case("a stage s3 lane off by one ulp fails the stage check", False,
         lambda i, r: [edit(r / f"exp_c512.s3_d.{k}.y.f16", lambda d: d.__setitem__(STRIDE * 40, np.nextafter(d[STRIDE * 40], np.float16(np.inf)))) for k in ("cal", "long")],
         needle="s3_d")
    case("a final lane outside the bound fails (stage checks consistent)", False,
         lambda i, r: [edit(r / f"exp_c512.s4_y.{k}.y.f16", lambda d: d.__setitem__(STRIDE * 40, d[STRIDE * 40] * np.float16(1.01))) for k in ("cal", "long")], needle="s4_y")
    case("x = 0 at lane 0: a nonzero c there fails the exact rule", False,
         lambda i, r: [edit(r / f"exp_c512.s2_c.{k}.y.f16", lambda d: d.__setitem__(0, np.float16(2.0 ** -24))) for k in ("cal", "long")], needle="s2_c")
    case("a median above 0.60 ms fails", False, lambda i, r: (r / "results.txt").write_text(GOOD_RESULTS.replace("median_ms=0.5", "median_ms=0.7", 1)), needle="median")
    case("a loop duty below 90 percent fails", False, lambda i, r: (r / "results.txt").write_text(GOOD_RESULTS.replace("99.6", "80.0", 1)), needle="loop duty")
    case("a ticket duty below 90 percent fails", False, lambda i, r: (r / "results.txt").write_text(GOOD_RESULTS.replace("ticket_duty_pct=99.0", "ticket_duty_pct=50.0")), needle="ticket duty")
    case("a total wall time above 900 s fails", False, lambda i, r: (r / "results.txt").write_text(GOOD_RESULTS.replace("300.0", "901.0")), needle="wall time")
    case("a SUMMARY with a failure fails", False, lambda i, r: (r / "results.txt").write_text(GOOD_RESULTS.replace("pass=8 fail=0", "pass=7 fail=1")), needle="SUMMARY")
    case("a results.txt without SUMMARY fails", False, lambda i, r: (r / "results.txt").write_text("nothing\n"))
    case("a missing results.txt fails", False, lambda i, r: (r / "results.txt").unlink())
    return out


def mutants():
    """(label, old text, new text) edits to compare-expchain.py; the bound itself lives in compare-unary.py, so these target the gate's own logic."""
    return [
        ("final bound not enforced", "    if outside:\n        problems.append", "    if False:\n        problems.append"),
        ("final bound loosened tenfold", "err > bound * np.abs(ref) + cu.FLOOR", "err > 10 * bound * np.abs(ref) + cu.FLOOR"),
        ("stage check skipped", "        if bad:\n            problems.append(f\"{stage}: {bad} lane(s) differ", "        if False:\n            problems.append(f\"{stage}: {bad} lane(s) differ"),
        ("stage check against the wrong inputs (s3 from x)", '("s3_d", d, e * cc)', '("s3_d", d, x * cc)'),
        ("cal/long identity not checked", "        if a.tobytes() != b.tobytes():", "        if False:"),
        ("padding not checked", "        if pad:\n            problems.append(f\"{stage}: {pad}", "        if False:\n            problems.append(f\"{stage}: {pad}"),
        ("non-finite not checked", "        if not np.isfinite(lanes(a, c).astype(np.float64)).all():", "        if False:"),
        ("x not compared to the saved input", "    if not saved_x.exists() or saved_x.read_bytes() != x_s.tobytes():", "    if False:"),
        ("K not checked", "    if not np.array_equal(lanes(k_s, c).view(np.uint16), np.full(c, K_BITS, dtype=np.uint16)) or padding_nonzero(k_s, c):", "    if False:"),
        ("missing files tolerated", "        if gone:\n            problems.append", "        if False:\n            problems.append"),
        ("wrong output size tolerated", "        if len(a) != tile_halves:", "        if False:"),
        ("zero lane compared with a tolerance", "bad = int(np.count_nonzero(got != rounded(want).astype(np.float64)))",
         "bad = int(np.count_nonzero(np.abs(got - rounded(want).astype(np.float64)) > 1e-3))"),
        ("median limit raised", "MAX_MEDIAN_MS, MIN_DUTY_PCT, MAX_WALL_S = 0.60, 90.0, 900.0", "MAX_MEDIAN_MS, MIN_DUTY_PCT, MAX_WALL_S = 6.0, 90.0, 900.0"),
        ("duty limit lowered", "MAX_MEDIAN_MS, MIN_DUTY_PCT, MAX_WALL_S = 0.60, 90.0, 900.0", "MAX_MEDIAN_MS, MIN_DUTY_PCT, MAX_WALL_S = 0.60, 10.0, 900.0"),
        ("wall limit raised", "MAX_MEDIAN_MS, MIN_DUTY_PCT, MAX_WALL_S = 0.60, 90.0, 900.0", "MAX_MEDIAN_MS, MIN_DUTY_PCT, MAX_WALL_S = 0.60, 90.0, 9000.0"),
        ("SUMMARY failure ignored", "    if failed or skipped or passed != 2 * len(SIZES) * 2:", "    if False:"),
        ("timing result ignored in the exit status", "return 1 if (numeric or timing) else 0", "return 1 if numeric else 0"),
        ("numerical result ignored in the exit status", "return 1 if (numeric or timing) else 0", "return 1 if timing else 0"),
    ]


def main():
    print("== checks against the script")
    results = checks(SCRIPT)
    failed = 0
    for label, ok in results:
        print(("PASS " if ok else "FAIL ") + label)
        failed += not ok
    print("== mutants: each broken copy of the script must be caught by at least one check")
    text = SCRIPT.read_text()
    for label, old, new in mutants():
        if text.count(old) != 1:
            print(f"FAIL mutant '{label}': the text to break was not found exactly once ({text.count(old)} times)")
            failed += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compare-expchain.py"
            path.write_text(text.replace(old, new))
            caught = [lab for lab, ok in checks(path) if not ok]
        print(("PASS " if caught else "FAIL ") + f"mutant '{label}' " + (f"caught by: {caught[0]}" if caught else "was NOT caught"))
        failed += not caught
    print(f"failures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
