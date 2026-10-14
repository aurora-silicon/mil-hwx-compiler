#!/usr/bin/env python3
"""Tests of compare-expchain.py's exit status on made-up run directories. No device.

Usage: test-compare-expchain.py [path-to-compare-expchain.py]
Part 1 runs the checks against the script. Part 2 builds copies of the script with one known fault each and requires the checks to catch
every copy. A gate whose tests cannot fail is not a gate. Each limit is bracketed closely: a value just inside passes and a value just outside fails.
The made-up run uses the saved device output of the failed exp program as stage s1 (real device lanes) and computes stages s2 to s4 in
numpy. That is a simulated device: a pass on it shows the gate accepts a well-formed run, not that a device run would pass."""
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "compare-expchain.py"
UNARY = HERE.parent / "2026-10-13-h14-unary-m2"
SUMS = HERE / "SHA256SUMS-inputs"
K_BITS = 0x0B18
STRIDE = 32
STAGES = ("s1_e", "s2_c", "s3_d", "s4_y")
PROG = {"s1_e": "exp", "s2_c": "mul", "s3_d": "mul", "s4_y": "add"}


def sums():
    return {name: digest for digest, _, name in (line.partition("  ") for line in SUMS.read_text().splitlines())}


def results_text(rundir):
    """A clean results.txt for the files in rundir: identity lines, eight stages, output hashes, SUMMARY."""
    s = sums()
    lines = ["H14 exp chain cell start=2026-10-14T00:00:00Z kernel=test target_s=30",
             "chip=apple,j414capple,t6021apple,arm-platform runner_sha256=943ae8c67598175a", "other_ane_run_before=0"]
    for c in (64, 512):
        for st in STAGES:
            name = f"exp_c{c}.{st}"
            lines += [f"\n== {name} program_sha256={s[f'prog/{PROG[st]}_c{c}.anec'][:16]} inputs=a b ==",
                      f"rc_cal({name})=0 median_ms=0.5", f"rc_long({name})=0 repeat=60000 wall_s=30.1 loop_duty_pct=99.6"]
    lines.append("\nother_ane_run_after=0")
    lines += [f"OUTPUT_SHA256 {f.name} {hashlib.sha256(f.read_bytes()).hexdigest()}" for f in sorted(rundir.glob("*.y.f16"))]
    lines.append("\nSUMMARY pass=8 fail=0 skipped_loops=0 total_wall_s=300.0 ticket_duty_pct=99.0")
    return "\n".join(lines) + "\n"


def fixture(root, corrected=True):
    """Write INDIR and RUNDIR for a clean simulated run; return their paths."""
    indir, rundir = root / "in", root / "run"
    indir.mkdir()
    rundir.mkdir()
    for c in (64, 512):
        x_bytes = (UNARY / "inputs" / f"unary_exp_c{c}" / "x.f16").read_bytes()
        (indir / f"x_c{c}.f16").write_bytes(x_bytes)
        k = np.zeros(len(x_bytes) // 2, dtype=np.float16)
        k[::STRIDE][:c] = np.array([K_BITS], dtype=np.uint16).view(np.float16)[0] if corrected else 0
        (indir / f"k_c{c}.f16").write_bytes(k.tobytes())
        e = np.fromfile(UNARY / "device-output" / f"unary_exp_c{c}.cal.y.f16", dtype=np.float16)
        write_chain(indir, rundir, c, e)
    (rundir / "results.txt").write_text(results_text(rundir))
    return indir, rundir


def write_chain(indir, rundir, c, e):
    """Write s1 = e and the s2 to s4 outputs computed from it with correctly rounded fp16 arithmetic."""
    x = np.fromfile(indir / f"x_c{c}.f16", dtype=np.float16).astype(np.float64)
    k = np.fromfile(indir / f"k_c{c}.f16", dtype=np.float16).astype(np.float64)
    cc = (x * k).astype(np.float16)
    d = (e.astype(np.float64) * cc.astype(np.float64)).astype(np.float16)
    y = (e.astype(np.float64) + d.astype(np.float64)).astype(np.float16)
    for stage, data in (("s1_e", e), ("s2_c", cc), ("s3_d", d), ("s4_y", y)):
        for run in ("cal", "long"):
            (rundir / f"exp_c{c}.{stage}.{run}.y.f16").write_bytes(data.tobytes())


def gate(script, indir, rundir, unary=UNARY):
    return subprocess.run([sys.executable, str(script), str(indir), str(rundir), "--unary-dir", str(unary), "--sums", str(SUMS)],
                          capture_output=True, text=True, check=False)


def edit(path, fn):
    data = np.fromfile(path, dtype=np.float16)
    fn(data)
    path.write_bytes(data.tobytes())


def tweak(rundir, old, new, count=1):
    """Change results.txt text; the output hashes in it stay as they are."""
    path = rundir / "results.txt"
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new, count))


def lane_near_zero(indir, c=512):
    x = np.fromfile(indir / f"x_c{c}.f16", dtype=np.float16)[::STRIDE][:c].astype(np.float64)
    return int(np.argmin(np.where(x == 0, np.inf, np.abs(x)))), x


def final_error(percent):
    """Setup: s1 at the lane nearest zero is exp(x) times (1 + percent/100); s2 to s4 are recomputed, so only the final bound can object."""
    def setup(indir, rundir):
        j, x = lane_near_zero(indir)

        def change(e):
            e[STRIDE * j] = np.float16(np.exp(x[j]) * (1 + percent / 100))
        e = np.fromfile(rundir / "exp_c512.s1_e.cal.y.f16", dtype=np.float16)
        change(e)
        write_chain(indir, rundir, 512, e)
    return setup


def both(rundir, c, stage, fn):
    for run in ("cal", "long"):
        edit(rundir / f"exp_c{c}.{stage}.{run}.y.f16", fn)


def rehash(rundir):
    """Make results.txt describe the files as they now are (so a test about one fault does not also trip the hash check)."""
    path = rundir / "results.txt"
    if not path.exists() or "SUMMARY" not in path.read_text():
        return
    text = re.sub(r"^OUTPUT_SHA256 .*\n", "", path.read_text(), flags=re.M)
    lines = "".join(f"OUTPUT_SHA256 {f.name} {hashlib.sha256(f.read_bytes()).hexdigest()}\n" for f in sorted(rundir.glob("*.y.f16")))
    path.write_text(text.replace("\nSUMMARY", "\n" + lines + "SUMMARY", 1) if "\nSUMMARY" in text else text + lines)


def checks(script):
    """Return a list of (label, passed)."""
    out = []

    def case(label, expect_ok, setup=None, corrected=True, needle=None, rehash_after=True, unary=UNARY):  # unary: a path, or a function returning one
        with tempfile.TemporaryDirectory() as tmp:
            indir, rundir = fixture(Path(tmp), corrected)
            if setup:
                setup(indir, rundir)
            if rehash_after:
                rehash(rundir)
            r = gate(script, indir, rundir, unary() if callable(unary) else unary)
        out.append((label, (r.returncode == 0) == expect_ok and (needle is None or needle in r.stdout)))

    # numerical
    case("a well-formed corrected run passes", True)
    case("the same run without the correction (K = 0) fails", False, corrected=False)
    case("K = 0 is refused as an input, not only by the bound", False, corrected=False, needle="K is not fp16")
    case("a different x file fails", False, lambda i, r: edit(i / "x_c512.f16", lambda d: d.__setitem__(STRIDE * 5, np.float16(1.5))), needle="saved input")
    case("an x file that differs only in padding fails", False, lambda i, r: edit(i / "x_c64.f16", lambda d: d.__setitem__(7, np.float16(1.0))), needle="saved input")
    case("a missing output file fails", False, lambda i, r: (r / "exp_c512.s3_d.long.y.f16").unlink(), needle="missing")
    case("a wrong-sized output fails", False, lambda i, r: (r / "exp_c64.s1_e.cal.y.f16").write_bytes(b"\0" * 100), needle="halves")
    case("calibration and loop outputs that differ fail", False,
         lambda i, r: edit(r / "exp_c512.s4_y.long.y.f16", lambda d: d.__setitem__(STRIDE * 3, d[STRIDE * 3] + np.float16(0.25))), needle="differ")
    case("nonzero padding fails", False, lambda i, r: both(r, 64, "s2_c", lambda d: d.__setitem__(7, np.float16(1.0))), needle="padding")
    case("a NaN in s1 fails", False, lambda i, r: both(r, 512, "s1_e", lambda d: d.__setitem__(STRIDE * 9, np.float16(np.nan))), needle="non-finite")
    case("a stage s3 lane off by one ulp fails the stage check", False,
         lambda i, r: both(r, 512, "s3_d", lambda d: d.__setitem__(STRIDE * 40, np.nextafter(d[STRIDE * 40], np.float16(np.inf)))), needle="s3_d")
    case("a stage s4 lane off by one ulp fails the stage check", False,
         lambda i, r: both(r, 512, "s4_y", lambda d: d.__setitem__(STRIDE * 40, np.nextafter(d[STRIDE * 40], np.float16(np.inf)))), needle="s4_y")
    case("x = 0 at lane 0: a nonzero c there fails the exact rule", False,
         lambda i, r: both(r, 512, "s2_c", lambda d: d.__setitem__(0, np.float16(2.0 ** -24))), needle="s2_c")
    # the final bound, bracketed: 0.391 percent
    case("a final error of about 0.30 percent passes", True, final_error(0.30))
    case("a final error of about 0.35 percent passes", True, final_error(0.35))
    case("a final error of about 0.45 percent fails on the final bound only", False, final_error(0.45), needle="outside the exp bound")
    case("a final error of about 1 percent fails on the final bound only", False, final_error(1.0), needle="outside the exp bound")

    def doctored(tmp_name, edit_text):
        def make():
            d = Path(tempfile.mkdtemp(prefix=tmp_name))
            for name in ("compare-unary.py", "unary_cases.py"):
                shutil.copy(UNARY / name, d / name)
            (d / "inputs").symlink_to(UNARY / "inputs")
            (d / "compare-unary.py").write_text(edit_text((d / "compare-unary.py").read_text()))
            return d
        return make

    def moved_unary():
        d = Path(tempfile.mkdtemp(prefix="cu-x"))
        for name in ("compare-unary.py", "unary_cases.py"):
            shutil.copy(UNARY / name, d / name)
        shutil.copytree(UNARY / "inputs", d / "inputs")
        edit(d / "inputs/unary_exp_c512/x.f16", lambda a: a.__setitem__(STRIDE * 5, np.float16(1.5)))
        return d

    def change_x_too(indir, rundir):
        edit(indir / "x_c512.f16", lambda a: a.__setitem__(STRIDE * 5, np.float16(1.5)))

    case("x and its saved copy changed together fail the sha256 pin", False, change_x_too, unary=moved_unary, needle="pinned sha256")
    case("a compare-unary.py that differs only by a comment fails the pin", False, unary=doctored("cu-comment", lambda t: t + "\n# harmless\n"), needle="pinned file")
    case("a compare-unary.py with a loosened exp bound fails (a 0.5 percent lane)", False, final_error(0.5),
         unary=doctored("cu-loose", lambda t: t.replace('"exp": ("rel", 2.0 ** -8)', '"exp": ("rel", 2.0 ** -7)')))
    # run identity
    case("a chip that is not T6021 fails", False, lambda i, r: tweak(r, "t6021", "t8103"), needle="chip")
    case("another ane-run process before the run fails", False, lambda i, r: tweak(r, "other_ane_run_before=0", "other_ane_run_before=1"), needle="other ane-run")
    case("another ane-run process after the run fails", False, lambda i, r: tweak(r, "other_ane_run_after=0", "other_ane_run_after=2"), needle="other ane-run")
    case("an unknown ane-run count fails", False, lambda i, r: tweak(r, "other_ane_run_before=0", "other_ane_run_before=unknown"), needle="other ane-run")
    case("a program hash that differs from SHA256SUMS-inputs fails", False,
         lambda i, r: tweak(r, sums()["prog/exp_c512.anec"][:16], "0123456789abcdef"), needle="program hash")
    case("a stage that ran the wrong program (s2 with the add hash) fails", False,
         lambda i, r: tweak(r, f"exp_c64.s2_c program_sha256={sums()['prog/mul_c64.anec'][:16]}", f"exp_c64.s2_c program_sha256={sums()['prog/add_c64.anec'][:16]}"), needle="program hash")
    case("a stage line missing from results.txt fails", False,
         lambda i, r: tweak(r, "== exp_c64.s4_y program_sha256", "== skipped.s4_y program_sha256"), needle="eight stages")
    case("a results.txt from another run (a hash edited) fails", False,
         lambda i, r: tweak(r, f"OUTPUT_SHA256 exp_c512.s4_y.cal.y.f16 {hashlib.sha256((r / 'exp_c512.s4_y.cal.y.f16').read_bytes()).hexdigest()}",
                            "OUTPUT_SHA256 exp_c512.s4_y.cal.y.f16 " + "0" * 64), rehash_after=False, needle="sha256 differs")
    case("an output with no hash line fails", False,
         lambda i, r: tweak(r, f"OUTPUT_SHA256 exp_c64.s3_d.long.y.f16 {hashlib.sha256((r / 'exp_c64.s3_d.long.y.f16').read_bytes()).hexdigest()}\n", ""),
         rehash_after=False, needle="no OUTPUT_SHA256")
    # duty and time, bracketed
    case("a median of 0.55 ms passes", True, lambda i, r: tweak(r, "median_ms=0.5", "median_ms=0.55"))
    case("a median of 0.65 ms fails", False, lambda i, r: tweak(r, "median_ms=0.5", "median_ms=0.65"), needle="median")
    case("a loop duty of 91 percent passes", True, lambda i, r: tweak(r, "loop_duty_pct=99.6", "loop_duty_pct=91.0"))
    case("a loop duty of 87 percent fails", False, lambda i, r: tweak(r, "loop_duty_pct=99.6", "loop_duty_pct=87.0"), needle="loop duty")
    case("a ticket duty of 91 percent passes", True, lambda i, r: tweak(r, "ticket_duty_pct=99.0", "ticket_duty_pct=91.0"))
    case("a ticket duty of 87 percent fails", False, lambda i, r: tweak(r, "ticket_duty_pct=99.0", "ticket_duty_pct=87.0"), needle="ticket duty")
    case("a total wall time of 890 s passes", True, lambda i, r: tweak(r, "total_wall_s=300.0", "total_wall_s=890.0"))
    case("a total wall time of 910 s fails", False, lambda i, r: tweak(r, "total_wall_s=300.0", "total_wall_s=910.0"), needle="wall time")
    case("a SUMMARY with a failure fails", False, lambda i, r: tweak(r, "pass=8 fail=0", "pass=7 fail=1"), needle="SUMMARY")
    case("a SUMMARY with a skipped loop and fail=0 fails", False, lambda i, r: tweak(r, "skipped_loops=0", "skipped_loops=1"), needle="SUMMARY")
    case("a SUMMARY with pass=7 and fail=0 fails", False, lambda i, r: tweak(r, "pass=8 fail=0", "pass=7 fail=0"), needle="SUMMARY")
    case("seven calibration medians fail", False, lambda i, r: tweak(r, "median_ms=0.5", "median=0.5"), needle="calibration medians")
    case("seven loop duty values fail", False, lambda i, r: tweak(r, "loop_duty_pct=99.6", "loop_duty=99.6"), needle="loop duty values")
    case("two SUMMARY lines fail", False, lambda i, r: (r / "results.txt").write_text((r / "results.txt").read_text() + "SUMMARY pass=8 fail=0 skipped_loops=0 total_wall_s=1.0 ticket_duty_pct=99.0\n"), needle="exactly one")
    case("a results.txt without SUMMARY fails", False, lambda i, r: (r / "results.txt").write_text("nothing\n"))
    case("a missing results.txt fails", False, lambda i, r: (r / "results.txt").unlink())
    return [c for c in out if c is not None]


def mutants():
    """(label, old text, new text) edits to compare-expchain.py."""
    limits = "MAX_MEDIAN_MS, MIN_DUTY_PCT, MAX_WALL_S = 0.60, 90.0, 900.0"
    return [
        ("final bound not enforced", "    if outside:\n        problems.append", "    if False:\n        problems.append"),
        ("final bound loosened 1.5 times", "err > bound * np.abs(ref) + cu.FLOOR", "err > 1.5 * bound * np.abs(ref) + cu.FLOOR"),
        ("final bound loosened twice", "err > bound * np.abs(ref) + cu.FLOOR", "err > 2 * bound * np.abs(ref) + cu.FLOOR"),
        ("compare-unary.py hash not pinned", "hashlib.sha256(unary_file.read_bytes()).hexdigest() != PIN_UNARY_SHA256", "False"),
        ("compare-unary.py neither pinned nor its bound asserted", [("hashlib.sha256(unary_file.read_bytes()).hexdigest() != PIN_UNARY_SHA256", "False"),
                                                                   ('if cu.BOUNDS.get("exp") != ("rel", 2.0 ** -8) or cu.FLOOR != 2.0 ** -24:', "if False:")]),
        ("stage check skipped", "        if bad:\n            problems.append(f\"{stage}: {bad} lane(s) differ", "        if False:\n            problems.append(f\"{stage}: {bad} lane(s) differ"),
        ("stage check against the wrong inputs (s3 from x)", '("s3_d", d, e * cc)', '("s3_d", d, x * cc)'),
        ("stage s4 against e + c", '("s4_y", y, e + d)', '("s4_y", y, e + cc)'),
        ("cal/long identity not checked", "        if a.tobytes() != b.tobytes():", "        if False:"),
        ("padding not checked", "        if pad:\n            problems.append(f\"{stage}: {pad}", "        if False:\n            problems.append(f\"{stage}: {pad}"),
        ("non-finite not checked", "        if not np.isfinite(lanes(a, c).astype(np.float64)).all():", "        if False:"),
        ("x not compared to the saved input", "    if not saved_x.exists() or saved_x.read_bytes() != x_s.tobytes():", "    if False:"),
        ("x hash not pinned", "    if hashlib.sha256(x_s.tobytes()).hexdigest() != PIN_X_SHA256[c]:", "    if False:"),
        ("K not checked", "    if not np.array_equal(lanes(k_s, c).view(np.uint16), np.full(c, K_BITS, dtype=np.uint16)) or padding_nonzero(k_s, c):", "    if False:"),
        ("missing files tolerated", "        if gone:\n            problems.append", "        if False:\n            problems.append"),
        ("wrong output size tolerated", "        if len(a) != tile_halves:", "        if False:"),
        ("zero lane compared with a tolerance", "bad = int(np.count_nonzero(got != rounded(want).astype(np.float64)))",
         "bad = int(np.count_nonzero(np.abs(got - rounded(want).astype(np.float64)) > 1e-3))"),
        ("median limit raised to 0.69", limits, limits.replace("0.60", "0.69")),
        ("duty limit lowered to 85", limits, limits.replace("90.0, 900", "85.0, 900")),
        ("wall limit raised to 1000", limits, limits.replace("900.0", "1000.0")),
        ("SUMMARY failure ignored", "    if failed or skipped or passed != stages:", "    if False:"),
        ("skipped loops ignored", "    if failed or skipped or passed != stages:", "    if failed or passed != stages:"),
        ("pass count not checked", "    if failed or skipped or passed != stages:", "    if failed or skipped:"),
        ("median count not checked", "    if len(medians) != stages:", "    if False:"),
        ("loop duty count not checked", "    if len(loops) != stages:", "    if False:"),
        ("loop duty not checked", '    problems += [f"loop duty {v}% is below', '    problems += [] and [f"loop duty {v}% is below'),
        ("ticket duty not checked", "    if duty < MIN_DUTY_PCT:", "    if False:"),
        ("only one SUMMARY not required", '    if len(re.findall(r"^SUMMARY ", text, re.M)) != 1:', "    if False:"),
        ("chip not checked", "    if not chip or CHIP not in chip.group(1).lower():", "    if False:"),
        ("other ane-run processes not checked", '    if others.get("before") != "0" or others.get("after") != "0":', "    if False:"),
        ("program hashes not checked", "        if not want or not digest or not want.startswith(digest) or len(digest) < 16:", "        if False:"),
        ("stage set not checked", "    if sorted((int(c), st) for c, st, _ in seen) != sorted((c, st) for c in SIZES for st in STAGES):", "    if False:"),
        ("output hashes not compared", "elif path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:", "elif False:"),
        ("missing output hash line tolerated", "                if name not in hashes:", "                if False:"),
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
    for label, *edits in mutants():
        pairs = edits[0] if isinstance(edits[0], list) else [(edits[0], edits[1])]
        broken, missing = text, [old for old, _ in pairs if text.count(old) != 1]
        if missing:
            print(f"FAIL mutant '{label}': the text to break was not found exactly once ({text.count(missing[0])} times)")
            failed += 1
            continue
        for old, new in pairs:
            broken = broken.replace(old, new)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compare-expchain.py"
            path.write_text(broken)
            caught = [lab for lab, ok in checks(path) if not ok]
        print(("PASS " if caught else "FAIL ") + f"mutant '{label}' " + (f"caught by: {caught[0]}" if caught else "was NOT caught"))
        failed += not caught
    print(f"failures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
