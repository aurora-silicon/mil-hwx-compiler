#!/usr/bin/env python3
"""Tests of compare-unary.py's exit status, on made-up outputs. No device.

Usage: test-compare-unary.py [path-to-compare-unary.py]
Part 1 runs the checks against the script. Part 2 builds copies of the script with one known fault each (a failure that does not count,
a skipped missing output, ignored padding, ignored bound) and requires the checks to catch every copy. A gate whose tests cannot fail
is not a gate."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "compare-unary.py"
STRIDE = 32
COUNT = 64


def surface(values, tiles=1):
    s = np.zeros(tiles * 16384 // 2, dtype=np.float16)
    s[::STRIDE][:len(values)] = values
    return s


def run(script, set_dir, out_dir):
    return subprocess.run([sys.executable, str(script), str(set_dir), str(out_dir)], capture_output=True, text=True, check=False)


def checks(script):
    """Return a list of (name, passed) for one script."""
    results = []

    def check(name, ok):
        results.append((name, bool(ok)))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = root / "set" / "unary_silu_c64"
        case.mkdir(parents=True)
        x = np.random.default_rng(3).uniform(-8, 8, COUNT).astype(np.float16)
        surface(x).tofile(case / "x.f16")
        xd = x.astype(np.float64)
        good = surface((xd / (1 + np.exp(-xd))).astype(np.float16))

        def put(name, cal, long=None):
            d = root / name
            d.mkdir()
            cal.tofile(d / "unary_silu_c64.cal.y.f16")
            (long if long is not None else cal).tofile(d / "unary_silu_c64.long.y.f16")
            return d

        r = run(script, root / "set", put("good", good))
        check("a valid output exits 0 and prints PASS", r.returncode == 0 and "PASS" in r.stdout)
        r = run(script, root / "set", put("zeros", np.zeros_like(good)))
        check("a wrong output (all zeros) exits nonzero", r.returncode != 0)
        (root / "empty").mkdir()
        r = run(script, root / "set", root / "empty")
        check("a missing output exits nonzero", r.returncode != 0 and "MISSING" in r.stdout)
        half = root / "half"
        half.mkdir()
        good.tofile(half / "unary_silu_c64.cal.y.f16")
        r = run(script, root / "set", half)
        check("a missing long-loop output exits nonzero", r.returncode != 0)
        (root / "nocase").mkdir()
        r = run(script, root / "nocase", root / "good")
        check("an empty set of cases exits nonzero", r.returncode != 0)
        padded = good.copy()
        padded[COUNT * STRIDE + 7] = np.float16(1.0)
        r = run(script, root / "set", put("padded", padded))
        check("a nonzero padding value exits nonzero", r.returncode != 0)
        nan = good.copy()
        nan[5 * STRIDE] = np.float16("nan")
        r = run(script, root / "set", put("nan", nan))
        check("a NaN lane exits nonzero", r.returncode != 0)
        off = good.copy()
        off[9 * STRIDE] = np.float16(float(off[9 * STRIDE]) + 0.5)
        r = run(script, root / "set", put("off", off))
        check("one lane outside the bound exits nonzero", r.returncode != 0 and "outside the bound" in r.stdout)
        r = run(script, root / "set", put("differ", good, off))
        check("calibration and long-loop outputs that differ exit nonzero", r.returncode != 0)
        unknown = root / "set2" / "unary_frob_c64"
        unknown.mkdir(parents=True)
        surface(x).tofile(unknown / "x.f16")
        r = run(script, root / "set2", root / "good")
        check("an unknown operation exits nonzero", r.returncode != 0 and "UNKNOWN" in r.stdout)
    return results


MUTANTS = {
    "failures do not count": ("return 1 if (failed or not cases) else 0", "return 0"),
    "missing output is skipped": ('return False, f"{case_dir.name}: MISSING output file(s) {\', \'.join(gone)}"', 'return True, "skipped"'),
    "padding is ignored": ("    pad = padding_nonzero(y_cal, count)\n", "    pad = 0\n"),
    "the bound is ignored": ("    out_of_band = int((err > tol).sum())\n", "    out_of_band = 0\n"),
    "the long-loop output is never compared with the calibration output":
        ("    if y_cal.tobytes() != y_long.tobytes():\n", "    if False:\n"),
    "an empty set passes": ("return 1 if (failed or not cases) else 0", "return 1 if failed else 0"),
}


def main():
    bad = 0
    print(f"== checks against {SCRIPT.name}")
    for name, ok in checks(SCRIPT):
        print(("PASS " if ok else "FAIL ") + name)
        bad += not ok
    print("== mutants: each broken copy of the script must be caught by at least one check")
    text = SCRIPT.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        for label, (old, new) in MUTANTS.items():
            if text.count(old) != 1:
                print(f"FAIL mutant '{label}': the line to break was not found exactly once")
                bad += 1
                continue
            path = Path(tmp) / "mutant.py"
            path.write_text(text.replace(old, new, 1))
            caught = [n for n, ok in checks(path) if not ok]
            print(("PASS " if caught else "FAIL ") + f"mutant '{label}' " + (f"caught by: {caught[0]}" if caught else "was NOT caught"))
            bad += not caught
    print(f"failures: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
