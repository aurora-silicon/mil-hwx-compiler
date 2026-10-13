#!/usr/bin/env python3
"""Tests of compare-offline.py's exit status, on made-up 2 x 256 x 256 shapes. No device.

Usage: test-compare-offline.py [path-to-compare-offline.py]
Part 1 runs the checks against the script. Part 2 builds copies of the script with one known fault each and requires the checks to catch
every copy. A gate that passes a wrong, missing, padded, empty, NaN or infinite output, or any output when the exact result is zero, is a
faulty gate, so each of those must give a nonzero exit status. A valid output, including a correct all-zero output, must give 0."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "compare-offline.py"
M, K, N = 2, 256, 256
TILE = 16384


def make_shape(root, set_name, x_values, w_values, n=N):
    d = root / set_name / f"m{M}k{K}n{n}"
    d.mkdir(parents=True, exist_ok=True)
    x = np.zeros(1 << 17, dtype=np.float16)
    x[:M * K] = x_values
    x.tofile(d / "x.f16")
    np.asarray(w_values, dtype=np.float16).tofile(d / "w.raw.f16")
    return x[:M * K].astype(np.float64).reshape(M, K), np.asarray(w_values, dtype=np.float16).astype(np.float64).reshape(n, K)


def surface(values):
    s = np.zeros(TILE // 2, dtype=np.float16)
    s[:len(values)] = values
    return s


def run(script, set_dir, out_dir, results=None):
    cmd = [sys.executable, str(script), str(set_dir), str(out_dir)] + ([str(results)] if results else [])
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def checks(script):
    """Return a list of (name, passed) for one script."""
    out = []

    def check(name, ok):
        out.append((name, bool(ok)))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(1)
        xv = rng.uniform(-1, 1, M * K).astype(np.float16)
        wv = rng.uniform(-1, 1, (N, K)).astype(np.float16)
        x, w = make_shape(root, "set", xv, wv)
        good = surface((x @ w.T).reshape(-1).astype(np.float16))

        def put(name, arr, shape_name=f"m{M}k{K}n{N}"):
            d = root / name
            d.mkdir(exist_ok=True)
            arr.tofile(d / f"{shape_name}.cal.y.f16")
            return d

        r = run(script, root / "set", put("good", good))
        check("a valid output exits 0 and prints PASS", r.returncode == 0 and "PASS" in r.stdout)
        check("a wrong output (all zeros) exits nonzero", run(script, root / "set", put("zeros", np.zeros_like(good))).returncode != 0)
        (root / "empty").mkdir()
        r = run(script, root / "set", root / "empty")
        check("a missing output file exits nonzero", r.returncode != 0 and "MISSING" in r.stdout)
        (root / "nocase").mkdir()
        check("an empty set of shapes exits nonzero", run(script, root / "nocase", root / "good").returncode != 0)
        padded = good.copy()
        padded[M * N + 5] = np.float16(1.0)
        check("a nonzero padding lane exits nonzero", run(script, root / "set", put("padded", padded)).returncode != 0)
        nan = good.copy()
        nan[7] = np.float16("nan")
        check("a NaN lane exits nonzero", run(script, root / "set", put("nan", nan)).returncode != 0)
        inf = good.copy()
        inf[7] = np.float16("inf")
        check("an infinite lane exits nonzero", run(script, root / "set", put("inf", inf)).returncode != 0)
        short = good[:M * N - 4]
        check("a short output exits nonzero", run(script, root / "set", put("short", short)).returncode != 0)
        exact = int((good[:M * N].view(np.uint16) == good[:M * N].view(np.uint16)).sum())
        line = f"matvec M={M} K={K} N={N}: {M * N}/{M * N} lanes within 2 ulp or 4 cond-units (%d bit-exact), max 0.000 ulp, max 0.000 cond-units; padding lanes zero: PASS\n"
        (root / "differ.txt").write_text(line % (exact - 7))
        r = run(script, root / "set", root / "good", root / "differ.txt")
        check("printed ane-run counts that differ exit nonzero", r.returncode != 0 and "DIFFER" in r.stdout)

        # two shapes in one set: one has a valid output, the other has none (the set is not empty, so only the missing output can fail it)
        x2, w2 = make_shape(root, "twoset", xv, wv)
        wv2 = rng.uniform(-1, 1, (512, K)).astype(np.float16)
        make_shape(root, "twoset", xv, wv2, n=512)
        r = run(script, root / "twoset", put("one_of_two", good))
        check("one valid output and one missing output exits nonzero", r.returncode != 0 and "MISSING" in r.stdout)
        # ane-run printed counts for a shape that is not in the set at all
        (root / "ghost.txt").write_text(line.replace(f"M={M} K={K} N={N}", "M=8 K=256 N=256") % 1)  # only the ghost shape is printed
        r = run(script, root / "set", root / "good", root / "ghost.txt")
        check("printed ane-run counts for a shape without an output exit nonzero", r.returncode != 0)

        # zero input: every product is 0, the exact result is 0, and only a zero output is acceptable
        make_shape(root, "zset", np.zeros(M * K), np.ones((N, K)))
        zero_ok = surface(np.zeros(M * N))
        r = run(script, root / "zset", put("z_ok", zero_ok))
        check("a correct all-zero output for zero input exits 0", r.returncode == 0 and "PASS" in r.stdout)
        ones = surface(np.ones(M * N))
        r = run(script, root / "zset", put("z_ones", ones))
        check("an all-ones output for zero input exits nonzero", r.returncode != 0 and "FAIL" in r.stdout)
        nans = surface(np.full(M * N, np.nan))
        check("an all-NaN output for zero input exits nonzero", run(script, root / "zset", put("z_nan", nans)).returncode != 0)
        infs = surface(np.full(M * N, np.inf))
        check("an all-infinity output for zero input exits nonzero", run(script, root / "zset", put("z_inf", infs)).returncode != 0)
        short_zero = zero_ok[:M * N - 4]
        check("a short output for zero input exits nonzero", run(script, root / "zset", put("z_short", short_zero)).returncode != 0)
    return out


MUTANTS = {
    "failures do not count": ("sys.exit(1 if (bad or failed or missing or unmatched or not rows) else 0)", "sys.exit(1 if bad else 0)"),
    "missing outputs do not count": ("(bad or failed or missing or unmatched or not rows)", "(bad or failed or unmatched or not rows)"),
    "an empty set passes": ("(bad or failed or missing or unmatched or not rows)", "(bad or failed or missing or unmatched)"),
    "printed shapes without output are ignored": ("(bad or failed or missing or unmatched or not rows)", "(bad or failed or missing or not rows)"),
    "a zero condition sum counts as zero error":
        ("np.where(err == 0, 0.0, np.inf))", "0.0)"),
    "a short output counts as zeros": ("np.full(m * n, np.nan, dtype=np.float16)", "np.zeros(m * n, dtype=np.float16)"),
    "padding is not counted as a failure": ("status = \"PASS\" if in_band == lanes and pad == 0 else \"FAIL\"", "status = \"PASS\" if in_band == lanes else \"FAIL\""),
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
                print(f"FAIL mutant '{label}': the text to break was not found exactly once ({text.count(old)} times)")
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
