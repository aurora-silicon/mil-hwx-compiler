#!/usr/bin/env python3
"""Tests of compare-offline.py's exit status, on a made-up 2 x 256 x 256 shape. No device.

Usage: test-compare-offline.py [path-to-compare-offline.py]
A comparator that passes a wrong output, a missing output, a nonzero padding lane or an empty shape set is a faulty gate, so each of
those must give a nonzero exit status. A valid output must give 0, and printed ane-run counts that differ must still give nonzero."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

script = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "compare-offline.py")
M, K, N = 2, 256, 256
TILE = 16384
fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += not ok


def run(set_dir, out_dir, results=None):
    cmd = [sys.executable, str(script), str(set_dir), str(out_dir)] + ([str(results)] if results else [])
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    shape = root / "set" / f"m{M}k{K}n{N}"
    shape.mkdir(parents=True)
    rng = np.random.default_rng(1)
    w = rng.uniform(-1, 1, (N, K)).astype(np.float16)
    x = np.zeros(1 << 17, dtype=np.float16)
    x[:M * K] = rng.uniform(-1, 1, M * K).astype(np.float16)
    w.tofile(shape / "w.raw.f16")
    x.tofile(shape / "x.f16")
    good = np.zeros(TILE // 2, dtype=np.float16)
    good[:M * N] = (x[:M * K].astype(np.float64).reshape(M, K) @ w.astype(np.float64).T).reshape(-1).astype(np.float16)

    def put(name, arr):
        d = root / name
        d.mkdir(exist_ok=True)
        arr.tofile(d / f"m{M}k{K}n{N}.cal.y.f16")
        return d

    r = run(root / "set", put("good", good))
    check("a valid output exits 0 and prints PASS", r.returncode == 0 and "PASS" in r.stdout)

    r = run(root / "set", put("wrong", np.zeros_like(good)))
    check("a wrong output (all zeros) exits nonzero", r.returncode != 0)

    (root / "empty").mkdir()
    r = run(root / "set", root / "empty")
    check("a missing output file exits nonzero", r.returncode != 0 and "missing" in (r.stdout + r.stderr).lower())

    (root / "none").mkdir()
    r = run(root / "none", root / "good")
    check("an empty set of shapes exits nonzero", r.returncode != 0)

    padded = good.copy()
    padded[M * N + 5] = np.float16(1.0)
    r = run(root / "set", put("padded", padded))
    check("a nonzero padding lane exits nonzero", r.returncode != 0)

    nan = good.copy()
    nan[3] = np.float16("nan")
    r = run(root / "set", put("nan", nan))
    check("a NaN lane exits nonzero", r.returncode != 0)

    line = f"matvec M={M} K={K} N={N}: {M * N}/{M * N} lanes within 2 ulp or 4 cond-units (%d bit-exact), max 0.000 ulp, max 0.000 cond-units; padding lanes zero: PASS\n"
    exact = int((good[:M * N].view(np.uint16) == good[:M * N].view(np.uint16)).sum())
    (root / "differ.txt").write_text(line % (exact - 7))
    r = run(root / "set", root / "good", root / "differ.txt")
    check("printed ane-run counts that differ exit nonzero", r.returncode != 0 and "DIFFER" in r.stdout)

print(f"failures: {fails}")
sys.exit(1 if fails else 0)
