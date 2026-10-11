#!/usr/bin/env python3
"""Each H13 elementwise MIL under tests/fixtures/h13 compiles to the exact ANEC bytes pinned here.

Unlike the H14 test, these hashes pin this compiler's own output: the new H13 encoder's add, mul and mul-scalar
programs (4,736 bytes) are not byte-identical to the older 20,992-byte add fixture in omarchy-ane. mul and mul-scalar
are the same program (the scalar travels as a second operand surface). The test proves the output is stable and
nothing about device behavior; the device runs are in receipts/2026-10-11-h13-elementwise-m1/.
Usage: tests/test_h13_fixture_programs.py [path-to-mil-hwxc]
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
FIXTURES = ROOT / "tests/fixtures/h13"
MODEL_ROOT = ROOT / "tests/models/conv_relu"  # the programs have no BLOBFILE constants

EXPECTED = {
    "add": ("ac97b46653e5f0dd85ce0f45338b35aabd6049f9612f46ae756e4c224bd0bcb5", 4736),
    "mul": ("99f082c20036cfa2c75c4000c3bc828417245a437164745ec4b94345fe07bfc3", 4736),
    "relu": ("c7cc9781af1bd2c00583291a4cdaca7db90462674abcaa6011d2622d39f3902f", 20992),
    "add-scalar": ("44c9d1d5f4a1ce0a84862105e9d755574c60a0d79dee6feda0dd2655dcddaf72", 20992),
    "mul-scalar": ("99f082c20036cfa2c75c4000c3bc828417245a437164745ec4b94345fe07bfc3", 4736),
    "real-div-scalar": ("7920c55dfc4f17b1930b1f9b159d64141f0b3d85972a8e543be6a5d0f60cfc86", 5376),
    "clip-low": ("cfc9bee528fa9b458815eaa87cef8753f6dfeea265d169524caf05d2d472b3e7", 6784),
    "clip-high": ("1859b3760a246bbf158ddbfaaf2dc65da53d78ab5e776cc7c5af10cfb9fcbbe8", 6784),
}


def compile_h13(mil):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "package"
        subprocess.run(
            [str(COMPILER), "--mil", str(mil), "--model-root", str(MODEL_ROOT),
             "--target", "H13", "--output", str(out)],
            check=True, capture_output=True, text=True, timeout=scaled_timeout(60))
        programs = sorted(out.glob("program-*.anec"))
        assert len(programs) == 1, f"{mil.name}: expected one program, got {len(programs)}"
        return programs[0].read_bytes()


def main():
    names = sorted(path.stem for path in FIXTURES.glob("*.mil"))
    assert names == sorted(EXPECTED), f"fixture set {names} differs from the pinned set {sorted(EXPECTED)}"
    bad = 0
    for name in names:
        data = compile_h13(FIXTURES / f"{name}.mil")
        digest = hashlib.sha256(data).hexdigest()
        want, size = EXPECTED[name]
        if digest == want and len(data) == size:
            print(f"PASS {name}")
        else:
            print(f"FAIL {name}: {digest} ({len(data)} bytes) is not the pinned {want} ({size} bytes)")
            bad += 1
    print(f"{len(names) - bad} of {len(names)} H13 fixture programs match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
