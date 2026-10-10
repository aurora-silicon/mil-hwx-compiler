#!/usr/bin/env python3
"""Each H14 elementwise MIL under tests/fixtures/h14 compiles to the exact ANEC bytes pinned here.

The pinned hashes are the hashes of the programs the omarchy-ane project checks in as H14 fixtures for the
M2 (fixtures/h14-anec/<op>/program-0.anec). The `add` program is proven on the device; the other seven are
compiled byte-identical to their fixtures and wait for a device run (see CHANGELOG.md for the status).
Usage: tests/test_h14_fixture_programs.py [path-to-mil-hwxc]
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
FIXTURES = ROOT / "tests/fixtures/h14"
MODEL_ROOT = ROOT / "tests/models/conv_relu"  # the programs have no BLOBFILE constants

EXPECTED = {
    "add": "b416b9d14059a0918f6b05bfcf4a8ba982dd92e7b25f2b042c1a15f1ab2a439a",
    "mul": "b32fef9f18eaf2a92bbef1ce14d92c37c430af70f08f6506b0993be9a1e3cc67",
    "relu": "429d991a07fcc8a1a4928390ff782c9351c855cd5e4299703b30799aa675e77d",
    "add-scalar": "f601839f49b8904c076c87a56b2dcc843bf98f8b6840b91aba438d41daf08f42",
    "mul-scalar": "a8c5a9441f5313df6a148d43486007205cd32c616c74cd0c0fc8b0d87a7ff998",
    "real-div-scalar": "05e3a0c1c60ba18079879a021abba6d7a8c71e18464a6def2ba20bf3aef18e25",
    "clip-low": "1b2c143310945eeebbb4ed8b2bf571d2d826f3b46988a43c3f417b94e7552329",
    "clip-high": "abe78b1292775e365a83e5e98c1aee7baed0eb83d0b165f7b559378755c40495",
}


def compile_h14(mil):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "package"
        subprocess.run(
            [str(COMPILER), "--mil", str(mil), "--model-root", str(MODEL_ROOT),
             "--target", "H14", "--output", str(out)],
            check=True, capture_output=True, text=True, timeout=scaled_timeout(60))
        programs = sorted(out.glob("program-*.anec"))
        assert len(programs) == 1, f"{mil.name}: expected one program, got {len(programs)}"
        return programs[0].read_bytes()


def main():
    names = sorted(path.stem for path in FIXTURES.glob("*.mil"))
    assert names == sorted(EXPECTED), f"fixture set {names} differs from the pinned set {sorted(EXPECTED)}"
    bad = 0
    for name in names:
        digest = hashlib.sha256(compile_h14(FIXTURES / f"{name}.mil")).hexdigest()
        if digest == EXPECTED[name]:
            print(f"PASS {name}")
        else:
            print(f"FAIL {name}: {digest} is not the pinned {EXPECTED[name]}")
            bad += 1
    print(f"{len(names) - bad} of {len(names)} H14 fixture programs match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
