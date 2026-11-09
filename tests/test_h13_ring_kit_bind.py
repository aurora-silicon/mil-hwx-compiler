#!/usr/bin/env python3
"""Every linked-ring device-kit ANEC passes libane's bind-overrun rule.

Usage: test_h13_ring_kit_bind.py [COMPILER]
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / 'build/mil-hwxc').resolve()


def main():
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / 'ring-kit'
        env = dict(os.environ, MILC=str(COMPILER))
        subprocess.run(
            [sys.executable, str(ROOT / 'tools/h13-mobilenet/ring_kit.py'),
             str(out), '--refs', '0'],
            check=True, env=env, timeout=scaled_timeout(600))
        manifest = json.loads((out / 'kit.json').read_text())
        if manifest.get('libaneBindOverrunProgramsChecked') != 34:
            raise SystemExit('ring kit did not check all 34 ANEC programs')
    print('PASS linked-ring kit: all 34 ANEC programs fit libane bind extents')


if __name__ == '__main__':
    main()
