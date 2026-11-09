#!/usr/bin/env python3
"""Build the M2 ladder and apply the H14 libane BAR overrun check."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / 'build/mil-hwxc').resolve()


def main():
    if importlib.util.find_spec('numpy') is None:
        print('SKIP H14 ring kit bind test (NumPy unavailable)')
        return
    with tempfile.TemporaryDirectory(prefix='h14-ring-kit-') as directory:
        out = Path(directory) / 'kit'
        subprocess.run([sys.executable,
                        str(ROOT / 'tools/h14-mobilenet/ring_kit.py'),
                        str(out), '--refs', '0'], check=True,
                       env=dict(os.environ, MILC=str(COMPILER)),
                       timeout=scaled_timeout(1200))
        manifest = json.loads((out / 'kit.json').read_text())
        assert manifest['libaneBindOverrunProgramsChecked'] == 35
        assert all((out / name).exists()
                   for name in ('run.c', 'run.py', 'run_kit.py'))
    print('PASS H14 linked-ring kit: 35 ANEC programs are bind-safe')


if __name__ == '__main__':
    main()
