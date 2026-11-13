#!/usr/bin/env python3
"""Build the M2 test sequence and apply the H14 libane BAR overrun check."""
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
        raise RuntimeError('H14 ring kit bind test requires NumPy')
    with tempfile.TemporaryDirectory(prefix='h14-ring-kit-') as directory:
        out = Path(directory) / 'kit'
        subprocess.run([sys.executable,
                        str(ROOT / 'tools/h14-mobilenet/ring_kit.py'),
                        str(out), '--refs', '0'], check=True,
                       env=dict(os.environ, MILC=str(COMPILER)),
                       timeout=scaled_timeout(1200))
        manifest = json.loads((out / 'kit.json').read_text())
        assert manifest['libaneBindOverrunProgramsChecked'] == 72
        assert manifest['refusedCases'] == {
            'singles/r6-L02': 'h14.linked-ring-outside-arena'}
        programs = {str(path.relative_to(out))
                    for path in out.rglob('program-0.anec')}
        expected = {
            f'prefixes/p{index:02d}/linked/program-0.anec'
            for index in range(1, 29)
        }
        expected.update({
            f'prefixes6/p{index:02d}/linked/program-0.anec'
            for index in range(1, 29)
        })
        expected.update({
            'full-relu/linked/program-0.anec',
            'full-relu6/linked/program-0.anec',
            'singles/stem-k3-s2-224/linked/program-0.anec',
            *(f'singles/dw-k3-s2-{size}/linked/program-0.anec'
              for size in (14, 28, 56, 112)),
            *(f'singles/r6-L{index:02d}/linked/program-0.anec'
              for index in (0, 1, 3, 5, 7, 13, 25, 26, 27)),
        })
        assert programs == expected
        refusal = (out / 'singles/r6-L02/refusal.txt').read_text()
        assert 'h14.linked-ring-outside-arena' in refusal
        assert '--schedule linked' in refusal
        assert (out / 'singles/r6-L02/refs.npz').exists()
        assert all((out / name).exists()
                   for name in ('run.c', 'run.py', 'run_kit.py',
                                'relu6-field-changes.json'))
    print('PASS H14 linked-ring kit: 72 ANEC programs are bind-safe; '
          'one over-arena standalone is refused')


if __name__ == '__main__':
    main()
