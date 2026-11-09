#!/usr/bin/env python3
"""Build every supplied ResNet ladder rung and enforce bind-overrun."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    model_value = os.environ.get('MILHWXC_RESNET50_MODEL')
    if not model_value:
        print('H13 ResNet-50 bind-overrun: SKIP (model env unset)')
        return
    model = Path(model_value)
    compiler = Path(sys.argv[1] if len(sys.argv) > 1 else
                    ROOT / 'build/mil-hwxc').resolve()
    with tempfile.TemporaryDirectory(prefix='h13-resnet50-bind-') as tmp:
        subprocess.run([
            sys.executable, str(ROOT / 'tools/h13-resnet50/kit.py'),
            str(model), str(Path(tmp) / 'kit'), '--compiler', str(compiler)],
            check=True)
    print('H13 ResNet-50: every compiler-built rung fits channel allocations PASS')


if __name__ == '__main__':
    main()
