#!/usr/bin/env python3
"""Compile the supplied ResNet model and enforce libane bind-overrun."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import libane_overrun  # noqa: E402


def main():
    model_value = os.environ.get('MILHWXC_RESNET50_MODEL')
    if not model_value:
        print('H13 ResNet-50 bind-overrun: SKIP (model env unset)')
        return
    model = Path(model_value)
    compiler = Path(sys.argv[1] if len(sys.argv) > 1 else
                    ROOT / 'build/mil-hwxc').resolve()
    with tempfile.TemporaryDirectory(prefix='h13-resnet50-bind-') as tmp:
        out = Path(tmp) / 'out'
        subprocess.run([
            str(compiler), '--target', 'H13', '--format', 'anec',
            '--schedule', 'linked-ring', '--mil', str(model / 'model.mil'),
            '--model-root', str(model), '--output', str(out)], check=True)
        failures = libane_overrun.overruns((out / 'program-0.anec').read_bytes())
        if failures:
            raise AssertionError(f'libane bind overruns: {failures}')
    print('H13 ResNet-50: all ANEC DMA extents fit channel allocations PASS')


if __name__ == '__main__':
    main()
