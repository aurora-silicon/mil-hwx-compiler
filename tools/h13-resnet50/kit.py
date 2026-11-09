#!/usr/bin/env python3
"""Build the safe full-network H13 ResNet-50 ANECOMP device kit."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import libane_overrun  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compiler', default=os.environ.get('MILC', 'build/mil-hwxc'))
    parser.add_argument('--skip-reference', action='store_true')
    args = parser.parse_args()
    shutil.rmtree(args.output, ignore_errors=True)
    full = args.output / 'full'
    full.mkdir(parents=True)
    subprocess.run([
        args.compiler, '--target', 'H13', '--format', 'anec',
        '--schedule', 'linked-ring', '--mil', str(args.model / 'model.mil'),
        '--model-root', str(args.model), '--output', str(full / 'package')],
        check=True)
    program = full / 'package/program-0.anec'
    failures = libane_overrun.overruns(program.read_bytes())
    if failures:
        raise RuntimeError(f'libane bind overruns: {failures}')
    if not args.skip_reference:
        subprocess.run([sys.executable, str(HERE / 'reference.py'),
                        str(args.model), str(full / 'reference.npz')], check=True)
    manifest = {
        'target': 'H13', 'schedule': 'linked-ring', 'tasks': 80,
        'libaneBindOverrunProgramsChecked': 1,
        'safeCases': ['full'],
        'qualificationOrder': [
            'one-bottleneck-block', 'maxpool', 'add+relu', 'reduce_mean',
            'linear', 'block-prefixes', 'full'],
        'note': 'Only the byte-exact full program is emitted. Standalone and '
                'prefix tasks remain deliberately absent until their boundary '
                'DMA forms are byte-exact; do not run synthesized boundary forms.',
        'reference': 'float64 with TensorFlow SAME padding, including stride 2',
    }
    (args.output / 'kit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
