#!/usr/bin/env python3
"""Build the smallest-first, compiler-terminated H13 ResNet-50 kit."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import libane_overrun  # noqa: E402


def compile_package(args, package: Path, terminate_after: str | None = None):
    command = [
        args.compiler, '--target', 'H13', '--format', 'anec',
        '--schedule', 'linked-ring', '--mil', str(args.model / 'model.mil'),
        '--model-root', str(args.model), '--output', str(package),
    ]
    if terminate_after:
        command += ['--terminate-after', terminate_after]
    subprocess.run(command, check=True)
    manifest = json.loads((package / 'manifest.json').read_text())
    program = package / manifest['programs'][0]['file']
    failures = libane_overrun.overruns(program.read_bytes())
    if failures:
        raise RuntimeError(f'{package.parent.name} libane bind overruns: {failures}')
    return manifest, program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compiler', default=os.environ.get('MILC',
                                                             'build/mil-hwxc'))
    args = parser.parse_args()
    shutil.rmtree(args.output, ignore_errors=True)
    args.output.mkdir(parents=True)

    reference_path = args.output / 'all-reference.npz'
    subprocess.run([sys.executable, str(HERE / 'reference.py'),
                    str(args.model), str(reference_path)], check=True)
    input_path = reference_path.with_name('input.npy')
    with np.load(reference_path, allow_pickle=False) as archive:
        references = {key: archive[key] for key in archive.files}

    full_dir = args.output / 'full'
    full_dir.mkdir()
    full_manifest, _ = compile_package(args, full_dir / 'package')
    boundaries = {item['name']: item for item in full_manifest['boundaries']}
    expected = ['pool'] + [f'block{index:02d}' for index in range(1, 17)] + \
        ['mean', 'logits']
    if list(references) != expected or set(boundaries) != set(expected):
        raise RuntimeError('compiler manifest and reference boundary keys differ')

    specifications = [('maxpool', 'pool'),
                      ('one-bottleneck-block', 'block01')]
    specifications += [(f'block-prefix-{index:02d}', f'block{index:02d}')
                       for index in range(2, 17)]
    specifications += [('reduce_mean', 'mean'), ('full', 'logits')]
    rungs = []
    for name, key in specifications:
        directory = args.output / name
        if name == 'full':
            manifest = full_manifest
        else:
            directory.mkdir()
            manifest, _ = compile_package(
                args, directory / 'package', boundaries[key]['tensor'])
        shutil.copy2(input_path, directory / 'input.npy')
        np.savez(directory / 'reference.npz', **{key: references[key]})
        logical = manifest['logicalResults'][0]
        rungs.append({
            'name': name,
            'program': f'{name}/package/{manifest["programs"][0]["file"]}',
            'manifest': f'{name}/package/manifest.json',
            'input': f'{name}/input.npy',
            'reference': f'{name}/reference.npz',
            'reference_key': key,
            'tolerance_relL2': 0.02,
            'tasks': manifest['taskDescriptors'],
            'output_shape': logical['shape'],
            'allocationBytes': manifest['outputs'][0]['allocationBytes'],
        })

    reference_path.unlink()
    input_path.unlink()
    manifest = {
        'target': 'H13',
        'schedule': 'linked-ring',
        'rungs': rungs,
        'qualificationOrder': [item['name'] for item in rungs],
        'libaneBindOverrunProgramsChecked': len(rungs),
        'reference': 'float64 TensorFlow SAME; one plain array per rung npz',
    }
    (args.output / 'kit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
