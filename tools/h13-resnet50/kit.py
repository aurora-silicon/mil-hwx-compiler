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


def compile_package(args, package: Path, terminate_after: str | None = None,
                    pool_via_dram: bool = False):
    command = [
        args.compiler, '--target', 'H13', '--format', 'anec',
        '--schedule', 'linked-ring', '--mil', str(args.model / 'model.mil'),
        '--model-root', str(args.model), '--output', str(package),
    ]
    if pool_via_dram:
        command += ['--terminate-after', 'pool-via-dram']
    elif terminate_after:
        command += ['--terminate-after', terminate_after]
    subprocess.run(command, check=True)
    manifest = json.loads((package / 'manifest.json').read_text())
    program = package / manifest['programs'][0]['file']
    failures = libane_overrun.overruns(program.read_bytes())
    if failures:
        raise RuntimeError(f'{package.parent.name} libane bind overruns: {failures}')
    return manifest, program


POOL_SWEEPS = [
    ('control', 'pool-sweep-control',
     'Measured round-12 map: low-nibble 4 gives 16-column pieces; destination +0x08 is the 128-byte row stride.'),
    ('piece-width-20', 'pool-sweep-piece-width-20',
     'Common +0x2c low-nibble 5 emits 20-column pieces.'),
    ('piece-width-24', 'pool-sweep-piece-width-24',
     'Common +0x2c low-nibble 6 emits 24-column pieces.'),
    ('piece-width-28', 'pool-sweep-piece-width-28',
     'Common +0x2c low-nibble 7 emits 28-column pieces.'),
    ('best-guess', 'pool-sweep-best-guess',
     'Chosen shape rule: low-nibble 8 emits contiguous 32-column (64-byte) pieces while +0x08 stays 128.'),
    ('row-256', 'pool-sweep-row-256',
     'Rejected alternative: coherent 256-byte package row, plane and extents with 16-column pieces.'),
    ('width-28-control', 'pool-sweep-width-28-control',
     'Width 28 with the measured 16-column-piece control, ordered late because its timed loop hung in round 12.'),
    ('width-28-best-guess', 'pool-sweep-width-28-best-guess',
     'Width 28 with one 32-column piece and the 128-byte destination row.'),
]


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
    expected = ['stem', 'pool', 'projection01'] + \
        [f'block{index:02d}' for index in range(1, 17)] + \
        ['mean', 'logits']
    if list(references) != expected or set(boundaries) != set(expected):
        raise RuntimeError('compiler manifest and reference boundary keys differ')

    specifications = [('stem', 'stem', False),
                      ('stem-pool-via-dram', 'pool', True),
                      ('maxpool', 'pool', False),
                      ('one-bottleneck-block', 'block01', False)]
    specifications += [(f'block-prefix-{index:02d}', f'block{index:02d}', False)
                       for index in range(2, 17)]
    specifications += [('reduce_mean', 'mean', False),
                       ('full', 'logits', False)]

    def kit_entry(name, key, manifest, section=None, description=None):
        logical = manifest['logicalResults'][0]
        item = {
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
        }
        if section:
            item['sweep'] = section
        if description:
            item['description'] = description
        return item

    def materialize(name, key, manifest, section=None, description=None,
                    input_value=None):
        directory = args.output / name
        if input_value is None:
            shutil.copy2(input_path, directory / 'input.npy')
        else:
            np.save(directory / 'input.npy', input_value)
        np.savez(directory / 'reference.npz', **{key: references[key]})
        return kit_entry(name, key, manifest, section, description)

    rungs = []
    for name, key, pool_via_dram in specifications:
        directory = args.output / name
        if name == 'full':
            manifest = full_manifest
        else:
            directory.mkdir()
            manifest, _ = compile_package(
                args, directory / 'package', boundaries[key]['tensor'],
                pool_via_dram)
        rungs.append(materialize(name, key, manifest))

    next_name = 'next-stall-projection'
    next_dir = args.output / next_name
    next_dir.mkdir()
    next_manifest, _ = compile_package(
        args, next_dir / 'package', boundaries['projection01']['tensor'])
    next_stall = materialize(
        next_name, 'projection01', next_manifest, description=
        'Three-task stem, pool, then the first projection: expected task index 2 after pool is repaired.')
    next_stall['expectedTaskIndex'] = 2
    next_stall['expectedTaskClass'] = 'projection'

    diagnostics = []
    prefix_name = 'block-prefix-14-only'
    prefix_dir = args.output / prefix_name
    prefix_dir.mkdir()
    prefix_manifest, _ = compile_package(
        args, prefix_dir / 'package', boundaries['block14']['tensor'])
    diagnostics.append(materialize(
        prefix_name, 'block14', prefix_manifest, description=
        'Focused copy of the first 7x7 block prefix; OCG-5 removal raises it from 70 to 77 tasks.'))

    standalone_name = 'stage-4-first-block-standalone'
    standalone_dir = args.output / standalone_name
    standalone_dir.mkdir()
    standalone_manifest, _ = compile_package(
        args, standalone_dir / 'package', standalone_name)
    diagnostics.append(materialize(
        standalone_name, 'block14', standalone_manifest, description=
        'Thirteen-task 1024x14x14 to 2048x7x7 block with the block13 reference as its external input.',
        input_value=references['block13']))

    sweep = []
    for field, boundary, description in POOL_SWEEPS:
        name = f'pool-sweep-{field}'
        directory = args.output / name
        directory.mkdir()
        sweep_manifest, _ = compile_package(
            args, directory / 'package', boundary)
        sweep.append(materialize(
            name, 'pool', sweep_manifest, field, description))

    reference_path.unlink()
    input_path.unlink()
    manifest = {
        'target': 'H13',
        'schedule': 'linked-ring',
        'rungs': rungs,
        'next_stall': next_stall,
        'diagnostics': diagnostics,
        'sweep': sweep,
        'qualificationOrder': [item['name'] for item in rungs],
        'sweepOrder': [item['name'] for item in sweep],
        'libaneBindOverrunProgramsChecked':
            len(rungs) + len(diagnostics) + len(sweep) + 1,
        'reference': 'float64 TensorFlow SAME; one plain array per rung npz',
    }
    (args.output / 'kit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
