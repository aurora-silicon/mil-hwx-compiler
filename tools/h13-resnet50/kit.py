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
     'Round-10 scratch-fed pool control; no candidate field is changed.'),
    ('destination-path', 'pool-sweep-destination-path',
     'Destination Cfg selects the external linear writer instead of the scratch interleave writer.'),
    ('destination-dma', 'pool-sweep-destination-dma',
     'The complete destination DMA record uses the proven depthwise config, strides, extents, and format.'),
    ('source-dma', 'pool-sweep-source-dma',
     'Source TileDMA config, column/row advances, extents, format, and valid tail use the proven task.'),
    ('l2-source', 'pool-sweep-l2-source',
     'L2 source config, base, channel, row, depth, and group advances use the proven task.'),
    ('common-traversal', 'pool-sweep-common-traversal',
     'Common traversal adds the geometry-derived last eight-column piece field.'),
    ('group-config', 'pool-sweep-group-config',
     'Common GroupCfg uses the linked depthwise traversal group field.'),
    ('column-tile', 'pool-sweep-column-tile',
     'Common PEColumnTileCfg uses the proven task value, the round-10 negative control.'),
    ('output-size', 'pool-sweep-output-size',
     'Common output channels, dimensions, and tile height use the proven task values.'),
    ('halo-padding', 'pool-sweep-halo-padding',
     'Common horizontal and vertical padding fields test the one-cell trailing-halo alternative.'),
    ('pe-config', 'pool-sweep-pe-config',
     'PE Cfg uses the proven depthwise task value while the executing pipeline word stays guarded.'),
    ('width-28', 'pool-sweep-width-28',
     'Only the pool Common output width changes from 56 columns to 28.'),
    ('width-16', 'pool-sweep-width-16',
     'Only the pool Common output width changes from 56 columns to 16.'),
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

    def materialize(name, key, manifest, section=None, description=None):
        directory = args.output / name
        shutil.copy2(input_path, directory / 'input.npy')
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
        'sweep': sweep,
        'qualificationOrder': [item['name'] for item in rungs],
        'sweepOrder': [item['name'] for item in sweep],
        'libaneBindOverrunProgramsChecked': len(rungs) + len(sweep) + 1,
        'reference': 'float64 TensorFlow SAME; one plain array per rung npz',
    }
    (args.output / 'kit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
