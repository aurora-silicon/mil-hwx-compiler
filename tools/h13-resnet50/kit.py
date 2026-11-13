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

BLOCK_SUBRUNGS = [
    ('block01-conv1-only', 'block01-conv1-only', ['pool'],
     'Standalone first bottleneck 1x1 conv with the CPU pool reference as input.'),
    ('block01-conv1-conv2', 'block01-conv1-conv2', ['pool'],
     'Standalone conv1+conv2 chain; this isolates the first scratch producer/consumer handoff.'),
    ('block01-conv1-conv2-conv3', 'block01-conv1-conv2-conv3', ['pool'],
     'Standalone three-convolution main branch with the CPU pool reference as input.'),
    ('block01-projection-only', 'block01-projection-only', ['pool'],
     'Standalone projection branch with the CPU pool reference as input.'),
    ('block01-add-only', 'block01-add-only',
     ['block01-main3', 'projection01'],
     'Standalone fused residual add+ReLU with CPU main3 and projection inputs.'),
]

INPUT_MORPHS = [
    ('r14-baseline',
     'Round-14 control: 112/6272/401408 TileDMA source geometry and zero L2 row phase.'),
    ('dead-l2-row',
     'Known-dead control: changes only L2 source-row 7168 to 7248; round 15 was byte-identical on device.'),
    ('source-group',
     'Changes only TileDMA source group/extent from 401408 to 458752.'),
    ('source-depth',
     'Changes only TileDMA source depth/extent from 401408 to 458752.'),
    ('source-plane',
     'Changes only TileDMA source plane stride from 6272 to 7168.'),
    ('source-row',
     'Changes only TileDMA source row stride from 112 to the legal 128-byte row class.'),
]

ADD_MORPHS = [
    ('r16-baseline',
     'Byte-exact round-16 add-only control: coherent physical surfaces, unphased L2 rows, and the four binary configuration words retain their r16 values.'),
    ('dead-main-source-group',
     'Dead-field control: only main source TileDMA group/extent takes the legacy 112-byte-row value; round 16 showed this word is not consumed.'),
    ('l2-second-enable',
     'Only L2 SourceCfg bit 24 changes to the decoded binary second-source form.'),
    ('pipeline-second-enable',
     'Only Common Cfg bit 1 changes, producing the decoded add pipeline word 0x33.'),
    ('second-dma-config',
     'Only TileDMA source-2 Cfg changes from source-1 0x33881 to decoded source-2 0x33880.'),
    ('result-config',
     'Only L2 ResultCfg changes from the r16 convolution-style value to decoded PE external result 0x50017a.'),
    ('main-l2-row',
     'Main/conv3 L2 row sensitivity probe: only the row takes the 32-byte PE bank phase.'),
    ('skip-l2-row',
     'Projected-skip L2 row sensitivity probe: only the row takes the 32-byte PE bank phase.'),
    ('main-source-base',
     'Main/conv3 TileDMA base sensitivity probe: only the base advances by one coherent 128-byte row.'),
    ('main-source-row',
     'Main/conv3 TileDMA row sensitivity probe: only row returns to the legacy 112-byte class.'),
    ('main-source-plane',
     'Main/conv3 TileDMA plane sensitivity probe: only plane returns to 6272 bytes.'),
    ('main-source-depth',
     'Main/conv3 TileDMA depth sensitivity probe: only depth returns to 1605632 bytes.'),
    ('skip-source-base',
     'Projected-skip TileDMA base sensitivity probe: only the base advances by one coherent 128-byte row.'),
    ('skip-source-row',
     'Projected-skip TileDMA row sensitivity probe: only row returns to the legacy 112-byte class.'),
    ('skip-source-plane',
     'Projected-skip TileDMA plane sensitivity probe: only plane returns to 6272 bytes.'),
    ('skip-source-depth',
     'Projected-skip TileDMA depth sensitivity probe: only depth returns to 1605632 bytes.'),
    ('result-row',
     'Add-result TileDMA row sensitivity probe: only row returns to the legacy 112-byte class.'),
    ('result-plane',
     'Add-result TileDMA plane sensitivity probe: only plane returns to 6272 bytes; run last among geometry probes.'),
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
    sub_reference_keys = [key for _, key, _, _ in BLOCK_SUBRUNGS] + \
        ['block01-main3']
    if ([key for key in references if key in expected] != expected or
            not set(sub_reference_keys).issubset(references) or
            set(boundaries) != set(expected)):
        raise RuntimeError('compiler manifest and reference boundary keys differ')

    specifications = [('stem', 'stem', False),
                      ('stem-pool-via-dram', 'pool', True),
                      ('maxpool', 'pool', False),
                      ('one-bottleneck-block', 'block01', False)]
    specifications += [(f'block-prefix-{index:02d}', f'block{index:02d}', False)
                       for index in range(2, 17)]
    specifications += [('reduce_mean', 'mean', False),
                       ('full', 'logits', False)]

    def kit_entry(name, key, manifest, section=None, description=None,
                  input_files=None):
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
        if input_files:
            item.pop('input')
            item['inputs'] = input_files
        return item

    def materialize(name, key, manifest, section=None, description=None,
                    input_value=None, input_values=None, reference_value=None):
        directory = args.output / name
        input_files = None
        if input_values is not None:
            input_files = []
            for index, value in enumerate(input_values):
                filename = f'input-{index}.npy'
                np.save(directory / filename, value)
                input_files.append(f'{name}/{filename}')
        elif input_value is None:
            shutil.copy2(input_path, directory / 'input.npy')
        else:
            np.save(directory / 'input.npy', input_value)
        np.savez(directory / 'reference.npz',
                 **{key: references[key] if reference_value is None
                    else reference_value})
        return kit_entry(name, key, manifest, section, description,
                         input_files)

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

    sub_rungs = []
    for name, key, input_keys, description in BLOCK_SUBRUNGS:
        directory = args.output / name
        directory.mkdir()
        sub_manifest, _ = compile_package(
            args, directory / 'package', name)
        entry = materialize(
            name, key, sub_manifest, description=description,
            input_values=[references[input_key]
                          for input_key in input_keys])
        entry['repeatability'] = {
            'coldRuns': 3,
            'comparison': 'uint16-bit-identical',
            'judge': 'verify_cold.py',
        }
        sub_rungs.append(entry)

    row_probe_name = 'block01-conv1-row1'
    row_probe_dir = args.output / row_probe_name
    row_probe_dir.mkdir()
    row_probe_manifest, _ = compile_package(
        args, row_probe_dir / 'package', row_probe_name)
    row_probe = materialize(
        row_probe_name, 'block01-conv1-only', row_probe_manifest,
        description=
        'H=1 first-row probe: removes every inter-row advance while retaining the 64-channel plane walk.',
        input_values=[references['pool'][:, :1, :]],
        reference_value=references['block01-conv1-only'][:, :1, :])
    row_probe['repeatability'] = {
        'coldRuns': 3,
        'comparison': 'uint16-bit-identical',
        'judge': 'verify_cold.py',
    }

    morphs = []
    for short, boundary, key in (
            ('conv1', 'block01-conv1-only', 'block01-conv1-only'),
            ('projection', 'block01-projection-only',
             'block01-projection-only')):
        for field, description in INPUT_MORPHS:
            name = f'morph-{short}-{field}'
            directory = args.output / name
            directory.mkdir()
            morph_manifest, _ = compile_package(
                args, directory / 'package', f'{boundary}-morph-{field}')
            entry = materialize(
                name, key, morph_manifest, field, description,
                input_values=[references['pool']])
            entry['candidateField'] = field
            entry['repeatability'] = {
                'coldRuns': 3,
                'comparison': 'uint16-bit-identical',
                'judge': 'verify_cold.py',
            }
            morphs.append(entry)
    for field, description in ADD_MORPHS:
        name = f'morph-add-{field}'
        directory = args.output / name
        directory.mkdir()
        morph_manifest, _ = compile_package(
            args, directory / 'package',
            f'block01-add-only-morph-{field}')
        entry = materialize(
            name, 'block01-add-only', morph_manifest, field, description,
            input_values=[references['block01-main3'],
                          references['projection01']])
        entry['candidateField'] = field
        entry['repeatability'] = {
            'coldRuns': 3,
            'comparison': 'uint16-bit-identical',
            'judge': 'verify_cold.py',
        }
        morphs.append(entry)

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
        'subRungs': sub_rungs,
        'rowProbe': row_probe,
        'morphs': morphs,
        'sweep': sweep,
        'qualificationOrder': [item['name'] for item in rungs],
        'sweepOrder': [item['name'] for item in sweep],
        'subRungOrder': [item['name'] for item in sub_rungs],
        'morphOrder': [item['name'] for item in morphs],
        'libaneBindOverrunProgramsChecked':
            len(rungs) + len(diagnostics) + len(sub_rungs) + len(sweep) +
            len(morphs) + 2,
        'reference': 'float64 TensorFlow SAME; one plain array per rung npz',
    }
    shutil.copy2(HERE / 'verify_cold.py', args.output / 'verify_cold.py')
    (args.output / 'kit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
