#!/usr/bin/env python3
"""Audit round-16 input geometry and model the H13 external-source walk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from round15 import (anec_parts, decoded_fields, rel_l2,
                     unpack_dense_1x1)


INPUT_FIELDS = [
    'input_dimensions', 'input_channels', 'convolution_config',
    'kernel_width', 'kernel_height', 'stride_width', 'stride_height',
    'padding_width', 'padding_height', 'common_traversal', 'group_config',
    'tile_height', 'l2_source_config', 'l2_source_base',
    'l2_source_channel_stride', 'l2_source_row_stride',
    'l2_source_depth_stride', 'l2_source_group_stride',
    'source_dma_config', 'source_surface_base',
    'source_surface_row_stride', 'source_surface_plane_stride',
    'source_surface_depth_stride', 'source_surface_group_stride',
]

MORPH_FIELDS = {
    'dead-l2-row': 'l2_source_row_stride',
    'source-row': 'source_surface_row_stride',
    'source-plane': 'source_surface_plane_stride',
    'source-depth': 'source_surface_depth_stride',
    'source-group': 'source_surface_group_stride',
}


def package_parts(package: Path):
    manifest = json.loads((package / 'manifest.json').read_text())
    program = package / manifest['programs'][0]['file']
    tasks, constants = anec_parts(program)
    return manifest, tasks, constants


def package_fields(package: Path, task_index: int = 0) -> dict:
    manifest, tasks, _ = package_parts(package)
    fields = decoded_fields(tasks[task_index])
    return {
        'fields': {name: fields[name] for name in INPUT_FIELDS
                   if name in fields},
        'input_nchw': manifest['inputs'][0]['nchw']
        if manifest.get('inputs') else None,
    }


def effective_dma_row(row: int) -> int:
    """H13 external TileDMA consumes whole 64-byte source-row beats."""
    return row & ~63


def predict(package: Path, input_path: Path):
    manifest, tasks, constants = package_parts(package)
    if len(tasks) != 1:
        raise ValueError('external-source model requires a one-task package')
    fields = decoded_fields(tasks[0])
    weights, bias = unpack_dense_1x1(fields, constants)
    source = np.load(input_path).astype(np.float32)
    if source.ndim == 4:
        source = source[0]
    channels, height, width = source.shape
    binding = manifest['inputs'][0]['nchw']
    binding_plane, binding_row = map(int, binding[-2:])
    physical = np.zeros(binding_plane * channels // 2, dtype=np.float32)
    for channel in range(channels):
        for row in range(height):
            at = (channel * binding_plane + row * binding_row) // 2
            physical[at:at + width] = source[channel, row]

    task_row = effective_dma_row(int(fields['source_surface_row_stride']))
    task_plane = int(fields['source_surface_plane_stride'])
    staged = np.empty((channels, height, width), dtype=np.float32)
    for channel in range(channels):
        for row in range(height):
            at = (channel * task_plane + row * task_row) // 2
            staged[channel, row] = physical[at:at + width]
    output = np.einsum('oc,chw->ohw', weights.astype(np.float32), staged,
                       optimize=True)
    output += bias.astype(np.float32)[:, None, None]
    return np.maximum(output, 0), {
        'task_row_bytes': int(fields['source_surface_row_stride']),
        'effective_task_row_bytes': task_row,
        'binding_row_bytes': binding_row,
        'task_plane_bytes': task_plane,
        'binding_plane_bytes': binding_plane,
    }


def task_word_differences(left: Path, right: Path) -> list[dict]:
    _, left_tasks, _ = package_parts(left)
    _, right_tasks, _ = package_parts(right)
    if len(left_tasks) != 1 or len(right_tasks) != 1 or \
            len(left_tasks[0]) != len(right_tasks[0]):
        raise ValueError('morph comparison requires equal one-task streams')
    before = decoded_fields(left_tasks[0])
    after = decoded_fields(right_tasks[0])
    return [{'field': key, 'before': before.get(key), 'after': after.get(key)}
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('kit', type=Path)
    parser.add_argument('--device', type=Path,
                        help='optional retained r14/r15 conv1 device output')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    packages = {
        'stem': (args.kit / 'stem' / 'package', 0),
        'stem-pool-via-dram':
            (args.kit / 'stem-pool-via-dram' / 'package', 1),
        'maxpool': (args.kit / 'maxpool' / 'package', 1),
        'conv1-fixed': (args.kit / 'block01-conv1-only' / 'package', 0),
        'projection-fixed':
            (args.kit / 'block01-projection-only' / 'package', 0),
        'row1-fixed': (args.kit / 'block01-conv1-row1' / 'package', 0),
        'conv1-r14': (args.kit / 'morph-conv1-r14-baseline' / 'package', 0),
    }
    report = {'input_side_fields': {
        name: package_fields(package, index)
        for name, (package, index) in packages.items()
    }}

    fixed_package = args.kit / 'block01-conv1-only' / 'package'
    fixed_input = args.kit / 'block01-conv1-only' / 'input-0.npy'
    fixed_reference = np.load(
        args.kit / 'block01-conv1-only' / 'reference.npz',
        allow_pickle=False)['block01-conv1-only']
    if fixed_reference.ndim == 4:
        fixed_reference = fixed_reference[0]
    fixed_prediction, fixed_walk = predict(fixed_package, fixed_input)
    fixed_walk['reference_rel_l2'] = rel_l2(fixed_prediction,
                                             fixed_reference)

    baseline_package = args.kit / 'morph-conv1-r14-baseline' / 'package'
    baseline_input = args.kit / 'morph-conv1-r14-baseline' / 'input-0.npy'
    baseline_prediction, baseline_walk = predict(baseline_package,
                                                  baseline_input)
    baseline_walk['reference_rel_l2'] = rel_l2(baseline_prediction,
                                                fixed_reference)
    if args.device:
        device = np.load(args.device).reshape(baseline_prediction.shape)
        baseline_walk['device_rel_l2'] = rel_l2(baseline_prediction, device)
    report['source_dma_model'] = {
        'r14_baseline': baseline_walk,
        'round16_fixed': fixed_walk,
    }

    morph_audit = {}
    for short in ('conv1', 'projection'):
        baseline = args.kit / f'morph-{short}-r14-baseline' / 'package'
        rows = {}
        for field in MORPH_FIELDS:
            variant = args.kit / f'morph-{short}-{field}' / 'package'
            differences = task_word_differences(baseline, variant)
            if len(differences) != 1 or \
                    differences[0]['field'] != MORPH_FIELDS[field]:
                raise ValueError(f'{short} {field} is not a one-field morph')
            rows[field] = differences[0]
        morph_audit[short] = rows
    report['one_field_morphs'] = morph_audit

    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
