#!/usr/bin/env python3
"""Audit round-17 residual-add fields and its one-word device morphs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from round15 import decoded_fields
from round16 import package_parts, task_word_differences


ADD_FIELDS = [
    'input_dimensions', 'input_channels', 'output_channels',
    'output_dimensions', 'column_piece_config', 'schedule_config',
    'l2_source_config', 'l2_source_base', 'l2_source_channel_stride',
    'l2_source_row_stride', 'l2_source_depth_stride',
    'l2_source_group_stride', 'l2_second_base',
    'l2_second_channel_stride', 'l2_second_row_stride',
    'l2_second_depth_stride', 'l2_second_group_stride',
    'l2_result_config', 'l2_result_base', 'l2_result_channel_stride',
    'l2_result_row_stride', 'l2_result_depth_stride',
    'l2_result_group_stride', 'source_dma_config', 'second_dma_config',
    'source_surface_base', 'source_surface_row_stride',
    'source_surface_plane_stride', 'source_surface_depth_stride',
    'source_surface_group_stride', 'second_surface_base',
    'second_surface_row_stride', 'second_surface_plane_stride',
    'second_surface_depth_stride', 'second_surface_group_stride',
    'source_surface_format', 'second_surface_extension',
    'destination_dma_config', 'destination_surface_base',
    'destination_surface_row_stride', 'destination_surface_plane_stride',
    'destination_surface_depth_stride',
    'destination_surface_group_stride', 'destination_surface_format',
]

ADD_MORPH_FIELDS = {
    'dead-main-source-group': 'source_surface_group_stride',
    'l2-second-enable': 'l2_source_config',
    'pipeline-second-enable': 'schedule_config',
    'second-dma-config': 'second_dma_config',
    'result-config': 'l2_result_config',
    'main-l2-row': 'l2_source_row_stride',
    'main-source-base': 'source_surface_base',
    'main-source-row': 'source_surface_row_stride',
    'main-source-plane': 'source_surface_plane_stride',
    'main-source-depth': 'source_surface_depth_stride',
    'skip-l2-row': 'l2_second_row_stride',
    'skip-source-base': 'second_surface_base',
    'skip-source-row': 'second_surface_row_stride',
    'skip-source-plane': 'second_surface_plane_stride',
    'skip-source-depth': 'second_surface_depth_stride',
    'result-row': 'destination_surface_row_stride',
    'result-plane': 'destination_surface_plane_stride',
}

ROUND16_STABLE = [
    'stem', 'stem-pool-via-dram', 'maxpool',
    'block01-conv1-only', 'block01-conv1-conv2',
    'block01-conv1-conv2-conv3', 'block01-projection-only',
    'block01-conv1-row1', 'next-stall-projection',
]


def program_path(package: Path) -> Path:
    manifest = json.loads((package / 'manifest.json').read_text())
    return package / manifest['programs'][0]['file']


def fields(package: Path, task_index: int) -> dict:
    manifest, tasks, _ = package_parts(package)
    decoded = decoded_fields(tasks[task_index])
    return {
        'task': task_index,
        'fields': {name: decoded[name] for name in ADD_FIELDS
                   if name in decoded},
        'inputs': [item.get('nchw') for item in manifest.get('inputs', [])],
        'output': manifest.get('outputs', [{}])[0].get('nchw'),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_stable(round16: Path, round17: Path) -> dict:
    result = {}
    for name in ROUND16_STABLE:
        before = program_path(round16 / name / 'package')
        after = program_path(round17 / name / 'package')
        left, right = sha256(before), sha256(after)
        if left != right:
            raise ValueError(f'{name} changed from round 16')
        result[name] = left
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('kit', type=Path)
    parser.add_argument('--round16', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    table = {
        'conv1_external': fields(
            args.kit / 'block01-conv1-only' / 'package', 0),
        'conv3_external_result': fields(
            args.kit / 'block01-conv1-conv2-conv3' / 'package', 2),
        'projection_external': fields(
            args.kit / 'block01-projection-only' / 'package', 0),
        'add_round16_baseline': fields(
            args.kit / 'morph-add-r16-baseline' / 'package', 0),
        'add_round17_fixed': fields(
            args.kit / 'block01-add-only' / 'package', 0),
        'block_add': fields(
            args.kit / 'one-bottleneck-block' / 'package', 6),
    }

    baseline = args.kit / 'morph-add-r16-baseline' / 'package'
    morphs = {}
    for short, expected in ADD_MORPH_FIELDS.items():
        variant = args.kit / f'morph-add-{short}' / 'package'
        differences = task_word_differences(baseline, variant)
        if len(differences) != 1 or differences[0]['field'] != expected:
            raise ValueError(f'add {short} is not a one-field {expected} morph')
        morphs[short] = differences[0]

    fixed = args.kit / 'block01-add-only' / 'package'
    coherent = task_word_differences(baseline, fixed)
    coherent_names = {item['field'] for item in coherent}
    expected_coherent = {
        'schedule_config', 'l2_source_config', 'l2_result_config',
        'second_dma_config', 'l2_source_row_stride',
        'l2_second_row_stride',
    }
    if coherent_names != expected_coherent:
        raise ValueError(
            f'coherent add fix changed {sorted(coherent_names)}, '
            f'expected {sorted(expected_coherent)}')

    _, fixed_tasks, _ = package_parts(fixed)
    fixed_fields = decoded_fields(fixed_tasks[0])
    expected_values = {
        'schedule_config': 0x33,
        'l2_source_config': 0x01500172,
        'l2_result_config': 0x0050017a,
        'second_dma_config': 0x00033880,
        'l2_source_row_stride': 28704,
        'l2_second_row_stride': 28704,
        'source_surface_row_stride': 128,
        'source_surface_plane_stride': 7168,
        'source_surface_depth_stride': 1835008,
        'second_surface_row_stride': 128,
        'second_surface_plane_stride': 7168,
        'second_surface_depth_stride': 1835008,
        'destination_surface_row_stride': 128,
        'destination_surface_plane_stride': 7168,
        'destination_surface_depth_stride': 1835008,
    }
    for name, expected in expected_values.items():
        if fixed_fields.get(name) != expected:
            raise ValueError(
                f'fixed add {name}={fixed_fields.get(name)}, expected {expected}')

    report = {
        'add_operand_field_table': table,
        'round17_coherent_fix': coherent,
        'one_field_morphs': morphs,
    }
    if args.round16:
        report['round16_byte_identical'] = audit_stable(
            args.round16, args.kit)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
