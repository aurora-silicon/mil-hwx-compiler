#!/usr/bin/env python3
"""Decode the round-15 geometry and model the round-14 zero-phase row walk.

The model consumes only an ANEC package, its public kit input, and an optional
device output.  It reconstructs the dense 1x1 weights from the package's
constant section; no compiler capture or target access is involved.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


ANEC_HEADER_BYTES = 0x1000


FIELD_NAMES = {
    0x00000: 'input_dimensions',
    0x0000c: 'input_channels',
    0x00010: 'output_channels',
    0x00014: 'output_dimensions',
    0x0001c: 'convolution_config',
    0x00020: 'common_traversal',
    0x00024: 'group_config',
    0x00028: 'tile_height',
    0x0002c: 'column_piece_config',
    0x00034: 'schedule_config',
    0x04804: 'l2_source_config',
    0x04808: 'l2_source_base',
    0x0480c: 'l2_source_channel_stride',
    0x04810: 'l2_source_row_stride',
    0x04814: 'l2_source_depth_stride',
    0x04818: 'l2_source_group_stride',
    0x0481c: 'l2_second_base',
    0x04820: 'l2_second_channel_stride',
    0x04824: 'l2_second_row_stride',
    0x04828: 'l2_second_depth_stride',
    0x0482c: 'l2_second_group_stride',
    0x04830: 'l2_result_config',
    0x04834: 'l2_result_base',
    0x04838: 'l2_result_channel_stride',
    0x0483c: 'l2_result_row_stride',
    0x04840: 'l2_result_depth_stride',
    0x04844: 'l2_result_group_stride',
    0x13800: 'source_dma_config',
    0x13804: 'second_dma_config',
    0x13808: 'source_surface_base',
    0x1380c: 'source_surface_row_stride',
    0x13810: 'source_surface_plane_stride',
    0x13814: 'source_surface_depth_stride',
    0x13818: 'source_surface_group_stride',
    0x17800: 'destination_dma_config',
    0x17804: 'destination_surface_base',
    0x17808: 'destination_surface_row_stride',
    0x1780c: 'destination_surface_plane_stride',
    0x17810: 'destination_surface_depth_stride',
    0x17814: 'destination_surface_group_stride',
    0x0c800: 'kernel_config',
    0x0c804: 'mac_config',
}


def registers(task: bytes) -> dict[int, int]:
    words = struct.unpack(f'<{len(task) // 4}I', task)
    cursor = 10 + (1 if words[9] & 3 == 3 else 0)
    result = {}
    while cursor < len(words):
        header = words[cursor]
        count = (header >> 26) + 1
        address = header & 0x03ffffff
        for index in range(count):
            result[address + 4 * index] = words[cursor + 1 + index]
        cursor += count + 1
    if cursor != len(words):
        raise ValueError('task register records are truncated')
    return result


def anec_parts(path: Path) -> tuple[list[bytes], bytes]:
    data = path.read_bytes()
    _, first_bytes, count, stream_bytes, constant_bytes, _, _ = \
        struct.unpack_from('<QIIQQII', data)
    stream = data[ANEC_HEADER_BYTES:ANEC_HEADER_BYTES + stream_bytes]
    tasks = []
    offset, size = 0, first_bytes
    for index in range(count):
        task = stream[offset:offset + size]
        if len(task) != size:
            raise ValueError(f'task {index} is truncated')
        tasks.append(task)
        if index + 1 != count:
            header = struct.unpack_from('<10I', task)
            offset = header[7]
            size = (((header[1] >> 16) & 0x1ff) + 1) * 4
    constant_offset = (stream_bytes + 63) & ~63
    constants = data[ANEC_HEADER_BYTES + constant_offset:
                     ANEC_HEADER_BYTES + constant_offset + constant_bytes]
    if len(constants) != constant_bytes:
        raise ValueError('constant section is truncated')
    return tasks, constants


def decoded_fields(task: bytes) -> dict[str, int | list[int]]:
    words = registers(task)
    conv = words[0x0001c]
    fields = {name: words[address] for address, name in FIELD_NAMES.items()
              if address in words}
    fields.update({
        'kernel_width': conv & 31,
        'kernel_height': (conv >> 5) & 31,
        'output_channel_group_log2': (conv >> 10) & 7,
        'stride_width': (conv >> 13) & 3,
        'stride_height': (conv >> 15) & 3,
        'padding_width': (conv >> 17) & 31,
        'padding_height': (conv >> 22) & 31,
        'input_hw': [words[0x00000] >> 16, words[0x00000] & 0xffff],
        'output_hw': [words[0x00014] >> 16, words[0x00014] & 0xffff],
    })
    return fields


def l2_skew(stride: int, width: int, channels: int) -> int:
    if stride == 2:
        return 0x30 if width >= 64 else 0x20
    if width >= 64:
        return 0
    if width > 16:
        return 0x50
    return 0x30 if channels >= 32 else 0x20


def unpack_dense_1x1(fields: dict[str, int | list[int]],
                     constants: bytes) -> tuple[np.ndarray, np.ndarray]:
    inputs = int(fields['input_channels'])
    outputs = int(fields['output_channels'])
    if fields['kernel_height'] != 1 or fields['kernel_width'] != 1:
        raise ValueError('offline row-walk model requires a 1x1 convolution')
    lanes = 1 << int(fields['output_channel_group_log2'])
    if outputs != 16 * lanes:
        raise ValueError('offline row-walk model requires one output chunk')
    group_halfwords = len(constants) // 16 // 2
    required = (inputs + 1) * lanes
    if group_halfwords < required:
        raise ValueError('constant plane is too small')
    packed = np.frombuffer(constants, dtype='<f2')
    weights = np.empty((outputs, inputs), dtype=np.float16)
    bias = np.empty(outputs, dtype=np.float16)
    for group in range(16):
        base = group * group_halfwords
        for lane in range(lanes):
            output = group * lanes + lane
            bias[output] = packed[base + lane]
            weights[output] = packed[
                base + lanes + lane:base + lanes + inputs * lanes:lanes]
    return weights, bias


def effective_row_walk(fields: dict[str, int | list[int]]) -> int:
    width = int(fields['input_hw'][1])
    channels = int(fields['input_channels'])
    stride = int(fields['stride_width'])
    expected = l2_skew(stride, width, channels)
    actual = int(fields['l2_source_row_stride']) - \
        int(fields['l2_source_depth_stride'])
    if actual == expected:
        return width
    if actual == 0 and expected:
        # A zero bank phase repeats the first half of the next power-of-two
        # spatial row.  For W=56 this is the observed 32-pixel row walk.
        return 1 << max(0, (width - 1).bit_length() - 1)
    raise ValueError(f'unmodelled L2 bank phase: actual={actual}, expected={expected}')


def predict(package: Path, input_path: Path) -> tuple[np.ndarray, dict]:
    manifest = json.loads((package / 'manifest.json').read_text())
    program = package / manifest['programs'][0]['file']
    tasks, constants = anec_parts(program)
    if len(tasks) != 1:
        raise ValueError('offline row-walk model requires a one-task package')
    fields = decoded_fields(tasks[0])
    weights, bias = unpack_dense_1x1(fields, constants)
    source = np.load(input_path).astype(np.float32).reshape(
        int(fields['input_channels']), -1)
    height, width = map(int, fields['output_hw'])
    walk = effective_row_walk(fields)
    indices = (np.arange(height)[:, None] * walk +
               np.arange(width)[None, :]).reshape(-1)
    if indices.max() >= source.shape[1]:
        raise ValueError('row-walk model reads beyond the input')
    staged = source[:, indices].reshape(source.shape[0], height, width)
    output = np.einsum('oc,chw->ohw', weights.astype(np.float32), staged,
                       optimize=True)
    output += bias.astype(np.float32)[:, None, None]
    return np.maximum(output, 0), {'fields': fields, 'effective_row_walk': walk}


def rel_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm((actual.astype(np.float64) -
                                 expected.astype(np.float64)).ravel()) /
                 np.linalg.norm(expected.astype(np.float64).ravel()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('kit', type=Path)
    parser.add_argument('--device', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = {}
    for name, task_index in (('stem', 0), ('maxpool', 1),
                             ('block01-conv1-only', 0),
                             ('block01-projection-only', 0),
                             ('block01-add-only', 0)):
        package = args.kit / name / 'package'
        manifest = json.loads((package / 'manifest.json').read_text())
        tasks, _ = anec_parts(package / manifest['programs'][0]['file'])
        rows[name] = decoded_fields(tasks[task_index])
    prediction, model = predict(
        args.kit / 'block01-conv1-only' / 'package',
        args.kit / 'block01-conv1-only' / 'input-0.npy')
    report = {'decoded_fields': rows, 'conv1_model': model}
    if args.device:
        device = np.load(args.device).reshape(prediction.shape)
        report['conv1_model']['device_rel_l2'] = rel_l2(prediction, device)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
