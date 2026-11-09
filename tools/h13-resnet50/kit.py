#!/usr/bin/env python3
"""Build the smallest-first, bind-safe H13 ResNet-50 device kit."""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
import libane_overrun  # noqa: E402

HEADER = 0x1000
TILE = 0x4000


def word(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


def store(data, offset, value):
    struct.pack_into('<I', data, offset, value)


def task_offsets(anec):
    size, count = struct.unpack_from('<II', anec, 8)
    result, offset = [], 0
    for index in range(count):
        result.append((offset, size))
        if index + 1 != count:
            size = (((word(anec, HEADER + offset + 4) >> 16) & 0x1ff) + 1) * 4
            offset = word(anec, HEADER + offset + 28)
    return result


def register_offset(task, address):
    cursor = 40
    while cursor < len(task):
        header = word(task, cursor)
        count, base = (header >> 26) + 1, header & 0x03ffffff
        if base <= address < base + 4 * count:
            return cursor + 4 + address - base
        cursor += 4 * (count + 1)
    raise ValueError(f'task has no register 0x{address:x}')


def external_layout(channels, height, width):
    row = (width * 2 + 63) // 64 * 64
    plane = row * height
    allocation = (plane * channels + TILE - 1) // TILE * TILE
    return [1, channels, height, width, plane, row], allocation


def prefix(full, count, shape):
    """Make a bind-safe terminal boundary from the same rule-built stream."""
    spans = task_offsets(full)
    if not 1 <= count <= len(spans):
        raise ValueError('prefix task count is out of range')
    old_task_bytes = struct.unpack_from('<Q', full, 16)[0]
    all_constants_bytes = struct.unpack_from('<Q', full, 24)[0]
    old_constant = (old_task_bytes + 63) // 64 * 64
    last_offset, last_bytes = spans[count - 1]
    task_bytes = last_offset + last_bytes
    constant_offset = (task_bytes + 63) // 64 * 64
    constants_bytes = 0
    for offset, size in spans[:count]:
        task = memoryview(full)[HEADER + offset:HEADER + offset + size]
        for lane in range(16):
            try:
                begin = word(task, register_offset(task, 0x1f848 + 4 * lane))
                length = word(task, register_offset(task, 0x1f888 + 4 * lane))
            except ValueError:
                break
            constants_bytes = max(constants_bytes, begin + length)
    constants_bytes = min((constants_bytes + 63) // 64 * 64,
                          all_constants_bytes)
    constants = full[HEADER + old_constant:HEADER + old_constant + constants_bytes]
    result = bytearray(full[:HEADER])
    result += full[HEADER:HEADER + task_bytes]
    result += bytes(constant_offset - task_bytes)
    result += constants

    last = HEADER + last_offset
    store(result, last, (word(result, last) & ~0x03000000) |
          (0x01000000 if count > 1 else 0) | 0x02000000)
    store(result, last + 4, word(result, last + 4) & 0xffff)
    store(result, last + 8, word(result, last + 8) | 0x22)
    store(result, last + 16, 0x00fff86a)
    store(result, last + 24, word(result, last + 24) | 0x20009000)
    store(result, last + 28, 0)
    selectors = word(result, last + 32)
    selectors &= ~((31 << 12) | (1 << 17) | (1 << 23))
    store(result, last + 32, selectors | (4 << 12) | (1 << 17))
    task = memoryview(result)[last:last + last_bytes]
    store(task, register_offset(task, 0x04830), 0x14a)
    store(task, register_offset(task, 0x04834), 0)
    for address in (0x04838, 0x0483c, 0x04840, 0x04844):
        store(task, register_offset(task, address), 0)
    layout, allocation = external_layout(*shape)
    store(task, register_offset(task, 0x17800), 0x040000c1)
    store(task, register_offset(task, 0x17804), 0)
    store(task, register_offset(task, 0x17808), layout[5])
    store(task, register_offset(task, 0x1780c), layout[4])
    store(task, register_offset(task, 0x17810), allocation)
    store(task, register_offset(task, 0x17814), allocation)
    store(task, register_offset(task, 0x17818), 0x01302031)

    struct.pack_into('<QIIQQ', result, 0,
                     constant_offset + constants_bytes,
                     spans[0][1], count, task_bytes, constants_bytes)
    tiles = (allocation + TILE - 1) // TILE
    store(result, 40 + 4 * 4, tiles)
    struct.pack_into('<6Q', result, 168 + 4 * 48, *layout)
    return bytes(result)


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
    full_bytes = program.read_bytes()
    cases = {
        'maxpool': (2, (64, 56, 56)),
        'one-bottleneck-block': (7, (256, 56, 56)),
        'add+relu': (7, (256, 56, 56)),
        'reduce_mean': (79, (2048, 1, 1)),
    }
    block_counts = [7, 11, 15, 20, 24, 28, 32, 39, 44, 49, 54, 59,
                    64, 70, 74, 78]
    block_shapes = ([(256, 56, 56)] * 3 + [(512, 28, 28)] * 4 +
                    [(1024, 14, 14)] * 6 + [(2048, 7, 7)] * 3)
    for index, (count, shape) in enumerate(zip(block_counts, block_shapes), 1):
        cases[f'block-prefix-{index:02d}'] = (count, shape)
    checked = 1
    for name, (count, shape) in cases.items():
        case = args.output / name
        case.mkdir()
        payload = prefix(full_bytes, count, shape)
        target = case / 'program-0.anec'
        target.write_bytes(payload)
        failures = libane_overrun.overruns(payload)
        if failures:
            raise RuntimeError(f'{name} libane bind overruns: {failures}')
        checked += 1
    # The linear rung is the exact full program; keep a separate path so the
    # device runner can exercise it immediately before the final full case.
    linear = args.output / 'linear'
    linear.mkdir()
    shutil.copy2(program, linear / 'program-0.anec')
    checked += 1
    manifest = {
        'target': 'H13', 'schedule': 'linked-ring', 'tasks': 80,
        'libaneBindOverrunProgramsChecked': checked,
        'safeCases': list(cases) + ['linear', 'full'],
        'qualificationOrder': [
            'one-bottleneck-block', 'maxpool', 'add+relu', 'reduce_mean',
            'linear', 'block-prefixes', 'full'],
        'note': 'The full and linear rungs retain the byte-exact 80-task form. '
                'The smaller rungs terminate the same rule-built stream with '
                'a conservative external boundary and pass the libane extent gate.',
        'reference': 'float64 with TensorFlow SAME padding, including stride 2',
    }
    (args.output / 'kit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
