#!/usr/bin/env python3
"""Byte-exact private parity for the H13 MobileNet linked ring.

The private t-5959 store is optional and is never copied into the repository.
When ``MILHWXC_T5959_PRIVATE`` is set, this gate truncates the retained source
MIL at p01..p28, reuses its exact weight blob, compiles each prefix, and
compares every task byte and the complete constant section with Apple's HWX.
Without the private store it reports SKIP.
"""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def cstring(raw):
    return raw.split(b'\0', 1)[0].decode('ascii')


def hwx_parts(path):
    data = path.read_bytes()
    if len(data) < 32 or struct.unpack_from('<I', data)[0] != 0xbeefface:
        raise AssertionError(f'{path}: not an HWX Mach-O')
    command_count, command_bytes = struct.unpack_from('<2I', data, 16)
    cursor, command_end = 32, 32 + command_bytes
    sections = {}
    task_count = first_words = None
    for _ in range(command_count):
        command, size = struct.unpack_from('<2I', data, cursor)
        if command == 0x19:
            fields = struct.unpack_from('<2I16s4Q4I', data, cursor)
            segment = cstring(fields[2])
            section_cursor = cursor + 72
            for _ in range(fields[-2]):
                entry = struct.unpack_from('<16s16s2Q8I', data, section_cursor)
                sections[(segment, cstring(entry[0]))] = (entry[4], entry[3])
                section_cursor += 80
        elif command == 4 and size >= 0x820:
            kind = struct.unpack_from('<I', data, cursor + 8)[0]
            if kind == 1:
                first_words = struct.unpack_from('<I', data, cursor + 0x818)[0]
                task_count = struct.unpack_from('<I', data, cursor + 0x81c)[0]
        cursor += size
    if cursor != command_end or task_count is None:
        raise AssertionError(f'{path}: malformed H13 load commands')
    text_at, text_size = sections[('__TEXT', '__text')]
    const_at, const_size = sections[('__TEXT', '__const')]
    stream = data[text_at:text_at + text_size]
    tasks, offset, size = [], 0, (first_words + 1) * 4
    for index in range(task_count):
        tasks.append(stream[offset:offset + size])
        if len(tasks[-1]) != size:
            raise AssertionError(f'{path}: task {index} is truncated')
        if index + 1 != task_count:
            words = struct.unpack_from(f'<{size // 4}I', stream, offset)
            size = (((words[1] >> 16) & 0x1ff) + 1) * 4
            offset = words[7]
    return tasks, data[const_at:const_at + const_size]


def prefix_mil(source, prefix):
    result = 'conv_27' if prefix == 28 else f'relu_{prefix - 1}'
    needle = f'> {result} = '
    lines, found = [], False
    for line in source.splitlines():
        if line.lstrip().startswith('} -> ('):
            break
        lines.append(line)
        if needle in line:
            found = True
            break
    if not found:
        raise AssertionError(f'private source MIL has no {result}')
    lines.extend([f'        }} -> ({result});', '}'])
    return '\n'.join(lines) + '\n'


def difference_count(a, b):
    common = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:common], b[:common])) + abs(len(a) - len(b))


def compare(prefix, ours, apple, same_blob_constants):
    ours_tasks, ours_constants = hwx_parts(ours)
    apple_tasks, apple_constants = hwx_parts(apple)
    if len(ours_tasks) != prefix or len(apple_tasks) != prefix:
        raise AssertionError(
            f'p{prefix:02d}: task count ours={len(ours_tasks)} '
            f'apple={len(apple_tasks)} expected={prefix}')
    word_diffs = 0
    for ours_task, apple_task in zip(ours_tasks, apple_tasks):
        if len(ours_task) != len(apple_task):
            word_diffs += (abs(len(ours_task) - len(apple_task)) + 3) // 4
        for at in range(0, min(len(ours_task), len(apple_task)), 4):
            word_diffs += ours_task[at:at + 4] != apple_task[at:at + 4]
    # The retained prefix captures prove task scheduling but their model
    # packages were not retained and their coefficient payloads differ from
    # the retained full-model blob.  Compare constants with the matching
    # prefix of Apple's full program, which was compiled from the exact MIL
    # and weight blob used above. Section boundaries are independently fixed
    # by the prefix capture's constant-section length.
    expected_constants = same_blob_constants[:len(apple_constants)]
    constant_diffs = difference_count(ours_constants, expected_constants)
    if word_diffs or constant_diffs:
        raise AssertionError(
            f'p{prefix:02d}: task_word_diffs={word_diffs} '
            f'constant_byte_diffs={constant_diffs} '
            f'constant_sizes={len(ours_constants)}/{len(apple_constants)}')
    print(f'p{prefix:02d}: task_word_diffs=0 constant_byte_diffs=0')


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: h13_mobilenet_ring_parity.py MILHWXC')
    private_value = os.environ.get('MILHWXC_T5959_PRIVATE')
    if not private_value:
        print('H13 linked-ring private byte parity: SKIP (MILHWXC_T5959_PRIVATE unset)')
        return
    private = Path(private_value)
    model = private / 'coremlc' / 'mobilenet_linux.mlmodelc'
    source_path = model / 'model.mil'
    weight_path = model / 'weights' / 'weight.bin'
    prefix_dir = private / 'hwx' / 'prefixes'
    if not source_path.is_file() or not weight_path.is_file() or not prefix_dir.is_dir():
        print('H13 linked-ring private byte parity: SKIP (source MIL/blob or HWX prefixes absent)')
        return
    source = source_path.read_text()
    compiler = str(Path(sys.argv[1]).resolve())
    _, same_blob_constants = hwx_parts(private / 'hwx' / 'mobilenet_linux.H13.hwx')
    with tempfile.TemporaryDirectory(prefix='h13-linked-ring-') as temporary:
        temporary = Path(temporary)
        for prefix in range(1, 29):
            case = temporary / f'p{prefix:02d}'
            weights = case / 'weights'
            weights.mkdir(parents=True)
            os.symlink(weight_path, weights / 'weight.bin')
            mil = case / 'model.mil'
            mil.write_text(prefix_mil(source, prefix))
            output = case / 'out'
            run = subprocess.run(
                [compiler, '--target', 'H13', '--schedule', 'linked-ring',
                 '--format', 'hwx', '--mil', str(mil), '--model-root', str(case),
                 '--output', str(output)], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ, MILHWXC_H13_CONV_GENERIC='1'))
            if run.returncode:
                raise AssertionError(
                    f'p{prefix:02d} compile failed: {run.stderr[-800:]}')
            compare(prefix, output / 'program-0.hwx',
                    prefix_dir / f'mn_p{prefix:02d}.H13.hwx',
                    same_blob_constants)
    print('H13 linked-ring private byte parity: PASS')


if __name__ == '__main__':
    main()
