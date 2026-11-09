#!/usr/bin/env python3
"""Optional byte-exact parity against the private t-5959 H13 ResNet HWX.

Set MILHWXC_T5959_PRIVATE and MILHWXC_RESNET50_MODEL.  The test compares all
80 task descriptors and the complete constant section; private inputs are
never copied into source or emitted in diagnostics.
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
        raise AssertionError('not an H13 HWX object')
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
            if struct.unpack_from('<I', data, cursor + 8)[0] == 1:
                first_words = struct.unpack_from('<I', data, cursor + 0x818)[0]
                task_count = struct.unpack_from('<I', data, cursor + 0x81c)[0]
        cursor += size
    if cursor != command_end or task_count is None:
        raise AssertionError('malformed H13 load commands')
    text_at, text_size = sections[('__TEXT', '__text')]
    const_at, const_size = sections[('__TEXT', '__const')]
    stream = data[text_at:text_at + text_size]
    tasks, offset, size = [], 0, (first_words + 1) * 4
    for index in range(task_count):
        tasks.append(stream[offset:offset + size])
        if len(tasks[-1]) != size:
            raise AssertionError(f'task {index} is truncated')
        if index + 1 != task_count:
            words = struct.unpack_from(f'<{size // 4}I', stream, offset)
            size = (((words[1] >> 16) & 0x1ff) + 1) * 4
            offset = words[7]
    return tasks, data[const_at:const_at + const_size]


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: h13_resnet50_parity.py MILHWXC')
    private = os.environ.get('MILHWXC_T5959_PRIVATE')
    model_value = os.environ.get('MILHWXC_RESNET50_MODEL')
    if not private or not model_value:
        print('H13 ResNet-50 private byte parity: SKIP (private/model env unset)')
        return
    private = Path(private)
    model = Path(model_value)
    apple = private / 'hwx' / 'resnet50.H13.hwx'
    if not apple.is_file() or not (model / 'model.mil').is_file():
        print('H13 ResNet-50 private byte parity: SKIP (oracle/model absent)')
        return
    with tempfile.TemporaryDirectory(prefix='h13-resnet50-parity-') as tmp:
        out = Path(tmp) / 'out'
        run = subprocess.run([
            str(Path(sys.argv[1]).resolve()), '--target', 'H13',
            '--schedule', 'linked-ring', '--format', 'hwx',
            '--mil', str(model / 'model.mil'), '--model-root', str(model),
            '--output', str(out)], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        if run.returncode:
            raise AssertionError(f'compile failed: {run.stderr[-800:]}')
        ours_tasks, ours_constants = hwx_parts(out / 'program-0.hwx')
        apple_tasks, apple_constants = hwx_parts(apple)
        if len(ours_tasks) != 80 or len(apple_tasks) != 80:
            raise AssertionError('expected 80 tasks')
        task_diffs = sum(
            sum(a != b for a, b in zip(ours, expected)) +
            abs(len(ours) - len(expected))
            for ours, expected in zip(ours_tasks, apple_tasks))
        constant_diffs = sum(a != b for a, b in
                             zip(ours_constants, apple_constants)) + abs(
            len(ours_constants) - len(apple_constants))
        if task_diffs or constant_diffs:
            raise AssertionError(
                f'task_byte_diffs={task_diffs} '
                f'constant_byte_diffs={constant_diffs}')
    print('H13 ResNet-50: 80 tasks and complete constant section byte-exact PASS')


if __name__ == '__main__':
    main()
