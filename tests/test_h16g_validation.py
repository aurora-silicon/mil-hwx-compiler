#!/usr/bin/env python3
"""Synthetic adversarial container checks. No captured HWX bytes are used."""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from h16g_validate import ValidationError, task_stream, validate


def put(data, offset, value, width=4):
    struct.pack_into({1: '<B', 4: '<I', 8: '<Q'}[width], data, offset, value)


def segment(label, vm, allocation, protection, fields, fileoff=0, filesize=0):
    result = bytearray(struct.pack('<2I16s4Q4I', 0x19, 72 + 80 * len(fields),
        label, vm, allocation, fileoff, filesize, protection, protection,
        len(fields), 6 if label == b'__FVMLIB' else 4 if label == b'__DATA' else 0))
    for name, addr, size, offset, reloff, nreloc, flags in fields:
        result.extend(struct.pack('<16s16s2Q8I', name, label, addr, size,
            offset, 2, reloff, nreloc, flags, 0, 0, 0))
    return result


def fixture(*, mixed=False, kernel=False, batch=1):
    """A tiny legal-shaped object, deliberately unrelated to an Apple program.

    Task is a nine-word header and one write to a valid common register.
    It proves structural validation only, never a runnable add operation.
    """
    base = 0x30000000
    vm = base + (0x4000 if mixed else 0)
    textvm = vm + 0x8000
    kernvm = textvm + 0x8000
    task = struct.pack('<16I', 0, 0, 0, 0, 11 << 16 | 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    commands = []
    if mixed:
        commands.append(segment(b'__DATA', base, 0x4000, 3,
            [(b'__bss', base, 0x4000, 0, 0, 0, 0x25)]))
    for i, direction in enumerate((1, 2)):
        commands.append(segment(b'__FVMLIB', vm + i * 0x4000, 0x4000, direction,
            [(b'__const' if direction == 1 else b'__data', vm + i * 0x4000,
              0x4000, 0, 0, 0, 0x21 if direction == 1 else 0x23)]))
    # Header-adjacent tables live at 0x3000, text at 0x4000, kernel at 0xc000.
    commands.append(segment(b'__TEXT', textvm, 0x8000, 5,
        [(b'__text', textvm, len(task), 0x4000, 0x3000, int(kernel), 0x128 if kernel else 0x28),
         (b'__const', textvm + len(task), 0x4000, 0x4000 + len(task), 0, 0, 0x26)],
        0x4000, 0x8000))
    if kernel:
        commands.append(segment(b'__KERN_0', kernvm, 0x4000, 1,
            [(b'__kern_0', kernvm, 16, 0xc000, 0, 0, 0x26)], 0xc000, 0x4000))
    for i, short in enumerate((b'x', b'y')):
        commands.append(struct.pack('<2I2Q8s', 0x40, 32, 0x18,
                                    vm + i * 0x4000, short))
    p = bytearray(0x8c0 if mixed else 0x8a0)
    fields = {0: 4, 4: len(p), 8: 4, 12: 0x22a if mixed else 0x222,
              0x820: 4, 0x824: len(task) // 4, 0x830: 1, 0x838: 0xffffffff,
              0x83c: 0xffffffff, 0x844: 0xffff,
              0x848: 0x8b8 if mixed else 0x898, 0x860: 0x10, 0x868: 4,
              0x870: 4 if mixed else 2}
    fields.update({0x858: 128, 0x878: 2, 0x87c: 8, 0x880: 128,
                   0x888: 3, 0x88c: 8, 0x890: 0x08020004, 0x894: 8,
                   0x898: 7, 0x89c: 8, 0x8a0: 0x461c4000, 0x8a8: 9,
                   0x8ac: 8, 0x8b0: 0xf} if mixed else
                  {0x878: 7, 0x87c: 8, 0x880: 0x461c4000, 0x888: 9,
                   0x88c: 8, 0x890: 0xf})
    for off, value in fields.items():
        put(p, off, value)
    for off, value in {0x10: textvm, 0x20: textvm + len(task),
                       0x810: textvm, 0x70: vm, 0x80: vm + 0x4000}.items():
        put(p, off, value, 8)
    if kernel:
        put(p, 0x90, kernvm, 8)
    if mixed:
        put(p, 0x40, base, 8)
    noff = 0x8b8 if mixed else 0x898
    p[noff:noff + 5] = b'main\0'
    commands.append(p)
    for direction, short in enumerate((b'x', b'y'), 1):
        d = bytearray(0xd48)
        for off, value in {0: 4, 4: len(d), 8: 3, 12: 0x34a, 16: 3,
                           20: direction, 24: 0xd38, 32: 0xd3d, 36: 5,
                           40: batch, 44: 1, 48: 2, 52: 3, 56: 1,
                           120: 1, 124: 0xd3f}.items():
            put(d, off, value)
        for off, value in {0x50: 12, 0x58: 12, 0x60: 6, 0x68: 2,
                           0x70: 12, 0x80: 12}.items():
            put(d, off, value, 8)
        d[0xd38:0xd3d] = b'main\0'
        d[0xd3d:0xd41] = short + b'\0' + short + b'\0'
        commands.append(d)
    commands.append(struct.pack('<6I', 2, 24, 0x3010, 2, 0x3030, 5))
    result = bytearray(0x10000 if kernel else 0xc000)
    struct.pack_into('<8I', result, 0, 0xbeefface, 0x80, 7, 2,
                     len(commands), sum(map(len, commands)), 0x200000, 0)
    pos, offsets = 32, []
    for c in commands:
        offsets.append(pos)
        result[pos:pos + len(c)] = c
        pos += len(c)
    result[0x4000:0x4000 + len(task)] = task
    ordinal = 1 + int(mixed)
    for i in range(2):
        struct.pack_into('<IBBHQ', result, 0x3010 + i * 16, 1 + i * 2,
                         0xf, ordinal + i, 0, vm + i * 0x4000)
    result[0x3030:0x3035] = b'\0x\0y\0'
    if kernel:
        struct.pack_into('<2I', result, 0x3000, 56, 0x07000000 | (5 + int(mixed)))
    return result, offsets


def command(data, offsets, cmd, kind=None):
    return next(off for off in offsets if struct.unpack_from('<I', data, off)[0] == cmd and
                (kind is None or struct.unpack_from('<I', data, off + 8)[0] == kind))


class ValidationTests(unittest.TestCase):
    def test_public_prefix_is_not_driver_compatible(self):
        raw, _ = fixture()
        raw[0x4000] = 1
        self.assertFalse(validate(raw, task_policy='public')['driver_task_gate'])
        with self.assertRaisesRegex(ValidationError, 'task-envelope'):
            validate(raw)
        raw[0x4001] = 1
        with self.assertRaisesRegex(ValidationError, 'task-envelope'):
            validate(raw, task_policy='public')

    def test_positive_forms(self):
        for mixed in (False, True):
            for kernel in (False, True):
                for batch in (1, 3):
                    with self.subTest(mixed=mixed, kernel=kernel, batch=batch):
                        result = validate(fixture(mixed=mixed, kernel=kernel, batch=batch)[0])
                        self.assertEqual(result['tasks'], 1)
                        self.assertEqual(result['scratch'], mixed)
                        self.assertFalse(result['owner_fixture'])

    def test_mutated_container(self):
        raw, offsets = fixture(kernel=True)
        p = command(raw, offsets, 4, 4)
        t = command(raw, offsets, 4, 3)
        s = command(raw, offsets, 0x19)
        ref = command(raw, offsets, 0x40)
        # Every mutation below was accepted by at least a header-only parser.
        mutations = {
            'subtype': (8, 9), 'command-count': (16, 10000),
            'command-table': (20, len(raw)), 'unaligned-command': (s + 4, 151),
            'permission': (s + 60, 3), 'allocation-overflow': (s + 28, 0xffffffff),
            'section-size': (s + 72 + 40, 0xffffffff),
            'tensor-code': (t + 12, 0x34b), 'direction': (t + 20, 2),
            'function-offset': (t + 24, 0xd37), 'tensor-element': (t + 36, 6),
            'zero-dimension': (t + 40, 0), 'row-stride': (t + 0x60, 4),
            'span-mismatch': (t + 0x70, 24), 'duplicate-span': (t + 0x80, 24),
            'short-offset': (t + 0x7c, 0xd40), 'tensor-reserved': (t + 0xc00, 1),
            'symbol-outside': (t + 32, len(raw)),
            'buffer-constant': (ref + 8, 0x19), 'buffer-interior': (ref + 16, 0x30000004),
            'aperture-interior': (p + 0x70, 0x30000004),
            'aperture-reserved': (p + 0x78, 1), 'aperture-hole': (p + 0x80, 0),
            'aperture-tail': (p + 0xb0, 0x30000000),
            'text-duplicate': (p + 0x810, 0), 'text-word-count': (p + 0x824, 15),
            'task-count': (p + 0x830, 2), 'record-code': (p + 0x860, 0x11),
            'format-code': (p + 0x890, 0x8), 'program-reserved': (p + 0x100, 1),
            'linear-scratch': (p + 0x40, 0x30000000),
            'relocation-type': (0x3004, 0x05000005),
            'relocation-ordinal': (0x3004, 0x07000001),
            'relocation-unaligned': (0x3000, 55), 'relocation-outside': (0x3000, 64),
            'relocation-addend': (0x4000 + 56, 16), 'task-prefix': (0x4000, 1),
            'task-size': (0x4010, 2047 << 16), 'task-register': (0x4034, 0x7fff),
            'task-padding': (0x403c, 1), 'symbol-ordinal': (0x3014, 0xff000f),
            'symbol-string': (0x3010, 5), 'symbol-interior': (0x3018, 0x30000004),
        }
        for label, (off, value) in mutations.items():
            with self.subTest(label=label):
                data = bytearray(raw)
                put(data, off, value)
                with self.assertRaises(ValidationError):
                    validate(data)
        for length in (0, 31, 32, 100, len(raw) - 1):
            with self.subTest(length=length), self.assertRaises(ValidationError):
                validate(raw[:length])
        with self.assertRaisesRegex(ValidationError, 'owner-fixture-identity'):
            validate(raw, owner=True)

    def test_mixed_scratch(self):
        raw, offsets = fixture(mixed=True)
        p = command(raw, offsets, 4, 4)
        for off, value in ((0x40, 0), (0x858, 0), (0x880, 256), (0x858, 0x4001),
                           (0x8b0, 8), (0x8bd, 1)):
            with self.subTest(off=off), self.assertRaises(ValidationError):
                data = bytearray(raw)
                put(data, p + off, value, 1 if off == 0x8bd else 4)
                validate(data)

    def test_tensor_product_overflow(self):
        raw, offsets = fixture()
        t = command(raw, offsets, 4, 3)
        put(raw, t + 0x28, 0xffffffff)
        for off in (0x50, 0x70, 0x80):
            put(raw, t + off, 0xffffffffffffffff, 8)
        with self.assertRaisesRegex(ValidationError, 'integer-overflow'):
            validate(raw)

    def test_parser_does_not_crash_on_single_byte_mutations(self):
        # Deterministic adversarial coverage of command, descriptor and table
        # boundaries. Mutations to ordinary data may remain structurally valid.
        raw, _ = fixture(kernel=True, mixed=True)
        import random
        rng = random.Random(3651)
        for _ in range(512):
            data = bytearray(raw)
            data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
            try:
                validate(data)
            except ValidationError:
                pass


if __name__ == '__main__':
    unittest.main()
