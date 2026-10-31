#!/usr/bin/env python3
"""Device-free, fail-closed H16G container gate for the initial fp16 ABI.

Implements the derived SPEC-r3 container rules. This is independent of the
writer and never opens an accelerator. Passing does not prove task semantics,
firmware identity, runtime ownership, or Linux execution readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

OWNER_SHA256 = "231f54c28ea6b14a2867fac78eac1a2cb22aa3b0eeaa28113ed03c520b83a1d3"
U64_MAX = (1 << 64) - 1
FORMATS = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17,
           18, 20, 21, 23, 27, 30, 31, 36, 41, 43, 72, 145}
REGISTER_BLOCKS = ((0, 0x5c), (0x4100, 0x41a4), (0x44d0, 0x44d4),
                   (0x4500, 0x453c), (0x4900, 0x4930), (0x4d00, 0x4e44),
                   (0x5100, 0x5154), (0x5500, 0x5620), (0x5900, 0x5930))


class ValidationError(ValueError):
    """A named refusal, safe to show without dumping container bytes."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValidationError("h16g." + code)


def region(offset: int, size: int, limit: int, code: str) -> None:
    require(0 <= offset <= limit and 0 <= size <= limit - offset, code)


def unpack(fmt: str, data: bytes, offset: int = 0) -> tuple:
    region(offset, struct.calcsize(fmt), len(data), "truncated")
    return struct.unpack_from(fmt, data, offset)


def u32(data: bytes, offset: int) -> int:
    return unpack("<I", data, offset)[0]


def u64(data: bytes, offset: int) -> int:
    return unpack("<Q", data, offset)[0]


def product(a: int, b: int) -> int:
    require(a >= 0 and b >= 0 and a * b <= U64_MAX, "integer-overflow")
    return a * b


def string(data: bytes, offset: int) -> tuple[bytes, int]:
    region(offset, 1, len(data), "string-outside-command")
    end = data.find(b"\0", offset)
    require(end >= offset and end > offset, "invalid-string")
    value = data[offset:end]
    require(all(0x21 <= c <= 0x7e for c in value), "invalid-string")
    return value, end + 1


def name(raw: bytes) -> bytes:
    value, end = string(raw, 0)
    require(not any(raw[end:]), "name-padding")
    return value


def no_overlap(ranges: list[tuple[int, int]], code: str) -> None:
    ranges = sorted((a, a + n) for a, n in ranges if n)
    require(all(end <= U64_MAX for _, end in ranges), "integer-overflow")
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), code)


def task_stream(data: bytes, expected: int, policy: str = 'driver') -> int:
    """Same bounded stream walk as ane_h16_task_stream_validate.

    Record-count and format words are opaque and never size this walk.
    """
    require(policy in ('driver', 'public'), "unknown-task-policy")
    prefix = bytes(16) if policy == 'driver' else b'\x01' + bytes(15)
    require(expected > 0 and 16 <= len(data) <= 0x60000 and
            len(data) % 4 == 0 and data[:16] == prefix, "task-envelope")
    pos, count = 16, 0
    while pos < len(data):
        region(pos, 16, len(data), "task-header")
        words = (u32(data, pos) >> 16) & 0x7ff
        if not words:
            require(not any(data[pos:pos + 16]), "task-padding")
            pos += 16
            continue
        require(9 <= words <= (len(data) - pos) // 4, "task-size")
        end = pos + words * 4
        header = 10 if u32(data, pos + 32) & 3 == 3 else 9
        require(words >= header, "task-header")
        rec = pos + header * 4
        while rec < end:
            h = u32(data, rec)
            base, mask = h & 0x7fff, (h >> 15) & 0xffff
            indices = [base] + ([base + i + 1 for i in range(16) if mask & (1 << i)]
                                if h & (1 << 31) else
                                list(range(base + 1, base + ((h >> 15) & 63) + 1)))
            rec += 4
            require(len(indices) <= (end - rec) // 4, "task-record-size")
            require(all(any(a <= i * 4 < b for a, b in REGISTER_BLOCKS)
                        for i in indices), "task-register")
            rec += len(indices) * 4
        padded = (end + 15) & ~15
        require(padded <= len(data) and not any(data[end:padded]), "task-padding")
        pos = padded
        count += 1
        require(count <= expected, "task-count")
    require(count == expected, "task-count")
    return count


def tensor(raw: bytes) -> dict:
    require(len(raw) >= 0xd48 and len(raw) % 8 == 0, "tensor-size")
    direction = u32(raw, 0x14)
    require(direction in (1, 2), "tensor-direction")
    require(u32(raw, 0x24) == 5 and u64(raw, 0x68) == 2, "tensor-element")
    dims = unpack("<4I", raw, 0x28)
    require(all(dims), "tensor-dimensions")
    batch, plane, row, _ = unpack("<4Q", raw, 0x50)
    require(row >= product(dims[3], 2) and plane >= product(dims[2], row)
            and batch >= product(dims[1], plane), "tensor-strides")
    require(u64(raw, 0x70) == batch == u64(raw, 0x80), "tensor-span")
    allocation = product(dims[0], batch)
    symbol, end = string(raw, 0xd3d)
    require(u32(raw, 0x7c) == end, "tensor-short-offset")
    short, end = string(raw, end)
    require(len(short) <= 7, "tensor-short-name")
    require(len(raw) == max(0xd48, (end + 7) & ~7), "tensor-size")
    # Reconstruct only proved fields; equality also checks every reserved byte.
    expected = bytearray(len(raw))
    for offset, value in {0: 4, 4: len(raw), 8: 3, 12: 0x34a, 16: 3,
                          20: direction, 24: 0xd38, 32: 0xd3d, 36: 5,
                          56: 1, 120: 1, 124: u32(raw, 124)}.items():
        struct.pack_into("<I", expected, offset, value)
    expected[0x28:0x38] = raw[0x28:0x38]
    expected[0x50:0x78] = raw[0x50:0x78]
    expected[0x80:0x88] = raw[0x80:0x88]
    expected[0xd38:0xd3d] = b"main\0"
    expected[0xd3d:end] = symbol + b"\0" + short + b"\0"
    require(raw == expected, "tensor-fixed-or-reserved")
    return dict(direction=direction, shape=dims, strides=(batch, plane, row, 2),
                span=batch, allocation=allocation, symbol=symbol, short=short)


def validate(data: bytes, *, owner: bool = False, task_policy: str = 'driver') -> dict:
    """Return bounded structural facts, or raise ValidationError.

    No descriptor value is used to allocate a tensor or a VM mapping.
    """
    data = bytes(data)  # Validate an immutable snapshot even for bytearray callers.
    magic, cpu, subtype, kind, ncmds, cmdbytes, flags, reserved = unpack("<8I", data)
    require((magic, cpu, subtype, kind, flags, reserved) ==
            (0xbeefface, 0x80, 7, 2, 0x200000, 0), "header")
    region(32, cmdbytes, len(data), "command-table")
    require(ncmds <= cmdbytes // 8, "command-count")
    end, pos = 32 + cmdbytes, 32
    segments, sections, tensors, refs, programs, symtabs = [], [], [], [], [], []
    for _ in range(ncmds):
        region(pos, 8, end, "command-header")
        cmd, size = unpack("<2I", data, pos)
        require(size >= 8 and size % 8 == 0, "command-size")
        region(pos, size, end, "command-size")
        raw = data[pos:pos + size]
        if cmd == 0x19:
            _, _, sname, vm, vmsize, off, fsize, maxprot, prot, ns, sf = unpack("<2I16s4Q4I", raw)
            require(size == 72 + ns * 80, "section-table")
            sname = name(sname)
            require(vm % 0x4000 == 0 and vmsize > 0 and vmsize % 0x4000 == 0,
                    "segment-alignment")
            region(vm, vmsize, U64_MAX, "segment-overflow")
            require(maxprot == prot, "segment-permissions")
            region(off, fsize, len(data), "segment-file-range")
            require(fsize <= vmsize and (not fsize or off >= end), "segment-file-range")
            seg = dict(name=sname, vm=vm, size=vmsize, off=off, fsize=fsize,
                       prot=prot, flags=sf, sections=[])
            for i in range(ns):
                sn, sg, addr, length, fo, align, ro, nr, fl, r1, r2, r3 = unpack("<16s16s2Q8I", raw, 72 + i * 80)
                require(name(sg) == sname and not (r1 or r2 or r3), "section-identity")
                require(align <= 31 and addr % (1 << align) == 0, "section-alignment")
                region(addr - vm, length, vmsize, "section-vm-range")
                if fsize:
                    region(fo - off, length, fsize, "section-file-range")
                    require(addr - vm == fo - off, "section-file-vm")
                else:
                    require(fo == 0, "section-file-range")
                region(ro, nr * 8, len(data), "relocation-table")
                sec = dict(name=name(sn), vm=addr, size=length, off=fo,
                           reloff=ro, nreloc=nr, flags=fl, ordinal=len(sections) + 1)
                seg['sections'].append(sec)
                sections.append(sec)
            no_overlap([(s['vm'], s['size']) for s in seg['sections']], "section-overlap")
            segments.append(seg)
        elif cmd == 4:
            k = u32(raw, 8)
            if k == 3:
                tensors.append(tensor(raw))
            elif k == 4:
                programs.append(raw)
            else:
                raise ValidationError("h16g.unknown-custom-command")
        elif cmd == 0x40:
            require(size == 0x20 and u64(raw, 8) == 0x18, "buffer-reference")
            refs.append((u64(raw, 16), name(raw[24:32])))
        elif cmd == 2:
            require(size == 24, "symbol-table")
            symtabs.append(unpack("<4I", raw, 8))
        elif cmd == 8:
            # Compiler provenance is digest-covered but has no mapping authority.
            require(size >= 16 and b"\0" in raw[8:], "compiler-metadata")
        else:
            raise ValidationError("h16g.unknown-command")
        pos += size
    require(pos == end, "command-table")
    require(len(programs) == 1 and len(symtabs) == 1, "program-or-symbol-count")
    no_overlap([(s['vm'], s['size']) for s in segments], "segment-overlap")
    file_regions = [(0, end)] + [(s['off'], s['fsize']) for s in segments]
    no_overlap(file_regions, "file-overlap")
    surface = [s for s in segments if s['name'] == b'__FVMLIB']
    require(2 <= len(surface) <= 4 and len(tensors) == len(surface) == len(refs), "surface-count")
    for s in surface:
        require(s['fsize'] == s['off'] == 0 and len(s['sections']) == 1 and
                s['prot'] in (1, 2) and s['flags'] == 6, "surface-layout")
        sec = s['sections'][0]
        require(sec['vm'] == s['vm'] and sec['size'] == s['size'] and
                sec['name'] == (b'__const' if s['prot'] == 1 else b'__data') and
                sec['flags'] == (0x21 if s['prot'] == 1 else 0x23), "surface-layout")
    require(sum(s['prot'] == 2 for s in surface) == 1, "surface-direction")
    require(len({t['symbol'] for t in tensors}) == len(tensors) and
            len({t['short'] for t in tensors}) == len(tensors), "duplicate-tensor")
    by_short = {t['short']: t for t in tensors}
    require(len(set(refs)) == len(refs), "duplicate-buffer-reference")
    bound = []
    for s, (vm, short) in zip(surface, refs):
        require(vm == s['vm'] and short in by_short, "buffer-reference-binding")
        t = by_short[short]
        require(t['direction'] == s['prot'] and t['allocation'] <= s['size'], "tensor-binding")
        bound.append(short)
    require(len(set(bound)) == len(tensors), "duplicate-buffer-reference")
    special = {}
    for s in segments:
        if s in surface:
            continue
        require(s['name'] in (b'__PAGEZERO', b'__TEXT', b'__DATA', b'__KERN_0')
                and s['name'] not in special, "segment-identity")
        special[s['name']] = s
    require(b'__TEXT' in special, "text-missing")
    textseg = special[b'__TEXT']
    require(textseg['prot'] == 5 and [s['name'] for s in textseg['sections']] ==
            [b'__text', b'__const'] and textseg['fsize'] > 0, "text-layout")
    text, const = textseg['sections']
    require(text['vm'] == textseg['vm'] and text['flags'] in (0x28, 0x128) and
            const['flags'] == 0x26, "text-layout")
    kernel = special.get(b'__KERN_0')
    if kernel:
        require(kernel['prot'] == 1 and len(kernel['sections']) == 1 and
                kernel['sections'][0]['name'] == b'__kern_0' and
                kernel['sections'][0]['vm'] == kernel['vm'] and
                kernel['sections'][0]['flags'] == 0x26 and
                kernel['fsize'] > 0 and kernel['sections'][0]['size'] > 0, "kernel-layout")
    scratch = special.get(b'__DATA')
    if scratch:
        require(scratch['prot'] == 3 and scratch['fsize'] == scratch['off'] == 0 and
                len(scratch['sections']) == 1 and scratch['sections'][0]['name'] == b'__bss' and
                scratch['sections'][0]['vm'] == scratch['vm'] and
                scratch['sections'][0]['size'] == scratch['size'] and
                scratch['sections'][0]['flags'] == 0x25, "scratch-layout")
    if b'__PAGEZERO' in special:
        s = special[b'__PAGEZERO']
        require(s['vm'] == s['off'] == s['fsize'] == s['prot'] == 0 and
                not s['sections'], "pagezero-layout")
    p = programs[0]
    require(len(p) in (0x8a0, 0x8c0), "program-size")
    mixed = len(p) == 0x8c0
    require(u32(p, 0x860) in (0x10, 0x12, 0x1f, 0x20), "program-record-code")
    fmt = u32(p, 0x8b0 if mixed else 0x890)
    require(fmt in FORMATS, "program-format-code")
    require(u64(p, 0x10) == u64(p, 0x810) == text['vm'] and
            u64(p, 0x20) == const['vm'] and u32(p, 0x824) * 4 == text['size'], "program-text")
    require(bool(scratch) == mixed, "program-scratch")
    if mixed:
        require(u64(p, 0x40) == scratch['vm'] and
                0 < u32(p, 0x858) == u32(p, 0x880) <= scratch['sections'][0]['size'], "program-scratch")
    apertures = [s['vm'] for s in surface] + ([kernel['vm']] if kernel else [])
    require(len(apertures) <= 5, "aperture-count")
    expected = bytearray(len(p))
    words = {0: 4, 4: len(p), 8: 4, 12: 0x22a if mixed else 0x222,
             0x820: 4, 0x824: text['size'] // 4, 0x830: u32(p, 0x830),
             0x838: 0xffffffff, 0x83c: 0xffffffff, 0x844: 0xffff,
             0x848: 0x8b8 if mixed else 0x898, 0x860: u32(p, 0x860),
             0x868: 4, 0x870: 4 if mixed else 2}
    words.update(({0x858: u32(p, 0x858), 0x878: 2, 0x87c: 8,
                   0x880: u32(p, 0x858), 0x888: 3, 0x88c: 8, 0x890: 0x08020004,
                   0x894: 8, 0x898: 7, 0x89c: 8, 0x8a0: 0x461c4000,
                   0x8a8: 9, 0x8ac: 8, 0x8b0: fmt} if mixed else
                  {0x878: 7, 0x87c: 8, 0x880: 0x461c4000, 0x888: 9,
                   0x88c: 8, 0x890: fmt}))
    for off, value in words.items():
        struct.pack_into('<I', expected, off, value)
    addresses = {0x10: text['vm'], 0x20: const['vm'], 0x810: text['vm']}
    if mixed:
        addresses[0x40] = scratch['vm']
    addresses.update({0x70 + i * 16: vm for i, vm in enumerate(apertures)})
    for off, value in addresses.items():
        struct.pack_into('<Q', expected, off, value)
    noff = 0x8b8 if mixed else 0x898
    expected[noff:noff + 5] = b'main\0'
    require(p == expected, "program-fixed-reserved-or-apertures")
    taskdata = data[text['off']:text['off'] + text['size']]
    tasks = task_stream(taskdata, u32(p, 0x830), task_policy)
    # Relocation fields follow the driver gate, including section ordinal,
    # unique aligned patch sites, and bounded unsigned kernel addends.
    seen = set()
    for sec in sections:
        if sec is not text:
            require(sec['nreloc'] == 0, "relocation-section")
    require(text['nreloc'] <= text['size'] // 4, "relocation-count")
    ro, rn = text['reloff'], text['nreloc']
    if rn:
        require(kernel is not None, "relocation-kernel")
        no_overlap(file_regions + [(ro, rn * 8)], "relocation-file-overlap")
    for i in range(rn):
        patch, info = unpack('<2I', data, ro + i * 8)
        require(patch % 4 == 0 and patch not in seen and patch + 4 <= len(taskdata), "relocation-site")
        seen.add(patch)
        require(info >> 24 == 7 and info & 0xffffff == kernel['sections'][0]['ordinal'], "relocation-kind")
        require(u32(taskdata, patch) < kernel['sections'][0]['size'], "relocation-addend")
    so, ns, st, ss = symtabs[0]
    region(so, ns * 16, len(data), "symbol-range")
    region(st, ss, len(data), "string-range")
    no_overlap(file_regions + [(ro, rn * 8), (so, ns * 16), (st, ss)], "table-file-overlap")
    names, symbols = set(), {}
    strings = data[st:st + ss]
    for i in range(ns):
        idx, typ, ordinal, desc, value = unpack('<IBBHQ', data, so + i * 16)
        sym, _ = string(strings, idx)
        require(sym not in names and typ == 0xf and 1 <= ordinal <= len(sections), "symbol-identity")
        names.add(sym)
        sec = sections[ordinal - 1]
        require(sec['vm'] <= value < sec['vm'] + sec['size'], "symbol-address")
        symbols[sym] = (ordinal, value)
    for s, (_, short) in zip(surface, refs):
        require(symbols.get(short) == (s['sections'][0]['ordinal'], s['vm']), "symbol-binding")
    digest = hashlib.sha256(data).hexdigest()
    if owner:
        require(digest == OWNER_SHA256, "owner-fixture-identity")
    return dict(target="H16G", sha256=digest, bytes=len(data), tasks=tasks,
                tensors=len(tensors), apertures=apertures, scratch=mixed,
                owner_fixture=digest == OWNER_SHA256,
                task_policy=task_policy, driver_task_gate=taskdata[:16] == bytes(16),
                validation="container-structure-only")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hwx', type=Path)
    parser.add_argument('--owner', action='store_true', help='require the pinned add-128 SHA-256')
    parser.add_argument('--task-policy', choices=('driver', 'public'), default='driver',
                        help='driver: current zero prefix; public: diagnostic pinned compiler prefix 1')
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.hwx.read_bytes(), owner=args.owner,
                                  task_policy=args.task_policy), sort_keys=True))
    except (OSError, ValidationError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
