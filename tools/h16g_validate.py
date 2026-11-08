#!/usr/bin/env python3
"""Device-free, fail-closed H16G container gate for the initial fp16 ABI.

Implements the derived SPEC-r3 container and SPEC-r4 resource rules.
This is independent of the writer and never opens an accelerator. Passing does not prove task semantics,
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
# Includes the public structured encoders' measured 1024/2048 square-view
# forms: LUT 0x3d/0xf5 and binary ALU 0x5c/0x170. These are opaque program
# metadata, not task sizes; task_stream remains the structural authority.
FORMATS = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17,
           18, 20, 21, 23, 27, 30, 31, 36, 41, 43, 61, 72, 92, 145,
           245, 368, 33, 44, 45, 66, 67, 68, 69, 101, 104, 167, 174,
           200, 209, 263, 266, 278, 395, 398, 419, 659, 791, 1054, 1583}
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


def task_stream(data: bytes, expected: int, policy: str = 'driver',
                facts: list[dict] | None = None) -> int:
    """Same bounded stream walk as ane_h16_task_stream_validate.

    Record-count and format words are opaque and never size this walk.
    """
    require(policy in ('driver', 'public'), "unknown-task-policy")
    # 'public' remains a CLI alias; both use the corrected H16 driver contract.
    prefix = b'\x01' + bytes(15)
    require(expected > 0 and 16 <= len(data) <= 0x60000 and
            len(data) % 4 == 0 and data[:16] == prefix, "task-envelope")
    pos, count = 16, 0
    while pos < len(data):
        region(pos, 16, len(data), "task-header")
        words = (u32(data, pos) >> 16) & 0x7ff
        if not words:
            require(not any(data[pos:pos + 16]), "task-padding")
            pos += 16
            require(pos < len(data), "task-final-padding")
            continue
        require(9 <= words <= (len(data) - pos) // 4, "task-size")
        end = pos + words * 4
        header = 10 if u32(data, pos + 32) & 3 == 3 else 9
        require(words >= header, "task-header")
        rec = pos + header * 4
        registers, dma = {}, []
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
            values = [u32(data, rec + 4 * i) for i in range(len(indices))]
            value_offsets = [rec + 4 * i for i in range(len(indices))]
            registers.update((index * 4, value) for index, value in zip(indices, values))
            # Bit 29 marks a BAR relocation record.  Single-register records
            # also carry configuration slots, so only address qword pairs are
            # DMA bases.  Retain their patch sites for the kernel-relocation
            # cross-check below.
            addresses = tuple(index * 4 for index in indices)
            if h & (1 << 29) and len(addresses) == 2 and addresses[1] == addresses[0] + 4:
                dma.append(dict(address=addresses[0], slot=(h >> 23) & 0x3f,
                                offset=values[0] | values[1] << 32,
                                patch=value_offsets[0]))
            rec += len(indices) * 4
        # Alignment and zero padding are mandatory between descriptors, but
        # the final descriptor must end exactly at EOF: trailing zero bytes
        # after it leave the engine armed and the next EXECUTE times out.
        padded = end if end == len(data) else (end + 15) & ~15
        require(padded <= len(data) and not any(data[end:padded]), "task-padding")
        require(padded < len(data) or padded == end, "task-final-padding")
        pos = padded
        count += 1
        if facts is not None:
            facts.append(dict(index=count - 1, registers=registers, dma=dma))
        require(count <= expected, "task-count")
    require(count == expected, "task-count")
    return count


def strided_extent(offset: int, inner: int, middle: int, outer: int,
                   middle_stride: int, outer_stride: int) -> int:
    """Last byte after a proved three-dimensional fp16 DMA walk."""
    require(all(value > 0 for value in (inner, middle, outer)), "dma-dimensions")
    extent = offset + product(middle - 1, middle_stride)
    require(extent <= U64_MAX, "integer-overflow")
    extent += product(outer - 1, outer_stride)
    require(extent <= U64_MAX, "integer-overflow")
    extent += product(inner, 2)
    require(extent <= U64_MAX, "integer-overflow")
    return extent


def dma_extent(dma: dict, registers: dict, target: dict,
               family: str | None = None) -> int | None:
    """Return an exclusive DMA end when the task exposes enough geometry.

    H16 source and result engines expose dimensions in the common 0x0004 and
    0x0014 blocks.  Some matrix tasks expose only a plane span; that still
    proves one complete DMA plane and is useful for bounding the selected
    surface.  Unknown engines remain subject to the base, direction, offset,
    and kernel-relocation checks in validate_dma().
    """
    address, offset = dma['address'], dma['offset']
    # Scratch-backed normalization stages reuse the common dimension words
    # for L2 tiles, not for a linear DRAM walk.  A result plane span at 0x511c
    # is nevertheless explicit and, with 0x24, proves the complete staged
    # allocation used by QK/PV loads.
    if target['kind'] == 'scratch':
        if (family == 'layer_norm' and address == 0x4d28 and
                all(key in registers for key in (4, 8, 12))):
            return offset + product(product(registers[4], registers[8]),
                                    product(registers[12], 2))
        if address == 0x5108 and registers.get(0x511c):
            planes = registers.get(0x24, 1)
            return offset + product(planes, registers[0x511c])
        return None
    if address == 0x4d10:
        if (family == 'positional' and
                all(key in registers for key in (4, 8, 0x4d18))):
            return strided_extent(offset, registers[4], registers[8], 1,
                                  registers[0x4d18], 0)
        if all(key in registers for key in (4, 8, 12, 0x4d18, 0x4d1c)):
            return strided_extent(offset, registers[4], registers[8], registers[12],
                                  registers[0x4d18], registers[0x4d1c])
        if registers.get(0x4d24):
            return offset + product(registers.get(0x24, 1), registers[0x4d24])
        if all(key in registers for key in (4, 12, 0x4d1c)):
            return strided_extent(offset, registers[4], 1, registers[12], 0,
                                  registers[0x4d1c])
    elif address == 0x4d28:
        if (family == 'positional' and
                all(key in registers for key in (4, 8, 0x4d30))):
            return strided_extent(offset, registers[4], registers[8], 1,
                                  registers[0x4d30], 0)
        if all(key in registers for key in (4, 8, 12, 0x4d34, 0x4d3c)):
            return strided_extent(offset, registers[4], registers[8], registers[12],
                                  registers[0x4d34], registers[0x4d3c])
        if (family == 'layer_norm' and
                all(key in registers for key in (4, 12, 0x4d34))):
            return strided_extent(offset, registers[4], 1, registers[12], 0,
                                  registers[0x4d34])
    elif address == 0x5108:
        if (family == 'positional' and
                all(key in registers for key in (0x14, 0x18, 0x5110))):
            return strided_extent(offset, registers[0x14], registers[0x18], 1,
                                  registers[0x5110], 0)
        if registers.get(0x511c):
            return offset + product(registers.get(0x24, 1), registers[0x511c])
        if (family == 'linear' and
                all(key in registers for key in (0x14, 0x1c, 0x5114))):
            # Linear ResultDMA writes one channel slice into each token row.
            # 0x5150 carries the first output channel in its high halfword.
            first_output = registers.get(0x5150, 0) >> 16
            return strided_extent(offset + product(first_output, 2),
                                  registers[0x1c], 1, registers[0x14], 0,
                                  registers[0x5114])
        if (family == 'whisper-stem' and
                all(key in registers for key in (0x14, 0x5114))):
            # Stem output tasks write one channel slice into padded token
            # rows. The measured one-lane tail omits 0x1c, which means 16.
            return strided_extent(offset, registers[0x14], 1,
                                  registers.get(0x1c, 16), 0,
                                  registers[0x5114])
        if (family == 'whisper-transpose' and
                all(key in registers for key in (0x14, 0x1c, 0x5114))):
            # Transpose materializes [T,C]: C contiguous halfwords for each
            # of T rows, unlike the stem's channel-major padded surfaces.
            return strided_extent(offset, registers[0x1c], 1,
                                  registers[0x14], 0, registers[0x5114])
        if (family == 'layer_norm' and
                all(key in registers for key in (0x14, 0x18, 0x1c, 0x5110))):
            return strided_extent(offset, registers[0x14], registers[0x18],
                                  registers[0x1c], registers[0x5110],
                                  product(registers[0x18], registers[0x5110]))
        if all(key in registers for key in (0x14, 0x18, 0x1c, 0x5110, 0x5114)):
            inner, middle, outer = (registers[0x14], registers[0x18],
                                    registers[0x1c])
            first, second = registers[0x5110], registers[0x5114]
            # Split-transpose stores [H,D,S]: a padded S row, D rows per
            # plane, then H planes.  Other layout tasks use D as the inner
            # dimension, the 0x18 dimension on 0x5110, and 0x1c on 0x5114.
            if second and first == inner * second and outer * 2 <= second:
                return strided_extent(offset, outer, inner, middle, second, first)
            return strided_extent(offset, inner, middle, outer, first, second)
        if all(key in registers for key in (0x14, 0x1c, 0x5114)):
            return strided_extent(offset, registers[0x14], 1, registers[0x1c], 0,
                                  registers[0x5114])
    elif address == 0x5618 and registers.get(0x5614):
        return offset + registers[0x5614]
    return None


def attention_program(task_count: int, record_code: int, format_code: int) -> bool:
    """Recognize the formula-built attention families whose DMA law is decoded."""
    for heads in range(1, 65):
        layout = heads * 11 // 2
        if task_count == 1 and record_code == 3 and format_code in (layout, layout + 1):
            return True
        qk_format = 209 if heads == 6 else 278 if heads == 8 else 35 * heads - 1
        if task_count == 3 and record_code == 7 and format_code == qk_format:
            return True
        qkt_format = 200 if heads == 6 else 266 if heads == 8 else 33 * heads + 2
        if task_count == 4 and record_code == 9 and format_code == qkt_format:
            return True
        tail_format = 791 if heads == 6 else 1054 if heads == 8 else 132 * heads - 1
        if format_code == tail_format and (
                (task_count == heads + 1 and record_code == 2 * heads + 3) or
                (task_count == 5 and record_code == 11) or
                (task_count == heads + 11 and record_code == 2 * heads + 23)):
            return True
    return False


def linear_program(tasks: list[dict], task_count: int,
                   record_code: int, format_code: int) -> bool:
    """Recognize formula-built 1x1 convolutions over transformer tokens.

    Their metadata overlaps older convolution forms, so require the decoded
    staging and compute record signatures as well as the record/format law.
    The BAR slots are deliberately absent from the signature: one-slot-early
    programs must still enter strict DMA validation and fail there.
    """
    if (len(tasks) != task_count or task_count < 2 or format_code != 0 or
            record_code not in (0x10, 0x20)):
        return False
    staging = tasks[0]
    staging_dma = {dma['address'] for dma in staging['dma']}
    if (staging['registers'].get(0x4904) != 0x103c0c or
            0x4d10 not in staging_dma):
        return False
    for task in tasks[1:]:
        registers = task['registers']
        addresses = {dma['address'] for dma in task['dma']}
        if ((registers.get(0x4904, 0) & ~0x20010) != 0x103c00 or
                not {0x5108, 0x5510}.issubset(addresses)):
            return False
    return True


def whisper_program(tasks: list[dict], task_count: int,
                    record_code: int, format_code: int) -> str | None:
    """Recognize the formula-built stem and encoder-layout task families.

    BAR slots are excluded deliberately so the v2 one-slot-early objects are
    recognized and refused by strict DMA binding checks.
    """
    if len(tasks) != task_count or format_code != 0:
        return None
    if task_count == 1 and record_code == 3:
        task = tasks[0]
        registers = task['registers']
        addresses = {dma['address'] for dma in task['dma']}
        if (registers.get(4) == registers.get(0x14) == 1500 and
                registers.get(0x30) == 0x84000 and
                {0x4d10, 0x5108}.issubset(addresses) and
                0x5510 not in addresses):
            return 'whisper-transpose'
    if record_code != {1: 16, 2: 31, 3: 45}.get(task_count):
        return None
    for task in tasks:
        registers = task['registers']
        addresses = {dma['address'] for dma in task['dma']}
        if (registers.get(0x28) not in (0x5002a043, 0x5000c043) or
                (registers.get(0x4904, 0) & ~0x20000) != 0x103c10 or
                not {0x4d10, 0x5108, 0x5510}.issubset(addresses)):
            return None
    return 'whisper-stem'


def positional_program(tasks: list[dict], task_count: int,
                       record_code: int, format_code: int) -> bool:
    """Recognize the 1024-square binary task used for positional addition."""
    if (len(tasks) != task_count or task_count != 1 or record_code != 0x10 or
            format_code != 0x5c):
        return False
    task = tasks[0]
    registers = task['registers']
    addresses = {dma['address'] for dma in task['dma']}
    return (registers.get(4) == registers.get(8) ==
            registers.get(0x14) == registers.get(0x18) == 1024 and
            {0x4d10, 0x4d28, 0x5108}.issubset(addresses))


def layer_norm_program(tasks: list[dict], task_count: int,
                       record_code: int, format_code: int) -> bool:
    """Recognize the standalone BERT/Whisper last-axis normalization family."""
    if (len(tasks) != task_count or task_count not in (5, 11) or
            record_code != 0x10 or format_code != 20):
        return False
    first_addresses = {dma['address'] for dma in tasks[0]['dma']}
    final_addresses = {dma['address'] for dma in tasks[-1]['dma']}
    boundary_addresses = {dma['address'] for dma in tasks[-2]['dma']}
    epsilon_tasks = [task for task in tasks
                     if task['registers'].get(0x450c) == 0x37280000]
    return (bool(first_addresses & {0x4d10, 0x4d28}) and
            0x4d28 in boundary_addresses and
            {0x4d28, 0x5108}.issubset(final_addresses) and
            bool(epsilon_tasks))


def validate_dma(tasks: list[dict], program: bytes, targets: list[dict],
                 relocations: set[int], family: str | None) -> int:
    """Bind decoded-family DMA records and bound every computable extent.

    Register addresses are reused with different roles by other H16G engines.
    Apply source/result direction semantics only to decoded attention,
    transformer-linear, Whisper stem/layout, and positional-add families;
    other families retain their existing relocation and container bounds.
    """
    if family is None:
        return 0
    by_vm = {target['vm']: target for target in targets}
    checked = 0
    source = {0x4d10, 0x4d28}
    result = {0x5108}
    known = source | result | {0x5510, 0x5618}
    for task in tasks:
        for dma in task['dma']:
            if dma['address'] not in known:
                continue
            descriptor_offset = 0x10 * (dma['slot'] + 1)
            region(descriptor_offset, 8, len(program), "dma-slot")
            base = u64(program, descriptor_offset)
            require(base != 0, "dma-unbound-slot")
            target = by_vm.get(base)
            require(target is not None, "dma-unknown-aperture")
            relocated = dma['patch'] in relocations
            if dma['address'] in source:
                require((family == 'positional' and dma['address'] == 0x4d28 and
                         target['kind'] == 'kernel') or
                        target['kind'] == 'scratch' or
                        target['kind'] == 'surface' and target['prot'] == 1,
                        "dma-direction")
            elif dma['address'] in result:
                require(target['kind'] == 'scratch' or
                        target['kind'] == 'surface' and target['prot'] == 2,
                        "dma-direction")
            elif dma['address'] == 0x5618 or relocated:
                require(target['kind'] == 'kernel', "dma-kernel-aperture")
            else:
                require(target['kind'] in ('scratch', 'kernel'), "dma-coefficient-aperture")
            require(dma['offset'] < target['size'], "dma-offset")
            extent = dma_extent(dma, task['registers'], target, family)
            if extent is not None:
                require(dma['offset'] <= extent <= target['size'], "dma-extent")
                checked += 1
    return checked


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
    # Materialized transposes and the transformer families add bounded
    # measured record codes to the convolution/linear set.
    require(u32(p, 0x860) in (3, 7, 9, 11, 13, 15, 16, 18, 19, 27, 31, 32,
                              35, 39, 45),
            "program-record-code")
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
    task_facts = []
    tasks = task_stream(taskdata, u32(p, 0x830), task_policy, task_facts)
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
    dma_targets = [dict(vm=s['vm'], size=s['size'], prot=s['prot'], kind='surface')
                   for s in surface]
    if scratch:
        dma_targets.append(dict(vm=scratch['vm'], size=scratch['size'],
                                prot=scratch['prot'], kind='scratch'))
    if kernel:
        dma_targets.append(dict(vm=kernel['vm'], size=kernel['size'],
                                prot=kernel['prot'], kind='kernel'))
    record_code = u32(p, 0x860)
    whisper_family = whisper_program(task_facts, tasks, record_code, fmt)
    family = ('attention' if attention_program(tasks, record_code, fmt) else
              'linear' if linear_program(task_facts, tasks, record_code, fmt) else
              whisper_family or
              ('positional' if positional_program(
                  task_facts, tasks, record_code, fmt) else
               'layer_norm' if layer_norm_program(
                   task_facts, tasks, record_code, fmt) else None))
    dma_extents = validate_dma(task_facts, p, dma_targets, seen, family)
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
    # The initial ABI maps only external input/output surfaces. Indices are
    # local to class, matching the separate input/output request arrays.
    ordered = [short for direction in (1, 2) for short in bound
               if by_short[short]['direction'] == direction]
    require([t['short'] for t in tensors] == ordered, "tensor-order")
    indices = {1: 0, 2: 0}
    buffers = []
    for i, s in enumerate(surface):
        buffers.append(dict(apertureSlot=i, index=indices[s['prot']],
                            **{'class': s['prot'] - 1}, offsetBytes=0,
                            sizeBytes=s['size'], mappingBytes=s['size']))
        indices[s['prot']] += 1
    resources = dict(version=1, commandBytes=0xca8, bufferOffset=0x60,
                     bufferStride=0x30, customBarOffset=0xca8, customBarStride=0x18,
                     buffers=buffers, customBars=[])
    digest = hashlib.sha256(data).hexdigest()
    if owner:
        require(digest == OWNER_SHA256, "owner-fixture-identity")
    return dict(target="H16G", sha256=digest, bytes=len(data), tasks=tasks,
                tensors=len(tensors), apertures=apertures, scratch=mixed,
                owner_fixture=digest == OWNER_SHA256,
                task_policy=task_policy, driver_task_gate=True, dma_extents=dma_extents,
                execute_resources=resources,
                validation="container-structure-only")


def validate_resources(metadata: dict, expected: dict) -> None:
    """Cross-check emitted metadata against facts from validate(HWX).

    JSON metadata is advisory, never mapping or selector authority. The current
    writer has no custom-BAR description, so any appended BAR is refused here.
    """
    require(json.dumps(metadata, sort_keys=True) == json.dumps(expected, sort_keys=True),
            "execute-resource-mapping")


def uint(value: int, bits: int, code: str) -> None:
    require(type(value) is int and 0 <= value < 1 << bits, code)


def validate_custom_bars(bars: list[dict]) -> None:
    """Check record geometry only; a validated program must authorize all fields."""
    require(type(bars) is list and len(bars) <= 2, "custom-bar-count")
    fields = dict(configuration=32, valueType=32, bar=32, bitOffset=8,
                  bitWidth=8, flags=16, value=64)
    for i, bar in enumerate(bars):
        require(type(bar) is dict and bar.keys() == fields.keys(), "custom-bar-fields")
        for key, bits in fields.items():
            uint(bar[key], bits, "custom-bar-field")
        end = bar['bitOffset'] + bar['bitWidth']
        require(bar['bitWidth'] > 0 and end <= 64, "custom-bar-bit-range")
        for previous in bars[:i]:
            require(previous['configuration'] != bar['configuration'] or
                    previous['bar'] != bar['bar'] or
                    previous['bitOffset'] >= end or
                    bar['bitOffset'] >= previous['bitOffset'] + previous['bitWidth'],
                    "custom-bar-overlap")


def validate_execute(command: bytes, resources: dict, mappings: list[dict], *,
                     uuid: int, program: int, process: int, procedure: int) -> None:
    """Device-free wire comparison against an already validated resource plan.

    mappings supplies test/consumer-owned DVA and length per ordinary buffer.
    This checks geometry; it cannot establish ownership, permissions or lifetime.
    Call validate_resources with validate(HWX)['execute_resources'] first.
    """
    bars, buffers = resources['customBars'], resources['buffers']
    validate_custom_bars(bars)
    require(0 < len(buffers) <= 64 and len(mappings) == len(buffers), "execute-buffer-count")
    size = 0xca8 + len(bars) * 0x18
    require(resources['commandBytes'] == size and resources['bufferOffset'] == 0x60 and
            resources['bufferStride'] == 0x30 and resources['customBarOffset'] == 0xca8 and
            resources['customBarStride'] == 0x18 and len(command) == size, "execute-size")
    uint(uuid, 64, "execute-key")
    for value in (program, process, procedure):
        uint(value, 32, "execute-key")
    expected = bytearray(size)
    struct.pack_into('<H', expected, 4, 0x20c)
    struct.pack_into('<3I', expected, 8, program, process, procedure)
    struct.pack_into('<I', expected, 0x18, 5)
    struct.pack_into('<Q', expected, 0x20, uuid)
    struct.pack_into('<I', expected, 0x28, len(buffers))
    for i, (buffer, mapping) in enumerate(zip(buffers, mappings)):
        for key in ('index', 'class'):
            uint(buffer[key], 32, "execute-buffer-field")
        for value in (mapping['dva'], mapping['size'], buffer['offsetBytes'],
                      buffer['sizeBytes'], buffer['mappingBytes']):
            uint(value, 64, "execute-buffer-field")
        require(mapping['dva'] + mapping['size'] <= U64_MAX and
                0 < buffer['sizeBytes'] <= mapping['size'] - buffer['offsetBytes'] and
                buffer['mappingBytes'] <= mapping['size'], "execute-buffer-range")
        off = 0x60 + i * 0x30
        struct.pack_into('<3I', expected, off, 1, buffer['index'], buffer['class'])
        struct.pack_into('<I', expected, off + 0x10, 3)
        struct.pack_into('<2Q', expected, off + 0x18,
                         mapping['dva'] + buffer['offsetBytes'], buffer['sizeBytes'])
    struct.pack_into('<I', expected, 0xca0, len(bars))
    for i, bar in enumerate(bars):
        struct.pack_into('<3IBBHQ', expected, 0xca8 + i * 0x18,
                         bar['configuration'], bar['valueType'], bar['bar'],
                         bar['bitOffset'], bar['bitWidth'], bar['flags'], bar['value'])
    require(command == expected, "execute-wire")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hwx', type=Path)
    parser.add_argument('--owner', action='store_true', help='require the pinned add-128 SHA-256')
    parser.add_argument('--task-policy', choices=('driver', 'public'), default='driver',
                        help='both aliases require the corrected H16 prologue (u32 1, then twelve zero bytes)')
    parser.add_argument('--manifest', type=Path, help='cross-check this artifact h16Execute metadata')
    args = parser.parse_args()
    try:
        result = validate(args.hwx.read_bytes(), owner=args.owner, task_policy=args.task_policy)
        if args.manifest:
            manifest = json.loads(args.manifest.read_text())
            require(manifest['target'] == 'H16G', "manifest-target")
            records = [r for r in manifest['artifacts'] if r['file'] == args.hwx.name]
            require(len(records) == 1, "manifest-artifact")
            validate_resources(records[0]['h16Execute'], result['execute_resources'])
        print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
