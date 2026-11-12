#!/usr/bin/env python3
"""Audit H16G fields consumed before or during firmware LOAD.

The general container validator proves bounds and ownership.  This companion
checker names the linked-task invariants which are easy for a structurally
valid writer to get wrong: descriptor length and id, terminal engine control,
record boundaries, BAR slots, task dependency ids, and complete __KERN_0
relocation coverage.  It never opens an accelerator.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

from h16g_validate import ValidationError, validate

HWX_MAGIC = int.from_bytes(b"\xce\xfa\xef\xbe", "little")
TASK_SIZE_MASK = (1 << 11) - 1
TASK_ID_MASK = (1 << 16) - 1
RECORD_BASE_MASK = (1 << 15) - 1


class GrammarError(ValueError):
    """One named LOAD-grammar violation."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise GrammarError("h16g.load-grammar." + code)


def u32(data: bytes, offset: int) -> int:
    require(0 <= offset <= len(data) - 4, "truncated")
    return struct.unpack_from("<I", data, offset)[0]


def u64(data: bytes, offset: int) -> int:
    require(0 <= offset <= len(data) - 8, "truncated")
    return struct.unpack_from("<Q", data, offset)[0]


def cstr(data: bytes) -> bytes:
    return data.split(b"\0", 1)[0]


def object_fields(data: bytes) -> dict:
    """Locate only the immutable fields needed by the grammar walk."""
    require(len(data) >= 32 and u32(data, 0) == HWX_MAGIC, "object")
    pos = 32
    text = program = kernel = None
    for _ in range(u32(data, 16)):
        cmd, size = u32(data, pos), u32(data, pos + 4)
        require(size >= 8 and pos + size <= len(data), "command")
        if cmd == 0x19:
            segment = cstr(data[pos + 8:pos + 24])
            for index in range(u32(data, pos + 64)):
                sec = pos + 72 + 80 * index
                require(sec + 80 <= pos + size, "section")
                name = cstr(data[sec:sec + 16])
                field = {
                    "vm": u64(data, sec + 32),
                    "size": u64(data, sec + 40),
                    "off": u32(data, sec + 48),
                    "reloff": u32(data, sec + 56),
                    "nreloc": u32(data, sec + 60),
                }
                if segment == b"__TEXT" and name == b"__text":
                    text = field
                elif segment == b"__KERN_0" and name == b"__kern_0":
                    kernel = field
        elif cmd == 4 and size >= 16 and u32(data, pos + 8) == 4:
            program = data[pos:pos + size]
        pos += size
    require(text is not None and program is not None, "object-fields")
    require(text["off"] + text["size"] <= len(data), "text-range")
    relocations = set()
    for index in range(text["nreloc"]):
        at = text["reloff"] + 8 * index
        require(at + 8 <= len(data), "relocation-table")
        patch, info = struct.unpack_from("<2I", data, at)
        require(info >> 24 == 7, "relocation-kind")
        relocations.add(patch)
    require(len(relocations) == text["nreloc"], "duplicate-relocation")
    mixed = len(program) == 0x8c0
    require(mixed or len(program) == 0x8a0, "program-size")
    return {
        "stream": data[text["off"]:text["off"] + text["size"]],
        "text_off": text["off"],
        "text_size": text["size"],
        "program": program,
        "relocations": relocations,
        "kernel_size": 0 if kernel is None else kernel["size"],
        "task_count": u32(program, 0x830),
        "record_code": u32(program, 0x860),
        "format_code": u32(program, 0x8b0 if mixed else 0x890),
    }


def record_indices(header: int) -> list[int]:
    base = header & RECORD_BASE_MASK
    if header & (1 << 31):
        return [base] + [base + delta for delta in range(1, 17)
                         if header & (1 << (delta + 14))]
    return list(range(base, base + ((header >> 15) & 63) + 1))


def walk(fields: dict, *, include_tasks: bool = False,
         enforce_residual: bool = True) -> dict:
    stream = fields["stream"]
    expected = fields["task_count"]
    require(len(stream) >= 16 and stream[:16] == b"\x01" + bytes(15),
            "prologue")
    pos, tasks, bars, kernel_sites = 16, [], set(), set()
    while pos < len(stream):
        if not u32(stream, pos):
            require(pos + 16 <= len(stream) and not any(stream[pos:pos + 16]),
                    "padding")
            pos += 16
            continue
        task_start = pos
        words = (u32(stream, pos) >> 16) & TASK_SIZE_MASK
        require(words >= 9 and pos + 4 * words <= len(stream), "task-size")
        end = pos + 4 * words
        header_words = 10 if u32(stream, pos + 32) & 3 == 3 else 9
        require(words >= header_words, "task-header")
        header = [u32(stream, pos + 4 * index) for index in range(header_words)]
        require((header[0] >> 16) == words, "task-length")
        require((header[0] & TASK_ID_MASK) == len(tasks), "task-link")
        rec, registers, register_offsets, records = \
            pos + 4 * header_words, {}, {}, []
        while rec < end:
            record_at = rec
            packed = u32(stream, rec)
            indices = record_indices(packed)
            rec += 4
            require(indices and rec + 4 * len(indices) <= end, "record-size")
            addresses = [4 * index for index in indices]
            require(len(addresses) == len(set(addresses)), "record-address")
            values = [u32(stream, rec + 4 * index)
                      for index in range(len(indices))]
            for index, (address, value) in enumerate(zip(addresses, values)):
                require(address not in registers, "duplicate-register")
                registers[address] = value
                register_offsets[address] = rec + 4 * index
                if address == 0x5510:
                    kernel_sites.add(rec + 4 * index)
            slot = None
            if (packed & (1 << 29)) and not (packed & (1 << 31)) and \
                    len(addresses) == 2 and addresses[1] == addresses[0] + 4:
                slot = (packed >> 23) & 0x3f
                bars.add(slot)
            records.append({"offset": record_at, "header": packed,
                            "addresses": addresses, "slot": slot})
            rec += 4 * len(indices)
        require(rec == end, "record-end")
        tasks.append({"offset": task_start, "words": words, "header": header,
                      "registers": registers, "register_offsets": register_offsets,
                      "records": records})
        padded = end if end == len(stream) else (end + 15) & ~15
        require(padded <= len(stream) and not any(stream[end:padded]), "padding")
        pos = padded
    require(len(tasks) == expected, "task-count")

    program = fields["program"]
    for slot in bars:
        at = 0x10 * (slot + 1)
        require(at + 8 <= len(program) and u64(program, at), "unbound-bar")
    require(kernel_sites == fields["relocations"], "kernel-relocations")

    residual = (expected >= 3 and fields["record_code"] == 2 * expected + 1 and
                fields["format_code"] == 0x4c)
    if residual and enforce_residual:
        last = tasks[-1]["header"]
        require(last[2] & (1 << 26), "terminal-route")
        require((last[8] & 9) == 9, "terminal-engine-control")
        for task in tasks[:-1]:
            require(not (task["header"][2] & (1 << 26)), "early-terminal-route")
        # These register families carry an eight-bit task dependency id at
        # bits 16..23. Higher bits in that halfword select the DMA endpoint.
        # KernelCfg uses id+1; the others use the zero-based task id.
        for task in tasks:
            for address in (0x4d00, 0x4d04, 0x5100, 0x5908):
                if address in task["registers"]:
                    require(((task["registers"][address] >> 16) & 0xff) < expected,
                            "dependency-task")
            if 0x5500 in task["registers"]:
                dependency = (task["registers"][0x5500] >> 16) & 0xff
                require(1 <= dependency <= expected, "kernel-dependency-task")
            for address in range(0x5520, 0x5560, 4):
                if address in task["registers"]:
                    require(((task["registers"][address] >> 16) & 0xff) < expected,
                            "kernel-lane-task")

    result = {
        "tasks": len(tasks),
        "records": sum(len(task["records"]) for task in tasks),
        "record_code": fields["record_code"],
        "format_code": fields["format_code"],
        "bar_slots": sorted(bars),
        "kernel_relocations": len(kernel_sites),
        "residual": residual,
        "terminal_control": f"0x{tasks[-1]['header'][8]:08x}",
    }
    if include_tasks:
        result["_tasks"] = tasks
    return result


def check(path: Path) -> dict:
    data = path.read_bytes()
    # Run the independent container/resource validator first.  The grammar
    # checker intentionally strengthens it; it is not an alternate parser.
    validated = validate(data)
    result = walk(object_fields(data))
    result.update(path=str(path), sha256=validated["sha256"],
                  driver_task_gate=validated["driver_task_gate"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hwx", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = []
    try:
        for path in args.hwx:
            results.append(check(path))
    except (OSError, ValidationError, GrammarError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print("PASS", result["path"],
                  f"tasks={result['tasks']}", f"records={result['records']}",
                  f"bars={result['bar_slots']}",
                  f"relocations={result['kernel_relocations']}",
                  f"terminal={result['terminal_control']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
