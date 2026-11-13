#!/usr/bin/env python3
"""Build a private one-field-group H16G LUT/reduction hang-bisection kit.

Both inputs are compiler-produced ANEC objects. The tool decodes their public
task record grammar, constructs a union record frame, and emits cumulative and
single-group morphs. It never reads an Apple binary or writes source fixtures.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys


CASES = (
    "exp-128x128-main-covered",
    "reduce_max-c1h64w64-a3-main-covered",
)
LOAD_ONLY_CASES = (
    "reduce_max-c7h33w19-a2-held-out",
    "reduce_mean-c7h33w19-a2-held-out",
    "reduce_sum-c7h33w19-a2-held-out",
    "reduce_max-c32h64w16-a2-main-covered",
    "reduce_mean-c32h64w16-a2-main-covered",
    "reduce_sum-c32h64w16-a2-main-covered",
)
GROUPS = (
    ("framing-noop", None),
    ("constant-region", None),
    ("Store", (0x5900, 0x592c)),
    ("task-header", None),
    ("Other", (0x44d0, 0x4538)),
    ("KernelDMA", (0x5500, 0x561c)),
    ("SourceDMA", (0x4d00, 0x4e40)),
    ("NE", (0x4900, 0x492c)),
    ("L2", (0x4100, 0x41a0)),
    ("Common", (0x0000, 0x0058)),
    ("ResultDMA", (0x5100, 0x5150)),
)
FIELD_PROBES = {
    "exp": (
        ("l2-input-plane-stride", 0x4118),
        ("l2-input-batch-stride", 0x411c),
        ("l2-input-surface-span", 0x4120),
        ("l2-output-tile-base", 0x414c),
        ("l2-output-row-stride", 0x4150),
        ("l2-output-plane-stride", 0x4154),
        ("l2-output-batch-stride", 0x4158),
        ("l2-output-surface-span", 0x415c),
        ("common-resident-layout", 0x0030),
        ("common-tile-or-partial-base", 0x0034),
        ("common-kernel-or-output-base", 0x003c),
        ("common-tile-count", 0x0040),
    ),
    "reduce-max": (
        ("other-reduction-selector", 0x4500),
        ("source-row-stride", 0x4d18),
        ("source-plane-stride", 0x4d1c),
        ("source-batch-span", 0x4d24),
        ("result-row-stride", 0x5110),
        ("result-plane-stride", 0x5114),
        ("result-batch-span", 0x511c),
        ("common-height", 0x0008),
        ("common-channels", 0x000c),
        ("common-output-width", 0x0014),
        ("common-output-height", 0x0018),
        ("common-output-channels", 0x001c),
        ("common-reduction-axis", 0x0024),
        ("common-tile-or-partial-base", 0x0034),
        ("common-kernel-or-output-base", 0x003c),
        ("common-tile-count", 0x0040),
        ("common-schedule-class", 0x0044),
        ("common-surface-limit", 0x004c),
    ),
}
HEADER_NAMES = {
    1: "execution-cycles",
    2: "output-record-format",
    3: "extended-header",
    4: "dispatch-mask-class-format",
    5: "schedule-hint-0",
    6: "schedule-hint-1",
    7: "dependencies",
    8: "task-control",
}
# Public linear-task grammar widths: 11 size bits, 15 register-base bits,
# and the low 16 bits of header word zero for the task identifier.
TASK_WORD_COUNT_MASK = (1 << 11) - 1
REGISTER_BASE_MASK = (1 << 15) - 1
TASK_IDENTIFIER_MASK = (1 << 16) - 1


def u32(data: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def put32(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", data, offset, value)


def cstring(raw: bytes) -> bytes:
    return raw.split(b"\0", 1)[0]


def macho_bytes(raw: bytes | bytearray) -> dict:
    data = bytearray(raw)
    command_count = u32(data, 16)
    cursor = 32
    sections = {}
    program = None
    for _ in range(command_count):
        command, size = struct.unpack_from("<2I", data, cursor)
        if command == 0x19:
            segment = cstring(bytes(data[cursor + 8:cursor + 24]))
            section_count = u32(data, cursor + 64)
            section_cursor = cursor + 72
            for _ in range(section_count):
                section = cstring(bytes(data[section_cursor:section_cursor + 16]))
                address, length, file_offset = struct.unpack_from(
                    "<2QI", data, section_cursor + 32)
                sections[(segment, section)] = {
                    "address": address,
                    "size": length,
                    "offset": file_offset,
                    "size_field": section_cursor + 40,
                    "reloff": u32(data, section_cursor + 56),
                    "nreloc": u32(data, section_cursor + 60),
                    "nreloc_field": section_cursor + 60,
                }
                section_cursor += 80
        elif command == 4 and u32(data, cursor + 8) == 4:
            program = cursor
        cursor += size
    if program is None or (b"__TEXT", b"__text") not in sections:
        raise ValueError("not a supported H16G ANEC object")
    return {"data": data, "sections": sections, "program": program}


def macho(path: Path) -> dict:
    try:
        return macho_bytes(path.read_bytes())
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def task(stream: bytes) -> dict:
    if stream[:16] != b"\1" + bytes(15):
        raise ValueError("task stream has no H16G linear prologue")
    words = (u32(stream, 16) >> 16) & TASK_WORD_COUNT_MASK
    end = 16 + words * 4
    if words < 9 or end != len(stream):
        raise ValueError("morph tool requires exactly one unpadded task")
    header = list(struct.unpack_from("<9I", stream, 16))
    registers = {}
    offsets = {}
    records = []
    cursor = 16 + 9 * 4
    while cursor < end:
        record_start = cursor
        record = u32(stream, cursor)
        base = record & REGISTER_BASE_MASK
        if record & (1 << 31):
            indices = [base] + [base + bit + 1 for bit in range(16)
                                if record & (1 << (15 + bit))]
        else:
            count = (record >> 15) & 63
            indices = list(range(base, base + count + 1))
        cursor += 4
        value_start = cursor
        for index in indices:
            address = index * 4
            registers[address] = u32(stream, cursor)
            offsets[address] = cursor
            cursor += 4
        records.append({
            "header": record,
            "addresses": [index * 4 for index in indices],
            "value_start": value_start,
            "start": record_start,
            "end": cursor,
            "raw": stream[record_start:cursor],
            "relocations": [],
        })
    return {"header": header, "registers": registers, "offsets": offsets,
            "records": records}


def decoded_task(image: dict) -> dict:
    text = image["sections"][(b"__TEXT", b"__text")]
    parsed = task(section_bytes(image, (b"__TEXT", b"__text")))
    for index in range(text["nreloc"]):
        patch, info = struct.unpack_from(
            "<2I", image["data"], text["reloff"] + index * 8)
        owner = next((record for record in parsed["records"]
                      if record["value_start"] <= patch < record["end"]), None)
        if owner is None or patch % 4:
            raise ValueError("task relocation is not owned by a record value")
        owner["relocations"].append((patch - owner["start"], info))
    return parsed


def section_bytes(image: dict, key: tuple[bytes, bytes]) -> bytes:
    section = image["sections"].get(key)
    if not section:
        return b""
    start = section["offset"]
    return bytes(image["data"][start:start + section["size"]])


def group_for(address: int) -> str | None:
    for name, bounds in GROUPS:
        if bounds and bounds[0] <= address <= bounds[1]:
            return name
    return None


def record_group(record: dict) -> str:
    groups = {group_for(address) for address in record["addresses"]}
    if len(groups) != 1 or None in groups:
        raise ValueError("task record crosses named field groups")
    return groups.pop()


def compose_task(old: dict, current: dict,
                 selected: set[str]) -> tuple[bytes, list[tuple[int, int]], int]:
    """Compose one task without inventing writes or dropping relocations.

    Unselected records remain byte-for-byte old-main records. A selected field
    group is replaced by that group's complete rule-built records, including
    BAR flags and Mach-O relocation ownership. The framing-only case selects
    no semantic group and consequently reproduces the old task exactly.
    """
    old_stream = section_bytes(old, (b"__TEXT", b"__text"))
    old_task, current_task = decoded_task(old), decoded_task(current)
    record_groups = {name for name, bounds in GROUPS if bounds is not None}
    if not (selected & record_groups):
        result = bytearray(old_stream)
        if "task-header" in selected:
            for index in range(1, 9):
                put32(result, 16 + index * 4, current_task["header"][index])
        relocations = [(record["start"] + relative, info)
                       for record in old_task["records"]
                       for relative, info in record["relocations"]]
        return bytes(result), relocations, len(old_task["records"])
    old_groups: dict[str, list[dict]] = {}
    current_groups: dict[str, list[dict]] = {}
    for parsed, grouped in ((old_task, old_groups),
                            (current_task, current_groups)):
        for record in parsed["records"]:
            grouped.setdefault(record_group(record), []).append(record)

    header = (current_task["header"] if "task-header" in selected
              else old_task["header"])
    result = bytearray(old_stream[:16])
    result.extend(struct.pack("<9I", *header))
    relocations: list[tuple[int, int]] = []
    record_count = 0

    def append_records(records: list[dict]) -> None:
        nonlocal record_count
        for record in records:
            start = len(result)
            result.extend(record["raw"])
            relocations.extend((start + relative, info)
                               for relative, info in record["relocations"])
            record_count += 1

    replaced: set[str] = set()
    for record in old_task["records"]:
        group = record_group(record)
        if group in selected:
            if group not in replaced:
                append_records(current_groups.get(group, []))
                replaced.add(group)
        else:
            append_records([record])
    for group, records in current_groups.items():
        if group in selected and group not in replaced:
            append_records(records)
            replaced.add(group)

    words = (len(result) - 16) // 4
    put32(result, 16, (words << 16) |
          (header[0] & TASK_IDENTIFIER_MASK))
    if selected <= {"framing-noop", "constant-region"}:
        if bytes(result) != old_stream:
            raise ValueError("framing no-op changed the old task stream")
    return bytes(result), relocations, record_count


def morph(old: dict, current: dict, selected: set[str]) -> bytes:
    stream, relocations, record_count = compose_task(old, current, selected)
    image = {"data": bytearray(current["data"]),
             "sections": copy.deepcopy(current["sections"]),
             "program": current["program"]}
    text = image["sections"][(b"__TEXT", b"__text")]
    later_offsets = [section["offset"] for section in image["sections"].values()
                     if section["offset"] > text["offset"]]
    capacity_end = min(later_offsets)
    if text["offset"] + len(stream) > capacity_end:
        raise ValueError("composed task records do not fit before the next section")
    image["data"][text["offset"]:capacity_end] = bytes(capacity_end - text["offset"])
    image["data"][text["offset"]:text["offset"] + len(stream)] = stream
    struct.pack_into("<Q", image["data"], text["size_field"], len(stream))
    put32(image["data"], image["program"] + 0x824, len(stream) // 4)
    put32(image["data"], image["program"] + 0x860, record_count)
    text["size"] = len(stream)

    if len(relocations) > text["nreloc"]:
        raise ValueError("composed task needs more relocation slots")
    for index in range(max(text["nreloc"], len(relocations))):
        struct.pack_into("<2I", image["data"], text["reloff"] + index * 8, 0, 0)
    for index, (patch, info) in enumerate(relocations):
        struct.pack_into("<2I", image["data"], text["reloff"] + index * 8,
                         patch, info)
    put32(image["data"], text["nreloc_field"], len(relocations))
    text["nreloc"] = len(relocations)

    key = (b"__KERN_0", b"__kern_0")
    kernel = image["sections"].get(key)
    source_kernel = section_bytes(
        current if "constant-region" in selected else old, key)
    if kernel and source_kernel:
        if len(source_kernel) != kernel["size"]:
            raise ValueError("constant regions have different sizes")
        image["data"][kernel["offset"]:kernel["offset"] + kernel["size"]] = source_kernel

    result = bytes(image["data"])
    check = macho_bytes(result)
    parsed = decoded_task(check)
    old_task, current_task = decoded_task(old), decoded_task(current)
    expected_registers = dict(old_task["registers"])
    for group, bounds in GROUPS:
        if bounds is None or group not in selected:
            continue
        expected_registers = {address: value for address, value in
                              expected_registers.items()
                              if group_for(address) != group}
        expected_registers.update({address: value for address, value in
                                  current_task["registers"].items()
                                  if group_for(address) == group})
    expected_header = (current_task["header"][1:] if "task-header" in selected
                       else old_task["header"][1:])
    if (parsed["registers"] != expected_registers or
            parsed["header"][1:] != expected_header):
        raise ValueError("composed task failed its decoded-field gate")
    return result


def install_task_stream(container: dict, kernel_source: dict, stream: bytes,
                        relocations: list[tuple[int, int]],
                        record_count: int) -> bytes:
    """Install a composed task stream in a compiler-produced container."""
    image = {"data": bytearray(container["data"]),
             "sections": copy.deepcopy(container["sections"]),
             "program": container["program"]}
    text = image["sections"][(b"__TEXT", b"__text")]
    later_offsets = [section["offset"] for section in image["sections"].values()
                     if section["offset"] > text["offset"]]
    capacity_end = min(later_offsets)
    if text["offset"] + len(stream) > capacity_end:
        raise ValueError("composed task records do not fit before the next section")
    image["data"][text["offset"]:capacity_end] = bytes(
        capacity_end - text["offset"])
    image["data"][text["offset"]:text["offset"] + len(stream)] = stream
    struct.pack_into("<Q", image["data"], text["size_field"], len(stream))
    put32(image["data"], image["program"] + 0x824, len(stream) // 4)
    put32(image["data"], image["program"] + 0x860, record_count)

    if len(relocations) > text["nreloc"]:
        raise ValueError("composed task needs more relocation slots")
    for index in range(max(text["nreloc"], len(relocations))):
        struct.pack_into("<2I", image["data"], text["reloff"] + index * 8, 0, 0)
    for index, (patch, info) in enumerate(relocations):
        struct.pack_into("<2I", image["data"], text["reloff"] + index * 8,
                         patch, info)
    put32(image["data"], text["nreloc_field"], len(relocations))

    key = (b"__KERN_0", b"__kern_0")
    kernel = image["sections"].get(key)
    source_kernel = section_bytes(kernel_source, key)
    if kernel and source_kernel:
        if len(source_kernel) != kernel["size"]:
            raise ValueError("constant regions have different sizes")
        image["data"][kernel["offset"]:kernel["offset"] + kernel["size"]] = \
            source_kernel
    return bytes(image["data"])


def old_relocations(parsed: dict) -> list[tuple[int, int]]:
    return [(record["start"] + relative, info)
            for record in parsed["records"]
            for relative, info in record["relocations"]]


def plain_records(values: list[tuple[int, int]]) -> list[bytes]:
    """Encode ordinary contiguous records without inventing reset writes."""
    result = []
    run: list[tuple[int, int]] = []
    for item in values:
        if run and (item[0] != run[-1][0] + 4 or len(run) == 64):
            header = run[0][0] // 4 | ((len(run) - 1) << 15)
            result.append(struct.pack("<I", header) +
                          b"".join(struct.pack("<I", value)
                                   for _, value in run))
            run = []
        run.append(item)
    if run:
        header = run[0][0] // 4 | ((len(run) - 1) << 15)
        result.append(struct.pack("<I", header) +
                      b"".join(struct.pack("<I", value)
                               for _, value in run))
    return result


def field_morph(container: dict, old: dict, probe: dict,
                address: int) -> bytes:
    """Switch one register field from old-main to the probe compiler output."""
    old_stream = section_bytes(old, (b"__TEXT", b"__text"))
    old_task, probe_task = decoded_task(old), decoded_task(probe)
    old_value = old_task["registers"].get(address)
    probe_value = probe_task["registers"].get(address)
    if old_value == probe_value:
        raise ValueError(f"field 0x{address:x} does not differ")
    target_group = group_for(address)
    result = bytearray(old_stream[:16 + 9 * 4])
    relocations: list[tuple[int, int]] = []
    record_count = 0
    inserted = False

    def append_raw(raw: bytes, record: dict | None = None) -> None:
        nonlocal record_count
        start = len(result)
        result.extend(raw)
        if record:
            relocations.extend((start + relative, info)
                               for relative, info in record["relocations"])
        record_count += 1

    for index, record in enumerate(old_task["records"]):
        if address in record["addresses"]:
            if probe_value is None:
                if record["relocations"] or record["header"] & (1 << 29):
                    raise ValueError("cannot remove a relocated field")
                kept = [(field, old_task["registers"][field])
                        for field in record["addresses"] if field != address]
                for raw in plain_records(kept):
                    append_raw(raw)
            else:
                raw = bytearray(record["raw"])
                value_index = record["addresses"].index(address)
                put32(raw, 4 + value_index * 4, probe_value)
                append_raw(bytes(raw), record)
            inserted = True
            continue
        append_raw(record["raw"], record)
        next_group = (record_group(old_task["records"][index + 1])
                      if index + 1 < len(old_task["records"]) else None)
        if (not inserted and old_value is None and
                record_group(record) == target_group and
                next_group != target_group):
            for raw in plain_records([(address, probe_value)]):
                append_raw(raw)
            inserted = True
    if not inserted:
        raise ValueError(f"no insertion point for field 0x{address:x}")
    words = (len(result) - 16) // 4
    put32(result, 16, (words << 16) |
          (old_task["header"][0] & TASK_IDENTIFIER_MASK))
    built = install_task_stream(container, old, bytes(result), relocations,
                                record_count)
    parsed = decoded_task(macho_bytes(built))
    expected = dict(old_task["registers"])
    if probe_value is None:
        expected.pop(address)
    else:
        expected[address] = probe_value
    if parsed["registers"] != expected or parsed["header"][1:] != old_task["header"][1:]:
        raise ValueError("single-field morph failed its decoded-field gate")
    return built


def task_positions(stream: bytes) -> list[int]:
    positions = []
    position = 16
    while position < len(stream):
        if stream[position:position + 16] == bytes(16):
            position += 16
            continue
        words = (u32(stream, position) >> 16) & TASK_WORD_COUNT_MASK
        if words < 9 or position + words * 4 > len(stream):
            raise ValueError("invalid multi-task stream")
        positions.append(position)
        end = position + words * 4
        position = end if end == len(stream) else (end + 15) & ~15
    return positions


def header_field_morph(container: dict, old: dict, probe: dict,
                       task_index: int, word_index: int) -> bytes:
    if not 1 <= word_index <= 8:
        raise ValueError("task word count/identifier is body-derived")
    old_stream = bytearray(section_bytes(old, (b"__TEXT", b"__text")))
    probe_stream = section_bytes(probe, (b"__TEXT", b"__text"))
    old_positions, probe_positions = task_positions(old_stream), task_positions(probe_stream)
    if len(old_positions) != len(probe_positions) or task_index >= len(old_positions):
        raise ValueError("header probe task mismatch")
    old_offset = old_positions[task_index] + word_index * 4
    probe_offset = probe_positions[task_index] + word_index * 4
    value = u32(probe_stream, probe_offset)
    if u32(old_stream, old_offset) == value:
        raise ValueError("header field does not differ")
    put32(old_stream, old_offset, value)
    parsed = decoded_task(old) if len(old_positions) == 1 else None
    if parsed:
        relocations = old_relocations(parsed)
        record_count = len(parsed["records"])
    else:
        text = old["sections"][(b"__TEXT", b"__text")]
        relocations = [struct.unpack_from("<2I", old["data"],
                       text["reloff"] + index * 8)
                       for index in range(text["nreloc"])]
        # The container metadata already bounds this hint; retain it.
        record_count = u32(old["data"], old["program"] + 0x860)
    return install_task_stream(container, old, bytes(old_stream), relocations,
                               record_count)


def validate(validator: Path, path: Path) -> None:
    result = subprocess.run([sys.executable, str(validator), str(path)],
                            text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"{path}: {result.stderr or result.stdout}")


def write_case(output: Path, name: str, program: bytes, source: Path,
               base: dict, validator: Path, morph_info: dict) -> dict:
    directory = output / name
    directory.mkdir()
    program_path = directory / "program.anec"
    program_path.write_bytes(program)
    shutil.copy2(source / "input.bin", directory / "input.bin")
    shutil.copy2(source / "reference.float64.bin",
                 directory / "reference.float64.bin")
    validate(validator, program_path)
    entry = {
        key: value for key, value in base.items()
        if key in ("tolerance", "inputShape", "outputShape")
    }
    entry.update(name=name, program=f"{name}/program.anec",
                 input=f"{name}/input.bin",
                 reference=f"{name}/reference.float64.bin",
                 coverage=("load-only" if morph_info.get("execute") is False
                           else "morph"),
                 tasks=base.get("tasks", 1), morph=morph_info)
    if morph_info.get("execute") is False:
        entry["execute"] = False
    (directory / "manifest.json").write_text(json.dumps(entry, indent=2) + "\n")
    return entry


def read_base_manifest(source: Path) -> dict:
    return json.loads((source / "manifest.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_kit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--probe-kit", type=Path,
        help="previous compiler kit whose failing field values are isolated")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    probe_kit = args.probe_kit or args.source_kit
    entries = []
    cases = {}
    for case_name in CASES:
        source = args.source_kit / case_name
        base = read_base_manifest(source)
        old_image = macho(source / "old-main.anec")
        current_image = macho(source / "program.anec")
        probe_image = macho(probe_kit / case_name / "program.anec")
        prefix = "exp" if case_name.startswith("exp-") else "reduce-max"
        cases[prefix] = (case_name, source, base, old_image, current_image,
                         probe_image)

    # Both no-op gates are first.  No field probe counts unless c01 reproduces
    # its exact 00 output bit-for-bit with zero DART faults.
    for prefix in ("exp", "reduce-max"):
        _, source, base, old_image, current_image, _ = cases[prefix]
        entries.append(write_case(args.output, f"{prefix}-00-old-main",
            bytes(old_image["data"]), source, base, args.validator,
            {"mode": "baseline", "source": "old-main"}))
        program = morph(old_image, current_image, {"framing-noop"})
        old_task = decoded_task(old_image)
        framed_task = decoded_task(macho_bytes(program))
        if (old_task["header"] != framed_task["header"] or
                old_task["registers"] != framed_task["registers"]):
            raise RuntimeError(f"{prefix}: c01 is not a decoded no-op")
        entries.append(write_case(args.output,
            f"{prefix}-c01-framing-noop", program, source, base,
            args.validator,
            {"mode": "gate", "gate": f"{prefix}-00-old-main",
             "requiredOutput": "bit-exact"}))

    for prefix in ("exp", "reduce-max"):
        _, source, base, old_image, current_image, probe_image = cases[prefix]
        for index, (field_name, address) in enumerate(FIELD_PROBES[prefix], 1):
            entries.append(write_case(args.output,
                f"{prefix}-f{index:02d}-{field_name}",
                field_morph(current_image, old_image, probe_image, address),
                source, base, args.validator,
                {"mode": "single-field", "step": index,
                 "group": group_for(address), "field": field_name,
                 "address": f"0x{address:04x}", "source": "rule-built-r4"}))
        if prefix == "reduce-max":
            entries.append(write_case(args.output,
                "reduce-max-f19-task-control",
                header_field_morph(current_image, old_image, probe_image, 0, 8),
                source, base, args.validator,
                {"mode": "single-field", "step": 19,
                 "group": "task-header", "field": "task-control",
                 "word": 8, "source": "rule-built-r4"}))
        entries.append(write_case(args.output, f"{prefix}-99-rule-built-exact",
            bytes(current_image["data"]), source, base, args.validator,
            {"mode": "endpoint", "source": "rule-built-r5"}))

    for case_name in LOAD_ONLY_CASES:
        source = args.source_kit / case_name
        base = read_base_manifest(source)
        short = case_name.replace("reduce_", "reduce-").replace("-main-covered", "")
        old_path = source / "old-main.anec"
        if old_path.exists():
            old_image = macho(old_path)
            current_image = macho(source / "program.anec")
            probe_image = macho(probe_kit / case_name / "program.anec")
            entries.append(write_case(args.output,
                f"load-{short}-00-old-main", old_path.read_bytes(),
                source, base, args.validator,
                {"mode": "load-only", "source": "old-main",
                 "execute": False}))
            old_stream = section_bytes(old_image, (b"__TEXT", b"__text"))
            probe_stream = section_bytes(probe_image, (b"__TEXT", b"__text"))
            old_positions = task_positions(old_stream)
            probe_positions = task_positions(probe_stream)
            for task_index, (old_position, probe_position) in enumerate(
                    zip(old_positions, probe_positions)):
                for word_index, field_name in HEADER_NAMES.items():
                    if (u32(old_stream, old_position + word_index * 4) ==
                            u32(probe_stream, probe_position + word_index * 4)):
                        continue
                    name = (f"load-{short}-t{task_index}-h{word_index}-"
                            f"{field_name}")
                    entries.append(write_case(args.output, name,
                        header_field_morph(current_image, old_image, probe_image,
                                           task_index, word_index),
                        source, base, args.validator,
                        {"mode": "load-only-single-header-field",
                         "source": "rule-built-r4", "task": task_index,
                         "word": word_index, "field": field_name,
                         "execute": False}))
        current_path = source / "program.anec"
        entries.append(write_case(args.output,
            f"load-{short}-99-rule-built-exact", current_path.read_bytes(),
            source, base, args.validator,
            {"mode": "load-only", "source": "rule-built-r5",
             "execute": False}))

    (args.output / "cases.json").write_text(json.dumps(entries, indent=2) + "\n")
    readme = """# H16G LUT/reduce one-field morph kit

This private kit is composed only from our old-main, r4 rule-built, and r5
rule-built compiler outputs. Run one case per boot, smallest/ordered first,
and recover after every LOAD or EXECUTE failure. `fNN` cases switch exactly
one named r4 field into the old-main task. `00` and `99` are exact old-main
and fixed r5 endpoints.

`c01-framing-noop` retains the exact old-main task
header, records, field presence, values, BAR flags and relocation target in a
roomier container. Their decoded fields are checked equal to `00` while the
kit is built. On device, c01 must reproduce 00's output bit-exactly with zero
DART faults before any later morph result counts. Stop the ladder if it does
not. No absent-side register is ever materialized as an explicit zero write.

Only the r4 failing groups are split: exp L2 and Common; reduction task
control, Other, SourceDMA, ResultDMA and Common. Wrong-output fields precede
known hang fields. Every case carries its field name and register address.

The `load-*` cases cover all axis-2 LOAD-timeout shapes and operations. Where
an old-main object exists, each task-header word that differed in r4 is
switched separately on the old-main task body. Their manifests say
`execute: false`: load and immediately unload; do not issue EXECUTE.
"""
    (args.output / "README.md").write_text(readme)
    sums = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sums.append(f"{digest}  {path.relative_to(args.output)}")
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"H16G morph kit: {len(entries)} validated cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
