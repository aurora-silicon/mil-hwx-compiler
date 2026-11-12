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
GROUPS = (
    ("header-record-framing", None),
    ("constant-region", None),
    ("Store", (0x5900, 0x592c)),
    ("Other", (0x44d0, 0x4538)),
    ("KernelDMA", (0x5500, 0x561c)),
    ("SourceDMA", (0x4d00, 0x4e40)),
    ("NE", (0x4900, 0x492c)),
    ("L2", (0x4100, 0x41a0)),
    ("Common", (0x0000, 0x0058)),
    ("ResultDMA", (0x5100, 0x5150)),
)
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


def macho(path: Path) -> dict:
    data = bytearray(path.read_bytes())
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
                }
                section_cursor += 80
        elif command == 4 and u32(data, cursor + 8) == 4:
            program = cursor
        cursor += size
    if program is None or (b"__TEXT", b"__text") not in sections:
        raise ValueError(f"{path}: not a supported H16G ANEC object")
    return {"data": data, "sections": sections, "program": program}


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
    cursor = 16 + 9 * 4
    while cursor < end:
        record = u32(stream, cursor)
        base = record & REGISTER_BASE_MASK
        if record & (1 << 31):
            indices = [base] + [base + bit + 1 for bit in range(16)
                                if record & (1 << (15 + bit))]
        else:
            count = (record >> 15) & 63
            indices = list(range(base, base + count + 1))
        cursor += 4
        for index in indices:
            address = index * 4
            registers[address] = u32(stream, cursor)
            offsets[address] = cursor
            cursor += 4
    return {"header": header, "registers": registers, "offsets": offsets}


def section_bytes(image: dict, key: tuple[bytes, bytes]) -> bytes:
    section = image["sections"].get(key)
    if not section:
        return b""
    start = section["offset"]
    return bytes(image["data"][start:start + section["size"]])


def append_old_only_records(current_stream: bytes, old_task: dict,
                            current_task: dict) -> bytes:
    result = bytearray(current_stream)
    missing = sorted(set(old_task["registers"]) - set(current_task["registers"]))
    for address in missing:
        # A one-value dense record makes presence explicit without changing any
        # existing record or relocation patch site.
        result += struct.pack("<2I", address // 4, old_task["registers"][address])
    words = (len(result) - 16) // 4
    put32(result, 16, (words << 16) |
          (current_task["header"][0] & TASK_IDENTIFIER_MASK))
    return bytes(result)


def seed_image(old: dict, current: dict) -> tuple[dict, dict, dict]:
    old_stream = section_bytes(old, (b"__TEXT", b"__text"))
    current_stream = section_bytes(current, (b"__TEXT", b"__text"))
    old_task, current_task = task(old_stream), task(current_stream)
    union_stream = append_old_only_records(current_stream, old_task, current_task)

    image = {"data": bytearray(current["data"]),
             "sections": copy.deepcopy(current["sections"]),
             "program": current["program"]}
    text = image["sections"][(b"__TEXT", b"__text")]
    later_offsets = [section["offset"] for section in image["sections"].values()
                     if section["offset"] > text["offset"]]
    capacity_end = min(later_offsets)
    if text["offset"] + len(union_stream) > capacity_end:
        raise ValueError("union task records do not fit before the next section")
    image["data"][text["offset"]:capacity_end] = bytes(capacity_end - text["offset"])
    image["data"][text["offset"]:text["offset"] + len(union_stream)] = union_stream
    struct.pack_into("<Q", image["data"], text["size_field"], len(union_stream))
    put32(image["data"], image["program"] + 0x824, len(union_stream) // 4)
    text["size"] = len(union_stream)

    union_task = task(union_stream)
    # The union seed represents old-main semantics. Missing writes are zero,
    # which is the descriptor reset value, while old-only writes were appended.
    for address, offset in union_task["offsets"].items():
        put32(image["data"], text["offset"] + offset,
              old_task["registers"].get(address, 0))
    for index in range(1, 9):
        put32(image["data"], text["offset"] + 16 + index * 4,
              old_task["header"][index])
    old_kernel = section_bytes(old, (b"__KERN_0", b"__kern_0"))
    kernel = image["sections"].get((b"__KERN_0", b"__kern_0"))
    if kernel and old_kernel:
        image["data"][kernel["offset"]:kernel["offset"] + kernel["size"]] = old_kernel
    return image, old_task, current_task


def group_for(address: int) -> str | None:
    for name, bounds in GROUPS:
        if bounds and bounds[0] <= address <= bounds[1]:
            return name
    return None


def morph(seed: dict, old: dict, current: dict,
          selected: set[str], current_image: dict) -> bytes:
    result = bytearray(seed["data"])
    text = seed["sections"][(b"__TEXT", b"__text")]
    parsed = task(bytes(result[text["offset"]:text["offset"] + text["size"]]))
    if "header-record-framing" in selected:
        for index in range(1, 9):
            put32(result, text["offset"] + 16 + index * 4,
                  current["header"][index])
    for address, offset in parsed["offsets"].items():
        source = current if group_for(address) in selected else old
        put32(result, text["offset"] + offset,
              source["registers"].get(address, 0))
    if "constant-region" in selected:
        key = (b"__KERN_0", b"__kern_0")
        kernel = seed["sections"].get(key)
        current_kernel = section_bytes(current_image, key)
        if kernel and current_kernel:
            result[kernel["offset"]:kernel["offset"] + kernel["size"]] = current_kernel
    return bytes(result)


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
                 coverage="morph", tasks=1, morph=morph_info)
    (directory / "manifest.json").write_text(json.dumps(entry, indent=2) + "\n")
    return entry


def read_base_manifest(source: Path) -> dict:
    return json.loads((source / "manifest.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_kit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validator", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    entries = []
    for case_name in CASES:
        source = args.source_kit / case_name
        base = read_base_manifest(source)
        old_image = macho(source / "old-main.anec")
        current_image = macho(source / "program.anec")
        seed, old_task, current_task = seed_image(old_image, current_image)
        prefix = "exp" if case_name.startswith("exp-") else "reduce-max"
        entries.append(write_case(args.output, f"{prefix}-00-old-main",
            bytes(old_image["data"]), source, base, args.validator,
            {"mode": "baseline", "source": "old-main"}))

        cumulative = set()
        for index, (group, _) in enumerate(GROUPS, 1):
            cumulative.add(group)
            entries.append(write_case(args.output,
                f"{prefix}-c{index:02d}-{group.lower()}",
                morph(seed, old_task, current_task, cumulative, current_image),
                source, base, args.validator,
                {"mode": "cumulative", "step": index,
                 "groups": sorted(cumulative)}))
        for index, (group, _) in enumerate(GROUPS, 1):
            entries.append(write_case(args.output,
                f"{prefix}-s{index:02d}-{group.lower()}",
                morph(seed, old_task, current_task, {group}, current_image),
                source, base, args.validator,
                {"mode": "single", "step": index, "groups": [group]}))
        entries.append(write_case(args.output, f"{prefix}-99-rule-built-exact",
            bytes(current_image["data"]), source, base, args.validator,
            {"mode": "endpoint", "source": "rule-built-r2"}))

    (args.output / "cases.json").write_text(json.dumps(entries, indent=2) + "\n")
    readme = """# H16G LUT/reduce one-field-group morph kit

This private kit is composed only from our old-main and rule-built compiler
outputs. Run one case per boot, smallest/ordered first, and reboot after every
LOAD or EXECUTE failure. `cNN` cases are cumulative; `sNN` cases switch only
the named group on the union record frame. `00` and `99` are the exact input
objects. The framing step adds explicit zero writes for fields absent on one
side and retains both sides' named register addresses.

The order is least likely to wedge first. Constant bytes affect LUT accuracy,
not task scheduling. Store is unchanged in both observed one-task failures.
Header execution hints and dependency words are equal apart from the derived
stream size. Other/Kernel/Source/NE then test data-path setup. L2 and Common
carry the largest geometry/allocation deltas. ResultDMA is last and is the
strongest hang hypothesis: r2 cleared its engine-enable bit in both failing
streams while Store waited for completion. The r3 compiler fixes that named
field; this morph set intentionally retains r2 values to identify the cause.
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
