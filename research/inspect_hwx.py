#!/usr/bin/env python3
"""Inspect and validate H13, H14, H15, H16G, H17, or H18 HWX object structure."""
from __future__ import annotations

import struct
import sys
from h13_td import BLOCKS, decode_task, split_h13_tasks, split_h14_tasks

HWX_MAGIC = 0xBEEFFACE
# ISA versions are the reference parser's table (freedomtan/coreml_to_ane_hwx
# ce54664e get_instruction_set_version); no HWX field carries one.
ARCHITECTURES = {
    5: ("H14", 11),
    4: ("H13", 7),
    6: ("H15", 8),
    7: ("H16G", 17),
    9: ("H17", 19),
    10: ("H18", 20),
    11: ("H18G", "unknown"),
}
# Subtypes whose __TEXT/__text is an H14-format aligned task stream.
TASK_TARGETS = {5: "h14", 6: "h15", 9: "h17", 10: "h18"}


def cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def unpack_from(fmt: str, data: bytes, offset: int, label: str):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise ValueError(f"{label} is truncated at 0x{offset:x}")
    return struct.unpack_from(fmt, data, offset)


def require_region(offset: int, size: int, limit: int, label: str) -> None:
    if offset < 0 or size < 0 or offset > limit or size > limit - offset:
        raise ValueError(
            f"{label} range 0x{offset:x}+0x{size:x} exceeds file size 0x{limit:x}")


def inspect_tasks(raw_tasks: list[bytes], target: str) -> int:
    tasks = [decode_task(task, target) for task in raw_tasks]
    observed = {name: set() for name, _, _ in BLOCKS[target]}
    for task_index, task in enumerate(tasks):
        print(
            f"  {target}_task[{task_index}] "
            f"size_words={task['size_bytes'] // 4} "
            f"records={len(task['records'])}")
        for key, block in task["blocks"].items():
            if block["name"].startswith("unknown_"):
                raise ValueError(
                    f"{target.upper()} task[{task_index}] writes {key} outside "
                    f"source-backed {target.upper()} blocks")
            print(
                f"    block address={key} name={block['name']} "
                f"words={len(block['words'])}")
            observed[block["name"]].update(block["words"])
    for name, start, count in BLOCKS[target]:
        print(
            f"  {target}_block name={name} range=0x{start:05x}-"
            f"0x{start + count * 4:05x} source_words={count} "
            f"observed_words={len(observed[name])}")
    print(f"  {target}_tasks count={len(tasks)}")
    return len(tasks)


def inspect_h13_tasks(
        section_data: bytes, task_words_minus_one: int, task_count: int) -> int:
    return inspect_tasks(
        split_h13_tasks(section_data, task_words_minus_one, task_count), "h13")


def inspect_segment(data: bytes, cursor: int, command_size: int, subtype: int) -> None:
    if command_size < 72:
        raise ValueError(f"LC_SEGMENT_64 at 0x{cursor:x} is shorter than 72 bytes")
    fields = unpack_from(
        "<2I16s4Q4I", data, cursor, f"LC_SEGMENT_64 at 0x{cursor:x}")
    segment = cstring(fields[2])
    segment_file_size = fields[6]
    section_count = fields[-2]
    section_bytes = section_count * 80
    if section_bytes > command_size - 72:
        raise ValueError(
            f"LC_SEGMENT_64 at 0x{cursor:x} declares {section_count} sections "
            f"outside command size 0x{command_size:x}")
    section_cursor = cursor + 72
    for section_index in range(section_count):
        section_fields = unpack_from(
            "<16s16s2Q8I", data, section_cursor,
            f"section[{section_index}] in {segment}")
        section = cstring(section_fields[0])
        size = section_fields[3]
        offset = section_fields[4]
        relocation_offset = section_fields[6]
        relocation_count = section_fields[7]
        if segment_file_size:
            require_region(offset, size, len(data), f"{segment}/{section}")
        require_region(
            relocation_offset, relocation_count * 8, len(data),
            f"{segment}/{section} relocation table")
        print(
            f"{segment}/{section} addr=0x{section_fields[2]:x} "
            f"size=0x{size:x} offset=0x{offset:x} "
            f"reloff=0x{relocation_offset:x} nreloc={relocation_count}")
        for relocation_index in range(relocation_count):
            address, info = unpack_from(
                "<2I", data, relocation_offset + relocation_index * 8,
                f"relocation[{relocation_index}] in {segment}/{section}")
            print(
                f"  relocation[{relocation_index}] "
                f"address=0x{address:x} info=0x{info:08x}")
        target = TASK_TARGETS.get(subtype)
        if target and segment == "__TEXT" and section in ("__text", "__TEXT"):
            inspect_tasks(split_h14_tasks(data[offset:offset + size]), target)
        section_cursor += 80


def h13_anec(data: bytes) -> tuple[bytes, dict[str, int]]:
    _, _, _, _, command_count, command_bytes, _, _ = unpack_from(
        "<8I", data, 0, "Mach-O header")
    cursor = 32
    command_end = cursor + command_bytes
    sections: dict[tuple[str, str], dict[str, int]] = {}
    surfaces: list[tuple[str, int]] = []
    tensors: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []
    task_words_minus_one = None
    task_count = None
    for command_index in range(command_count):
        command, command_size = unpack_from(
            "<2I", data, cursor, f"load command[{command_index}]")
        if command_size < 8 or command_size > command_end - cursor:
            raise ValueError(f"invalid H13 load command[{command_index}]")
        kind = unpack_from("<I", data, cursor + 8, "load command kind")[0] \
            if command_size >= 12 else None
        if command == 0x19:
            segment_fields = unpack_from(
                "<2I16s4Q4I", data, cursor, "H13 segment")
            segment = cstring(segment_fields[2])
            section_cursor = cursor + 72
            for section_index in range(segment_fields[-2]):
                fields = unpack_from(
                    "<16s16s2Q8I", data, section_cursor, "H13 section")
                section = cstring(fields[0])
                sections[(segment, section)] = {
                    "address": fields[2], "size": fields[3],
                    "offset": fields[4], "allocation": segment_fields[4],
                }
                if segment == "__FVMLIB":
                    surfaces.append((section, segment_fields[4]))
                section_cursor += 80
        elif command == 4 and kind == 1 and command_size >= 0x820:
            task_words_minus_one = unpack_from(
                "<I", data, cursor + 0x818, "H13 task words")[0]
            task_count = unpack_from(
                "<I", data, cursor + 0x81c, "H13 task count")[0]
        elif command == 4 and kind == 3 and command_size >= 0x80:
            tensors.append((
                unpack_from("<4I", data, cursor + 0x28, "H13 tensor shape"),
                unpack_from("<4Q", data, cursor + 0x50, "H13 tensor strides"),
                unpack_from("<Q", data, cursor + 0x70, "H13 tensor total")[0],
                unpack_from("<I", data, cursor + 0x24, "H13 tensor element")[0],
            ))
        cursor += command_size
    if cursor != command_end:
        raise ValueError("H13 load command table size does not match its header")
    text = sections.get(("__TEXT", "__text"))
    constants = sections.get(("__TEXT", "__const"))
    if not text or not constants:
        raise ValueError("H13 HWX requires __TEXT/__text and __TEXT/__const")
    if task_words_minus_one is None or task_count is None:
        raise ValueError("H13 HWX requires a program task descriptor")
    task_bytes = (task_words_minus_one + 1) * 4
    text_data = data[text["offset"]:text["offset"] + text["size"]]
    inspect_h13_tasks(text_data, task_words_minus_one, task_count)
    constant_offset = constants["offset"] - text["offset"]
    if constant_offset < text["size"] or \
            constants["address"] - text["address"] != constant_offset:
        raise ValueError("H13 __TEXT/__const overlaps or diverges from __text")
    input_allocations = [allocation for name, allocation in surfaces
                         if name == "__const"]
    output_allocations = [allocation for name, allocation in surfaces
                          if name == "__data"]
    if len(output_allocations) != 1 or len(input_allocations) not in (1, 2) or \
            len(tensors) != len(input_allocations) + 1:
        raise ValueError("H13 HWX requires one or two inputs and one output")
    allocations = [*input_allocations, output_allocations[0]]
    for index, ((shape, strides, total, element_code), allocation) in enumerate(
            zip(tensors, allocations)):
        batch, plane, row, element = strides
        # Apple's batched surfaces declare one batch element's span as the
        # descriptor's total size and reserve the whole batch in the surface
        # allocation, so the batch only shows up in the shape.
        if element_code != 5 or element != 2 or batch != shape[1] * plane or \
                total != batch or shape[0] * batch > allocation:
            raise ValueError(f"H13 tensor descriptor[{index}] has an invalid layout")
    content_size = constant_offset + constants["size"]
    content = data[text["offset"]:text["offset"] + content_size]
    if len(content) != content_size:
        raise ValueError("H13 task and constant content is truncated")
    header = bytearray(0x1000)
    struct.pack_into("<QIIQQII", header, 0, content_size, task_bytes, task_count,
                     text["size"], constants["size"], len(tensors) - 1, 1)
    tiles = [0] * 32
    tiles[0] = (content_size + 0x3fff) // 0x4000
    tiles[4] = (output_allocations[0] + 0x3fff) // 0x4000
    for index, allocation in enumerate(input_allocations, 5):
        tiles[index] = (allocation + 0x3fff) // 0x4000
    struct.pack_into("<32I", header, 40, *tiles)
    layouts = [0] * (32 * 6)
    ordered = [(4, tensors[-1]), *
               ((index + 5, tensor) for index, tensor in enumerate(tensors[:-1]))]
    for channel, (shape, strides, _, _) in ordered:
        layouts[channel * 6:channel * 6 + 6] = [
            *shape, strides[1], strides[2]]
    struct.pack_into("<192Q", header, 0xa8, *layouts)
    return bytes(header) + content, {
        "contentOffset": text["offset"], "contentBytes": content_size,
        "taskBytes": task_bytes, "taskContentBytes": text["size"],
        "tasks": task_count, "constantOffset": constant_offset,
        "constantBytes": constants["size"],
        "inputs": len(tensors) - 1, "outputs": 1,
    }


def main(path: str, extract_path: str | None = None) -> None:
    with open(path, "rb") as stream:
        data = stream.read()
    header = unpack_from("<8I", data, 0, "Mach-O header")
    magic, _, subtype, _, command_count, command_bytes, _, _ = header
    if magic != HWX_MAGIC:
        raise ValueError(
            f"invalid HWX magic 0x{magic:08x}; expected 0x{HWX_MAGIC:08x}")
    architecture = ARCHITECTURES.get(subtype)
    if architecture:
        print(
            f"architecture subtype=0x{subtype:04x} "
            f"name={architecture[0]} isa={architecture[1]}")
    else:
        print(f"architecture subtype=0x{subtype:04x} name=unknown isa=unknown")
    require_region(32, command_bytes, len(data), "load command table")
    command_end = 32 + command_bytes
    cursor = 32
    for command_index in range(command_count):
        command, command_size = unpack_from(
            "<2I", data, cursor, f"load command[{command_index}]")
        if command_size < 8:
            raise ValueError(
                f"load command[{command_index}] has invalid size {command_size}")
        if command_size > command_end - cursor:
            raise ValueError(
                f"load command[{command_index}] exceeds declared command table")
        kind = unpack_from("<I", data, cursor + 8, "load command kind")[0] \
            if command_size >= 12 else None
        print(
            f"command[{command_index}] offset=0x{cursor:x} "
            f"cmd=0x{command:x} size=0x{command_size:x} "
            f"kind={kind if kind is not None else '-'}")
        if command == 0x40 and command_size == 0x20:
            address = unpack_from(
                "<Q", data, cursor + 0x10, "buffer reference address")[0]
            name = cstring(data[cursor + 0x18:cursor + 0x20])
            print(f"  buffer_reference address=0x{address:x} name={name!r}")
        if command == 4 and kind == 4 and subtype in (5, 6) and command_size >= 0x838:
            task_words = unpack_from(
                "<I", data, cursor + 0x824, "program task words")[0]
            task_count = unpack_from(
                "<I", data, cursor + 0x830, "program task count")[0]
            text_address = unpack_from(
                "<Q", data, cursor + 0x10, "program text address")[0]
            constant_address = unpack_from(
                "<Q", data, cursor + 0x20, "program constant address")[0]
            print(
                f"  {TASK_TARGETS[subtype]}_program_descriptor "
                f"text=0x{text_address:x} text_const=0x{constant_address:x} "
                f"task_words={task_words} task_count={task_count}")
        if command == 4 and kind == 4 and subtype in (7, 9, 10) and \
                command_size >= 0x898:
            record_count = unpack_from(
                "<I", data, cursor + 0x860, "program record count")[0]
            format_code = unpack_from(
                "<I", data, cursor + 0x890, "program format code")[0]
            task_count = unpack_from(
                "<I", data, cursor + 0x830, "program task count")[0]
            addresses = [
                unpack_from("<Q", data, cursor + offset, "program address")[0]
                for offset in (0x10, 0x20, 0x40, 0x70, 0x80, 0x90)
            ]
            print(
                "  program_descriptor "
                f"tasks={task_count} records={record_count} "
                f"format=0x{format_code:x} "
                "text=0x%x text_const=0x%x scratch=0x%x "
                "slot70=0x%x slot80=0x%x slot90=0x%x" % tuple(addresses))
        if command == 4 and kind == 1 and command_size >= 0x880:
            code = unpack_from("<I", data, cursor + 0x0c, "H13 program code")[0]
            text_address, constant_address = unpack_from(
                "<2Q", data, cursor + 0x10, "H13 text addresses")
            resources = unpack_from(
                "<5Q", data, cursor + 0x30, "H13 resource addresses")
            task_words = unpack_from(
                "<I", data, cursor + 0x818, "H13 task words")[0]
            task_count = unpack_from(
                "<I", data, cursor + 0x81c, "H13 task count")[0]
            max_binding = unpack_from(
                "<I", data, cursor + 0x824, "H13 max binding")[0]
            name_offset = unpack_from(
                "<I", data, cursor + 0x83c, "H13 name offset")[0]
            field_offsets = (0x850, 0x854, 0x858, 0x85c, 0x860, 0x868, 0x86c)
            fields = tuple(
                unpack_from("<I", data, cursor + offset, "H13 trailing field")[0]
                for offset in field_offsets)
            name = cstring(data[cursor + 0x878:cursor + command_size])
            print(
                "  h13_program_descriptor "
                f"code=0x{code:x} text=0x{text_address:x} "
                f"text_const=0x{constant_address:x} resources={resources} "
                f"task_words_minus_one={task_words} task_count={task_count} "
                f"max_binding={max_binding} name_offset=0x{name_offset:x} "
                f"field850=0x{fields[0]:x} field854=0x{fields[1]:x} "
                f"field858={fields[2]} field85c={fields[3]} "
                f"field860={fields[4]} field868={fields[5]} "
                f"field86c={fields[6]} name={name!r}")
        if command == 4 and kind == 3 and command_size >= 0x80:
            binding_index = unpack_from(
                "<I", data, cursor + 0x14, "tensor binding index")[0]
            element_code = unpack_from(
                "<I", data, cursor + 0x24, "tensor element code")[0]
            shape = unpack_from("<4I", data, cursor + 0x28, "tensor shape")
            strides = unpack_from("<4Q", data, cursor + 0x50, "tensor strides")
            total = unpack_from("<Q", data, cursor + 0x70, "tensor total")[0]
            print(
                "  tensor_descriptor "
                f"binding={binding_index} element={element_code} "
                f"shape={shape} strides={strides} total={total}")
        if command == 0x19:
            inspect_segment(data, cursor, command_size, subtype)
        cursor += command_size
    if cursor != command_end:
        raise ValueError(
            f"load commands consume 0x{cursor - 32:x} bytes; "
            f"header declares 0x{command_bytes:x}")
    if subtype == 4:
        anec, summary = h13_anec(data)
        print(
            "h13_anec_content " + " ".join(
                f"{key}={hex(value)}" for key, value in summary.items()))
        if extract_path:
            with open(extract_path, "wb") as stream:
                stream.write(anec)
            print(f"extracted_anec path={extract_path!r} bytes=0x{len(anec):x}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path, extract_path = sys.argv[1], None
    elif len(sys.argv) == 4 and sys.argv[1] == "--extract-anec":
        path, extract_path = sys.argv[2], sys.argv[3]
    else:
        raise SystemExit(
            f"usage: {sys.argv[0]} [--extract-anec FILE.hwx OUTPUT.anec] FILE.hwx")
    try:
        main(path, extract_path)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
