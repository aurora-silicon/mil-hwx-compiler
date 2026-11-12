#!/usr/bin/env python3
"""Build safe forward and mirror EXECUTE morphs for the H16G ResNet stem.

Both inputs compile the same stem/probe graph.  ``--main-task`` is the
rule-built object whose first task is byte-identical to the accepted main
ResNet stem; ``--rb3`` is the round-3 stem that hangs.  Value-only groups are
patched at the existing operand offsets.  The result-span write-set group is
never hand-reframed: its cases use one of the two compiler-built containers
and patch every other group back to the requested endpoint.  This preserves
valid record headers, task lengths, later-task offsets, and relocations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

from h16g_load_grammar import GrammarError, check, object_fields, walk
from h16g_validate import ValidationError


GROUPS = (
    ("header-hints", "execution-cost hint and live-bank mask", (1, 7), ()),
    ("schedule", "Common issue/traversal schedule", (), (0x3C,)),
    ("result-span-record", "compiler-built implicit/explicit result-span write set",
     (), ()),
    ("result-layout", "sparse-phase result traversal and row stride", (),
     (0x4148, 0x4154)),
    ("source-layout", "compact sparse-phase source base and tuple", (),
     (0x4110, 0x4114, 0x4118, 0x411C, 0x4120)),
)
VALUE_GROUPS = tuple(group for group in GROUPS
                     if group[0] != "result-span-record")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task_bytes(endpoint: tuple[bytes, dict, list[dict]], index: int = 0) -> bytes:
    task = endpoint[2][index]
    start = task["offset"]
    return endpoint[1]["stream"][start:start + 4 * task["words"]]


def load(path: Path) -> tuple[bytes, dict, list[dict]]:
    data = path.read_bytes()
    fields = object_fields(data)
    result = walk(fields, include_tasks=True, enforce_residual=False)
    return data, fields, result["_tasks"]


def put32(image: bytearray, fields: dict, stream_offset: int, value: int) -> None:
    struct.pack_into("<I", image, fields["text_off"] + stream_offset, value)


def patch_value_group(data: bytes, fields: dict, target: dict,
                      source: dict, group: tuple) -> bytes:
    image = bytearray(data)
    _, _, header_words, registers = group
    for word in header_words:
        put32(image, fields, target["offset"] + 4 * word,
              source["header"][word])
    for address in registers:
        if address not in target["register_offsets"] or address not in source["registers"]:
            raise ValueError(f"group {group[0]} lacks shared register 0x{address:x}")
        put32(image, fields, target["register_offsets"][address],
              source["registers"][address])
    return bytes(image)


def build_case(main: tuple[bytes, dict, list[dict]],
               rb3: tuple[bytes, dict, list[dict]],
               structural_side: str, value_sides: dict[str, str]) -> bytes:
    endpoints = {"main": main, "rb3": rb3}
    data, fields, tasks = endpoints[structural_side]
    target = tasks[0]
    for group in VALUE_GROUPS:
        side = value_sides[group[0]]
        source = endpoints[side][2][0]
        data = patch_value_group(data, fields, target, source, group)
    return data


def diagnostic(path: Path) -> dict:
    try:
        facts = check(path)
        return {"load_grammar": "pass", "facts": facts}
    except (OSError, ValidationError, GrammarError) as error:
        return {"load_grammar": "fail", "error": str(error)}


def require_inputs(main: tuple[bytes, dict, list[dict]],
                   rb3: tuple[bytes, dict, list[dict]],
                   accepted: Path | None) -> None:
    main_tasks, rb3_tasks = main[2], rb3[2]
    if len(main_tasks) != len(rb3_tasks) or len(main_tasks) != 3:
        raise ValueError("stem morph endpoints must both contain three tasks")
    header_diffs = {index for index, values in enumerate(
                    zip(main_tasks[0]["header"], rb3_tasks[0]["header"]))
                    if values[0] != values[1]}
    register_diffs = {address for address in
                      set(main_tasks[0]["registers"]) | set(rb3_tasks[0]["registers"])
                      if main_tasks[0]["registers"].get(address) !=
                         rb3_tasks[0]["registers"].get(address)}
    if header_diffs != {0, 1, 7}:
        raise ValueError(f"unexpected stem header differences: {sorted(header_diffs)}")
    expected_registers = {0x3C, 0x4110, 0x4114, 0x4118, 0x411C, 0x4120,
                          0x4148, 0x4154, 0x4158, 0x415C}
    if register_diffs != expected_registers:
        raise ValueError("unexpected stem register differences: " +
                         ",".join(f"0x{x:x}" for x in sorted(register_diffs)))
    for index in (1, 2):
        if (main_tasks[index]["header"] != rb3_tasks[index]["header"] or
                main_tasks[index]["registers"] != rb3_tasks[index]["registers"]):
            raise ValueError(f"non-stem task {index} differs semantically")
    if accepted is not None:
        accepted_data, accepted_fields, accepted_tasks = load(accepted)
        del accepted_data
        accepted_endpoint = (b"", accepted_fields, accepted_tasks)
        a_bytes = task_bytes(accepted_endpoint)
        m_bytes = task_bytes(main)
        if a_bytes != m_bytes:
            raise ValueError("--main-task stem is not byte-identical to --accepted-main")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-task", type=Path, required=True)
    parser.add_argument("--rb3", type=Path, required=True)
    parser.add_argument("--accepted-main", type=Path)
    parser.add_argument("--refs", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir must not exist")

    main_endpoint, rb3_endpoint = load(args.main_task), load(args.rb3)
    try:
        require_inputs(main_endpoint, rb3_endpoint, args.accepted_main)
    except (OSError, ValueError, GrammarError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True)
    if args.refs:
        shutil.copytree(args.refs, args.out_dir / "refs")
    reference = "refs/logits_00.fp16" if args.refs else None
    manifest = {
        "purpose": "stem EXECUTE field bisection with compiler-built framing",
        "order": "least likely to wedge first",
        "accepted_main_task_checked": bool(args.accepted_main),
        "main_object_sha256": sha256(main_endpoint[0]),
        "rb3_object_sha256": sha256(rb3_endpoint[0]),
        "main_task_sha256": sha256(task_bytes(main_endpoint)),
        "rb3_task_sha256": sha256(task_bytes(rb3_endpoint)),
        "groups": [{"name": group[0], "description": group[1]}
                   for group in GROUPS],
        "cases": [],
    }

    case_hashes = {}

    def emit(name: str, direction: str, mode: str, group: str,
             payload: bytes) -> None:
        path = args.out_dir / f"{name}.hwx"
        path.write_bytes(payload)
        entry = {"name": name, "file": path.name, "direction": direction,
                 "mode": mode, "group": group, "sha256": sha256(payload),
                 **diagnostic(path)}
        if reference:
            entry["cpu_reference"] = reference
        manifest["cases"].append(entry)
        case_hashes[name] = entry["sha256"]

    emit("00-main-task-baseline", "baseline", "identity", "none",
         main_endpoint[0])
    emit("00-rb3-baseline", "baseline", "identity", "none", rb3_endpoint[0])

    main_values = {group[0]: "main" for group in VALUE_GROUPS}
    rb3_values = {group[0]: "rb3" for group in VALUE_GROUPS}
    cumulative = dict(main_values)
    structural = "main"
    for index, group in enumerate(GROUPS, 1):
        if group[0] == "result-span-record":
            structural = "rb3"
        else:
            cumulative[group[0]] = "rb3"
        payload = build_case(main_endpoint, rb3_endpoint, structural, cumulative)
        emit(f"f-c{index:02d}-{group[0]}", "main-to-rb3", "cumulative",
             group[0], payload)

    for index, group in enumerate(GROUPS, 1):
        values = dict(main_values)
        structural = "main"
        if group[0] == "result-span-record":
            structural = "rb3"
        else:
            values[group[0]] = "rb3"
        emit(f"f-s{index:02d}-{group[0]}", "main-to-rb3", "single",
             group[0], build_case(main_endpoint, rb3_endpoint, structural, values))

    cumulative = dict(rb3_values)
    structural = "rb3"
    for index, group in enumerate(GROUPS, 1):
        if group[0] == "result-span-record":
            structural = "main"
        else:
            cumulative[group[0]] = "main"
        payload = build_case(main_endpoint, rb3_endpoint, structural, cumulative)
        emit(f"m-c{index:02d}-{group[0]}", "rb3-to-main", "cumulative",
             group[0], payload)

    for index, group in enumerate(GROUPS, 1):
        values = dict(rb3_values)
        structural = "rb3"
        if group[0] == "result-span-record":
            structural = "main"
        else:
            values[group[0]] = "main"
        emit(f"m-s{index:02d}-{group[0]}", "rb3-to-main", "single",
             group[0], build_case(main_endpoint, rb3_endpoint, structural, values))

    if case_hashes["f-c05-source-layout"] != manifest["rb3_object_sha256"]:
        raise ValueError("forward cumulative endpoint is not rb3 byte-identical")
    if case_hashes["m-c05-source-layout"] != manifest["main_object_sha256"]:
        raise ValueError("mirror cumulative endpoint is not main-task byte-identical")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
