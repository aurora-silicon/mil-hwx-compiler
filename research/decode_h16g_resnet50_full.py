#!/usr/bin/env python3
"""Decode the exact public ResNet-50 full-model H16G program privately.

The input HWX remains in a tainted directory.  The output is decoded JSON:
semantic operation groups, task headers, register records, container facts,
and an explicit list of fields that one full graph cannot generalize.  No HWX
payload or kernel bytes are copied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mint_oracles import parse_hwx


# Apple schedules the four projection blocks around their main branches and
# splits several wide convolutions.  These are observed task groupings for the
# exact build_models.py graph, not a scheduling formula.
GROUPS = (
    ("conv_0", (0,)), ("max_pool_0", (1,)),
    ("conv_1", (2,)), ("conv_2", (3,)), ("conv_3", (4,)),
    ("conv_4", (5,)), ("add_0", (6,)),
    ("conv_5", (7,)), ("conv_6", (8,)), ("conv_7", (9,)),
    ("add_1", (10,)),
    ("conv_8", (11,)), ("conv_9", (12,)), ("conv_10", (13,)),
    ("add_2", (14,)),
    ("conv_14", (15,)), ("conv_11", (16,)), ("conv_12", (17,)),
    ("conv_13", (18,)), ("add_3", (19,)),
    ("conv_15", (20,)), ("conv_16", (21,)), ("conv_17", (22,)),
    ("add_4", (23,)),
    ("conv_18", (24,)), ("conv_19", (25,)), ("conv_20", (26,)),
    ("add_5", (27,)),
    ("conv_21", (28,)), ("conv_22", (29,)), ("conv_23", (30,)),
    ("add_6", (31,)),
    ("conv_24", (32,)), ("conv_25", (33, 34, 35)),
    ("conv_27", (36, 37, 38)), ("conv_26", (39,)),
    ("add_7", (40,)),
    ("conv_28", (41,)), ("conv_29", (42, 43)), ("conv_30", (44,)),
    ("add_8", (45,)),
    ("conv_31", (46,)), ("conv_32", (47, 48)), ("conv_33", (49,)),
    ("add_9", (50,)),
    ("conv_34", (51,)), ("conv_35", (52, 53)), ("conv_36", (54,)),
    ("add_10", (55,)),
    ("conv_37", (56,)), ("conv_38", (57, 58)), ("conv_39", (59,)),
    ("add_11", (60,)),
    ("conv_40", (61,)), ("conv_41", (62, 63)), ("conv_42", (64,)),
    ("add_12", (65,)),
    ("conv_46", (66,)), ("conv_43", (67, 68)), ("conv_44", (69,)),
    ("conv_45", (70,)), ("add_13", (71,)),
    ("conv_47", (72,)), ("conv_48", (73,)), ("conv_49", (74,)),
    ("add_14", (75,)),
    ("conv_50", (76,)), ("conv_51", (77,)), ("conv_52", (78,)),
    ("add_15", (79,)), ("reduce_mean_0", (80,)), ("linear_0", (81,)),
)


def register(task: dict, address: int) -> int | None:
    key = f"0x{address:05x}"
    for block in task.get("blocks", {}).values():
        value = block.get("words", {}).get(key)
        if value is not None:
            return int(value, 16)
    return None


def decode(hwx: Path, graph_path: Path) -> dict:
    decoded = parse_hwx(hwx.read_bytes(), "h16")
    graph = json.loads(graph_path.read_text())
    all_operations = {item["name"]: item for item in graph["operations"]}
    operations = {name: item for name, item in all_operations.items()
                  if item["op"] != "relu"}
    relu_by_input = {item["inputs"]["x"]: item["name"]
                     for item in all_operations.values()
                     if item["op"] == "relu"}
    tasks = decoded["task_descriptors"]
    if len(tasks) != 82 or decoded["program_count"] != 1:
        raise ValueError("full ResNet oracle must contain one 82-task program")
    if decoded["task_decode_errors"]:
        raise ValueError("full ResNet oracle has undecoded task records")
    expected = list(range(82))
    observed = sorted(index for _, indices in GROUPS for index in indices)
    if observed != expected:
        raise AssertionError("operation grouping does not cover every task once")
    if set(name for name, _ in GROUPS) != set(operations):
        missing = sorted(set(operations) - set(name for name, _ in GROUPS))
        extra = sorted(set(name for name, _ in GROUPS) - set(operations))
        raise ValueError(f"graph/group mismatch: missing={missing} extra={extra}")

    groups = []
    for name, indices in GROUPS:
        operation = operations[name]
        entry = {
            "operation": name,
            "kind": operation["op"],
            "tasks": list(indices),
            "output_shape": operation["type"]["shape"],
            "task_bytes": [tasks[index]["size_bytes"] for index in indices],
            "header_words": [tasks[index]["header_words"] for index in indices],
            "records": [tasks[index]["records"] for index in indices],
            "register_blocks": [tasks[index]["blocks"] for index in indices],
        }
        fused_relu = relu_by_input.get(operation["result"])
        if fused_relu:
            entry["fused_operation"] = fused_relu
        if operation["op"] == "conv":
            cin = graph["variables"][operation["inputs"]["x"]]["shape"][1]
            cout = operation["type"]["shape"][1]
            encoded_cout = sum(register(tasks[index], 0x1C) or 16
                               for index in indices)
            encoded_cin = {register(tasks[index], 0x0C) or 16
                           for index in indices}
            if encoded_cin != {cin} or encoded_cout != cout:
                raise ValueError(
                    f"{name}: task channel slices {encoded_cin}/{encoded_cout} "
                    f"do not match graph {cin}/{cout}")
            entry["input_channels"] = cin
            entry["output_channel_slices"] = [
                register(tasks[index], 0x1C) or 16 for index in indices]
        groups.append(entry)

    # Drop file identity and section hashes from the durable facts.  They are
    # useful for private provenance but are not encoder inputs.
    return {
        "schema": "mil-hwxc.h16g-resnet50-full-facts.v1",
        "source": "one full random-weight public ResNet-50 program",
        "program_count": decoded["program_count"],
        "task_count": len(tasks),
        "tensor_descriptors": decoded["tensor_descriptors"],
        "programs": decoded["programs"],
        "groups": groups,
        "unknowns": [
            "ExeCycles formula",
            "generic task dependency/live-mask formula",
            "generic L2 allocation and branch-lifetime formula",
            "standalone forms for max_pool, residual add, mean, and linear",
            "whether fields that vary only with task position belong to the operation or whole-program schedule",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwx", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    facts = decode(args.hwx, args.graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    print(f"tasks={facts['task_count']} groups={len(facts['groups'])} "
          f"unknowns={len(facts['unknowns'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
