#!/usr/bin/env python3
"""Build a private H16G LOAD-refusal morph set from two compiler outputs.

The base is the accepted object and the donor is the rule-built object.  The
first three arms patch one LOAD-visible group in place, preserving the base
object's task split and file layout.  The final donor control carries the new
task split because changed descriptor boundaries cannot be isolated by an
in-place patch.  No input object is embedded in this tool or its manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

from h16g_load_grammar import (GrammarError, check, object_fields, walk)
from h16g_validate import ValidationError


# Relocated 64-bit BAR operands.  The records arm excludes them so the BAR arm
# owns both their slot flags and addends.
BAR_VALUES = {0x4d10, 0x4d14, 0x4d28, 0x4d2c,
              0x5108, 0x510c, 0x5510, 0x5514}


def signature(task: dict) -> tuple:
    r = task["registers"]
    dimensions = tuple(r.get(address) for address in
                       (4, 8, 12, 0x14, 0x18, 0x1c))
    engines = tuple(address in r for address in
                    (0x4500, 0x4514, 0x4d04, 0x5100, 0x5500))
    return dimensions + engines


def align_tasks(base: list[dict], donor: list[dict]) -> list[tuple[int, int]]:
    """Stable LCS alignment by decoded geometry and engine role."""
    a, b = [signature(task) for task in base], [signature(task) for task in donor]
    score = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            score[i][j] = (1 + score[i + 1][j + 1] if a[i] == b[j] else
                           max(score[i + 1][j], score[i][j + 1]))
    pairs, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j] and score[i][j] == 1 + score[i + 1][j + 1]:
            pairs.append((i, j)); i += 1; j += 1
        elif score[i + 1][j] >= score[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def put32(image: bytearray, text_off: int, stream_off: int, value: int) -> None:
    struct.pack_into("<I", image, text_off + stream_off, value)


def header_arm(base_data: bytes, base_fields: dict, base_tasks: list[dict],
               donor_tasks: list[dict], pairs: list[tuple[int, int]]) -> bytes:
    image = bytearray(base_data)
    for base_index, donor_index in pairs:
        target, source = base_tasks[base_index], donor_tasks[donor_index]
        # Word 0 owns the base task's actual length and sequential link id.
        for word in range(1, min(len(target["header"]), len(source["header"]))):
            put32(image, base_fields["text_off"], target["offset"] + 4 * word,
                  source["header"][word])
    return bytes(image)


def records_arm(base_data: bytes, base_fields: dict, base_tasks: list[dict],
                donor_tasks: list[dict], pairs: list[tuple[int, int]]) -> bytes:
    image = bytearray(base_data)
    for base_index, donor_index in pairs:
        target, source = base_tasks[base_index], donor_tasks[donor_index]
        common = set(target["registers"]) & set(source["registers"])
        for address in common - BAR_VALUES:
            put32(image, base_fields["text_off"],
                  target["register_offsets"][address], source["registers"][address])
    return bytes(image)


def bars_arm(base_data: bytes, base_fields: dict, base_tasks: list[dict],
             donor_tasks: list[dict], pairs: list[tuple[int, int]]) -> bytes:
    image = bytearray(base_data)
    for base_index, donor_index in pairs:
        target, source = base_tasks[base_index], donor_tasks[donor_index]
        donor_records = {tuple(record["addresses"]): record
                         for record in source["records"] if record["slot"] is not None}
        for record in target["records"]:
            if record["slot"] is None:
                continue
            replacement = donor_records.get(tuple(record["addresses"]))
            if replacement is not None:
                put32(image, base_fields["text_off"], record["offset"],
                      replacement["header"])
        for address in set(target["registers"]) & set(source["registers"]) & BAR_VALUES:
            put32(image, base_fields["text_off"],
                  target["register_offsets"][address], source["registers"][address])
    return bytes(image)


def diagnostic(path: Path) -> dict:
    try:
        facts = check(path)
        return {"load_grammar": "pass", "facts": facts}
    except (OSError, ValidationError, GrammarError) as error:
        return {"load_grammar": "fail", "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True,
                        help="accepted compiler output")
    parser.add_argument("--donor", type=Path, required=True,
                        help="rule-built compiler output")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    base_data, donor_data = args.base.read_bytes(), args.donor.read_bytes()
    base_fields, donor_fields = object_fields(base_data), object_fields(donor_data)
    base_walk = walk(base_fields, include_tasks=True, enforce_residual=False)
    donor_walk = walk(donor_fields, include_tasks=True, enforce_residual=False)
    base_tasks, donor_tasks = base_walk.pop("_tasks"), donor_walk.pop("_tasks")
    pairs = align_tasks(base_tasks, donor_tasks)
    if len(pairs) < min(len(base_tasks), len(donor_tasks)) // 2:
        print("FAIL: fewer than half the tasks have a semantic alignment", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=False)
    arms = [
        ("01-bar-relocations.hwx", "BAR record flags and relocated addends",
         bars_arm(base_data, base_fields, base_tasks, donor_tasks, pairs)),
        ("02-records.hwx", "register values excluding BAR/relocation operands",
         records_arm(base_data, base_fields, base_tasks, donor_tasks, pairs)),
        ("03-headers.hwx", "semantic task header words; base lengths/ids retained",
         header_arm(base_data, base_fields, base_tasks, donor_tasks, pairs)),
        ("04-task-split-donor-control.hwx",
         "donor task split control; changed boundaries require the complete donor object",
         donor_data),
    ]
    manifest = {
        "purpose": "private LOAD refusal bisection; one case per reboot",
        "order": "least likely to be refused first",
        "base_sha256": hashlib.sha256(base_data).hexdigest(),
        "donor_sha256": hashlib.sha256(donor_data).hexdigest(),
        "base_tasks": len(base_tasks), "donor_tasks": len(donor_tasks),
        "aligned_tasks": len(pairs),
        "unmatched_base": sorted(set(range(len(base_tasks))) - {a for a, _ in pairs}),
        "unmatched_donor": sorted(set(range(len(donor_tasks))) - {b for _, b in pairs}),
        "arms": [],
    }
    for name, group, payload in arms:
        path = args.out_dir / name
        path.write_bytes(payload)
        manifest["arms"].append({"file": name, "group": group,
                                 "sha256": hashlib.sha256(payload).hexdigest(),
                                 **diagnostic(path)})
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(Path(__file__).with_name("h16g_load_grammar.py"),
                 args.out_dir / "h16g_load_grammar.py")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
