#!/usr/bin/env python3
"""Sanitized parity for native H16G last-axis layer normalization.

The sealed prefix digests cover the isolated layer_norm->projection programs.
The boundary digest covers the final standalone layer_norm in the full
Whisper-tiny encoder after renumbering it as a standalone program and rebasing
its scratch/output apertures to the object writer's BAR layout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from mint_oracles import decode_task, split_h14_tasks  # noqa: E402

DUMP = str(Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / "build/h16_transformer_layer_norm_dump").resolve())

CASES = [
    (128, 768, 3, [180, 132, 132, 176, 248],
     "1b9fe62201bfab3bf23e00fa7d9db9a89118e0ba5de670c45a48e3be4467d061"),
    (1500, 384, 9,
     [180, 172, 172, 168, 196, 196, 192, 192, 200, 176, 248],
     "8f1107a90c61a18787b6c8a7e266af65fc6a6adddbe43eeb37478479db932d2a"),
    (1500, 512, 9,
     [180, 172, 172, 168, 196, 196, 192, 192, 184, 176, 248],
     "b95ec9b09570c0b57ad532181f3f2e804cc19ad3c138729f96eac024b533f4f6"),
]
BOUNDARY_DIGEST = "0e50c51e244106441d454ca044038d814ddef9605b75ace0a04f5f6aa7c36c82"


def canonical(task: bytes) -> dict:
    decoded = decode_task(task, "h16")
    return {
        "header": decoded["header_words"][:1] + decoded["header_words"][2:],
        "blocks": decoded["blocks"],
        "records": decoded["records"],
    }


def digest(tasks: list[bytes]) -> str:
    value = json.dumps([canonical(task) for task in tasks], sort_keys=True,
                       separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def bar_slot(decoded: dict, address: str) -> int:
    record = next(record for record in decoded["records"]
                  if record["address"] == address)
    return (int(record["header"], 16) >> 23) & 0x3f


def main() -> None:
    for tokens, channels, prefix, sizes, expected in CASES:
        stream = subprocess.check_output([DUMP, str(tokens), str(channels)])
        assert stream[:16] == b"\x01" + bytes(15)
        tasks = split_h14_tasks(stream[16:])
        assert [len(task) for task in tasks] == sizes
        assert digest(tasks[:prefix]) == expected
        decoded = [decode_task(task, "h16") for task in tasks]
        for task in decoded[:prefix]:
            for record in task["records"]:
                if record["address"] in ("0x04d10", "0x04d28"):
                    assert bar_slot(task, record["address"]) == 6
        assert bar_slot(decoded[-2], "0x04d28") == 3
        assert bar_slot(decoded[-1], "0x04d28") == 3
        assert bar_slot(decoded[-1], "0x05108") == 7
    tiny = split_h14_tasks(subprocess.check_output([DUMP, "1500", "384"])[16:])
    assert digest(tiny[-2:]) == BOUNDARY_DIGEST
    print("H16G layer_norm parity PASS: three measured input prefixes, "
          "one measured standalone output boundary, two explicit boundary predictions")


if __name__ == "__main__":
    main()
