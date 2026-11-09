#!/usr/bin/env python3
"""Sanitized parity for native H16G last-axis layer normalization.

The sealed prefix digests cover the isolated layer_norm->projection programs.
The standalone assertions connect those measured producer layouts to the
affine/output boundary from the final full Whisper-tiny layer normalization.
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
def canonical(task: bytes) -> dict:
    decoded = decode_task(task, "h16")
    header = decoded["header_words"][:1] + decoded["header_words"][2:]
    # Prefix digests predate the standalone serialization correction.  Keep
    # the sealed register/record comparison while checking the schedule words
    # explicitly below.
    schedule = int(header[3], 16)
    if schedule in (0x00fff868, 0x00fff820, 0x00fff860):
        header[3] = "0x00000068" if schedule == 0x00fff868 else "0x00000028"
    return {
        "header": header,
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


def task_word(decoded: dict, address: str) -> int:
    key = f"0x{address:05x}"
    for block in decoded["blocks"].values():
        if key in block["words"]:
            return int(block["words"][key], 16)
    raise AssertionError(f"task has no register {key}")


def main() -> None:
    for tokens, channels, prefix, sizes, expected in CASES:
        stream = subprocess.check_output([DUMP, str(tokens), str(channels)])
        assert stream[:16] == b"\x01" + bytes(15)
        tasks = split_h14_tasks(stream[16:])
        assert [len(task) for task in tasks] == sizes
        assert digest(tasks[:prefix]) == expected
        decoded = [decode_task(task, "h16") for task in tasks]
        schedules = [int(task["header_words"][4], 16) for task in decoded]
        assert schedules == [0x00fff868] + [0x00fff820] * (len(tasks) - 2) + \
            [0x00fff860]
        for task in decoded[:prefix]:
            for record in task["records"]:
                if record["address"] in ("0x04d10", "0x04d28"):
                    assert bar_slot(task, record["address"]) == 6
        assert bar_slot(decoded[-2], "0x04d28") == 8
        assert bar_slot(decoded[-1], "0x04d28") == 8
        assert bar_slot(decoded[-1], "0x05108") == 7
        normalization_base = 0x33000 if tokens == 128 else \
            0x127380 if channels == 384 else 1500 * 32
        row = 2 * channels
        assert task_word(decoded[-2], 0x4124) == normalization_base
        assert task_word(decoded[-1], 0x4124) == normalization_base + 0x40
        assert task_word(decoded[-1], 0x414c) == normalization_base + row + 0x40
    print("H16G layer_norm parity PASS: three measured input prefixes, "
          "serialized standalone schedules and connected normalization tails")


if __name__ == "__main__":
    main()
