#!/usr/bin/env python3
"""Sanitized parity for native H16G last-axis layer normalization.

The sealed prefix digests cover the isolated layer_norm->projection programs.
The standalone assertions retain the measured task that actually normalizes
the prefix result, then append the zero-beta output writer. Identity gamma is
a no-op and remains omitted.
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
    (128, 768, 3, [180, 132, 132, 124, 252],
     "1b9fe62201bfab3bf23e00fa7d9db9a89118e0ba5de670c45a48e3be4467d061"),
    (1500, 384, 9,
     [180, 172, 172, 168, 196, 196, 192, 192, 200, 140, 252],
     "8f1107a90c61a18787b6c8a7e266af65fc6a6adddbe43eeb37478479db932d2a"),
    (1500, 512, 9,
     [180, 172, 172, 168, 196, 196, 192, 192, 184, 156, 252],
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


def has_task_word(decoded: dict, address: int) -> bool:
    key = f"0x{address:05x}"
    return any(key in block["words"] for block in decoded["blocks"].values())


def main() -> None:
    decoded_cases = {}
    for tokens, channels, prefix, sizes, expected in CASES:
        stream = subprocess.check_output([DUMP, str(tokens), str(channels)])
        assert stream[:16] == b"\x01" + bytes(15)
        tasks = split_h14_tasks(stream[16:])
        assert [len(task) for task in tasks] == sizes
        assert digest(tasks[:prefix]) == expected
        decoded = [decode_task(task, "h16") for task in tasks]
        decoded_cases[(tokens, channels)] = decoded
        schedules = [int(task["header_words"][4], 16) for task in decoded]
        normalize_schedule = 0x20 if channels != 512 else 0x28
        assert schedules == [0x00fff868] + [0x00fff820] * (prefix - 1) + \
            [normalize_schedule, 0x00fff860]
        for task in decoded[:prefix]:
            for record in task["records"]:
                if record["address"] in ("0x04d10", "0x04d28"):
                    assert bar_slot(task, record["address"]) == 6
        assert bar_slot(decoded[-1], "0x04d28") == 8
        assert bar_slot(decoded[-1], "0x05108") == 7
        source_base = 0x33000 if tokens == 128 else \
            0x127380 if channels == 384 else 0xbb80
        normalization_base = 0x33000 if tokens == 128 else \
            0xbb80 if channels == 384 else 0x1e0780
        row = 2 * channels
        assert task_word(decoded[-2], 0x4110) == source_base
        assert task_word(decoded[-2], 0x4904) == 0x103c0c
        assert task_word(decoded[-2], 0x4910) == 0x3c00
        if channels != 768:
            assert task_word(decoded[-2], 0x414c) == normalization_base
        if channels == 512:
            assert bar_slot(decoded[-2], "0x05108") == 3
        assert task_word(decoded[-1], 0x4124) == normalization_base
        assert task_word(decoded[-1], 0x414c) == normalization_base + row + 0x40
        prefix_info = json.loads(subprocess.check_output(
            [DUMP, str(tokens), str(channels), "prefix-info"]))
        assert len(prefix_info) == prefix
        assert [item["task_count"] for item in prefix_info] == \
            list(range(1, prefix + 1))
        for item in prefix_info:
            stream = subprocess.check_output(
                [DUMP, str(tokens), str(channels),
                 f"prefix-{item['task_count']}"])
            stage_tasks = split_h14_tasks(stream[16:])
            assert len(stage_tasks) == item["task_count"] + 1
            final = decode_task(stage_tasks[-1], "h16")
            assert int(final["header_words"][4], 16) == 0x00fff860
            assert task_word(final, 0x4124) == item["result_base"]
            assert task_word(final, 0x4128) == item["result_row_bytes"]
            assert task_word(final, 4) == item["result_row_bytes"] // 2
            assert task_word(final, 8) == item["result_rows"]
            assert task_word(final, 0x4d28) == 2 * channels
            assert task_word(final, 0x5114) == \
                item["result_rows"] * item["result_row_bytes"]
            assert bar_slot(final, "0x04d28") == 8
            assert bar_slot(final, "0x05108") == 7
        control_stream = subprocess.check_output(
            [DUMP, str(tokens), str(channels), "prefix-1-dead-control"])
        ordinary_stream = subprocess.check_output(
            [DUMP, str(tokens), str(channels), "prefix-1"])
        control_tasks = split_h14_tasks(control_stream[16:])
        ordinary_tasks = split_h14_tasks(ordinary_stream[16:])
        assert len(control_tasks) == len(ordinary_tasks) == 2
        assert control_tasks[0] != ordinary_tasks[0]
        assert control_tasks[1] == ordinary_tasks[1]
        control_mean = decode_task(control_tasks[0], "h16")
        ordinary_mean = decode_task(ordinary_tasks[0], "h16")
        assert task_word(control_mean, 0x4514) == task_word(
            ordinary_mean, 0x4514)
        assert control_mean["blocks"] == ordinary_mean["blocks"]
        stream = subprocess.check_output(
            [DUMP, str(tokens), str(channels), "normalized"])
        normalized_tasks = split_h14_tasks(stream[16:])
        assert len(normalized_tasks) == prefix + 2
        assert task_word(decode_task(normalized_tasks[-1], "h16"),
                         0x4124) == normalization_base
        for unsafe in ("centered", "rsqrt"):
            completed = subprocess.run(
                [DUMP, str(tokens), str(channels), unsafe],
                capture_output=True)
            assert completed.returncode != 0
    tiny = decoded_cases[(1500, 384)]
    base = decoded_cases[(1500, 512)]
    # Named wtiny/wbase hang candidates from their field diff. The reduce
    # tail has a distinct dependency/mode pair; the measured normalizers use
    # different contextual schedules; only base has an interleave-8 spill.
    assert [int(tiny[8]["header_words"][i], 16) for i in (7, 8)] == [0, 0]
    assert [int(base[8]["header_words"][i], 16) for i in (7, 8)] == \
        [0x1ffffff, 1]
    assert int(tiny[9]["header_words"][4], 16) == 0x20
    assert int(base[9]["header_words"][4], 16) == 0x28
    assert not has_task_word(tiny[9], 0x5108)
    assert has_task_word(base[9], 0x5108)
    print("H16G layer_norm parity PASS: three measured input prefixes, "
          "exact measured normalization tasks, zero-beta output boundaries, "
          "and producer-derived prefix boundaries per shape")


if __name__ == "__main__":
    main()
