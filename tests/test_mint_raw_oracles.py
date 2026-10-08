#!/usr/bin/env python3
"""Raw H15/H16 oracle record checks for research/mint_oracles.py."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import mint_oracles  # noqa: E402

MAGIC = 0xBEEFFACE


def raw_object(subtype: int) -> bytes:
    """A subtype-N object whose task stream no decoder can split."""
    text = struct.pack("<16I", *([0xFFFFFFFF] * 16))
    constant = bytes(16)
    command_size = 72 + 2 * 80
    file_offset = 32 + command_size
    header = struct.pack(
        "<8I", MAGIC, 0x80, subtype, 2, 1, command_size, 0, 0)
    segment = struct.pack(
        "<2I16s4Q4I", 0x19, command_size, b"__TEXT", 0, len(text) + len(constant),
        file_offset, len(text) + len(constant), 7, 5, 2, 0)
    text_section = struct.pack(
        "<16s16s2Q8I", b"__text", b"__TEXT", 0, len(text), file_offset,
        2, 0, 0, 0, 0, 0, 0)
    const_section = struct.pack(
        "<16s16s2Q8I", b"__const", b"__TEXT", len(text), len(constant),
        file_offset + len(text), 2, 0, 0, 0, 0, 0, 0)
    return header + segment + text_section + const_section + text + constant


record = mint_oracles.parse_hwx(raw_object(6), "h15")
assert record["decoded"] is False, record
assert record["mach_o_header"]["cpusubtype"] == 6, record["mach_o_header"]
assert record["task_section"]["word_count"] == 16, record["task_section"]
assert record["task_section"]["words"] == ["0xffffffff"] * 16, \
    record["task_section"]
assert record["constant_section"]["size"] == 16, record["constant_section"]
assert len(record["constant_section"]["sha256"]) == 64
assert "task_descriptors" not in record, sorted(record)
assert "tensor_descriptors" not in record, sorted(record)
assert "task_section_raw_words" not in record

try:
    mint_oracles.parse_hwx(raw_object(6), "h16")
    raise SystemExit("wrong subtype was accepted for h16")
except ValueError as error:
    assert "subtype 6 for h16" in str(error), error

selected = mint_oracles.select_cases(
    ["binary_add_1x64x1x1", "matmul_m1_k256_n512_ty1"])
assert [item["name"] for item in selected] == [
    "binary_add_1x64x1x1", "matmul_m1_k256_n512_ty1"], selected

print("Mint raw H15/H16 records: PASS (raw record shape, wrong-subtype "
      "rejection, multi-pattern selection)")
