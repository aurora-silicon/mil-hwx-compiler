#!/usr/bin/env python3
"""Device-free checks for H13/H14/H16G HWX inspection."""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout

INSPECTOR = Path(__file__).resolve().parents[1] / "research" / "inspect_hwx.py"
MAGIC = 0xBEEFFACE
H14_BLOCKS = (
    ("common", 0x0000, 19),
    ("l2", 0x0500, 25),
    ("pe", 0x0900, 5),
    ("ne", 0x0D00, 5),
    ("tile_dma_src", 0x1100, 53),
    ("tile_dma_dst", 0x1500, 10),
    ("kernel_dma_src", 0x1900, 70),
)


def dense_record(start: int, count: int, seed: int) -> list[int]:
    records = []
    written = 0
    while written < count:
        span = min(64, count - written)
        records.append(((span - 1) << 15) | (start // 4 + written))
        records.extend(seed + written + index for index in range(span))
        written += span
    return records


def h14_task() -> bytes:
    words = [0] * 8
    for index, (_, start, count) in enumerate(H14_BLOCKS, 1):
        words.extend(dense_record(start, count, index << 24))
    words[0] = 1 | (len(words) << 16)
    words[1] = 1
    payload = struct.pack(f"<{len(words)}I", *words)
    return payload + bytes((-len(payload)) % 16)


def envelope(subtype: int, text: bytes) -> bytes:
    command_size = 72 + 80
    file_offset = 32 + command_size
    header = struct.pack("<8I", MAGIC, 0x80, subtype, 1, 1, command_size, 0, 0)
    segment = struct.pack(
        "<2I16s4Q4I", 0x19, command_size, b"__TEXT", 0, len(text),
        file_offset, len(text), 7, 5, 1, 0)
    section = struct.pack(
        "<16s16s2Q8I", b"__text", b"__TEXT", 0, len(text), file_offset,
        2, 0, 0, 0x28, 0, 0, 0)
    return header + segment + section + text


def segment(name: bytes, section_name: bytes, size: int, offset: int,
            allocation: int) -> bytes:
    command_size = 72 + 80
    file_size = size if name == b"__TEXT" else 0
    command = struct.pack(
        "<2I16s4Q4I", 0x19, command_size, name, 0, allocation,
        offset if file_size else 0, file_size, 7, 5, 1, 0)
    section = struct.pack(
        "<16s16s2Q8I", section_name, name, 0, size,
        offset if file_size else 0, 2, 0, 0, 0, 0, 0, 0)
    return command + section


def tensor_descriptor(batch: int = 1, total: int | None = None) -> bytes:
    """One H13 surface. Apple records a batched surface's total size as one
    batch element's span, so `total` defaults to the batch stride."""
    command = bytearray(0x80)
    struct.pack_into("<III", command, 0, 4, len(command), 3)
    struct.pack_into("<I", command, 0x24, 5)
    struct.pack_into("<4I", command, 0x28, batch, 64, 1, 1)
    struct.pack_into("<4Q", command, 0x50, 128, 2, 2, 2)
    struct.pack_into("<Q", command, 0x70, 128 if total is None else total)
    return bytes(command)


def h13_task(next_offset: int, next_words: int = 0) -> bytes:
    words = [0] * 10
    words[0] = 1 | (12 << 16)
    words[1] = next_words << 16
    words[7] = next_offset
    words.extend([(1 << 26) | 0x00000, 0x00010001, 0x00000040])
    return struct.pack("<13I", *words)


def h13_multi_task_hwx(batch: int = 1, total: int | None = None) -> bytes:
    first = h13_task(0x40, 12)
    second = h13_task(0)
    text = first + bytes(0x40 - len(first)) + second
    constant_offset = (len(text) + 15) & ~15
    program = bytearray(0x880)
    struct.pack_into("<III", program, 0, 4, len(program), 1)
    struct.pack_into("<I", program, 0x818, 12)
    struct.pack_into("<I", program, 0x81C, 2)
    commands_size = 2 * 0x98 + 0xE8 + len(program) + 2 * 0x80
    file_offset = 32 + commands_size
    text_segment = bytearray(0xE8)
    struct.pack_into(
        "<2I16s4Q4I", text_segment, 0, 0x19, len(text_segment), b"__TEXT",
        0, constant_offset, file_offset, constant_offset, 7, 5, 2, 0)
    struct.pack_into(
        "<16s16s2Q8I", text_segment, 72, b"__text", b"__TEXT", 0,
        len(text), file_offset, 2, 0, 0, 0, 0, 0, 0)
    struct.pack_into(
        "<16s16s2Q8I", text_segment, 152, b"__const", b"__TEXT",
        constant_offset, 0, file_offset + constant_offset, 2,
        0, 0, 0, 0, 0, 0)
    commands = b"".join([
        segment(b"__FVMLIB", b"__const", 0, 0, 0x1000),
        segment(b"__FVMLIB", b"__data", 0, 0, 0x1000),
        bytes(text_segment), bytes(program),
        tensor_descriptor(batch, total), tensor_descriptor(batch, total),
    ])
    assert len(commands) == commands_size
    header = struct.pack("<8I", MAGIC, 0x80, 4, 1, 6, commands_size, 0, 0)
    return header + commands + text + bytes(constant_offset - len(text))


def inspect(path: Path, success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(INSPECTOR), str(path)], capture_output=True,
        text=True, timeout=scaled_timeout(10), check=False)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


with tempfile.TemporaryDirectory(prefix="mil-hwx-inspection-") as directory:
    root = Path(directory)
    source_derived = root / "source-derived-h14.hwx"
    source_derived.write_bytes(envelope(5, h14_task()))
    result = inspect(source_derived)
    assert "architecture subtype=0x0005 name=H14 isa=11" in result.stdout
    for name, start, count in H14_BLOCKS:
        end = start + count * 4
        expected = (
            f"h14_block name={name} range=0x{start:05x}-0x{end:05x} "
            f"source_words={count} observed_words={count}")
        assert expected in result.stdout, result.stdout
    assert "h14_tasks count=1" in result.stdout

    mixed = bytearray(envelope(5, h14_task()))
    text_offset = 32 + 72 + 80
    struct.pack_into("<I", mixed, text_offset + 8 * 4, 0x4100 // 4)
    mixed_path = root / "mixed-generation-h14.hwx"
    mixed_path.write_bytes(mixed)
    result = inspect(mixed_path, success=False)
    assert "outside source-backed H14 blocks" in result.stderr, result.stderr

    truncated_path = root / "truncated-h14.hwx"
    truncated_path.write_bytes(envelope(5, h14_task()[:-16]))
    result = inspect(truncated_path, success=False)
    assert "declares" in result.stderr and "beyond __TEXT/__text" in result.stderr

    h16g_path = root / "synthetic-h16g-envelope.hwx"
    h16g_path.write_bytes(envelope(7, b""))
    result = inspect(h16g_path)
    assert "architecture subtype=0x0007 name=H16G isa=17" in result.stdout
    assert "__TEXT/__text" in result.stdout

    h18g_path = root / "synthetic-h18g-envelope.hwx"
    h18g_path.write_bytes(envelope(11, b""))
    result = inspect(h18g_path)
    assert "architecture subtype=0x000b name=H18G isa=unknown" in result.stdout

    h13_path = root / "synthetic-h13-multi-task.hwx"
    h13_path.write_bytes(h13_multi_task_hwx())
    result = inspect(h13_path)
    assert "architecture subtype=0x0004 name=H13 isa=7" in result.stdout
    assert "h13_task[0] size_words=13 records=1" in result.stdout
    assert "h13_task[1] size_words=13 records=1" in result.stdout
    assert "h13_tasks count=2" in result.stdout
    assert "taskContentBytes=0x74" in result.stdout
    assert "tasks=0x2" in result.stdout

    broken = bytearray(h13_multi_task_hwx())
    command_bytes = struct.unpack_from("<I", broken, 20)[0]
    second_task_word7 = 32 + command_bytes + 0x40 + 7 * 4
    struct.pack_into("<I", broken, second_task_word7, 4)
    broken_path = root / "broken-h13-task-link.hwx"
    broken_path.write_bytes(broken)
    result = inspect(broken_path, success=False)
    assert "H13 final task points to 0x4" in result.stderr, result.stderr

    batched_path = root / "synthetic-h13-batched.hwx"
    batched_path.write_bytes(h13_multi_task_hwx(batch=8))
    result = inspect(batched_path)
    assert "architecture subtype=0x0004 name=H13 isa=7" in result.stdout
    assert "shape=(8, 64, 1, 1)" in result.stdout, result.stdout

    oversized_path = root / "synthetic-h13-batched-total.hwx"
    oversized_path.write_bytes(h13_multi_task_hwx(batch=8, total=8 * 128))
    result = inspect(oversized_path, success=False)
    assert "invalid layout" in result.stderr, result.stderr

print("HWX inspection: PASS (multi-task H13 plus source-derived H14 fixtures)")
