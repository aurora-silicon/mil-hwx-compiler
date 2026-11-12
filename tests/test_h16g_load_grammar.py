#!/usr/bin/env python3
"""Synthetic checks for the H16G firmware-LOAD grammar audit."""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from h16g_load_grammar import GrammarError, walk  # noqa: E402

TERMINAL_ENGINE_CONTROL = (5 << 16) | 9


def task(index, final=False, records=((4, (1,)),)):
    body = bytearray()
    for header, values in records:
        body += struct.pack("<I", header)
        body += struct.pack(f"<{len(values)}I", *values)
    words = 9 + len(body) // 4
    header = [words << 16 | index, 0, 1 << 26 if final else 0, 0,
              0, 0, 0, 0, TERMINAL_ENGINE_CONTROL if final else 1]
    return struct.pack("<9I", *header) + body


def fields(tasks, relocations=()):
    stream = bytearray(b"\x01" + bytes(15))
    for index, raw in enumerate(tasks):
        if index:
            stream += bytes((-len(stream)) % 16)
        stream += raw
    program = bytearray(0x8c0)
    struct.pack_into("<I", program, 0x830, len(tasks))
    struct.pack_into("<I", program, 0x860, 2 * len(tasks) + 1)
    struct.pack_into("<I", program, 0x8b0, 0x4c)
    struct.pack_into("<Q", program, 0x90, 1)  # bound kernel BAR slot 8
    return {"stream": bytes(stream), "program": bytes(program),
            "relocations": set(relocations), "kernel_size": 0,
            "task_count": len(tasks), "record_code": 2 * len(tasks) + 1,
            "format_code": 0x4c}


class LoadGrammarTests(unittest.TestCase):
    def test_residual_terminal_and_chain(self):
        result = walk(fields([task(0), task(1), task(2, True)]))
        self.assertTrue(result["residual"])
        self.assertEqual(result["terminal_control"],
                         f"0x{TERMINAL_ENGINE_CONTROL:08x}")

    def test_terminal_ready_is_required(self):
        broken = bytearray(task(2, True))
        struct.pack_into("<I", broken, 32, (5 << 16) | 8)
        with self.assertRaisesRegex(GrammarError, "terminal-engine-control"):
            walk(fields([task(0), task(1), bytes(broken)]))

    def test_task_link_is_sequential(self):
        with self.assertRaisesRegex(GrammarError, "task-link"):
            walk(fields([task(0), task(7), task(2, True)]))

    def test_dependency_cannot_name_missing_task(self):
        dma = (0x4d00 // 4, (0xff << 16,))
        with self.assertRaisesRegex(GrammarError, "dependency-task"):
            walk(fields([task(0, records=(dma,)), task(1), task(2, True)]))

    def test_every_kernel_base_has_a_relocation(self):
        kernel = (0x5510 // 4 | 1 << 15 | 1 << 29 | 8 << 23, (0, 0))
        with self.assertRaisesRegex(GrammarError, "kernel-relocations"):
            walk(fields([task(0, records=(kernel,)), task(1), task(2, True)]))


if __name__ == "__main__":
    unittest.main()
