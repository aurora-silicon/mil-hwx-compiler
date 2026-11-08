#!/usr/bin/env python3
"""Synthetic task-stream EOF/alignment cases; no compiler fixtures required."""
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from h16g_validate import ValidationError, task_stream

PREFIX = struct.pack('<I', 1) + bytes(12)


def task():
    # Nine header words plus a dense common-register write and its payload.
    return struct.pack('<11I', 11 << 16, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0)


class FinalTaskPadding(unittest.TestCase):
    def test_exact_final_end(self):
        self.assertEqual(task_stream(PREFIX + task(), 1), 1)

    def test_zero_final_padding(self):
        for pad in (4, 8, 12):
            with self.assertRaises(ValidationError):
                task_stream(PREFIX + task() + bytes(pad), 1)

    def test_zero_final_block(self):
        aligned = PREFIX + task() + bytes(4)
        with self.assertRaises(ValidationError):
            task_stream(aligned + bytes(16), 1)

    def test_zero_final_padding_after_aligned_tasks(self):
        with self.assertRaises(ValidationError):
            task_stream(PREFIX + task() + bytes(4) + task() + bytes(4), 2)

    def test_nonzero_final_padding(self):
        with self.assertRaises(ValidationError):
            task_stream(PREFIX + task() + b'\0\0\0\1', 1)

    def test_aligned_intertask_and_unpadded_final(self):
        self.assertEqual(task_stream(PREFIX + task() + bytes(4) + task(), 2), 2)

    def test_missing_intertask_padding(self):
        with self.assertRaises(ValidationError):
            task_stream(PREFIX + task() + task(), 2)

    def test_nonzero_intertask_padding(self):
        with self.assertRaises(ValidationError):
            task_stream(PREFIX + task() + b'\0\0\0\1' + task(), 2)

    def test_truncated_payload(self):
        with self.assertRaises(ValidationError):
            task_stream(PREFIX + task()[:-4], 1)

    def test_wrong_count(self):
        with self.assertRaises(ValidationError):
            task_stream(PREFIX + task(), 2)


if __name__ == '__main__':
    unittest.main()
