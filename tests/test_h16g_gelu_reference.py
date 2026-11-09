#!/usr/bin/env python3
"""Check the constructive numerical H16G GELU reference."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from h16g_gelu import GELU_TABLE, h16g_gelu  # noqa: E402


class H16GGELUReferenceTests(unittest.TestCase):
    def test_samples_follow_public_exact_gelu_rule(self):
        values = np.asarray([
            0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
            for x in np.linspace(-4.0, 4.0, 33)
        ], dtype=np.float16).astype(np.float64)
        np.testing.assert_array_equal(values, GELU_TABLE)

    def test_interpolation_and_tails(self):
        values = np.asarray([-5.0, -3.875, 0.125, 4.5], np.float32)
        expected = np.asarray([0.0, (GELU_TABLE[0] + GELU_TABLE[1]) / 2,
                               (GELU_TABLE[16] + GELU_TABLE[17]) / 2, 4.5],
                              np.float32)
        np.testing.assert_array_equal(h16g_gelu(values), expected)


if __name__ == '__main__':
    unittest.main()
