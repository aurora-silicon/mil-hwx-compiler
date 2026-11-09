#!/usr/bin/env python3
"""Keep the numerical H16G GELU reference tied to the encoded public table."""
from __future__ import annotations

from pathlib import Path
import re
import struct
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from h16g_gelu import GELU_TABLE, h16g_gelu  # noqa: E402


class H16GGELUReferenceTests(unittest.TestCase):
    def test_samples_match_encoder_words(self):
        source = (ROOT / 'plugins/H16G/Encoding/H16GLUTEncoderData.inc').read_text()
        body = re.search(r'kGeluKERNWords\[\] = \{(.*?)\};', source,
                         re.DOTALL).group(1)
        words = [int(value, 16) for value in re.findall(r'0x[0-9a-fA-F]+', body)]
        halves = []
        for word in words[2:19]:
            halves.extend(struct.unpack('<2e', struct.pack('<I', word)))
        np.testing.assert_array_equal(np.asarray(halves[:33]), GELU_TABLE)

    def test_interpolation_and_tails(self):
        values = np.asarray([-5.0, -3.875, 0.125, 4.5], np.float32)
        expected = np.asarray([0.0, (GELU_TABLE[0] + GELU_TABLE[1]) / 2,
                               (GELU_TABLE[16] + GELU_TABLE[17]) / 2, 4.5],
                              np.float32)
        np.testing.assert_array_equal(h16g_gelu(values), expected)


if __name__ == '__main__':
    unittest.main()
