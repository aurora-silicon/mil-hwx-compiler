#!/usr/bin/env python3
"""Independent analytical checks for the float64 layer references."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import layer_reference as ref
sys.path.insert(0, str(ROOT / "tools/h13-conv-bench"))
from repro_camera046 import parameters


class LayerReferences(unittest.TestCase):
    def test_ties_away(self):
        bits = np.arange(0x0400, 0x7800, 997, dtype=np.uint16)
        lower = bits.view(np.float16).astype(np.float64)
        upper = (bits + 1).view(np.float16).astype(np.float64)
        midpoint = (lower + upper) / 2
        np.testing.assert_array_equal(ref.f16_ties_away(midpoint), upper)
        np.testing.assert_array_equal(ref.f16_ties_away(-midpoint), -upper)
        np.testing.assert_array_equal(ref.f16_ties_away([2**-25, -2**-25]), [2**-24, -2**-24])

    def test_nearest_and_bilinear_coordinates(self):
        x = np.array([[[0, 4], [8, 12]]], dtype=np.float16)
        np.testing.assert_array_equal(ref.nearest2x(x)[0],
            [[0,0,4,4], [0,0,4,4], [8,8,12,12], [8,8,12,12]])
        np.testing.assert_array_equal(ref.bilinear_half2x(x)[0],
            [[0,1,3,4], [2,3,5,6], [6,7,9,10], [8,9,11,12]])
        for shape in ((3,1,1), (24,35,35)):
            np.testing.assert_array_equal(ref.bilinear_half2x(np.full(shape, -0.75)), -0.75)

    def test_concat_and_sigmoid(self):
        x, z = np.ones((3, 5, 7)), np.full((8, 5, 7), 2)
        y = ref.concat_channels(x, z)
        np.testing.assert_array_equal(y[:3], 1)
        np.testing.assert_array_equal(y[3:], 2)
        values = np.array([-1000, -3, 0, 3, 1000], dtype=np.float64)
        np.testing.assert_allclose(ref.sigmoid(values) + ref.sigmoid(-values), 1, rtol=0, atol=3e-16)

    def test_hardswish_not_default_hard_sigmoid(self):
        x = np.array([-4, -3, -2.75, 0, 2.75, 3, 4], dtype=np.float64)
        expected = np.array([0,0,-2.75/24,0,2.75*23/24,3,4])
        np.testing.assert_allclose(ref.hardswish(x, staged=False), expected, rtol=0, atol=1e-15)
        self.assertLess(ref.hardswish(x, staged=False)[2], 0)

    def test_small_channel_5x5(self):
        for channels in (3,8,16,24,32):
            x = np.arange(channels * 7 * 7).reshape(channels,7,7) / 64
            w = np.zeros((channels,1,5,5))
            w[:,0,2,2] = 1
            np.testing.assert_array_equal(ref.conv(x,w,groups=channels,pad=2),x)
        x = np.arange(3 * 7 * 7).reshape(3,7,7) / 64
        w = np.zeros((8,3,5,5))
        w[:,:,2,2] = np.arange(24).reshape(8,3) / 16
        expected = np.einsum("oc,chw->ohw",w[:,:,2,2],x)
        np.testing.assert_allclose(ref.conv(x,w,pad=2),expected,rtol=0,atol=1e-14)

    def test_leading_padding_changes_values(self):
        x = np.arange(25).reshape(1,5,5)
        w = np.zeros((1,1,3,3)); w[0,0,1,1] = 1
        same = ref.conv(x,w,pad=1)
        shifted = ref.conv(x,w,pad=(0,2,1,1))
        self.assertEqual(same.shape, shifted.shape)
        np.testing.assert_array_equal(shifted[:,:-1],same[:,1:])
        np.testing.assert_array_equal(shifted[:,-1],0)

    def test_exact_camera_geometry(self):
        x, weights, bias = parameters("gather",3991)
        out = ref.f16_ties_away(ref.conv(x,weights,bias,pad=1))
        expected = np.zeros((512,35,35),dtype=np.float16)
        padded = np.pad(x[:96],((0,0),(1,1),(1,1)))
        for block,(dy,dx) in enumerate(((0,0),(0,2),(2,0),(2,2))):
            expected[block*96:(block+1)*96] = padded[:,dy:dy+35,dx:dx+35]
        np.testing.assert_array_equal(out,expected)


if __name__ == "__main__":
    unittest.main()
