#!/usr/bin/env python3
"""CPU reference tests only. No hardware envelope or device-run claim."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import h13_reference as ref
from spatial_tiles import conv_tiles


def mil(parameters, body, outputs='y'):
    return ('program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
            f'func main<ios18>({parameters}) {{\n{body}\n}} -> ({outputs});\n}}\n')


def run(source, **inputs):
    return {name: ref.decode_fp16(data) for name, data in
            ref.evaluate(source, Path('.'), inputs).items()}


class ReferenceTests(unittest.TestCase):
    def test_batch8_conv_reference(self):
        values = tuple((i - 16) * .125 for i in range(32))
        result = ref._conv(ref.Tensor('fp16', (8, 1, 2, 2), values),
                           ref.Tensor('fp16', (1, 1, 1, 1), (2.,)),
                           ref.Tensor('fp16', (1,), (-1.,)),
                           (1, 1), (1, 1), (0, 0, 0, 0), 'valid', 1,
                           ref.MILType('fp16', (8, 1, 2, 2)))
        self.assertEqual(result.shape, (8, 1, 2, 2))
        self.assertEqual(result.values, tuple(2 * v - 1 for v in values))

    def test_concat_outer_inner_and_negative_axis(self):
        source = mil('tensor<fp16, [2, 1, 2]> a, tensor<fp16, [2, 2, 2]> b',
            'tensor<fp16, [2, 3, 2]> y = concat(values = (a, b), axis = int32(-2));')
        self.assertEqual(run(source, a=[1, 2, 3, 4], b=[5, 6, 7, 8, 9, 10, 11, 12])['y'],
                         (1, 2, 5, 6, 7, 8, 3, 4, 9, 10, 11, 12))
        scalar_tensor = source.replace('int32(-2)', 'tensor<int32, []>([-2])')
        self.assertEqual(run(scalar_tensor, a=[1, 2, 3, 4], b=list(range(8)))['y'],
                         (1, 2, 0, 1, 2, 3, 3, 4, 4, 5, 6, 7))
        with self.assertRaisesRegex(ValueError, 'concat-interleave'):
            run(source.replace('axis = int32(-2)', 'axis = int32(-2), interleave = bool(true)'),
                a=[1, 2, 3, 4], b=list(range(8)))

    def test_nearest_exact_samples(self):
        source = mil('tensor<fp16, [1, 1, 2, 2]> x',
            'tensor<fp16, [1, 1, 4, 6]> y = upsample_nearest_neighbor(x = x, '
            'scale_factor_height = int32(2), scale_factor_width = int32(3));')
        self.assertEqual(run(source, x=[1, 2, 3, 4])['y'],
                         (1, 1, 1, 2, 2, 2) * 2 + (3, 3, 3, 4, 4, 4) * 2)
        source = source.replace('upsample_nearest_neighbor', 'resize_nearest_neighbor').replace(
            'scale_factor_height = int32(2), scale_factor_width = int32(3)',
            'target_size_height = int32(4), target_size_width = int32(6)')
        self.assertEqual(run(source, x=[1, 2, 3, 4])['y'][:6], (1, 1, 1, 2, 2, 2))

    def test_bilinear_coordinate_systems_and_singleton(self):
        source = mil('tensor<fp16, [1, 1, 1, 2]> x',
            'tensor<fp16, [1, 1, 1, 5]> y = resize_bilinear(x = x, '
            'target_size_height = int32(1), target_size_width = int32(5), '
            'sampling_mode = string("MODE"));')
        for mode, expected in (
            ('STRICT_ALIGN_CORNERS', [0, .25, .5, .75, 1]),
            ('DEFAULT', [0, .4, .8, 1, 1]),
            ('UNALIGN_CORNERS', [0, .1, .5, .9, 1]),
        ):
            with self.subTest(mode=mode):
                self.assertEqual(run(source.replace('MODE', mode), x=[0, 1])['y'],
                                 tuple(map(ref.fp16, expected)))
        with self.assertRaisesRegex(ValueError, 'sampling-mode-outside-envelope'):
            run(source.replace('MODE', 'guessed'), x=[0, 1])
        upsample = mil('tensor<fp16, [1, 1, 2, 2]> x',
            'tensor<fp16, [1, 1, 4, 4]> y = upsample_bilinear(x = x, '
            'scale_factor_height = int32(2), scale_factor_width = int32(2), '
            'align_corners = bool(false));')
        expected = (0, .25, .75, 1, .5, .75, 1.25, 1.5,
                    1.5, 1.75, 2.25, 2.5, 2, 2.25, 2.75, 3)
        self.assertEqual(run(upsample, x=[0, 1, 2, 3])['y'], expected)

    def test_halo_seams(self):
        # Distinct signed input/weight values expose spatial shifts, missing
        # channels, halo errors, and swapped padding on both partial axes.
        for kernel in (1, 3):
            for channels, outputs, groups in ((3, 2, 1), (3, 3, 3)):
                with self.subTest(kernel=kernel, channels=channels, groups=groups):
                    x = ref.Tensor('fp16', (1, channels, 13, 15), tuple(
                        ((i * 7) % 23 - 11) * .125 for i in range(channels * 13 * 15)))
                    w = ref.Tensor('fp16', (outputs, channels // groups, kernel, kernel), tuple(
                        (i % 7 - 3) * .0625 for i in range(outputs * (channels // groups) * kernel ** 2)))
                    b = ref.Tensor('fp16', (outputs,), tuple(i * .125 for i in range(outputs)))
                    dense = ref._conv(x, w, b, (1, 1), (1, 1), (0, 0, 0, 0),
                                      'same', groups, ref.MILType('fp16', (1, outputs, 13, 15)))
                    tiled = ref.conv_tiled(x, w, b, groups, patch_size=8)
                    self.assertEqual(ref.encode_fp16(tiled.values), ref.encode_fp16(dense.values))
    def test_full_camera_frame_partition(self):
        for height, width in ((1080, 1920), (720, 1280)):
            for kernel in (1, 3):
                tiles = conv_tiles(height, width, kernel)
                core = 64 - kernel + 1
                self.assertEqual(len(tiles), math.ceil(height / core) * math.ceil(width / core))
                self.assertEqual(sum(t.output_height * t.output_width for t in tiles), height * width)
                # Disjoint ownership by rows; there are no gaps or overlaps.
                for y in range(0, height, core):
                    row = [t for t in tiles if t.output_top == y]
                    self.assertEqual(row[0].output_left, 0)
                    self.assertEqual(row[-1].output_left + row[-1].output_width, width)
                    self.assertTrue(all(a.output_left + a.output_width == b.output_left
                                        for a, b in zip(row, row[1:])))
                    for t in row:
                        self.assertEqual(t.input_top + t.crop_top, t.output_top)
                        self.assertEqual(t.input_left + t.crop_left, t.output_left)
                        self.assertLessEqual(t.crop_top + t.output_height + kernel // 2, 64)



if __name__ == '__main__':
    unittest.main()
