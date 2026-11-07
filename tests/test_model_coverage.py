#!/usr/bin/env python3
"""Offline real-model exporter, full refusal diagnostics and fp16 reference checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from mil_model_coverage import Exporter, report
from resnet_fp16_reference import Reference, convolution, pool, surface
import h13_reference


class ReferenceTests(unittest.TestCase):
    def test_convolution_against_scalar_sum(self):
        # Includes the 7x7/s2 stem's asymmetric SAME padding, dense 3x3/s2,
        # and pointwise bias. A separate scalar loop checks indexing/arithmetic.
        rng = np.random.default_rng(1234)
        for size, kernel, stride in [(8, 7, 2), (8, 3, 2), (7, 3, 1), (5, 1, 1)]:
            x = rng.uniform(-1, 1, (1, 3, size, size)).astype(np.float16)
            w = rng.uniform(-.2, .2, (4, 3, kernel, kernel)).astype(np.float16)
            b = rng.uniform(-.1, .1, 4).astype(np.float16)
            actual = convolution(x, w, b, strides=(stride, stride))
            ho = (size + stride - 1) // stride
            pad = max((ho - 1) * stride + kernel - size, 0) // 2
            expected = np.zeros((1, 4, ho, ho), dtype=np.float64)
            for o in range(4):
                for y in range(ho):
                    for z in range(ho):
                        total = float(b[o])
                        for c in range(3):
                            for ky in range(kernel):
                                for kx in range(kernel):
                                    iy, ix = y * stride + ky - pad, z * stride + kx - pad
                                    if 0 <= iy < size and 0 <= ix < size:
                                        total += float(x[0, c, iy, ix]) * float(w[o, c, ky, kx])
                        expected[0, o, y, z] = total
            np.testing.assert_array_equal(actual.astype(np.float16), expected.astype(np.float16))

    def test_pool_same_negative_and_asymmetric(self):
        x = -np.arange(1, 17, dtype=np.float16).reshape(1, 1, 4, 4)
        expected = np.array([[-1, -3], [-9, -11]], dtype=np.float64).reshape(1, 1, 2, 2)
        np.testing.assert_array_equal(pool(x, [3, 3], [2, 2], 'same'), expected)

    def test_surface_rows(self):
        a = np.arange(42, dtype=np.float16).reshape(1, 2, 3, 7)
        encoded = np.frombuffer(surface(a), dtype='<f2').reshape(2, 3, 32)
        np.testing.assert_array_equal(encoded[:, :, :7], a[0])
        self.assertTrue((encoded[:, :, 7:] == 0).all())

    def test_rounding_at_residual_and_reduce_boundaries(self):
        # The add must round before the mean: rounding only the final result
        # yields 0.00024414 instead of zero here.
        graph = {'inputs': ['x'], 'outputs': ['out'],
                 'variables': {'x': {'dtype': 'fp16', 'shape': [1, 2]}},
                 'constants': {'b': {'array': 'b'}, 'axes': {'array': 'axes'},
                               'keep': {'value': False}},
                 'operations': [
                     {'op': 'add', 'name': 'residual', 'result': 'a',
                      'inputs': {'x': 'x', 'y': 'b'}, 'type': {'dtype': 'fp16', 'shape': [1, 2]}},
                     {'op': 'reduce_mean', 'name': 'pool', 'result': 'out',
                      'inputs': {'x': 'a', 'axes': 'axes', 'keep_dims': 'keep'},
                      'type': {'dtype': 'fp16', 'shape': [1]}}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'graph.json').write_text(json.dumps(graph))
            np.savez(path / 'constants.npz', b=np.array([[1, -1]], np.float16), axes=np.array([1]))
            y = Reference(path).evaluate({'x': np.array([[1 / 2048, 0]], np.float16)})['out']
            np.testing.assert_array_equal(y, np.array([0], np.float16))


class ExportTests(unittest.TestCase):
    def test_export_and_compile_all_refusals(self):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types

        @mb.program(input_specs=[mb.TensorSpec((1, 3, 8, 8), dtype=types.fp16)],
                    opset_version=ct.target.macOS15)
        def program(x):
            weights = ((np.arange(16 * 3 * 7 * 7) % 31 - 15) / 128).astype(np.float16)
            y = mb.conv(x=x, weight=weights.reshape(16, 3, 7, 7), pad_type='same',
                        strides=[2, 2], bias=np.zeros(16, np.float16))
            y = mb.relu(x=y)
            y = mb.max_pool(x=y, kernel_sizes=[3, 3], strides=[2, 2], pad_type='same')
            y = mb.add(x=y, y=y)
            return mb.reduce_mean(x=y, axes=[2, 3], keep_dims=False)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            exporter = Exporter(program.functions['main'], root / 'model')
            text = (root / 'model/model.mil').read_text()
            self.assertIn('BLOBFILE', text)
            self.assertNotIn('...', text)
            # Decode the textual constants using the independent existing MIL
            # parser/blob reader, rather than just reading our NPZ sidecar.
            _, operations, _ = h13_reference.Parser(text).parse()
            environment = {}
            reference = Reference(root / 'model')
            for operation in operations:
                if operation.name != 'const':
                    continue
                decoded = h13_reference._execute(operation, environment, root / 'model')
                environment[operation.result] = decoded
                expected = reference.constants[operation.result]
                if isinstance(decoded, h13_reference.Tensor):
                    np.testing.assert_array_equal(np.asarray(decoded.values).reshape(decoded.shape), expected)
                else:
                    self.assertEqual(decoded, expected)
            x = np.ones((1, 3, 8, 8), np.float16)
            output = next(iter(reference.evaluate({exporter.graph['inputs'][0]: x}).values()))
            self.assertEqual(output.shape, (1, 16))
            compiler = os.environ.get('MILC')
            if not compiler:
                self.fail('MILC must point to the buildbox compiler for integration checks')
            for target, code in [('H16G', 'h16g.conv-outside-envelope'),
                                 ('H17', 'h17.outside-parity-envelope')]:
                path = root / target
                result = subprocess.run([compiler, '--mil', str(root / 'model/model.mil'),
                    '--model-root', str(root / 'model'), '--target', target, '--format', 'hwx',
                    '--schedule', 'linked', '--output', str(path)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stderr.count(f'[{code}]'), 4, result.stderr)
                for kind in ['conv', 'max_pool', 'add', 'reduce_mean']:
                    self.assertIn(kind, result.stderr)
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
