#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Coverage audits must distinguish sample data from compile-time values."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dfn3_coverage import observe, emit_probe, Unsupported
from h13_reference import evaluate


class CoverageTest(unittest.TestCase):
    def test_sample_is_not_a_constant(self):
        nodes = [h.make_node('Shape', ['x'], ['shape']),
                 h.make_node('Add', ['x', 'bias'], ['y'])]
        model = h.make_model(h.make_graph(nodes, 'static-shape',
            [h.make_tensor_value_info('x', T.FLOAT, [1, 'S', 4])],
            [h.make_tensor_value_info('y', T.FLOAT, [1, 'S', 4])],
            [numpy_helper.from_array(np.zeros((1, 1, 4), np.float32), 'bias')]),
            opset_imports=[h.make_opsetid('', 14)], ir_version=10)
        for frames in (1, 3):
            values, constants = observe(model, frames)
            self.assertEqual(set(constants), {'bias', 'shape'})
            self.assertEqual(values['shape'].tolist(), [1, frames, 4])
            self.assertEqual(values['y'].shape, (1, frames, 4))
            self.assertTrue(np.all(values['y'] == 0))

    def test_conv_padding_and_weights(self):
        # An asymmetric pad catches ONNX-to-MIL pad-order mistakes. Integer
        # inputs/weights keep the independent fp32 and fp16 references exact.
        x = np.arange(30, dtype=np.float32).reshape(1, 2, 3, 5) % 5 - 2
        w = np.arange(24, dtype=np.float32).reshape(4, 2, 1, 3) % 3 - 1
        node = h.make_node('Conv', ['x', 'w'], ['y'], pads=[0, 2, 0, 0], strides=[1, 1])
        model = h.make_model(h.make_graph([node], 'conv',
            [h.make_tensor_value_info('x', T.FLOAT, list(x.shape))],
            [h.make_tensor_value_info('y', T.FLOAT, [1, 4, 3, 5])],
            [numpy_helper.from_array(w, 'w')]), opset_imports=[h.make_opsetid('', 14)], ir_version=10)
        # Observe uses zero runtime inputs. Use a separate independent ONNX
        # evaluation on signed nonzero data for the semantic mapping check.
        import onnxruntime as ort
        session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        expected = session.run(None, {'x': x})[0].astype('<f2').tobytes()
        values = {'x': x, 'w': w, 'y': np.empty((1, 4, 3, 5), np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            emit_probe(node, values, {'w'}, directory)
            text = (directory / 'cell.mil').read_text()
            self.assertIn('[0, 0, 2, 0]', text)
            actual = evaluate(text, directory, {'x': x.astype('<f2').tobytes()})
            self.assertEqual(actual['result'], expected)

    def test_missing_lowering_is_explicit(self):
        node = h.make_node('ConvTranspose', ['x', 'w'], ['y'])
        arrays = {name: np.zeros((1, 1, 1, 1), np.float32) for name in ('x', 'w', 'y')}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Unsupported, 'coverage.onnx-convtranspose'):
                emit_probe(node, arrays, {'w'}, Path(tmp))
            self.assertFalse((Path(tmp) / 'cell.mil').exists())


if __name__ == '__main__':
    unittest.main()
