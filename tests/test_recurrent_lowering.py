# SPDX-License-Identifier: MIT
# Adapted from aurora-silicon/capture-ns tests/test_recurrent.py at f843fd3d053f716222caa7286a2095a9ffb8e762.
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from recurrent_lowering import gru, lstm


class RecurrentTest(unittest.TestCase):
    def check_cell(self, kind, reset=1):
        rng = np.random.default_rng(3654)
        I, H = 7, 11
        gates = 3 if kind == "GRU" else 4
        # Round the source weights once, exactly as the MIL exporter does.
        W, R, B = [rng.normal(0, .15, shape).astype(np.float16).astype(np.float32)
                   for shape in ((gates*H, I), (gates*H, H), (2*gates*H,))]
        g = gru(W, R, B, linear_before_reset=reset) if kind == "GRU" else lstm(W, R, B)
        params = [numpy_helper.from_array(v[None], name)
                  for name, v in (("W", W), ("R", R), ("B", B))]
        inputs = [h.make_tensor_value_info("X", T.FLOAT, [1, 1, I]),
                  h.make_tensor_value_info("H", T.FLOAT, [1, 1, H])]
        names = ["X", "W", "R", "B", "", "H"]
        outputs = [h.make_tensor_value_info("Y", T.FLOAT, [1, 1, 1, H]),
                   h.make_tensor_value_info("YH", T.FLOAT, [1, 1, H])]
        if kind == "LSTM":
            inputs.append(h.make_tensor_value_info("C", T.FLOAT, [1, 1, H]))
            names.append("C")
            outputs.append(h.make_tensor_value_info("YC", T.FLOAT, [1, 1, H]))
        attrs = {"hidden_size": H}
        if kind == "GRU": attrs["linear_before_reset"] = reset
        node = h.make_node(kind, names, [o.name for o in outputs], **attrs)
        graph = h.make_graph([node], "oracle", inputs, outputs, params)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)], ir_version=10)
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        state, cell = np.zeros((1, H), np.float32), np.zeros((1, H), np.float32)
        for _ in range(100):
            x = rng.normal(0, .5, (1, I)).astype(np.float32)
            feeds = {"X": x[None], "H": state[None]}
            if kind == "LSTM": feeds["C"] = cell[None]
            expected = session.run(None, feeds)
            actual = g.evaluate({"x": x, "h": state, "c": cell})
            state = actual["h_next"]
            np.testing.assert_allclose(state, expected[1][0], atol=2e-6, rtol=2e-6)
            if kind == "LSTM":
                cell = actual["c_next"]
                np.testing.assert_allclose(cell, expected[2][0], atol=2e-6, rtol=2e-6)
        with tempfile.TemporaryDirectory() as d:
            g.write(d)
            text = (Path(d) / "cell.mil").read_text()
            self.assertNotIn("= gru(", text)
            self.assertNotIn("= lstm(", text)
            self.assertTrue((Path(d) / "weights.bin").stat().st_size > 64)

    def test_gru_reset_after(self): self.check_cell("GRU", 1)
    def test_gru_reset_before(self): self.check_cell("GRU", 0)
    def test_lstm(self): self.check_cell("LSTM")

    def test_rnnoise_candidate_activation(self):
        W, R = np.zeros((6, 3), np.float32), np.zeros((6, 2), np.float32)
        B = np.array([0, 0, 0, 0, -2, 3] + [0]*6, np.float32)
        g = gru(W, R, B, linear_before_reset=0, candidate="relu")
        actual = g.evaluate({"x": np.zeros((1, 3)), "h": np.zeros((1, 2))})
        np.testing.assert_allclose(actual["h_next"], [[0, 1.5]])

    def test_refuse_invalid_mode(self):
        with self.assertRaises(ValueError):
            gru(np.zeros((6, 3)), np.zeros((6, 2)), linear_before_reset=2)

    def test_padding_preserves_state_and_zero_tail(self):
        rng = np.random.default_rng(3654)
        I, H, P = 7, 11, 512
        for factory, gates, options in ((gru, 3, {"linear_before_reset": 0, "candidate": "relu"}),
                                        (gru, 3, {"linear_before_reset": 1}), (lstm, 4, {})):
            with self.subTest(factory=factory.__name__, options=options):
                W, R, B = [rng.normal(0, .1, shape).astype(np.float16).astype(np.float32)
                           for shape in ((gates*H, I), (gates*H, H), (2*gates*H,))]
                plain = factory(W, R, B, **options)
                padded = factory(W, R, B, pad_to=P, **options)
                h, c = np.zeros((1, H), np.float32), np.zeros((1, H), np.float32)
                hp, cp = np.zeros((1, P), np.float32), np.zeros((1, P), np.float32)
                for _ in range(100):
                    x = rng.normal(0, .5, (1, I)).astype(np.float32)
                    xp = np.pad(x, ((0, 0), (0, P-I)))
                    expected = plain.evaluate({"x": x, "h": h, "c": c})
                    actual = padded.evaluate({"x": xp, "h": hp, "c": cp})
                    h, hp = expected["h_next"], actual["h_next"]
                    np.testing.assert_allclose(hp[:, :H], h, atol=2e-6, rtol=2e-6)
                    np.testing.assert_array_equal(hp[:, H:], 0)
                    if factory is lstm:
                        c, cp = expected["c_next"], actual["c_next"]
                        np.testing.assert_allclose(cp[:, :H], c, atol=2e-6, rtol=2e-6)
                        np.testing.assert_array_equal(cp[:, H:], 0)
                with tempfile.TemporaryDirectory() as d:
                    padded.write(d, runtime_vectors=True)
                    interface = json.loads((Path(d) / "interface.json").read_text())
                    self.assertEqual(interface["valid_sizes"], {"input": I, "hidden": H})
                    self.assertIn("Bi" if factory is lstm else "Bz", interface["fixed_inputs"])
                    self.assertFalse(any(name.startswith(("W", "R")) for name in interface["fixed_inputs"]))
                    with np.load(Path(d) / "fixed-inputs.npz", allow_pickle=False) as feeds:
                        self.assertTrue(all(feeds[name].dtype == np.float16 for name in feeds.files))


if __name__ == "__main__":
    unittest.main()
