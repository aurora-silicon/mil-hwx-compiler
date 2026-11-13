#!/usr/bin/env python3
"""Device-free layout checks for h16g_attention_stage_check.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h16g_attention_stage_check",
    ROOT / "tools" / "h16g_attention_stage_check.py")
CHECK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECK)


def main() -> int:
    heads, tokens, head_dim = 2, 3, 2
    physical_heads, score_columns = 3, 4
    q = np.arange(1, 13, dtype=np.float16).reshape(heads, tokens, head_dim)
    k = np.arange(1, 17, dtype=np.float16).reshape(
        heads, head_dim, score_columns)
    v = (np.arange(1, 13, dtype=np.float16) / 8).reshape(
        heads, tokens, head_dim)
    qk, probability, pv = CHECK.cpu_references(
        q, k, v, 0.5, physical_heads, score_columns)
    assert qk.shape == probability.shape == (3, 3, 4)
    assert pv.shape == (2, 3, 2)
    assert np.count_nonzero(qk[:, :, tokens:]) == 0
    assert np.allclose(probability[heads:, :, :tokens].astype(np.float32),
                       1.0 / tokens, atol=5e-4)

    with tempfile.TemporaryDirectory(prefix="h16g-attention-stage-") as temp:
        root = Path(temp)
        q.tofile(root / "q.fp16")
        k.tofile(root / "k.fp16")
        v.tofile(root / "v.fp16")
        for name, value in (("qk", qk), ("softmax", probability),
                            ("pv", pv)):
            directory = root / "device" / name
            directory.mkdir(parents=True)
            value.tofile(directory / "output_00.bin")
        loaded = CHECK.load_inputs(root / "q.fp16", root / "k.fp16",
                                   root / "v.fp16", heads, tokens,
                                   head_dim, score_columns)
        report = CHECK.analyze_device(root / "device", *loaded, 0.5,
                                      physical_heads, score_columns)
        assert report["qk"]["rel_l2"] < 1e-3
        assert report["softmax_from_device_qk"]["rel_l2"] < 1e-3
        assert report["pv_from_device_probabilities"]["rel_l2"] < 1e-3
        CHECK.write_references(root / "references", qk, probability, pv)
        assert (root / "references/qk.fp16").stat().st_size == qk.nbytes
    print("H16G attention stage layout checks PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
