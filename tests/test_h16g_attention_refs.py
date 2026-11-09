#!/usr/bin/env python3
"""Device-free checks for the H16G attention CPU-reference generator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h16g_attention_refs", ROOT / "tools" / "h16g_attention_refs.py")
REFS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REFS)


def check(sequence: int, heads: int, head_dim: int) -> None:
    arrays = REFS.reference_arrays(sequence, heads, head_dim)
    assert arrays["flat"].shape == (1, sequence, heads * head_dim)
    assert arrays["split"].shape == (1, heads, sequence, head_dim)
    assert arrays["split_t"].shape == (1, heads, head_dim, sequence)
    assert arrays["qk"].shape == (1, heads, sequence, sequence)
    assert arrays["pv"].shape == (1, heads, sequence, head_dim)
    assert np.array_equal(arrays["merge"], arrays["flat"])
    assert np.array_equal(arrays["split"], arrays["q"])
    assert np.array_equal(arrays["qk"], arrays["qk_transpose_y"])
    sums = arrays["probabilities"].astype(np.float32).sum(axis=-1)
    assert np.max(np.abs(sums - 1.0)) < 2e-3
    assert np.isfinite(arrays["attention"]).all()
    exact = REFS.comparison(arrays["attention"], arrays["attention"])
    assert exact["exact"] == arrays["attention"].size
    changed = arrays["attention"].copy()
    changed.reshape(-1)[0] = np.nextafter(
        changed.reshape(-1)[0], np.float16(np.inf))
    one_ulp = REFS.comparison(arrays["attention"], changed)
    assert one_ulp["exact"] == arrays["attention"].size - 1
    assert one_ulp["within_1ulp"] == arrays["attention"].size


def main() -> int:
    # Requested held-out axes, kept small enough for an ordinary unit test.
    for shape in ((64, 2, 64), (32, 3, 80), (16, 2, 128)):
        check(*shape)
    print("H16G attention CPU references PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

