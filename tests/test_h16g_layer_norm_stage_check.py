#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from h16g_layer_norm_stage_check import (  # noqa: E402
    STAGES, analyze_prefix_directory, analyze_stage_directory, cpu_references,
    write_references)


def main() -> None:
    rng = np.random.default_rng(6568)
    value = rng.normal(size=(8, 16)).astype(np.float16)
    references = cpu_references(value)
    assert tuple(references) == STAGES
    assert all(item.shape == value.shape and item.dtype == np.float16
               for item in references.values())
    assert np.max(np.abs(references["normalized"].astype(np.float32).mean(1))) < 2e-3
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_references(root / "refs", references)
        for stage in STAGES:
            target = root / "out" / stage
            target.mkdir(parents=True)
            references[stage].astype("<f2").tofile(target / "output_00.bin")
        result = analyze_stage_directory(value, root / "out")
        assert result["first_wrong_stage"] is None
        (root / "out" / "variance" / "output_00.bin").write_bytes(
            bytes(value.nbytes))
        result = analyze_stage_directory(value, root / "out")
        assert result["first_wrong_stage"] == "variance"
        prefixes = [
            {"task_count": 1, "result_rows": 8, "result_row_bytes": 2,
             "token_start": 0, "expected_stage": "mean"},
            {"task_count": 2, "result_rows": 8, "result_row_bytes": 32,
             "token_start": 0, "expected_stage": "normalized"},
        ]
        prefix_root = root / "prefix"
        for item in prefixes:
            target = prefix_root / f"prefix-{item['task_count']:02d}"
            target.mkdir(parents=True)
            expected = references[item["expected_stage"]]
            elements = item["result_row_bytes"] // 2
            expected[:, :elements].astype("<f2").tofile(target / "output_00.bin")
        result = analyze_prefix_directory(value, prefix_root, prefixes)
        assert result["first_failing_task"] is None
        (prefix_root / "prefix-02" / "output_00.bin").write_bytes(
            bytes(value.nbytes))
        result = analyze_prefix_directory(value, prefix_root, prefixes)
        assert result["first_failing_task"] == 2
    print("H16G layer_norm stage check PASS")


if __name__ == "__main__":
    main()
