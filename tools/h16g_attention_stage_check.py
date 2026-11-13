#!/usr/bin/env python3
"""Check H16G attention stage dumps with their physical IOSurface layouts.

Q and V are contiguous ``[H,S,D]`` fp16 surfaces. K is the runtime input
``[H,D,score_columns]`` with every logical S row padded to score_columns.
Scores and probabilities are ``[physical_heads,S,score_columns]``; only the
first H planes and first S columns are logical. The comparisons intentionally
isolate each device stage: QK uses CPU Q@K and the requested graph scale,
softmax uses the device QK values, and PV uses the device probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    denominator = np.linalg.norm(expected64)
    return float(np.linalg.norm(actual64 - expected64) /
                 max(float(denominator), 1e-300))


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    actual16 = np.asarray(actual, dtype=np.float16)
    expected16 = np.asarray(expected, dtype=np.float16)
    return {
        "elements": int(actual16.size),
        "exact": int(np.count_nonzero(actual16 == expected16)),
        "rel_l2": relative_l2(actual16, expected),
        "actual_norm": float(np.linalg.norm(actual16.astype(np.float64))),
        "expected_norm": float(np.linalg.norm(
            np.asarray(expected, dtype=np.float64))),
    }


def stable_softmax(value: np.ndarray) -> np.ndarray:
    work = np.asarray(value, dtype=np.float64)
    shifted = work - np.max(work, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def load_inputs(q_path: Path, k_path: Path, v_path: Path, heads: int,
                tokens: int, head_dim: int, score_columns: int):
    q = np.fromfile(q_path, dtype="<f2").reshape(heads, tokens, head_dim)
    k = np.fromfile(k_path, dtype="<f2").reshape(
        heads, head_dim, score_columns)
    v = np.fromfile(v_path, dtype="<f2").reshape(heads, tokens, head_dim)
    return q, k, v


def cpu_references(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                   scale: float, physical_heads: int,
                   score_columns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heads, tokens, _ = q.shape
    logical_qk = (q.astype(np.float64) @
                  k[:, :, :tokens].astype(np.float64)) * scale
    qk = np.zeros((physical_heads, tokens, score_columns), dtype=np.float16)
    qk[:heads, :, :tokens] = logical_qk.astype(np.float16)
    logical_probability = stable_softmax(qk[:heads, :, :tokens])
    probability = np.zeros_like(qk)
    probability[:heads, :, :tokens] = logical_probability.astype(np.float16)
    if physical_heads > heads:
        probability[heads:, :, :tokens] = np.float16(1.0 / tokens)
    pv = (logical_probability @ v.astype(np.float64)).astype(np.float16)
    return qk, probability, pv


def write_references(directory: Path, qk: np.ndarray,
                     probability: np.ndarray, pv: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    qk.astype("<f2").tofile(directory / "qk.fp16")
    probability.astype("<f2").tofile(directory / "softmax.fp16")
    pv.astype("<f2").tofile(directory / "pv.fp16")


def analyze_device(root: Path, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                   scale: float, physical_heads: int,
                   score_columns: int) -> dict:
    heads, tokens, head_dim = q.shape
    score_shape = (physical_heads, tokens, score_columns)
    device_qk = np.fromfile(root / "qk/output_00.bin", dtype="<f2").reshape(
        score_shape)
    device_p = np.fromfile(root / "softmax/output_00.bin", dtype="<f2").reshape(
        score_shape)
    device_pv = np.fromfile(root / "pv/output_00.bin", dtype="<f2").reshape(
        heads, tokens, head_dim)

    expected_qk = (q.astype(np.float64) @
                   k[:, :, :tokens].astype(np.float64)) * scale
    qk_heads = [metrics(device_qk[index, :, :tokens], expected_qk[index])
                for index in range(heads)]
    expected_p = stable_softmax(device_qk[:heads, :, :tokens])
    expected_pv = (device_p[:heads, :, :tokens].astype(np.float64) @
                   v.astype(np.float64))
    return {
        "layout": {
            "q": [heads, tokens, head_dim],
            "k_padded": [heads, head_dim, score_columns],
            "scores": [physical_heads, tokens, score_columns],
            "v": [heads, tokens, head_dim],
            "logical_score_columns": tokens,
        },
        "scale": scale,
        "qk": {
            **metrics(device_qk[:heads, :, :tokens], expected_qk),
            "per_head": qk_heads,
            "padding_nonzero": int(np.count_nonzero(
                device_qk[:heads, :, tokens:])),
        },
        "softmax_from_device_qk": metrics(
            device_p[:heads, :, :tokens], expected_p),
        "pv_from_device_probabilities": metrics(device_pv, expected_pv),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=Path, required=True)
    parser.add_argument("--k-padded", type=Path, required=True)
    parser.add_argument("--v", type=Path, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--physical-heads", type=int, required=True)
    parser.add_argument("--score-columns", type=int, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--device-root", type=Path)
    parser.add_argument("--write-references", type=Path)
    args = parser.parse_args()
    q, k, v = load_inputs(args.q, args.k_padded, args.v, args.heads,
                          args.tokens, args.head_dim, args.score_columns)
    if args.write_references:
        write_references(args.write_references, *cpu_references(
            q, k, v, args.scale, args.physical_heads, args.score_columns))
    if args.device_root:
        print(json.dumps(analyze_device(
            args.device_root, q, k, v, args.scale, args.physical_heads,
            args.score_columns), indent=2, sort_keys=True))
    if not args.device_root and not args.write_references:
        parser.error("one of --device-root or --write-references is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
