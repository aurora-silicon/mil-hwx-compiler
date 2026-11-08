#!/usr/bin/env python3
"""Generate deterministic fp16 CPU references for H16G attention probes.

Arrays use the logical row-major shapes in the MIL programs.  The runtime
manifest supplies the physical row/plane strides when these arrays are packed
into an IOSurface; keeping that packing outside this file makes each numerical
reference independent of a particular descriptor hypothesis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GEOMETRIES = {
    "bert": (128, 12, 64),
    "wtiny": (1500, 6, 64),
    "wbase": (1500, 8, 64),
}


def f16(value: np.ndarray) -> np.ndarray:
    """Materialize the fp16 boundary between H16G tasks."""
    return np.asarray(value, dtype=np.float16)


def matmul_f16(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Float32 accumulation followed by the observable fp16 task boundary."""
    return f16(np.matmul(left.astype(np.float32), right.astype(np.float32)))


def softmax_f16(value: np.ndarray) -> np.ndarray:
    """Stable last-axis softmax with float32 reductions and fp16 output."""
    work = value.astype(np.float32)
    work -= np.max(work, axis=-1, keepdims=True)
    np.exp(work, out=work)
    work /= np.sum(work, axis=-1, keepdims=True, dtype=np.float32)
    return f16(work)


def inputs(sequence: int, heads: int, head_dim: int, seed: int = 5727):
    rng = np.random.default_rng(seed + sequence * 17 + heads * 31 + head_dim)
    shape = (1, heads, sequence, head_dim)
    # Unit-variance Q/K keeps the scaled score range realistic.  V is smaller
    # so a context row remains well inside fp16 even at sequence 1500.
    q = f16(rng.standard_normal(shape, dtype=np.float32))
    k = f16(rng.standard_normal(shape, dtype=np.float32))
    v = f16(rng.standard_normal(shape, dtype=np.float32) * np.float32(0.25))
    return q, k, v


def split_reference(flat: np.ndarray, heads: int, head_dim: int):
    sequence = flat.shape[1]
    shaped = flat.reshape(1, sequence, heads, head_dim)
    split = shaped.transpose(0, 2, 1, 3).copy()
    split_t = shaped.transpose(0, 2, 3, 1).copy()
    return split, split_t


def merge_reference(split: np.ndarray) -> np.ndarray:
    batch, heads, sequence, head_dim = split.shape
    return split.transpose(0, 2, 1, 3).reshape(
        batch, sequence, heads * head_dim).copy()


def attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    head_dim = q.shape[-1]
    q32 = q.astype(np.float32)
    kt32 = k.transpose(0, 1, 3, 2).astype(np.float32)
    score_accumulator = np.matmul(q32, kt32)
    scores = f16(score_accumulator)
    # Apple's fused form puts 1/sqrt(dh) in the matmul output-scale field, so
    # it scales the accumulator before the one observable fp16 boundary.
    scaled = f16(score_accumulator / np.float32(np.sqrt(head_dim)))
    probabilities = softmax_f16(scaled)
    context = matmul_f16(probabilities, v)
    return scores, scaled, probabilities, context


def reference_arrays(sequence: int, heads: int, head_dim: int,
                     seed: int = 5727) -> dict[str, np.ndarray]:
    q, k, v = inputs(sequence, heads, head_dim, seed)
    flat = q.transpose(0, 2, 1, 3).reshape(1, sequence, heads * head_dim)
    split, split_t = split_reference(flat, heads, head_dim)
    k_t = k.transpose(0, 1, 3, 2).copy()
    scores, scaled, probabilities, context = attention_reference(q, k, v)
    # The explicit-K and transpose_y forms must agree at their fp16 boundary.
    qk_explicit = matmul_f16(q, k_t)
    qk_transpose_y = matmul_f16(q, k.transpose(0, 1, 3, 2))
    if not np.array_equal(qk_explicit, qk_transpose_y):
        raise AssertionError("explicit K^T and transpose_y CPU references differ")
    return {
        "flat": flat,
        "split": split,
        "split_t": split_t,
        "merge": merge_reference(split),
        "q": q,
        "k": k,
        "k_t": k_t,
        "v": v,
        "qk": qk_explicit,
        "qk_transpose_y": qk_transpose_y,
        "scaled": scaled,
        "probabilities": probabilities,
        "pv": context,
        "attention": context,
        "q1": q[:, :1].copy(),
        "k1_t": k_t[:, :1].copy(),
        "v1": v[:, :1].copy(),
        "qk1": scores[:, :1].copy(),
        "pv1": context[:, :1].copy(),
    }


def write_raw(directory: Path, arrays: dict[str, np.ndarray], metadata: dict):
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, array in arrays.items():
        path = directory / f"{name}.fp16"
        f16(array).tofile(path)
        records[name] = {
            "file": path.name,
            "shape": list(array.shape),
            "dtype": "float16-le",
            "bytes": path.stat().st_size,
        }
    (directory / "manifest.json").write_text(json.dumps(
        {**metadata, "arrays": records}, indent=2, sort_keys=True) + "\n")


def comparison(expected: np.ndarray, actual: np.ndarray) -> dict:
    expected = f16(expected).ravel()
    actual = f16(actual).ravel()
    if actual.shape != expected.shape:
        raise ValueError(f"actual has {actual.size} elements; expected "
                         f"{expected.size}")
    ai = actual.view(np.int16).astype(np.int32)
    ei = expected.view(np.int16).astype(np.int32)
    ai = np.where(ai < 0, -32768 - ai, ai)
    ei = np.where(ei < 0, -32768 - ei, ei)
    ulp = np.abs(ai - ei)
    equal = actual == expected
    actual64 = actual.astype(np.float64)
    expected64 = expected.astype(np.float64)
    delta = actual64 - expected64
    return {
        "elements": int(expected.size),
        "exact": int(equal.sum()),
        "within_1ulp": int((equal | (ulp <= 1)).sum()),
        "max_ulp": int(np.where(equal, 0, ulp).max(initial=0)),
        "max_abs": float(np.abs(delta).max(initial=0.0)),
        "rel_l2": float(np.linalg.norm(delta) /
                        max(np.linalg.norm(expected64), 1e-30)),
        "actual_nonfinite": int((~np.isfinite(actual64)).sum()),
    }


def compare_file(directory: Path, name: str, actual_path: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    record = manifest["arrays"][name]
    expected = np.fromfile(directory / record["file"], dtype="<f2")
    actual = np.fromfile(actual_path, dtype="<f2")
    return {"array": name, "actual": str(actual_path),
            **comparison(expected, actual)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", choices=GEOMETRIES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=5727)
    parser.add_argument("--compare", nargs=2, metavar=("ARRAY", "ACTUAL_FP16"))
    args = parser.parse_args()
    if args.compare:
        name, actual = args.compare
        print(json.dumps(compare_file(args.output, name, Path(actual)),
                         sort_keys=True))
        return 0
    sequence, heads, head_dim = GEOMETRIES[args.geometry]
    arrays = reference_arrays(sequence, heads, head_dim, args.seed)
    write_raw(args.output, arrays, {
        "geometry": args.geometry,
        "sequence": sequence,
        "heads": heads,
        "head_dim": head_dim,
        "seed": args.seed,
        "matmul_accumulator": "float32",
        "task_boundaries": "float16",
        "softmax_reduction": "float32",
    })
    print(args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

