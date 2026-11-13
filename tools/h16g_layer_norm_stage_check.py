#!/usr/bin/env python3
"""Build and judge H16G layer-normalization boundary references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGES = ("mean", "centered", "variance", "rsqrt", "normalized")


def cpu_references(value: np.ndarray) -> dict[str, np.ndarray]:
    """Return float64-statistics stages in external row-major [S,C] layout."""
    wide = value.astype(np.float64)
    mean_scalar = wide.mean(axis=-1, keepdims=True)
    centered = wide - mean_scalar
    variance_scalar = (centered * centered).mean(axis=-1, keepdims=True)
    epsilon = np.float16(1e-5).astype(np.float64)
    rsqrt_scalar = 1.0 / np.sqrt(variance_scalar + epsilon)
    shape = wide.shape
    return {
        "mean": np.broadcast_to(mean_scalar, shape).astype(np.float16),
        "centered": centered.astype(np.float16),
        "variance": np.broadcast_to(variance_scalar, shape).astype(np.float16),
        "rsqrt": np.broadcast_to(rsqrt_scalar, shape).astype(np.float16),
        "normalized": (centered * rsqrt_scalar).astype(np.float16),
    }


def write_references(output: Path, references: dict[str, np.ndarray]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        references[stage].astype("<f2", copy=False).tofile(output / f"{stage}.fp16")


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.linalg.norm(expected.astype(np.float64).ravel())
    if denominator == 0:
        return float(np.linalg.norm(actual.astype(np.float64).ravel()))
    return float(np.linalg.norm((actual.astype(np.float64) -
                                 expected.astype(np.float64)).ravel()) /
                 denominator)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    a = actual.astype(np.float64).ravel()
    e = expected.astype(np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(e)
    correlation = float("nan")
    if finite.any() and np.std(a[finite]) and np.std(e[finite]):
        correlation = float(np.corrcoef(a[finite], e[finite])[0, 1])
    return {
        "rel_l2": relative_l2(actual, expected),
        "correlation": correlation,
        "exact": int(np.count_nonzero(actual.view(np.uint16) ==
                                       expected.view(np.uint16))),
        "nonfinite": int(np.count_nonzero(~np.isfinite(a))),
    }


def analyze_final(actual: np.ndarray,
                  references: dict[str, np.ndarray]) -> dict[str, object]:
    rows, channels = actual.shape
    comparisons = {stage: metrics(actual, references[stage]) for stage in STAGES}
    nonzero = np.flatnonzero(actual)
    nonzero_rows = np.unique(nonzero // channels) if nonzero.size else np.array([], int)
    layout = {
        "assumed_external": "row-major fp16 [S,C], row stride 2*C",
        "nonzero_values": int(nonzero.size),
        "nonzero_rows": int(nonzero_rows.size),
        "first_nonzero_row": int(nonzero_rows[0]) if nonzero_rows.size else None,
        "last_nonzero_row": int(nonzero_rows[-1]) if nonzero_rows.size else None,
        "row_mean_abs_max": float(np.nanmax(np.abs(np.nanmean(actual, axis=1)))),
        "row_std_median": float(np.nanmedian(np.nanstd(actual, axis=1))),
    }
    # A flattened CxS interpretation is the principal row/channel-stride
    # alternative for these contiguous surfaces.
    transposed = actual.reshape(channels, rows).T
    layout["channel_major_vs_normalized"] = metrics(
        transposed, references["normalized"])
    return {"comparisons": comparisons, "layout": layout}


def analyze_stage_directory(input_value: np.ndarray, output: Path,
                            threshold: float = 1e-2) -> dict[str, object]:
    references = cpu_references(input_value)
    rows, channels = input_value.shape
    results: dict[str, object] = {}
    first_wrong = None
    for stage in STAGES:
        candidates = (output / stage / "output_00.bin", output / f"{stage}.bin")
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            results[stage] = {"missing": True}
            continue
        actual = np.fromfile(path, dtype="<f2")
        if actual.size != rows * channels:
            results[stage] = {"elements": int(actual.size),
                              "expected_elements": rows * channels}
            if first_wrong is None:
                first_wrong = stage
            continue
        result = metrics(actual.reshape(rows, channels), references[stage])
        results[stage] = result
        if first_wrong is None and (result["nonfinite"] or
                                    result["rel_l2"] > threshold):
            first_wrong = stage
    return {"stages": results, "first_wrong_stage": first_wrong,
            "threshold": threshold}


def load_input(path: Path, tokens: int, channels: int) -> np.ndarray:
    value = np.fromfile(path, dtype="<f2")
    if value.size != tokens * channels:
        raise ValueError(f"{path}: {value.size} elements, expected {tokens * channels}")
    return value.reshape(tokens, channels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("input", type=Path)
    write.add_argument("tokens", type=int)
    write.add_argument("channels", type=int)
    write.add_argument("output", type=Path)
    final = sub.add_parser("analyze-final")
    final.add_argument("input", type=Path)
    final.add_argument("device", type=Path)
    final.add_argument("tokens", type=int)
    final.add_argument("channels", type=int)
    stages = sub.add_parser("analyze-stages")
    stages.add_argument("input", type=Path)
    stages.add_argument("output", type=Path)
    stages.add_argument("tokens", type=int)
    stages.add_argument("channels", type=int)
    args = parser.parse_args()
    value = load_input(args.input, args.tokens, args.channels)
    references = cpu_references(value)
    if args.command == "write":
        write_references(args.output, references)
        return 0
    if args.command == "analyze-final":
        actual = load_input(args.device, args.tokens, args.channels)
        result = analyze_final(actual, references)
    else:
        result = analyze_stage_directory(value, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
