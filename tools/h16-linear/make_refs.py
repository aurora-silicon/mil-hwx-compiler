#!/usr/bin/env python3
"""Generate deterministic fp16 transformer-linear device references.

The files are logical contiguous ``[1, S, C]`` tensors.  The HWX manifest
describes any physical allocation padding; ``ane-run.py`` compares only the
logical expected-output byte span supplied here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h16g_gelu import h16g_gelu  # noqa: E402

GEOMETRIES = {
    "bert": (128, 768, 3072),
    "wtiny": (1500, 384, 1536),
    "wbase": (1500, 512, 2048),
}


def cases() -> list[tuple[str, int, int, int, bool, str]]:
    result = []
    for tag, (sequence, width, hidden) in GEOMETRIES.items():
        for reduction, columns in ((width, width), (width, hidden),
                                   (hidden, width)):
            stem = f"{tag}_{reduction}x{columns}"
            result.append((f"lin_{stem}", sequence, reduction, columns,
                           False, "none"))
            result.append((f"linb_{stem}", sequence, reduction, columns,
                           True, "none"))
        result.append((f"lingelu_{tag}", sequence, width, hidden,
                       True, "gelu"))
        # d_head is 64 for all three requested transformer families.
        result.append((f"linscale_{tag}", sequence, width, width,
                       True, "scale-0.125"))
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(case, root: Path) -> dict:
    name, sequence, reduction, columns, with_bias, post_op = case
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    weights = (rng.standard_normal((columns, reduction)) /
               math.sqrt(reduction)).astype(np.float16)
    bias = ((rng.standard_normal(columns) * 0.05).astype(np.float16)
            if with_bias else None)
    source = (rng.standard_normal((sequence, reduction)) * 0.25).astype(np.float16)
    expected = np.empty((sequence, columns), dtype=np.float16)
    weight32 = weights.astype(np.float32)
    bias32 = bias.astype(np.float32) if bias is not None else None
    for start in range(0, sequence, 64):
        block = source[start:start + 64].astype(np.float32) @ weight32.T
        if bias32 is not None:
            block += bias32
        if post_op == "gelu":
            block = h16g_gelu(block)
        elif post_op == "scale-0.125":
            block *= np.float32(0.125)
        expected[start:start + block.shape[0]] = block.astype(np.float16)

    destination = root / name
    destination.mkdir(parents=True, exist_ok=False)
    arrays = {"input.bin": source, "weights.bin": weights,
              "expected.bin": expected}
    if bias is not None:
        arrays["bias.bin"] = bias
    for filename, array in arrays.items():
        (destination / filename).write_bytes(array.tobytes())
    return {
        "name": name,
        "shape": {"sequence": sequence, "inputChannels": reduction,
                  "outputChannels": columns},
        "bias": with_bias,
        "postOp": post_op,
        "files": {filename: {"bytes": (destination / filename).stat().st_size,
                             "sha256": digest(destination / filename)}
                  for filename in arrays},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", help="regular expression selecting case names")
    args = parser.parse_args()
    selected = [case for case in cases()
                if not args.case or re.search(args.case, case[0])]
    if not selected:
        parser.error("case selector matched no references")
    args.output.mkdir(parents=True, exist_ok=True)
    records = [generate(case, args.output) for case in selected]
    manifest = {"format": "h16g-linear-fp16-reference-v1",
                "accumulation": "float32 then fp16 store",
                "cases": records}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"generated {len(records)} reference cases in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
