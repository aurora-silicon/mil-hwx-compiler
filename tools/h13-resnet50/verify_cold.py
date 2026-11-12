#!/usr/bin/env python3
"""Verify three cold sub-rung outputs bit-for-bit and against the CPU reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('kit', type=Path)
    parser.add_argument('rung')
    parser.add_argument('outputs', type=Path, nargs=3)
    args = parser.parse_args()
    manifest = json.loads((args.kit / 'kit.json').read_text())
    entry = next(item for item in manifest['subRungs']
                 if item['name'] == args.rung)
    runs = [np.load(path, allow_pickle=False).astype(np.float16, copy=False)
            for path in args.outputs]
    repeatable = all(run.shape == runs[0].shape and
                     np.array_equal(run.view(np.uint16),
                                    runs[0].view(np.uint16))
                     for run in runs[1:])
    with np.load(args.kit / entry['reference'], allow_pickle=False) as archive:
        reference = archive[entry['reference_key']].astype(np.float64)
    output = runs[0].astype(np.float64).reshape(reference.shape)
    finite = bool(np.isfinite(output).all())
    rel_l2 = (float(np.linalg.norm(output - reference) /
                    max(np.linalg.norm(reference), 1e-30))
              if finite else None)
    passed = repeatable and finite and rel_l2 <= entry['tolerance_relL2']
    print(json.dumps({'rung': args.rung, 'coldRuns': 3,
                      'bitIdentical': repeatable, 'finite': finite,
                      'relL2': rel_l2, 'pass': passed}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
