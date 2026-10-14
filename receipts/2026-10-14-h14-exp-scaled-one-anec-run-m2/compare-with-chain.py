#!/usr/bin/env python3
"""Read-only comparison of the one-ANEC output with the five-call chain's final output and the failed exp program's output. It judges nothing: compare-expchain.py holds the verdict.

Usage: compare-with-chain.py [ARCHIVE_DIR]"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
CHAIN = HERE.parent / "2026-10-14-h14-exp-chain-d4-run-m2/device-output"
FAILED = HERE.parent / "2026-10-13-h14-unary-m2/device-output"


def lanes(path, count):
    return np.fromfile(path, dtype=np.float16)[::32][:count]


for c in (64, 512):
    one = HERE / f"device-output/exp_c{c}.s1_y.cal.y.f16"
    chain = CHAIN / f"exp_c{c}.s5_y.cal.y.f16"
    failed = FAILED / f"unary_exp_c{c}.cal.y.f16"
    same_bytes = one.read_bytes() == chain.read_bytes()
    y, s5, e = lanes(one, c), lanes(chain, c), lanes(failed, c)
    print(f"c{c}: file bytes equal to the five-call chain's final output: {same_bytes}; lanes differing from it: {int((y != s5).sum())} of {c}; "
          f"lanes differing from the failed exp program's output: {int((y != e).sum())} of {c}")
