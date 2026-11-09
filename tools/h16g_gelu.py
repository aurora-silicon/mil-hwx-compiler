#!/usr/bin/env python3
"""Numerical reference for the rule-built H16G GELU lookup table."""
from __future__ import annotations

import math
import numpy as np


# The encoder evaluates the public exact-GELU equation at 33 evenly spaced
# points, then rounds each sample to fp16. Keep this reference constructive too.
GELU_GRID = np.linspace(-4.0, 4.0, 33, dtype=np.float64)
GELU_TABLE = np.asarray([
    0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))
    for value in GELU_GRID
], dtype=np.float16).astype(np.float64)


def h16g_gelu(values: np.ndarray) -> np.ndarray:
    """Apply the decoded piecewise-linear H16G GELU in the input dtype."""
    source = np.asarray(values)
    wide = source.astype(np.float64)
    result = np.interp(wide, GELU_GRID, GELU_TABLE)
    result = np.where(wide <= -4.0, 0.0, result)
    result = np.where(wide >= 4.0, wide, result)
    return result.astype(source.dtype)
