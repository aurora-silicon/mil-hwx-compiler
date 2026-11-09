#!/usr/bin/env python3
"""Numerical reference for the decoded H16G GELU lookup table."""
from __future__ import annotations

import numpy as np


# The 33 fp16 samples in H16GLUTEncoderData.inc cover [-4, 4] at 0.25
# intervals. H16G linearly interpolates between them, clamps the negative
# tail to zero, and passes the positive tail through. This is the hardware
# implementation selected by MIL's GELU EXACT form; it is intentionally not
# math.erf evaluated at full precision.
GELU_TABLE = np.array([
    0.0, -0.0003991127014160156, -0.00075531005859375,
    -0.0017919540405273438, -0.003875732421875, -0.00794219970703125,
    -0.01512908935546875, -0.02703857421875, -0.044891357421875,
    -0.06964111328125, -0.10003662109375, -0.1324462890625,
    -0.159912109375, -0.17236328125, -0.15673828125,
    -0.10638427734375, -0.0007538795471191406, 0.1435546875,
    0.34326171875, 0.57763671875, 0.83984375, 1.1171875,
    1.400390625, 1.6806640625, 1.955078125, 2.22265625, 2.484375,
    2.7421875, 2.99609375, 3.248046875, 3.5, 3.75, 4.0,
], dtype=np.float64)
GELU_GRID = np.linspace(-4.0, 4.0, GELU_TABLE.size, dtype=np.float64)


def h16g_gelu(values: np.ndarray) -> np.ndarray:
    """Apply the decoded piecewise-linear H16G GELU in the input dtype."""
    source = np.asarray(values)
    wide = source.astype(np.float64)
    result = np.interp(wide, GELU_GRID, GELU_TABLE)
    result = np.where(wide <= -4.0, 0.0, result)
    result = np.where(wide >= 4.0, wide, result)
    return result.astype(source.dtype)
