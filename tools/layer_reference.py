"""Float64 CPU references for the native H13/H14 layer envelope.

All arithmetic is float64; surface writes use ties-away fp16 rounding.
These are numerical references, not a simulation of task-descriptor fields.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "h13-conv-bench"))
from surf import f16_ties_away


def nearest2x(x):
    return np.repeat(np.repeat(np.asarray(x, dtype=np.float64), 2, axis=-2), 2, axis=-1)


def bilinear_half2x(x):
    x = np.asarray(x, dtype=np.float64)
    h, w = x.shape[-2:]
    yy = np.clip((np.arange(2 * h, dtype=np.float64) + 0.5) / 2 - 0.5, 0, h - 1)
    xx = np.clip((np.arange(2 * w, dtype=np.float64) + 0.5) / 2 - 0.5, 0, w - 1)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    dy, dx = yy - y0, xx - x0
    horizontal = x[..., x0] * (1 - dx) + x[..., x1] * dx
    return horizontal[..., y0, :] * (1 - dy[:, None]) + horizontal[..., y1, :] * dy[:, None]


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    out[~positive] = exp / (1 + exp)
    return out


def hardswish(x, staged=True):
    x = np.asarray(x, dtype=np.float64)
    a = x + 3
    if staged:
        a = f16_ties_away(a).astype(np.float64)
    h = np.clip(a, 0, 6) / 6
    if staged:
        h = f16_ties_away(h).astype(np.float64)
    return x * h


def concat_channels(x, z):
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    if x.ndim != 3 or z.ndim != 3 or x.shape[1:] != z.shape[1:]:
        raise ValueError("concat needs CHW inputs with equal spatial dimensions")
    return np.concatenate((x, z), axis=0)


def conv(x, weights, bias=None, *, groups=1, stride=1, pad=0):
    x, weights = np.asarray(x, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    cin, height, width = x.shape
    cout, cig, kh, kw = weights.shape
    pt,pb,pl,pr = (pad,pad,pad,pad) if isinstance(pad,int) else tuple(pad)
    if groups < 1 or cin % groups or cout % groups or cig != cin // groups or stride < 1 or min(pt,pb,pl,pr)<0:
        raise ValueError("invalid convolution geometry")
    ho, wo = (height + pt + pb - kh) // stride + 1, (width + pl + pr - kw) // stride + 1
    if ho < 1 or wo < 1:
        raise ValueError("kernel does not fit")
    source = np.pad(x, ((0,0),(pt,pb),(pl,pr)))
    out = np.zeros((cout, ho, wo), dtype=np.float64)
    cog = cout // groups
    for group in range(groups):
        a, b = group * cig, group * cog
        for y in range(kh):
            for z in range(kw):
                patch = source[a:a + cig, y:y + stride * ho:stride, z:z + stride * wo:stride]
                out[b:b + cog] += (weights[b:b + cog, :, y, z] @ patch.reshape(cig, -1)).reshape(cog, ho, wo)
    if bias is not None:
        out += np.asarray(bias, dtype=np.float64)[:, None, None]
    return out
