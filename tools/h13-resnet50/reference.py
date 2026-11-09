#!/usr/bin/env python3
"""Float64 ResNet-50 reference for the supplied coremltools MIL model."""
from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # Padding rules stay host-testable without NumPy.
    np = None


CONST = re.compile(
    r'tensor<fp16, \[([^]]+)\]> (conv_\d+_(?:weight|bias)_0|linear_0_(?:weight|bias)_0) = const\(\).*?'
    r'offset = uint64\((\d+)\)', re.S)


def tf_same_pad(extent: int, kernel: int, stride: int) -> tuple[int, int]:
    """TensorFlow SAME: the odd extra cell is placed after the surface."""
    output = (extent + stride - 1) // stride
    total = max((output - 1) * stride + kernel - extent, 0)
    return total // 2, total - total // 2


def conv2d_python(x, weight, bias, stride=1):
    """Small dependency-free float64 path used by the mandatory host test."""
    channels, height, width = len(x), len(x[0]), len(x[0][0])
    outputs, kernel = len(weight), len(weight[0][0])
    top, bottom = tf_same_pad(height, kernel, stride)
    left, right = tf_same_pad(width, kernel, stride)
    out_height = (height + stride - 1) // stride
    out_width = (width + stride - 1) // stride
    result = [[[float(bias[o]) for _ in range(out_width)]
               for _ in range(out_height)] for o in range(outputs)]
    for o in range(outputs):
        for oy in range(out_height):
            for ox in range(out_width):
                total = float(bias[o])
                for c in range(channels):
                    for ky in range(kernel):
                        iy = oy * stride + ky - top
                        if not 0 <= iy < height:
                            continue
                        for kx in range(kernel):
                            ix = ox * stride + kx - left
                            if 0 <= ix < width:
                                total += (float(x[c][iy][ix]) *
                                          float(weight[o][c][ky][kx]))
                result[o][oy][ox] = total
    return result


def reduce_mean_python(x):
    count = len(x[0]) * len(x[0][0])
    return [sum(sum(row) for row in channel) / count for channel in x]


def constants(model: Path) -> dict[str, np.ndarray]:
    if np is None:
        raise RuntimeError('NumPy is required to generate ResNet references')
    text = (model / 'model.mil').read_text()
    blob = (model / 'weights' / 'weight.bin').read_bytes()
    result = {}
    for shape_text, name, offset_text in CONST.findall(text):
        shape = tuple(int(value.strip()) for value in shape_text.split(','))
        count = int(np.prod(shape))
        offset = int(offset_text)
        payload = struct.unpack_from('<Q', blob, offset + 16)[0]
        result[name] = np.frombuffer(blob, dtype='<f2', count=count,
                                     offset=payload).astype(np.float64).reshape(shape)
    if len(result) != 108:
        raise ValueError(f'expected 108 ResNet constants, found {len(result)}')
    return result


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
           stride: int) -> np.ndarray:
    kernel = weight.shape[-1]
    top, bottom = tf_same_pad(x.shape[1], kernel, stride)
    left, right = tf_same_pad(x.shape[2], kernel, stride)
    padded = np.pad(x, ((0, 0), (top, bottom), (left, right)))
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel, kernel), axis=(1, 2))[:, ::stride, ::stride]
    return np.tensordot(weight, windows, axes=((1, 2, 3), (0, 3, 4))) + \
        bias[:, None, None]


def maxpool(x: np.ndarray, kernel: int = 3, stride: int = 2) -> np.ndarray:
    top, bottom = tf_same_pad(x.shape[1], kernel, stride)
    left, right = tf_same_pad(x.shape[2], kernel, stride)
    padded = np.pad(x, ((0, 0), (top, bottom), (left, right)),
                    constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel, kernel), axis=(1, 2))[:, ::stride, ::stride]
    return windows.max(axis=(-1, -2))


def evaluate_boundaries(model: Path, x: np.ndarray) -> dict[str, np.ndarray]:
    c = constants(model)
    result = {}

    def conv(index: int, value: np.ndarray, stride: int = 1) -> np.ndarray:
        return conv2d(value, c[f'conv_{index}_weight_0'],
                      c[f'conv_{index}_bias_0'], stride)

    y = np.maximum(conv(0, x.astype(np.float64), 2), 0)
    y = maxpool(y)
    result['pool'] = y.copy()
    stages = [
        ([1, 2, 3], 4, [[5, 6, 7], [8, 9, 10]]),
        ([11, 12, 13], 14, [[15, 16, 17], [18, 19, 20], [21, 22, 23]]),
        ([24, 25, 26], 27,
         [[28,29,30],[31,32,33],[34,35,36],[37,38,39],[40,41,42]]),
        ([43, 44, 45], 46, [[47,48,49],[50,51,52]]),
    ]
    for stage_index, (first, projection, rest) in enumerate(stages):
        stride = 1 if stage_index == 0 else 2
        shortcut = conv(projection, y, stride)
        z = np.maximum(conv(first[0], y), 0)
        z = np.maximum(conv(first[1], z, stride), 0)
        z = conv(first[2], z)
        y = np.maximum(z + shortcut, 0)
        result[f'block{len(result):02d}'] = y.copy()
        for block in rest:
            shortcut = y
            z = np.maximum(conv(block[0], y), 0)
            z = np.maximum(conv(block[1], z), 0)
            z = conv(block[2], z)
            y = np.maximum(z + shortcut, 0)
            block = sum(key.startswith('block') for key in result) + 1
            result[f'block{block:02d}'] = y.copy()
    mean = y.mean(axis=(1, 2), dtype=np.float64)
    logits = c['linear_0_weight_0'] @ mean + c['linear_0_bias_0']
    result['mean'] = mean
    result['logits'] = logits
    return result


def evaluate(model: Path, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compatibility view used by the host reference test."""
    result = evaluate_boundaries(model, x)
    prefixes = [result['pool']] + [result[f'block{index:02d}']
                                   for index in range(1, 17)]
    return result['logits'], prefixes


def main() -> None:
    if np is None:
        raise SystemExit('NumPy is required to generate ResNet references')
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=5975)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    sample = rng.uniform(-1, 1, (3, 224, 224)).astype(np.float16)
    references = evaluate_boundaries(args.model, sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **references)
    np.save(args.output.with_name('input.npy'), sample)


if __name__ == '__main__':
    main()
