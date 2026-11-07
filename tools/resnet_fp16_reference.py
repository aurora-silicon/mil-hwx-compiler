#!/usr/bin/env python3
"""CPU fp16-chain reference for an exported ResNet MIL graph.

Convolution/linear/reduction accumulate in float64; EVERY MIL operation rounds
to its declared dtype. SAME padding is asymmetric when necessary. This models
surface boundaries, not the ANE's narrower accumulator. No hardware access.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def same_padding(size, kernel, stride):
    output = (size + stride - 1) // stride
    total = max((output - 1) * stride + kernel - size, 0)
    return total // 2, total - total // 2


def windows(x, kernel, strides, pad_type, pad=None, fill=0):
    if len(kernel) != 2 or len(strides) != 2:
        raise ValueError('reference supports 2D spatial operations only')
    if pad_type == 'same':
        padding = [same_padding(n, k, s) for n, k, s in zip(x.shape[-2:], kernel, strides)]
    elif pad_type in ('valid', 'custom'):
        p = [0, 0, 0, 0] if pad is None else list(pad)
        padding = [(p[0], p[1]), (p[2], p[3])]
    else:
        raise ValueError(f'unsupported pad_type {pad_type}')
    padded = np.pad(x, [(0, 0), (0, 0), *padding], constant_values=fill)
    view = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=(-2, -1))
    return view[:, :, ::strides[0], ::strides[1]]


def convolution(x, weight, bias=None, strides=(1, 1), pad_type='same',
                pad=None, dilations=(1, 1), groups=1):
    if list(dilations) != [1, 1] or groups != 1:
        raise ValueError('ResNet reference requires dense unit-dilation convolution')
    x, weight = x.astype(np.float64), weight.astype(np.float64)
    if x.ndim != 4 or weight.ndim != 4 or x.shape[1] != weight.shape[1]:
        raise ValueError('invalid NCHW/OIHW convolution shapes')
    patches = windows(x, weight.shape[-2:], strides, pad_type, pad)
    n, _, ho, wo, _, _ = patches.shape
    matrix = patches.transpose(1, 4, 5, 0, 2, 3).reshape(-1, n * ho * wo)
    y = (weight.reshape(weight.shape[0], -1) @ matrix)
    y = y.reshape(weight.shape[0], n, ho, wo).transpose(1, 0, 2, 3)
    if bias is not None:
        y += np.asarray(bias, dtype=np.float64)[None, :, None, None]
    return y


def pool(x, kernel_sizes, strides, pad_type='valid', pad=None,
         ceil_mode=False, dilations=(1, 1)):
    if ceil_mode or list(dilations) != [1, 1]:
        raise ValueError('unsupported pooling variant')
    return windows(x.astype(np.float64), kernel_sizes, strides, pad_type,
                   pad, fill=-np.inf).max(axis=(-2, -1))


def surface(array):
    """Return the documented external NCHW convolution surface, align64 rows."""
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError('surface requires a batch-one NCHW tensor')
    _, c, h, w = array.shape
    row_elements = (2 * w + 63) // 64 * 32
    result = np.zeros((c, h, row_elements), dtype='<f2')
    result[:, :, :w] = array[0]
    return result.tobytes()


class Reference:
    def __init__(self, directory):
        directory = Path(directory)
        self.graph = json.loads((directory / 'graph.json').read_text())
        with np.load(directory / 'constants.npz', allow_pickle=False) as arrays:
            self.constants = {name: arrays[value['array']].copy() if 'array' in value
                              else value['value']
                              for name, value in self.graph['constants'].items()}

    def evaluate(self, inputs, trace=None):
        if set(inputs) != set(self.graph['inputs']):
            raise ValueError('reference input names differ from MIL boundary')
        values = dict(self.constants)
        for name, array in inputs.items():
            typ = self.graph['variables'][name]
            if list(array.shape) != typ['shape'] or array.dtype != np.float16:
                raise ValueError('reference needs the declared fp16 input shape')
            values[name] = array
        # Keep residual inputs until their last consumer, release all others.
        remaining = Counter(v for op in self.graph['operations'] for v in op['inputs'].values())
        remaining.update(self.graph['outputs'])
        for op in self.graph['operations']:
            a = {key: values[name] for key, name in op['inputs'].items()}
            kind = op['op']
            if kind == 'conv':
                y = convolution(**a)
            elif kind == 'relu':
                y = np.maximum(a['x'], 0)
            elif kind == 'add':
                y = a['x'].astype(np.float64) + a['y'].astype(np.float64)
            elif kind == 'max_pool':
                y = pool(**a)
            elif kind == 'reduce_mean':
                y = np.mean(a['x'].astype(np.float64), axis=tuple(int(x) for x in a['axes']),
                            keepdims=bool(a['keep_dims']))
            elif kind == 'linear':
                y = a['x'].astype(np.float64) @ a['weight'].astype(np.float64).T
                if 'bias' in a:
                    y += np.asarray(a['bias'], dtype=np.float64)
            elif kind == 'reshape':
                y = a['x'].reshape(tuple(int(x) for x in a['shape']))
            elif kind == 'cast':
                y = a['x']
            else:
                raise ValueError(f"reference has no implementation for {kind} ({op['name']})")
            if op['type']['dtype'] != 'fp16':
                raise ValueError('ResNet reference expects fp16 operation results')
            y = np.asarray(y, dtype=np.float16)
            if list(y.shape) != op['type']['shape'] or not np.isfinite(y).all():
                raise ValueError(f"invalid output shape/value at {op['name']}")
            values[op['result']] = y
            if trace is not None:
                trace.append({'op': op['op'], 'name': op['name'], 'shape': list(y.shape),
                              'sha256': hashlib.sha256(y.astype('<f2').tobytes()).hexdigest(),
                              'min': float(y.min()), 'max': float(y.max())})
            for name in op['inputs'].values():
                remaining[name] -= 1
                if not remaining[name] and name not in self.constants:
                    del values[name]
        return {name: values[name] for name in self.graph['outputs']}


def make_references(model, output, count=4, seed=1234):
    model, output = Path(model), Path(output)
    ref = Reference(model)
    if len(ref.graph['inputs']) != 1 or len(ref.graph['outputs']) != 1:
        raise ValueError('ResNet package needs one input and one output')
    name = ref.graph['inputs'][0]
    shape = ref.graph['variables'][name]['shape']
    if shape != [1, 3, 224, 224]:
        raise ValueError('expected ResNet 224 input')
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    tops, traces = [], []
    for index in range(count):
        x = rng.uniform(-1, 1, shape).astype(np.float16)
        trace = []
        y = next(iter(ref.evaluate({name: x}, trace).values()))
        if y.shape != (1, 1000):
            raise ValueError(f'expected 1000 logits, got {y.shape}')
        (output / f'input_{index:02d}.bin').write_bytes(surface(x))
        # Dense logits have no hardware layout claim until the head is encoded.
        (output / f'logits_{index:02d}.fp16').write_bytes(y.astype('<f2').tobytes())
        tops.append(int(y.argmax()))
        traces.append(trace)
        if index == 0:
            repeat = next(iter(ref.evaluate({name: x}).values()))
            if not np.array_equal(y.view(np.uint16), repeat.view(np.uint16)):
                raise ValueError('reference is not deterministic')
        print(f'image {index}: top1={tops[-1]}, {len(trace)} fp16 boundaries', flush=True)
    (output / 'top1.json').write_text(json.dumps({'top1': tops, 'seed': seed,
        'input': 'uniform [-1,1], rounded fp16', 'classes': 'random-weight model; no label semantics',
        'arithmetic': 'float64 accumulation; fp16 at every MIL operation'}, indent=2) + '\n')
    (output / 'trace.json').write_text(json.dumps(traces, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--images', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if args.images < 1:
        parser.error('--images must be positive')
    make_references(args.model, args.out, args.images, args.seed)


if __name__ == '__main__':
    main()
