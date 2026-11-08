#!/usr/bin/env python3
"""Generate deterministic fp16 references for the Whisper convolution stem.

The reference follows Core ML's rank-3 ``[1,C,T]`` spelling: kernel size 3,
SAME padding, stride 1 followed by stride 2, a bias and exact GELU after each
convolution, then ``[1,C,1500] -> [1,1500,C]`` and a constant positional add.
All multiply-accumulates and nonlinearities are evaluated in float64 and each
hardware-visible edge is rounded to fp16.

The output is deliberately generated, not checked in.  It contains one NPZ
per model plus row-padded external-surface files suitable for the Linux ANE
runner.  Numpy is the only dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np


MODELS = {"tiny": 384, "base": 512}


def tensor_type(shape) -> str:
    return "tensor<fp16, [" + ", ".join(map(str, shape)) + "]>"


def blob_write(path: Path, arrays: list[np.ndarray]) -> list[int]:
    """Write source-owned fp16 arrays in the repository's BLOBFILE grammar."""
    output = bytearray(struct.pack("<II", 1, 2) + bytes(56))
    offsets = []
    for array in arrays:
        payload = np.ascontiguousarray(array, dtype=np.float16).tobytes()
        metadata = len(output)
        output += struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload),
                              metadata + 64) + bytes(40) + payload
        output += bytes((-len(output)) % 64)
        offsets.append(metadata)
    path.write_bytes(output)
    return offsets


def conv_mil(cin: int, cout: int, length: int, stride: int,
             weight_offset: int, bias_offset: int, gelu: bool) -> str:
    output_length = (length + stride - 1) // stride
    x_type = tensor_type((1, cin, length))
    weight_type = tensor_type((cout, cin, 3))
    bias_type = tensor_type((cout,))
    y_type = tensor_type((1, cout, output_length))
    result = "y" if gelu else "c"
    activation = (f'    {y_type} y = gelu(x = c, mode = string("EXACT"))'
                  '[name = string("gelu")];\n' if gelu else "")
    blob = 'BLOBFILE(path = string("@model_path/parameters.bin"), offset = uint64({}))'
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x_type} x) {{
    string pt = const()[name = string("pt"), val = string("same")];
    tensor<int32, [1]> st = const()[name = string("st"), val = tensor<int32, [1]>([{stride}])];
    tensor<int32, [2]> pd = const()[name = string("pd"), val = tensor<int32, [2]>([0, 0])];
    tensor<int32, [1]> dl = const()[name = string("dl"), val = tensor<int32, [1]>([1])];
    int32 gp = const()[name = string("gp"), val = int32(1)];
    {weight_type} w = const()[name = string("w"), val = {weight_type}({blob.format(weight_offset)})];
    {bias_type} b = const()[name = string("b"), val = {bias_type}({blob.format(bias_offset)})];
    {y_type} c = conv(bias = b, dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string("conv")];
{activation}  }} -> ({result});
}}
'''


def transpose_mil(channels: int) -> str:
    source = tensor_type((1, channels, 1500))
    result = tensor_type((1, 1500, channels))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({source} x) {{
    tensor<int32, [3]> p = const()[name = string("p"), val = tensor<int32, [3]>([0, 2, 1])];
    {result} y = transpose(perm = p, x = x)[name = string("transpose")];
  }} -> (y);
}}
'''


def position_mil(channels: int, position_offset: int) -> str:
    value_type = tensor_type((1, 1500, channels))
    blob = ('BLOBFILE(path = string("@model_path/parameters.bin"), '
            f'offset = uint64({position_offset}))')
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} x) {{
    {value_type} p = const()[name = string("p"), val = {value_type}({blob})];
    {value_type} y = add(x = x, y = p)[name = string("position")];
  }} -> (y);
}}
'''


def conv1d_same(x16: np.ndarray, weight16: np.ndarray,
                bias16: np.ndarray, stride: int) -> np.ndarray:
    """Float64 NCT convolution with Core ML SAME_UPPER padding."""
    x = x16.astype(np.float64)
    weight = weight16.astype(np.float64)
    bias = bias16.astype(np.float64)
    length = x.shape[1]
    output_length = (length + stride - 1) // stride
    total_pad = max((output_length - 1) * stride + 3 - length, 0)
    low = total_pad // 2
    padded = np.pad(x, ((0, 0), (low, total_pad - low)))
    output = np.zeros((weight.shape[0], output_length), dtype=np.float64)
    for tap in range(3):
        patch = padded[:, tap:tap + stride * output_length:stride]
        output += weight[:, :, tap] @ patch
    output += bias[:, None]
    return output.astype(np.float16)


def exact_gelu(x16: np.ndarray) -> np.ndarray:
    """Core ML's default EXACT GELU, rounded once at the output edge."""
    x = x16.astype(np.float64)
    erf = np.frompyfunc(lambda v: math.erf(v / math.sqrt(2.0)), 1, 1)
    return (0.5 * x * (1.0 + erf(x).astype(np.float64))).astype(np.float16)


def surface(array: np.ndarray) -> bytes:
    """Pack a CxT fp16 tensor with each external row aligned to 64 bytes."""
    if array.ndim != 2 or array.dtype != np.float16:
        raise ValueError("surface expects a CxT fp16 array")
    channels, length = array.shape
    row_halfwords = ((2 * length + 63) // 64 * 64) // 2
    packed = np.zeros((channels, row_halfwords), dtype=np.float16)
    packed[:, :length] = array
    return packed.tobytes()


def parameters(rng: np.random.Generator, channels: int,
               output_channels: int) -> tuple[np.ndarray, np.ndarray]:
    fan = channels * 3
    weight = (rng.standard_normal((output_channels, channels, 3)) /
              math.sqrt(fan)).astype(np.float16)
    bias = (rng.standard_normal(output_channels) * 0.05).astype(np.float16)
    return weight, bias


def generate_model(output: Path, name: str, channels: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((80, 3000)) * 0.2).astype(np.float16)
    w1, b1 = parameters(rng, 80, channels)
    w2, b2 = parameters(rng, channels, channels)
    position = (rng.standard_normal((1500, channels)) * 0.02).astype(np.float16)

    conv1 = conv1d_same(x, w1, b1, 1)
    gelu1 = exact_gelu(conv1)
    conv2 = conv1d_same(gelu1, w2, b2, 2)
    gelu2 = exact_gelu(conv2)
    tokens = np.ascontiguousarray(gelu2.T)
    positional = (tokens.astype(np.float64) +
                  position.astype(np.float64)).astype(np.float16)

    case = output / name
    case.mkdir(parents=True, exist_ok=False)
    offsets = blob_write(case / "parameters.bin", [w1, b1, w2, b2,
                                                     position[None, :, :]])
    for filename, text in (
            ("stem1.mil", conv_mil(80, channels, 3000, 1,
                                    offsets[0], offsets[1], False)),
            ("stem1-gelu.mil", conv_mil(80, channels, 3000, 1,
                                         offsets[0], offsets[1], True)),
            ("stem2.mil", conv_mil(channels, channels, 3000, 2,
                                    offsets[2], offsets[3], False)),
            ("stem2-gelu.mil", conv_mil(channels, channels, 3000, 2,
                                         offsets[2], offsets[3], True)),
            ("transpose.mil", transpose_mil(channels)),
            ("position.mil", position_mil(channels, offsets[4]))):
        (case / filename).write_text(text)
    np.savez(case / "reference.npz", input=x, weight1=w1, bias1=b1,
             conv1=conv1, gelu1=gelu1, weight2=w2, bias2=b2, conv2=conv2,
             gelu2=gelu2, position=position, tokens=tokens,
             positional=positional)
    for filename, array in (("input.surface", x),
                            ("conv1.surface", conv1),
                            ("gelu1.surface", gelu1),
                            ("conv2.surface", conv2),
                            ("gelu2.surface", gelu2)):
        (case / filename).write_bytes(surface(array))
    (case / "tokens.fp16").write_bytes(tokens.tobytes())
    (case / "positional.fp16").write_bytes(positional.tobytes())

    metadata = {
        "model": f"whisper-{name}",
        "seed": seed,
        "dtype": "fp16 at every hardware-visible edge; float64 reference math",
        "input": [1, 80, 3000],
        "conv1": {"weight": [channels, 80, 3], "stride": 1,
                  "output": [1, channels, 3000], "bias": True,
                  "postOperation": "gelu_exact"},
        "conv2": {"weight": [channels, channels, 3], "stride": 2,
                  "output": [1, channels, 1500], "bias": True,
                  "postOperation": "gelu_exact"},
        "transpose": {"permutation": [0, 2, 1],
                      "output": [1, 1500, channels]},
        "positionalAdd": {"constant": [1, 1500, channels],
                          "output": [1, 1500, channels]},
        "externalSurface": "NCHW [1,C,1,T], fp16 rows aligned to 64 bytes",
        "milPrograms": ["stem1.mil", "stem1-gelu.mil", "stem2.mil",
                        "stem2-gelu.mil", "transpose.mil", "position.mil"],
        "parameterOffsets": offsets,
    }
    (case / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=["tiny", "base", "all"],
                        default="all")
    parser.add_argument("--seed", type=int, default=5728)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = MODELS if args.model == "all" else {
        args.model: MODELS[args.model]}
    index = [generate_model(args.output, name, channels,
                            args.seed + index)
             for index, (name, channels) in enumerate(selected.items())]
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"generated {len(index)} Whisper stem reference set(s) in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
