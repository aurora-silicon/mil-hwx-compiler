#!/usr/bin/env python3
"""Mint and decode H13-H18 ANE compiler oracles without retaining HWX bytes."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

from h13_td import HEADER_WORDS, decode_task, split_h13_tasks, split_h14_tasks

MAGIC = 0xBEEFFACE
SUBTYPES = {"h13": 4, "h14": 5, "h15": 6, "h16": 7, "h17": 9, "h18": 10}
DEFAULT_TOOL = "/tmp/h13-oracle/bin/ane-compile-hwx"
COMPILE_TIMEOUT_SECONDS = 900


def decode_constant_half_words(data: bytes) -> list[dict[str, Any]] | None:
    if len(data) > 256 or len(data) % 2:
        return None
    words = []
    for index, (value,) in enumerate(struct.iter_unpack("<e", data)):
        if value == 0.0:
            continue
        if value != value:
            decoded: float | str = "nan"
        elif value == float("inf"):
            decoded = "+inf"
        elif value == float("-inf"):
            decoded = "-inf"
        else:
            decoded = value
        words.append({"index": index, "value": decoded})
    return words


def shape_text(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(map(str, shape)) + "]"


def tensor_type(shape: tuple[int, ...]) -> str:
    return f"tensor<fp16, {shape_text(shape)}>"


def program(arguments: str, body: list[str], result: str) -> str:
    lines = [
        "program(1.3)",
        "[buildInfo = dict<string, string>({})]",
        "{",
        f"  func main<ios18>({arguments}) {{",
        *(f"    {line}" for line in body),
        f"  }} -> ({result});",
        "}",
    ]
    return "\n".join(lines) + "\n"


def blob(payload: bytes) -> bytes:
    result = bytearray(128 + len(payload))
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<IQQ", result, 64, 0xDEADBEEF, len(payload), 128)
    result[128:] = payload
    return bytes(result)


def half_payload(elements: int) -> bytes:
    return b"\x00\x38" * elements


def case(name: str, family: str, parameters: dict[str, Any], mil: str,
         weights: bytes | int | None = None,
         weights_description: dict[str, Any] | None = None) -> dict[str, Any]:
    """A campaign entry. `weights` is raw bytes or an element count for a
    uniform fp16 payload that `run_case` materializes at compile time."""
    return {
        "name": name,
        "family": family,
        "parameters": parameters,
        "mil": mil,
        "weights": weights,
        "weights_description": weights_description or {"storage": "none"},
    }


def binary_runtime(operation: str, shape: tuple[int, ...]) -> dict[str, Any]:
    kind = tensor_type(shape)
    name = f"binary_{operation}_{'x'.join(map(str, shape))}"
    mil = program(
        f"{kind} x, {kind} z",
        [f'{kind} y = {operation}(x = x, y = z)[name = string("y")];'], "y")
    return case(name, "binary_runtime", {"operation": operation, "shape": shape}, mil)


def binary_constant(operation: str, channels: int, storage: str) -> dict[str, Any]:
    shape = (1, channels, 1, 1)
    kind = tensor_type(shape)
    name = f"binary_{operation}_c{channels}_constant_{storage}"
    if storage == "scalar":
        constant = 'fp16 z = const()[name = string("z"), val = fp16(0x1p-1)];'
        weights = None
        description = {"storage": "inline_scalar", "value": "fp16(0x1p-1)"}
    elif storage == "inline":
        values = ", ".join(["fp16(0x1p-1)"] * channels)
        constant = (
            f'{kind} z = const()[name = string("z"), '
            f'val = {kind}([{values}])];')
        weights = None
        description = {"storage": "inline_tensor", "shape": shape,
                       "value": "fp16(0x1p-1)"}
    else:
        constant, weights = blobfile("z", shape)
        description = {"storage": "BLOBFILE", "shape": shape,
                       "payload_bytes": weights * 2, "value": "fp16(0x1p-1)"}
    mil = program(
        f"{kind} x", [constant,
        f'{kind} y = {operation}(x = x, y = z)[name = string("y")];'], "y")
    return case(name, "binary_constant",
                {"operation": operation, "shape": shape, "constant": storage},
                mil, weights, description)


def unary(operation: str, channels: int) -> dict[str, Any]:
    shape = (1, channels, 1, 1)
    kind = tensor_type(shape)
    attributes = {
        "gelu": ', mode = string("EXACT")',
        "rsqrt": ", epsilon = fp32(0.000001)",
        "clip": ", alpha = fp32(-1.0), beta = fp32(1.0)",
        "leaky_relu": ", alpha = fp32(0.125)",
    }.get(operation, "")
    mil = program(
        f"{kind} x",
        [f'{kind} y = {operation}(x = x{attributes})[name = string("y")];'], "y")
    return case(f"unary_{operation}_c{channels}", "unary",
                {"operation": operation, "shape": shape}, mil)


def reduction(operation: str, axes: tuple[int, ...]) -> dict[str, Any]:
    shape = (1, 64, 8, 8)
    output = tuple(1 if index in axes else value for index, value in enumerate(shape))
    axis_values = ", ".join(map(str, axes))
    kind = tensor_type(shape)
    out_kind = tensor_type(output)
    axis_kind = f"tensor<int32, [{len(axes)}]>"
    mil = program(f"{kind} x", [
        f'{axis_kind} axes = const()[name = string("axes"), '
        f'val = {axis_kind}([{axis_values}])];',
        f'{out_kind} y = {operation}(x = x, axes = axes, '
        'keep_dims = bool(true))[name = string("y")];',
    ], "y")
    suffix = "_".join(map(str, axes))
    return case(f"{operation}_axes_{suffix}", "reduction",
                {"operation": operation, "shape": shape, "axes": axes}, mil)


def normalization(operation: str, shape: tuple[int, ...]) -> dict[str, Any]:
    kind = tensor_type(shape)
    body = []
    if operation == "softmax":
        body.extend([
            'int32 axis = const()[name = string("axis"), val = int32(-1)];',
            f'{kind} y = softmax(x = x, axis = axis)[name = string("y")];',
        ])
    else:
        axes = tuple(range(1, len(shape)))
        axis_kind = f"tensor<int32, [{len(axes)}]>"
        body.extend([
            f'{axis_kind} axes = const()[name = string("axes"), '
            f'val = {axis_kind}([{", ".join(map(str, axes))}])];',
            f'{kind} y = layer_norm(x = x, axes = axes, '
            'epsilon = fp32(0.00001))[name = string("y")];',
        ])
    name = f"{operation}_{'x'.join(map(str, shape))}"
    return case(name, "normalization", {"operation": operation, "shape": shape},
                program(f"{kind} x", body, "y"))


def blobfile(name: str, shape: tuple[int, ...]) -> tuple[str, int]:
    """A BLOBFILE constant declaration and the element count it needs."""
    kind = tensor_type(shape)
    return (
        f'{kind} {name} = const()[name = string("{name}"), val = {kind}'
        '(BLOBFILE(path = string("@model_path/weights.bin"), '
        'offset = uint64(64)))];'), math.prod(shape)


def weight_constant(name: str, shape: tuple[int, ...]) -> tuple[str, int, dict[str, Any]]:
    source, elements = blobfile(name, shape)
    return source, elements, {
        "storage": "BLOBFILE", "shape": shape, "payload_bytes": elements * 2,
        "value": "fp16(0x1p-1)",
    }


def convolution(kernel: int, inputs: int, outputs: int, spatial: int,
                bias: bool, chain_relu: bool = False) -> dict[str, Any]:
    x_shape = (1, inputs, spatial, spatial)
    y_shape = (1, outputs, spatial, spatial)
    w_shape = (outputs, inputs, kernel, kernel)
    x_kind, y_kind = tensor_type(x_shape), tensor_type(y_shape)
    weight, weights, description = weight_constant("w", w_shape)
    body = [
        'string pt = const()[name = string("pt"), val = string("same")];',
        'tensor<int32, [2]> st = const()[name = string("st"), '
        'val = tensor<int32, [2]>([1, 1])];',
        'tensor<int32, [4]> pd = const()[name = string("pd"), '
        'val = tensor<int32, [4]>([0, 0, 0, 0])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), '
        'val = tensor<int32, [2]>([1, 1])];',
        'int32 gp = const()[name = string("gp"), val = int32(1)];', weight,
    ]
    bias_arg = ""
    if bias:
        values = ", ".join(["fp16(0x0p+0)"] * outputs)
        bias_kind = tensor_type((outputs,))
        body.append(f'{bias_kind} b = const()[name = string("b"), '
                    f'val = {bias_kind}([{values}])];')
        bias_arg = ", bias = b"
    result = "c" if chain_relu else "y"
    body.append(
        f'{y_kind} {result} = conv(dilations = dl, groups = gp, pad = pd, '
        f'pad_type = pt, strides = st, weight = w, x = x{bias_arg})'
        f'[name = string("{result}")];')
    if chain_relu:
        body.append(f'{y_kind} y = relu(x = c)[name = string("y")];')
    name = f"conv_k{kernel}_c{inputs}_n{outputs}_s{spatial}_bias{int(bias)}"
    if chain_relu:
        name += "_relu"
    return case(name, "chain" if chain_relu else "convolution", {
        "kernel": kernel, "input_channels": inputs, "output_channels": outputs,
        "spatial": spatial, "bias": bias, "relu": chain_relu,
    }, program(f"{x_kind} x", body, "y"), weights, description)


def matmul(reduction: int, columns: int, rows: int, transpose: bool,
           chain_bias_relu: bool = False) -> dict[str, Any]:
    x_shape = (rows, reduction)
    w_shape = (columns, reduction) if transpose else (reduction, columns)
    y_shape = (rows, columns)
    x_kind, y_kind = tensor_type(x_shape), tensor_type(y_shape)
    weight, weights, description = weight_constant("w", w_shape)
    flag = "true" if transpose else "false"
    body = [
        'bool f = const()[name = string("f"), val = bool(false)];',
        f'bool ty = const()[name = string("ty"), val = bool({flag})];', weight,
        f'{y_kind} product = matmul(transpose_x = f, transpose_y = ty, '
        'x = x, y = w)[name = string("product")];',
    ]
    result = "product"
    if chain_bias_relu:
        bias_kind = tensor_type((columns,))
        values = ", ".join(["fp16(0x0p+0)"] * columns)
        body.extend([
            f'{bias_kind} b = const()[name = string("b"), '
            f'val = {bias_kind}([{values}])];',
            f'{y_kind} biased = add(x = product, y = b)[name = string("biased")];',
            f'{y_kind} y = relu(x = biased)[name = string("y")];',
        ])
        result = "y"
    name = f"matmul_m{rows}_k{reduction}_n{columns}_ty{int(transpose)}"
    if chain_bias_relu:
        name += "_bias_relu"
    return case(name, "chain" if chain_bias_relu else "matmul", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "transpose_y": transpose, "bias": chain_bias_relu, "relu": chain_bias_relu,
    }, program(f"{x_kind} x", body, result), weights, description)


def linear(reduction: int, columns: int) -> dict[str, Any]:
    x_shape, w_shape, y_shape = (1, reduction), (columns, reduction), (1, columns)
    x_kind, y_kind = tensor_type(x_shape), tensor_type(y_shape)
    weight, weights, description = weight_constant("w", w_shape)
    bias_kind = tensor_type((columns,))
    values = ", ".join(["fp16(0x0p+0)"] * columns)
    body = [weight,
        f'{bias_kind} b = const()[name = string("b"), val = {bias_kind}([{values}])];',
        f'{y_kind} y = linear(x = x, weight = w, bias = b)[name = string("y")];']
    return case(f"linear_k{reduction}_n{columns}", "linear",
                {"reduction": reduction, "columns": columns},
                program(f"{x_kind} x", body, "y"), weights, description)


def add_relu_chain() -> dict[str, Any]:
    shape = (1, 512, 1, 1)
    kind = tensor_type(shape)
    body = [
        f'{kind} added = add(x = x, y = z)[name = string("added")];',
        f'{kind} y = relu(x = added)[name = string("y")];',
    ]
    return case("chain_add_relu_c512", "chain", {"operations": ["add", "relu"],
                "shape": shape}, program(f"{kind} x, {kind} z", body, "y"))


def dims(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def boolean(value: bool) -> str:
    return "true" if value else "false"


def env_case(name: str, family: str, parameters: dict[str, Any], arguments: str,
             body: list[str], result: str, elements: int | None = None,
             shapes: list[tuple[int, ...]] | None = None) -> dict[str, Any]:
    """An envelope case whose BLOBFILE constants share one uniform payload."""
    description: dict[str, Any] = {"storage": "none"}
    if elements:
        description = {
            "storage": "BLOBFILE", "shapes": [list(shape) for shape in shapes or []],
            "payload_bytes": elements * 2, "value": "fp16(0x1p-1)",
        }
    return case(name, family, parameters, program(arguments, body, result),
                elements, description)


def env_broadcast(operation: str, x_shape: tuple[int, ...],
                  z_shape: tuple[int, ...], mode: str = "runtime") -> dict[str, Any]:
    """`operation` over an input and a second operand of a broadcastable shape."""
    out_shape = tuple(max(left, right) for left, right in zip(x_shape, z_shape))
    x_kind, out_kind = tensor_type(x_shape), tensor_type(out_shape)
    arguments, body, elements, shapes = f"{x_kind} x", [], None, []
    if mode == "runtime":
        arguments += f", {tensor_type(z_shape)} z"
    elif mode == "scalar":
        body.append('fp16 z = const()[name = string("z"), val = fp16(0x1p-1)];')
    else:
        source, elements = blobfile("z", z_shape)
        body.append(source)
        shapes = [z_shape]
    body.append(f'{out_kind} y = {operation}(x = x, y = z)[name = string("y")];')
    operand = "scalar" if mode == "scalar" else f"{mode}_{dims(z_shape)}"
    return env_case(f"env_bcast_{operation}_{dims(x_shape)}_{operand}",
                    "env_broadcast", {
                        "operation": operation, "shape": x_shape,
                        "operand_shape": None if mode == "scalar" else z_shape,
                        "operand": mode, "output_shape": out_shape,
                    }, arguments, body, "y", elements, shapes)


def env_matmul(rows: int, reduction: int, columns: int, batch: int | None = None,
               transpose_x: bool = False, transpose_y: bool = True,
               x_storage: str = "runtime", w_storage: str = "blob") -> dict[str, Any]:
    """One matmul with either operand runtime or a BLOBFILE constant."""
    prefix = () if batch is None else (batch,)
    x_shape = prefix + ((reduction, rows) if transpose_x else (rows, reduction))
    w_matrix = (columns, reduction) if transpose_y else (reduction, columns)
    w_shape = prefix + w_matrix if w_storage == "runtime" else w_matrix
    out_shape = prefix + (rows, columns)
    body = [
        f'bool tx = const()[name = string("tx"), val = bool({boolean(transpose_x)})];',
        f'bool ty = const()[name = string("ty"), val = bool({boolean(transpose_y)})];',
    ]
    arguments, elements, shapes = [], None, []
    for operand, shape, storage in (("x", x_shape, x_storage), ("w", w_shape, w_storage)):
        if storage == "runtime":
            arguments.append(f"{tensor_type(shape)} {operand}")
        else:
            source, count = blobfile(operand, shape)
            body.append(source)
            elements = max(elements or 0, count)
            shapes.append(shape)
    body.append(f'{tensor_type(out_shape)} product = matmul(transpose_x = tx, '
                'transpose_y = ty, x = x, y = w)[name = string("product")];')
    form = f"{'r2' if batch is None else 'r3'}{x_storage[0]}{w_storage[0]}"
    name = (f"env_mm_{form}_m{rows}_k{reduction}_n{columns}"
            f"_tx{int(transpose_x)}_ty{int(transpose_y)}")
    if batch is not None:
        name += f"_b{batch}"
    return env_case(name, "env_matmul", {
        "rows": rows, "reduction": reduction, "columns": columns, "batch": batch,
        "transpose_x": transpose_x, "transpose_y": transpose_y,
        "x_storage": x_storage, "w_storage": w_storage, "x_shape": x_shape,
        "w_shape": w_shape, "output_shape": out_shape,
    }, ", ".join(arguments), body, "product", elements, shapes)


def env_conv(kernel: int, inputs: int, outputs: int, spatial: int, stride: int,
             groups: int, bias: bool) -> dict[str, Any]:
    """A convolution sweeping kernel, stride, grouping, and a BLOBFILE bias."""
    out_spatial = (spatial + stride - 1) // stride
    x_shape = (1, inputs, spatial, spatial)
    y_shape = (1, outputs, out_spatial, out_spatial)
    w_shape = (outputs, inputs // groups, kernel, kernel)
    weight, elements = blobfile("w", w_shape)
    shapes = [w_shape]
    body = [
        'string pt = const()[name = string("pt"), val = string("same")];',
        'tensor<int32, [2]> st = const()[name = string("st"), '
        f'val = tensor<int32, [2]>([{stride}, {stride}])];',
        'tensor<int32, [4]> pd = const()[name = string("pd"), '
        'val = tensor<int32, [4]>([0, 0, 0, 0])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), '
        'val = tensor<int32, [2]>([1, 1])];',
        f'int32 gp = const()[name = string("gp"), val = int32({groups})];', weight,
    ]
    bias_argument = ""
    if bias:
        source, count = blobfile("b", (outputs,))
        body.append(source)
        elements = max(elements, count)
        shapes.append((outputs,))
        bias_argument = "bias = b, "
    body.append(
        f'{tensor_type(y_shape)} y = conv({bias_argument}dilations = dl, '
        'groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)'
        '[name = string("y")];')
    name = (f"env_conv_k{kernel}_c{inputs}_n{outputs}_s{spatial}"
            f"_st{stride}_g{groups}_bias{int(bias)}")
    return env_case(name, "env_conv", {
        "kernel": kernel, "input_channels": inputs, "output_channels": outputs,
        "spatial": spatial, "stride": stride, "groups": groups, "bias": bias,
        "input_shape": x_shape, "output_shape": y_shape, "weight_shape": w_shape,
    }, f"{tensor_type(x_shape)} x", body, "y", elements, shapes)


def env_activation(operation: str, shape: tuple[int, ...],
                   mode: str | None = None) -> dict[str, Any]:
    """gelu or silu on a spatial shape, optionally selecting the gelu mode."""
    kind = tensor_type(shape)
    attributes = f', mode = string("{mode}")' if mode else ""
    body = [f'{kind} y = {operation}(x = x{attributes})[name = string("y")];']
    name = f"env_act_{operation}_{dims(shape)}"
    if mode:
        name += f"_{mode.lower()}"
    return env_case(name, "env_activation",
                    {"operation": operation, "shape": shape, "mode": mode},
                    f"{kind} x", body, "y")


def env_chain_mlp(rows: int, reduction: int, hidden: int,
                  batch: int | None = None) -> dict[str, Any]:
    """matmul -> gelu -> matmul, the transformer feed-forward block."""
    prefix = () if batch is None else (batch,)
    x_shape = prefix + (rows, reduction)
    inner_shape = prefix + (rows, hidden)
    first, first_elements = blobfile("w1", (hidden, reduction))
    second, second_elements = blobfile("w2", (reduction, hidden))
    body = [
        'bool f = const()[name = string("f"), val = bool(false)];',
        'bool t = const()[name = string("t"), val = bool(true)];', first,
        f'{tensor_type(inner_shape)} h = matmul(transpose_x = f, transpose_y = t, '
        'x = x, y = w1)[name = string("h")];',
        f'{tensor_type(inner_shape)} a = gelu(x = h, mode = string("EXACT"))'
        '[name = string("a")];', second,
        f'{tensor_type(x_shape)} y = matmul(transpose_x = f, transpose_y = t, '
        'x = a, y = w2)[name = string("y")];',
    ]
    name = f"env_chain_mlp_m{rows}_k{reduction}_h{hidden}"
    if batch is not None:
        name += f"_b{batch}"
    return env_case(name, "env_chain", {
        "operations": ["matmul", "gelu", "matmul"], "rows": rows,
        "reduction": reduction, "hidden": hidden, "batch": batch,
        "input_shape": x_shape,
    }, f"{tensor_type(x_shape)} x", body, "y",
        max(first_elements, second_elements), [(hidden, reduction), (reduction, hidden)])


def env_chain_softmax(heads: int, sequence: int) -> dict[str, Any]:
    """Attention-style softmax over the last axis of [1, H, S, S]."""
    shape = (1, heads, sequence, sequence)
    kind = tensor_type(shape)
    body = [
        'int32 axis = const()[name = string("axis"), val = int32(-1)];',
        f'{kind} y = softmax(x = x, axis = axis)[name = string("y")];',
    ]
    return env_case(f"env_chain_softmax_h{heads}_s{sequence}", "env_chain",
                    {"operations": ["softmax"], "heads": heads,
                     "sequence": sequence, "shape": shape, "axis": -1},
                    f"{kind} x", body, "y")


def env_chain_layer_norm_matmul(reduction: int, columns: int, rows: int = 1,
                                batch: int | None = None) -> dict[str, Any]:
    """layer_norm over the last axis feeding a constant-weight matmul."""
    prefix = () if batch is None else (batch,)
    x_shape = prefix + (rows, reduction)
    out_shape = prefix + (rows, columns)
    kind = tensor_type(x_shape)
    weight, elements = blobfile("w", (columns, reduction))
    body = [
        'tensor<int32, [1]> axes = const()[name = string("axes"), '
        'val = tensor<int32, [1]>([-1])];',
        'bool f = const()[name = string("f"), val = bool(false)];',
        'bool t = const()[name = string("t"), val = bool(true)];',
        f'{kind} n = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001))'
        '[name = string("n")];', weight,
        f'{tensor_type(out_shape)} y = matmul(transpose_x = f, transpose_y = t, '
        'x = n, y = w)[name = string("y")];',
    ]
    name = f"env_chain_ln_matmul_m{rows}_k{reduction}_n{columns}"
    if batch is not None:
        name += f"_b{batch}"
    return env_case(name, "env_chain", {
        "operations": ["layer_norm", "matmul"], "rows": rows,
        "reduction": reduction, "columns": columns, "batch": batch,
        "input_shape": x_shape, "output_shape": out_shape,
    }, f"{kind} x", body, "y", elements, [(columns, reduction)])


def env_chain_residual(operation: str, shape: tuple[int, ...]) -> dict[str, Any]:
    """The residual form x + f(x)."""
    kind = tensor_type(shape)
    attributes = ', mode = string("EXACT")' if operation == "gelu" else ""
    body = [
        f'{kind} f = {operation}(x = x{attributes})[name = string("f")];',
        f'{kind} y = add(x = x, y = f)[name = string("y")];',
    ]
    return env_case(f"env_chain_residual_{operation}_{dims(shape)}", "env_chain",
                    {"operations": [operation, "add"], "operation": operation,
                     "shape": shape}, f"{kind} x", body, "y")


def env_chain_attention(sequence: int, depth: int) -> dict[str, Any]:
    """matmul -> softmax -> matmul over three runtime inputs."""
    operand = (1, sequence, depth)
    scores = (1, sequence, sequence)
    operand_kind, scores_kind = tensor_type(operand), tensor_type(scores)
    body = [
        'bool f = const()[name = string("f"), val = bool(false)];',
        'bool t = const()[name = string("t"), val = bool(true)];',
        'int32 axis = const()[name = string("axis"), val = int32(-1)];',
        f'{scores_kind} s = matmul(transpose_x = f, transpose_y = t, x = q, y = k)'
        '[name = string("s")];',
        f'{scores_kind} p = softmax(x = s, axis = axis)[name = string("p")];',
        f'{operand_kind} y = matmul(transpose_x = f, transpose_y = f, x = p, y = v)'
        '[name = string("y")];',
    ]
    arguments = ", ".join(f"{operand_kind} {name}" for name in ("q", "k", "v"))
    return env_case(f"env_chain_attention_s{sequence}_d{depth}", "env_chain",
                    {"operations": ["matmul", "softmax", "matmul"],
                     "sequence": sequence, "depth": depth,
                     "operand_shape": operand, "scores_shape": scores},
                    arguments, body, "y")


def envelope_campaign() -> list[dict[str, Any]]:
    """Probes for the outer edge of Apple's accepted single-program forms."""
    cases = []
    for operation in ("add", "mul"):
        for channels in (64, 96, 768):
            for spatial in (8, 16):
                shape = (1, channels, spatial, spatial)
                cases.extend(env_broadcast(operation, shape, operand) for operand in
                             ((1, channels, 1, 1), (1, 1, spatial, spatial), (1, 1, 1, 1)))
        for channels in (64, 768):
            for spatial in (8, 16):
                shape = (1, channels, spatial, spatial)
                cases.append(env_broadcast(operation, shape, (1, channels, 1, 1), "blob"))
                cases.append(env_broadcast(operation, shape, (1, 1, 1, 1), "scalar"))
        for batch in (2, 8):
            for channels in (64, 512, 1024):
                shape = (batch, channels, 1, 1)
                cases.append(env_broadcast(operation, shape, shape))
            spatial_shape = (batch, 64, 8, 8)
            cases.append(env_broadcast(operation, spatial_shape, spatial_shape))
            cases.append(env_broadcast(operation, spatial_shape, (1, 64, 1, 1)))
    for operation in ("add", "mul", "sub"):
        for channels in (96, 200, 300, 768, 3072, 8192, 16384):
            shape = (1, channels, 1, 1)
            cases.append(env_broadcast(operation, shape, shape))
    for reduction in (2048, 4096, 8192):
        for columns in (2048, 4096, 8192):
            cases.extend(env_matmul(rows, reduction, columns)
                         for rows in (1, 16, 32, 128, 256, 512))
            cases.extend(env_matmul(rows, reduction, columns, batch=1)
                         for rows in (1, 32, 256))
    for size in (2048, 4096):
        for rows in (1, 32, 256):
            cases.append(env_matmul(rows, size, size, transpose_x=True))
        for rows in (1, 32):
            cases.append(env_matmul(rows, size, size, x_storage="blob",
                                    w_storage="runtime"))
        for rows in (16, 128):
            cases.append(env_matmul(rows, size, size, w_storage="runtime"))
            cases.append(env_matmul(rows, size, size, w_storage="runtime",
                                    transpose_y=False))
    cases.append(env_matmul(128, 2048, 2048, w_storage="runtime", transpose_x=True))
    cases.append(env_matmul(128, 2048, 2048, w_storage="runtime", transpose_x=True,
                            transpose_y=False))
    for sequence in (64, 128, 256, 512):
        for depth in (64, 128):
            cases.append(env_matmul(sequence, depth, sequence, batch=1,
                                    w_storage="runtime"))
            cases.append(env_matmul(sequence, sequence, depth, batch=1,
                                    w_storage="runtime", transpose_y=False))
    for kernel in (1, 3):
        for stride in (1, 2):
            for bias in (False, True):
                cases.extend(env_conv(kernel, 64, 64, 16, stride, groups, bias)
                             for groups in (1, 4))
    cases.extend([
        env_conv(1, 256, 256, 8, 1, 1, True),
        env_conv(1, 768, 768, 1, 1, 1, True),
        env_conv(1, 64, 256, 8, 1, 1, False),
        env_conv(3, 128, 128, 16, 2, 1, True),
        env_conv(3, 64, 64, 16, 1, 64, False),
        env_conv(3, 64, 64, 16, 1, 64, True),
    ])
    for operation in ("gelu", "silu"):
        mode = "EXACT" if operation == "gelu" else None
        cases.extend(env_activation(operation, shape, mode) for shape in (
            (1, 64, 8, 8), (1, 128, 16, 16), (1, 256, 32, 32), (1, 768, 16, 16),
            (1, 3072, 1, 1)))
    cases.extend(env_activation("gelu", (1, 768, 1, 1), mode) for mode in
                 ("EXACT", "TANH_APPROXIMATION", "SIGMOID_APPROXIMATION"))
    cases.extend(env_chain_mlp(rows, 2048, 4096) for rows in (1, 32, 128))
    cases.append(env_chain_mlp(32, 4096, 4096))
    cases.append(env_chain_mlp(1, 2048, 2048, batch=1))
    for heads in (1, 8, 12):
        cases.extend(env_chain_softmax(heads, sequence)
                     for sequence in (64, 128, 256))
    for reduction in (2048, 4096):
        cases.extend(env_chain_layer_norm_matmul(reduction, columns)
                     for columns in (2048, 4096))
    cases.append(env_chain_layer_norm_matmul(2048, 2048, rows=32, batch=1))
    for operation in ("gelu", "silu", "relu"):
        cases.extend(env_chain_residual(operation, shape)
                     for shape in ((1, 768, 1, 1), (1, 64, 8, 8)))
    cases.extend(env_chain_attention(sequence, 64) for sequence in (64, 128, 256))
    cases.append(env_chain_attention(128, 128))
    return cases


def exp_scaled_chain(channels: int) -> dict[str, Any]:
    """exp with the scale correction in scaled form, as one MIL program (the five calls of the D4 cell, in one program).

    c = x * 16 K, g = c + 16, p = exp(x) * g, y = p * 0.0625. K = 0x0b18 is the relative error of the exp program's fp16 scale word 0x3dc5.
    The constants are scalars, so Apple may fold them into register words. Whether the result keeps the stage arithmetic of the five separate
    programs is what a device run of the minted program has to show."""
    kind = tensor_type((1, channels, 1, 1))
    mil = program(f"{kind} x", [
        'fp16 k16 = const()[name = string("k16"), val = fp16(0x1.c6p-9)];',
        'fp16 sixteen = const()[name = string("sixteen"), val = fp16(0x1p+4)];',
        'fp16 sixteenth = const()[name = string("sixteenth"), val = fp16(0x1p-4)];',
        f'{kind} e = exp(x = x)[name = string("e")];',
        f'{kind} c = mul(x = x, y = k16)[name = string("c")];',
        f'{kind} g = add(x = c, y = sixteen)[name = string("g")];',
        f'{kind} p = mul(x = e, y = g)[name = string("p")];',
        f'{kind} y = mul(x = p, y = sixteenth)[name = string("y")];',
    ], "y")
    return case(f"chain_exp_d4_c{channels}", "chain",
                {"operations": ["exp", "mul", "add", "mul", "mul"], "shape": (1, channels, 1, 1), "scale_shift": 4}, mil)


def campaign() -> list[dict[str, Any]]:
    cases = []
    shapes = [(1, channels, 1, 1) for channels in
              (64, 128, 256, 512, 1024, 2048, 4096)]
    shapes += [(1, 64, 8, 8), (1, 128, 16, 16), (1, 3, 224, 224)]
    for operation in ("add", "mul", "maximum", "minimum", "sub", "real_div", "pow"):
        cases.extend(binary_runtime(operation, shape) for shape in shapes)
    cases.extend(unary("abs", shape[1]) for shape in shapes[:7])
    for operation in ("add", "mul", "maximum", "minimum", "sub", "real_div"):
        for channels in (64, 512):
            cases.extend(binary_constant(operation, channels, storage)
                         for storage in ("inline", "blob", "scalar"))
    for operation in ("relu", "sigmoid", "tanh", "gelu", "silu", "exp", "log",
                      "sqrt", "rsqrt", "clip", "leaky_relu"):
        cases.extend(unary(operation, channels) for channels in (64, 512))
    for operation in ("reduce_sum", "reduce_max", "reduce_mean"):
        cases.extend(reduction(operation, axes) for axes in ((1,), (2, 3)))
    for operation in ("softmax", "layer_norm"):
        cases.extend(normalization(operation, shape)
                     for shape in ((1, 512, 1, 1), (1, 64, 8, 8)))
    for inputs in (256, 512, 1024):
        for outputs in (512, 1024):
            cases.extend(convolution(1, inputs, outputs, 1, bias)
                         for bias in (False, True))
    cases.append(convolution(3, 64, 64, 8, False))
    for rows in (1, 2, 8, 64):
        for inner in (256, 512, 1024):
            for columns in (256, 512, 1024):
                cases.extend(matmul(inner, columns, rows, transpose)
                             for transpose in (False, True))
    cases.extend(linear(inner, columns)
                 for inner, columns in ((256, 512), (512, 1024)))
    cases.extend([
        add_relu_chain(),
        matmul(512, 512, 1, False, True),
        convolution(1, 64, 64, 8, False, True),
    ])
    cases.extend(envelope_campaign())
    cases.extend(exp_scaled_chain(channels) for channels in (64, 512))
    names = [item["name"] for item in cases]
    if len(names) != len(set(names)):
        raise AssertionError("campaign contains duplicate case names")
    return cases


def cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def program_regions(programs: list[dict[str, Any]], text: dict[str, int],
                    text_bytes: int) -> list[tuple[int, int]]:
    """The __TEXT/__text byte range each program descriptor owns."""
    starts = []
    for index, program in enumerate(programs):
        start = int(program["text_address"], 16) - text["address"]
        if start < 0 or start >= text_bytes or start % 4:
            raise ValueError(
                f"program[{index}] text address {program['text_address']} is "
                "outside __TEXT/__text")
        starts.append(start)
    if starts != sorted(starts):
        raise ValueError("HWX program descriptors are not in text order")
    return [(start, next_start) for start, next_start in
            zip(starts, starts[1:] + [text_bytes])]


def split_program_tasks(region: bytes, program: dict[str, Any],
                        target: str) -> list[bytes]:
    if target == "h13":
        return split_h13_tasks(region, program["task_words_minus_one"],
                               program["task_count"])
    return split_h14_tasks(region)


def decode_task_safely(task: bytes, target: str) -> dict[str, Any]:
    """Decode one task, or record why its register stream is undecodable.

    A register-stream gap means Apple accepted the program and this parser
    cannot yet split every record; it is not an Apple rejection."""
    try:
        return decode_task(task, target)
    except ValueError as error:
        header = HEADER_WORDS[target]
        words = struct.unpack_from(f"<{min(len(task) // 4, header)}I", task)
        return {
            "size_bytes": len(task),
            "header_words": [f"0x{word:08x}" for word in words],
            "decode_error": str(error),
        }


def nonzero_words(data: bytes, cursor: int, size: int) -> dict[str, str]:
    """Every nonzero 32-bit word of one load command, keyed by command offset."""
    return {f"0x{index * 4:03x}": f"0x{value:08x}"
            for index, value in enumerate(
                struct.unpack_from(f"<{size // 4}I", data, cursor)) if value}


def parse_hwx(data: bytes, target: str) -> dict[str, Any]:
    if len(data) < 32:
        raise ValueError("HWX Mach-O header is truncated")
    magic, _, subtype, _, command_count, command_bytes, _, _ = struct.unpack_from(
        "<8I", data)
    if magic != MAGIC:
        raise ValueError(f"invalid HWX magic 0x{magic:08x}")
    if subtype != SUBTYPES[target]:
        raise ValueError(f"compiler returned subtype {subtype} for {target}")
    sections: dict[tuple[str, str], dict[str, int]] = {}
    tensors = []
    programs: list[dict[str, Any]] = []
    descriptor_words: list[dict[str, str]] = []
    cursor = 32
    command_end = cursor + command_bytes
    if command_end > len(data):
        raise ValueError("HWX load command table is truncated")
    for command_index in range(command_count):
        if cursor + 8 > command_end:
            raise ValueError(f"load command[{command_index}] is truncated")
        command, size = struct.unpack_from("<2I", data, cursor)
        if size < 8 or cursor + size > command_end:
            raise ValueError(f"load command[{command_index}] has invalid size")
        kind = struct.unpack_from("<I", data, cursor + 8)[0] if size >= 12 else None
        if command == 0x19:
            if size < 72:
                raise ValueError("LC_SEGMENT_64 is truncated")
            fields = struct.unpack_from("<2I16s4Q4I", data, cursor)
            segment = cstring(fields[2])
            section_cursor = cursor + 72
            if fields[-2] * 80 > size - 72:
                raise ValueError("LC_SEGMENT_64 section table is truncated")
            for _ in range(fields[-2]):
                entry = struct.unpack_from("<16s16s2Q8I", data, section_cursor)
                section = cstring(entry[0])
                sections[(segment, section)] = {
                    "address": entry[2], "size": entry[3], "offset": entry[4],
                }
                section_cursor += 80
        elif command == 4 and kind == 3 and size >= 0x78:
            tensors.append({
                "binding": struct.unpack_from("<I", data, cursor + 0x14)[0],
                "element_code": struct.unpack_from("<I", data, cursor + 0x24)[0],
                "shape": list(struct.unpack_from("<4I", data, cursor + 0x28)),
                "strides": list(struct.unpack_from("<4Q", data, cursor + 0x50)),
                "total_bytes": struct.unpack_from("<Q", data, cursor + 0x70)[0],
            })
        elif command == 4 and kind == 1 and size >= 0x820:
            programs.append({
                "kind": kind, "command_size": size,
                "code": struct.unpack_from("<I", data, cursor + 0x0C)[0],
                "text_address": f"0x{struct.unpack_from('<Q', data, cursor + 0x10)[0]:x}",
                "constant_address": f"0x{struct.unpack_from('<Q', data, cursor + 0x18)[0]:x}",
                "resource_addresses": [
                    f"0x{value:x}" for value in struct.unpack_from("<5Q", data, cursor + 0x30)
                ],
                "task_words_minus_one": struct.unpack_from("<I", data, cursor + 0x818)[0],
                "task_count": struct.unpack_from("<I", data, cursor + 0x81C)[0],
            })
            descriptor_words.append(nonzero_words(data, cursor, size))
        elif command == 4 and kind == 4 and size >= 0x838:
            programs.append({
                "kind": kind, "command_size": size,
                "text_address": f"0x{struct.unpack_from('<Q', data, cursor + 0x10)[0]:x}",
                "constant_address": f"0x{struct.unpack_from('<Q', data, cursor + 0x20)[0]:x}",
                "resource_addresses": [
                    f"0x{value:x}" for value in struct.unpack_from("<5Q", data, cursor + 0x30)
                ],
                "text_words": struct.unpack_from("<I", data, cursor + 0x824)[0],
                "task_count": struct.unpack_from("<I", data, cursor + 0x830)[0],
                "trailing_words": [
                    f"0x{value:08x}" for value in struct.unpack_from(
                        f"<{(size - 0x810) // 4}I", data, cursor + 0x810)
                ],
            })
            descriptor_words.append(nonzero_words(data, cursor, size))
        cursor += size
    if cursor != command_end:
        raise ValueError("HWX load command sizes do not match the header")
    text = sections.get(("__TEXT", "__text")) or sections.get(("__TEXT", "__TEXT"))
    constants = sections.get(("__TEXT", "__const"))
    kernel = sections.get(("__KERN_0", "__kern_0"))
    if not text or not constants or not programs:
        raise ValueError("HWX lacks text, constant, or program descriptor metadata")
    text_end = text["offset"] + text["size"]
    if text_end > len(data):
        raise ValueError("HWX task section is truncated")
    text_data = data[text["offset"]:text_end]
    raw_tasks = []
    for index, (program, region) in enumerate(
            zip(programs, program_regions(programs, text, len(text_data)))):
        start, end = region
        tasks = split_program_tasks(text_data[start:end], program, target)
        if len(tasks) != program["task_count"]:
            raise ValueError(
                f"program[{index}] declares {program['task_count']} tasks but "
                f"decoded {len(tasks)}")
        program["task_section"] = {"offset": start, "size": end - start}
        raw_tasks.extend(tasks)
    descriptors = [decode_task_safely(task, target) for task in raw_tasks]
    return {
        "hwx_sha256": hashlib.sha256(data).hexdigest(),
        "hwx_bytes": len(data),
        "program_descriptor": programs[-1],
        "program_count": len(programs),
        "programs": programs,
        "program_descriptor_words": descriptor_words,
        "tensor_descriptors": tensors,
        "constant_section": section_summary(data, constants, "constant"),
        # H16-H18 move the kernel table (LUT and scalar operands) that H13-H15
        # keep in __TEXT/__const into its own __KERN_0 segment.
        "kernel_section": section_summary(data, kernel, "kernel") if kernel else None,
        "task_descriptors": descriptors,
        "task_decode_errors": sum("decode_error" in item for item in descriptors),
    }


def section_summary(data: bytes, section: dict[str, int],
                    label: str) -> dict[str, Any]:
    """Hashes and nonzero statistics of one section; the decoded fp16 words
    only when the section is at most 256 bytes."""
    end = section["offset"] + section["size"]
    if end > len(data):
        raise ValueError(f"HWX {label} section is truncated")
    content = data[section["offset"]:end]
    prefix = content[:128]
    chunk_bytes = 0x800
    chunks = [
        {
            "offset": offset,
            "size": len(chunk),
            "sha256": hashlib.sha256(chunk).hexdigest(),
            "nonzero_bytes": sum(value != 0 for value in chunk),
        }
        for offset in range(0, len(content), chunk_bytes)
        for chunk in (content[offset:offset + chunk_bytes],)
    ] if len(content) <= 0x10000 else []
    return {
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "nonzero_bytes": sum(value != 0 for value in content),
        "prefix_128_sha256": hashlib.sha256(prefix).hexdigest(),
        "prefix_128_nonzero_bytes": sum(value != 0 for value in prefix),
        "tail_after_128_nonzero_bytes": sum(value != 0 for value in content[128:]),
        "nonzero_fp16_words": decode_constant_half_words(content),
        "chunk_bytes": chunk_bytes,
        "chunk_count": (len(content) + chunk_bytes - 1) // chunk_bytes,
        "chunks": chunks,
    }


def run_case(item: dict[str, Any], target: str, output: Path,
             oracle_tool: Path, source_commit: str) -> str:
    destination = output / target / f"{item['name']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "case": item["name"],
        "family": item["family"],
        "parameters": item["parameters"],
        "mil": item["mil"],
        "weights": item["weights_description"],
        "target": target,
        "source_commit": source_commit,
        "compiler": {
            "tool": str(oracle_tool),
            "tool_sha256": hashlib.sha256(oracle_tool.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "decoder_sha256": hashlib.sha256(
                Path(__file__).with_name("h13_td.py").read_bytes()).hexdigest(),
            "host": platform.node(),
            "platform": platform.platform(),
        },
    }
    with tempfile.TemporaryDirectory(prefix=f"mil-hwx-{target}-{item['name']}-") as root:
        root_path = Path(root)
        capture = root_path / "capture"
        compiled = root_path / "compiled"
        capture.mkdir()
        compiled.mkdir()
        (capture / "model.mil").write_text(item["mil"])
        weights = item["weights"]
        if isinstance(weights, int):
            weights = blob(half_payload(weights))
        (capture / "weights.bin").write_bytes(weights or b"")
        command = [str(oracle_tool), str(capture), str(compiled), target]
        record["compiler"]["command"] = [str(oracle_tool), "CAPTURE_DIR", "OUTPUT_DIR", target]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=COMPILE_TIMEOUT_SECONDS, check=False)
            error = (result.stderr or result.stdout).strip()
            hwx = compiled / "model.hwx"
            if result.returncode == 0 and hwx.is_file():
                record.update(parse_hwx(hwx.read_bytes(), target))
                record["error"] = None
                status = "decoded"
            else:
                record.update({
                    "hwx_sha256": None, "hwx_bytes": None,
                    "program_descriptor": None, "tensor_descriptors": [],
                    "constant_section": None, "task_descriptors": [],
                    "error": error or f"compiler exited {result.returncode}",
                })
                status = "rejected"
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            record.update({
                "hwx_sha256": None, "hwx_bytes": None,
                "program_descriptor": None, "tensor_descriptors": [],
                "constant_section": None, "task_descriptors": [],
                "error": str(error),
            })
            status = "rejected"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return status


def source_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def local_run(args: argparse.Namespace) -> int:
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.limit is not None:
        selected = selected[:args.limit]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)} targets={len(args.targets)}")
        return 0
    tool = Path(args.oracle_tool)
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    if not os.access(tool, os.X_OK):
        raise SystemExit(f"oracle tool is not executable: {tool}")
    decoded = rejected = 0
    output = Path(args.output)
    for target in args.targets:
        for item in selected:
            destination = output / target / f"{item['name']}.json"
            if destination.exists() and not args.force:
                existing = json.loads(destination.read_text())
                decoded += existing.get("error") is None
                rejected += existing.get("error") is not None
                continue
            status = run_case(item, target, output, tool, args.source_commit)
            decoded += status == "decoded"
            rejected += status == "rejected"
            print(f"{target} {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected) * len(args.targets)} decoded={decoded} rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root_result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-oracles.XXXXXX"],
        capture_output=True, text=True, check=True)
    remote_root = root_result.stdout.strip()
    if not remote_root.startswith("/tmp/mil-hwx-oracles."):
        raise RuntimeError(f"unexpected remote temporary path: {remote_root!r}")
    script = Path(__file__).resolve()
    decoder = script.with_name("h13_td.py")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", str(script), str(decoder),
                        f"{args.host}:{remote_root}/"], check=True)
        command = [
            "python3", f"{remote_root}/{script.name}", "--local",
            "--oracle-tool", args.oracle_tool, "--output", f"{remote_root}/oracles",
            "--source-commit", args.source_commit, "--targets", *args.targets,
        ]
        if args.case:
            command.extend(["--case", args.case])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.force:
            command.append("--force")
        if args.list:
            command.append("--list")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host,
             " ".join(shlex.quote(value) for value in command)], check=False)
        if not args.list:
            with tempfile.TemporaryDirectory(prefix="mil-hwx-oracle-json-") as staging:
                subprocess.run(["scp", "-q", "-r",
                                f"{args.host}:{remote_root}/oracles/.", staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(remote_root)}"], check=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="run the Apple compiler locally")
    mode.add_argument("--host", help="copy the worker to this SSH Mac and retrieve JSON")
    # The shipped corpus is H13 and H14; later generations are opt-in.
    parser.add_argument("--targets", nargs="+", choices=sorted(SUBTYPES),
                        default=["h13", "h14"])
    parser.add_argument("--oracle-tool", default=DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--case", help="shell pattern selecting case names")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local else remote_run(arguments))
