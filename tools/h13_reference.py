#!/usr/bin/env python3
"""Numpy-free fp16 MIL reference, including CPU-only tiling, resize and concat references.

Reference support does not expand any target's hardware encoding envelope.

Matmul products accumulate in float32 and round once to fp16. H13 reductions
larger than 512 elements use chunked-fp16 partial sums, so device results may
have one extra fp16 rounding per chunk.
"""

import argparse
import ast
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


_TOKEN = re.compile(
    r"\s+|//[^\n]*|/\*.*?\*/|->|"
    r'"(?:\\.|[^"\\])*"|'
    r"(?:0[xX][0-9a-fA-F]+|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"[A-Za-z_$][A-Za-z0-9_.$]*|[][(){}<>,;=]",
    re.DOTALL,
)


@dataclass(frozen=True)
class MILType:
    dtype: str
    shape: tuple = ()


@dataclass(frozen=True)
class Call:
    name: str
    call_type: MILType
    positional: tuple
    named: dict


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple
    values: tuple


@dataclass(frozen=True)
class Operation:
    result_type: MILType
    result: str
    name: str
    arguments: dict
    attributes: dict


def fp16(value):
    value = float(value)
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def add_fp16(left, right):
    return fp16(float(left) + float(right))


def dot_fp32(left, right):
    if len(left) != len(right):
        raise ValueError("matmul reduction dimensions differ")
    total = 0.0
    for x, y in zip(left, right):
        total = fp32(total + fp32(float(x) * float(y)))
    return fp16(total)


def encode_fp16(items):
    return b"".join(struct.pack("<e", value) for value in items)


def decode_fp16(data):
    if len(data) % 2:
        raise ValueError("fp16 data must have an even byte count")
    return tuple(value[0] for value in struct.iter_unpack("<e", data))


def _tokenize(text):
    tokens = []
    offset = 0
    while offset < len(text):
        match = _TOKEN.match(text, offset)
        if not match:
            raise ValueError(f"unsupported MIL token at byte {offset}")
        token = match.group(0)
        offset = match.end()
        if not token.isspace() and not token.startswith(("//", "/*")):
            tokens.append(token)
    return tokens


class Parser:
    def __init__(self, text):
        self.tokens = _tokenize(text)
        self.index = 0

    def peek(self, token=None):
        if self.index >= len(self.tokens):
            return False if token is not None else None
        return self.tokens[self.index] == token if token is not None else self.tokens[self.index]

    def take(self, token=None):
        value = self.peek()
        if value is None or token is not None and value != token:
            raise ValueError(f"expected {token or 'token'}, got {value or 'end of input'}")
        self.index += 1
        return value

    def parse_type(self):
        name = self.take()
        if not re.match(r"[A-Za-z_$]", name):
            raise ValueError(f"expected MIL type, got {name}")
        dtype = name
        shape = ()
        if self.peek("<"):
            self.take("<")
            depth = 1
            inner = []
            while depth:
                token = self.take()
                if token == "<":
                    depth += 1
                elif token == ">":
                    depth -= 1
                    if not depth:
                        break
                inner.append(token)
            if name == "tensor":
                if not inner:
                    raise ValueError("tensor type lacks an element type")
                dtype = inner[0]
                try:
                    opening = inner.index("[")
                    closing = len(inner) - 1 - inner[::-1].index("]")
                except ValueError as error:
                    raise ValueError("tensor type lacks a shape") from error
                shape = tuple(int(item, 0) for item in inner[opening + 1:closing] if item != ",")
        return MILType(dtype, shape)

    def parse_arguments(self, closing):
        positional = []
        named = {}
        while not self.peek(closing):
            if (self.index + 1 < len(self.tokens) and
                    re.match(r"[A-Za-z_$]", self.tokens[self.index]) and
                    self.tokens[self.index + 1] == "="):
                name = self.take()
                self.take("=")
                named[name] = self.parse_expression()
            else:
                positional.append(self.parse_expression())
            if not self.peek(","):
                break
            self.take(",")
        self.take(closing)
        return tuple(positional), named

    def parse_expression(self):
        if self.peek("("):
            self.take("(")
            positional, named = self.parse_arguments(")")
            if named:
                raise ValueError("MIL tuple cannot contain named values")
            return list(positional)
        if self.peek("["):
            self.take("[")
            positional, named = self.parse_arguments("]")
            if named:
                raise ValueError("MIL list cannot contain named values")
            return list(positional)
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end of MIL expression")
        if token.startswith('"'):
            self.take()
            return ast.literal_eval(token)
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", token):
            self.take()
            return int(token, 0)
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
            self.take()
            return float(token) if any(character in token for character in ".eE") else int(token)
        if token in ("true", "false"):
            self.take()
            return token == "true"
        expression_type = self.parse_type()
        if not self.peek("("):
            if expression_type.shape or expression_type.dtype != token:
                raise ValueError("a constructed MIL type requires a value")
            return Ref(token)
        self.take("(")
        positional, named = self.parse_arguments(")")
        return Call(token, expression_type, positional, named)

    def parse_operation(self):
        result_type = self.parse_type()
        result = self.take()
        self.take("=")
        name = self.take()
        self.take("(")
        _, arguments = self.parse_arguments(")")
        attributes = {}
        if self.peek("["):
            self.take("[")
            _, attributes = self.parse_arguments("]")
        self.take(";")
        return Operation(result_type, result, name, arguments, attributes)

    def parse(self):
        try:
            self.index = self.tokens.index("func")
        except ValueError as error:
            raise ValueError("MIL program has no function") from error
        self.take("func")
        self.take()
        self.take("<")
        while not self.peek(">"):
            self.take()
        self.take(">")
        self.take("(")
        parameters = []
        while not self.peek(")"):
            parameters.append((self.parse_type(), self.take()))
            if not self.peek(","):
                break
            self.take(",")
        self.take(")")
        self.take("{")
        operations = []
        while not self.peek("}"):
            operations.append(self.parse_operation())
        self.take("}")
        self.take("->")
        self.take("(")
        returns = []
        while not self.peek(")"):
            returns.append(self.take())
            if not self.peek(","):
                break
            self.take(",")
        self.take(")")
        self.take(";")
        return parameters, operations, returns


def _tensor(value, expected_type=None):
    if isinstance(value, Tensor):
        result = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        if expected_type is None or expected_type.dtype != "fp16":
            raise ValueError("raw input requires an fp16 tensor type")
        result = Tensor("fp16", expected_type.shape, decode_fp16(bytes(value)))
    elif isinstance(value, (list, tuple)):
        if expected_type is None:
            raise ValueError("input sequences require a tensor type")
        converted = tuple(fp16(item) for item in value) if expected_type.dtype == "fp16" else tuple(value)
        result = Tensor(expected_type.dtype, expected_type.shape, converted)
    else:
        raise ValueError("tensor value must be dense bytes or a sequence")
    if expected_type is not None:
        if result.dtype != expected_type.dtype or result.shape != expected_type.shape:
            raise ValueError("tensor value does not match its MIL type")
        if len(result.values) != math.prod(expected_type.shape):
            raise ValueError("tensor byte count does not match its MIL shape")
    return result


def _flatten(items):
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(_flatten(item))
        elif isinstance(item, Tensor):
            result.extend(item.values)
        else:
            result.append(item)
    return result


def _blob(call, result_type, model_root):
    path = _resolve(call.named.get("path"), {}, model_root)
    offset = _resolve(call.named.get("offset"), {}, model_root)
    if not isinstance(path, str) or not path.startswith("@model_path/") or not isinstance(offset, int):
        raise ValueError("BLOBFILE requires @model_path and an integer offset")
    root = Path(model_root).resolve()
    candidate = (root / path[len("@model_path/"):]).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("BLOBFILE path escapes model root") from error
    data = candidate.read_bytes()
    if offset < 0 or offset + 24 > len(data):
        raise ValueError("BLOBFILE chunk header exceeds the file")
    magic, _, payload_length, payload_offset = struct.unpack_from("<IIQQ", data, offset)
    expected = math.prod(result_type.shape) * (2 if result_type.dtype == "fp16" else 4)
    if magic != 0xDEADBEEF or payload_length < expected or payload_offset + expected > len(data):
        raise ValueError("BLOBFILE subheader has invalid magic, size, or payload offset")
    payload = data[payload_offset:payload_offset + expected]
    if result_type.dtype == "fp16":
        values = decode_fp16(payload)
    elif result_type.dtype in ("int32", "uint32"):
        code = "i" if result_type.dtype == "int32" else "I"
        values = struct.unpack(f"<{math.prod(result_type.shape)}{code}", payload)
    else:
        raise ValueError(f"unsupported BLOBFILE element type {result_type.dtype}")
    return Tensor(result_type.dtype, result_type.shape, tuple(values))


def _resolve(expression, environment, model_root):
    if isinstance(expression, Ref):
        if expression.name not in environment:
            raise ValueError(f"unknown MIL value {expression.name}")
        return environment[expression.name]
    if isinstance(expression, list):
        return [_resolve(item, environment, model_root) for item in expression]
    if not isinstance(expression, Call):
        return expression
    arguments = [_resolve(item, environment, model_root) for item in expression.positional]
    if expression.name == "BLOBFILE":
        return expression
    if expression.name == "string":
        return str(arguments[0])
    if expression.name in ("int32", "uint32", "uint64"):
        return int(arguments[0])
    if expression.name == "bool":
        return bool(arguments[0])
    if expression.name == "fp16":
        return fp16(arguments[0])
    if expression.name == "fp32":
        return fp32(arguments[0])
    if expression.name == "tensor":
        if len(expression.positional) != 1:
            raise ValueError("tensor literal requires one payload")
        payload = expression.positional[0]
        if isinstance(payload, Call) and payload.name == "BLOBFILE":
            return _blob(payload, expression.call_type, model_root)
        resolved = _flatten([_resolve(payload, environment, model_root)])
        if len(resolved) != math.prod(expression.call_type.shape):
            raise ValueError("tensor literal element count does not match its shape")
        if expression.call_type.dtype == "fp16":
            resolved = [fp16(item) for item in resolved]
        elif expression.call_type.dtype.startswith(("int", "uint")):
            resolved = [int(item) for item in resolved]
        else:
            raise ValueError(f"unsupported tensor element type {expression.call_type.dtype}")
        return Tensor(expression.call_type.dtype, expression.call_type.shape, tuple(resolved))
    raise ValueError(f"unsupported MIL constructor {expression.name}")


def _broadcast_shape(left, right):
    if not left:
        return tuple(right)
    if not right:
        return tuple(left)
    result = []
    for a, b in zip(reversed(left), reversed(right)):
        if a != b and a != 1 and b != 1:
            raise ValueError("elementwise tensor shapes do not broadcast")
        result.append(max(a, b))
    result.extend(reversed(left[:-len(right)] if len(left) > len(right) else right[:-len(left)]))
    return tuple(reversed(result))


def _broadcast_values(value, shape):
    if not isinstance(value, Tensor):
        return [value] * math.prod(shape)
    if value.shape == shape:
        return list(value.values)
    padded = (1,) * (len(shape) - len(value.shape)) + value.shape
    if any(source not in (1, target) for source, target in zip(padded, shape)):
        raise ValueError("elementwise tensor shapes do not broadcast")
    source_strides = []
    stride = 1
    for dimension in reversed(padded):
        source_strides.append(stride)
        stride *= dimension
    source_strides.reverse()
    result = []
    for flat in range(math.prod(shape)):
        remainder = flat
        source = 0
        target_stride = math.prod(shape)
        for dimension, source_dimension, source_stride in zip(shape, padded, source_strides):
            target_stride //= dimension
            coordinate, remainder = divmod(remainder, target_stride)
            if source_dimension != 1:
                source += coordinate * source_stride
        result.append(value.values[source])
    return result


def _binary(name, left, right, result_type):
    """Evaluate the measured H13 binary datapath, including its unsigned zero products."""
    left_shape = left.shape if isinstance(left, Tensor) else ()
    right_shape = right.shape if isinstance(right, Tensor) else ()
    shape = _broadcast_shape(left_shape, right_shape)
    if shape != result_type.shape:
        raise ValueError(f"{name} result shape differs from its declaration")
    operations = {
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "maximum": max,
        "minimum": min,
        "sub": lambda a, b: a - b,
        "real_div": lambda a, b: a / b,
    }
    values = []
    for left_value, right_value in zip(_broadcast_values(left, shape),
                                       _broadcast_values(right, shape)):
        value = fp16(operations[name](left_value, right_value))
        values.append(0.0 if name == "mul" and value == 0.0 else value)
    return Tensor("fp16", shape, tuple(values))


def _matmul(left, right, transpose_x, transpose_y, result_type):
    """CPU MIL matmul with trailing matrix axes and broadcast batch axes.

    This is a mathematical reference, not an expansion of a target envelope.
    Products and sums round to fp32; each result rounds once to fp16.
    """
    left, right = _tensor(left), _tensor(right)
    if left.dtype != "fp16" or right.dtype != "fp16" or len(right.shape) < 2:
        raise ValueError("matmul requires fp16 tensors and a rank-2 or higher y")
    vector = len(left.shape) == 1
    if not left.shape or (vector and transpose_x):
        raise ValueError("transpose_x is invalid for rank-one x")
    lshape = (1, left.shape[0]) if vector else left.shape
    rows, reduction = lshape[-2:][::-1] if transpose_x else lshape[-2:]
    rk, columns = right.shape[-2:][::-1] if transpose_y else right.shape[-2:]
    batches = _broadcast_shape(lshape[:-2], right.shape[:-2])
    expected = batches + ((columns,) if vector else (rows, columns))
    if reduction != rk or result_type.dtype != "fp16" or result_type.shape != expected:
        raise ValueError("matmul shapes differ from its declaration")
    batch_count = math.prod(batches)
    # Broadcast indices instead of copying either runtime tensor.
    lindices = _broadcast_values(Tensor("int32", lshape[:-2],
        tuple(range(math.prod(lshape[:-2])))), batches)
    rindices = _broadcast_values(Tensor("int32", right.shape[:-2],
        tuple(range(math.prod(right.shape[:-2])))), batches)
    result = []
    for batch in range(batch_count):
        lb = lindices[batch] * rows * reduction
        rb = rindices[batch] * reduction * columns
        for row in range(rows):
            lhs = tuple(left.values[lb + (k * rows + row if transpose_x else row * reduction + k)]
                        for k in range(reduction))
            for column in range(columns):
                rhs = tuple(right.values[rb + (column * reduction + k if transpose_y else k * columns + column)]
                            for k in range(reduction))
                result.append(dot_fp32(lhs, rhs))
    return Tensor("fp16", expected, tuple(result))


def _transpose(value, perm, result_type):
    value = _tensor(value)
    rank = len(value.shape)
    if len(perm) != rank or any(type(n) is not int or not -rank <= n < rank for n in perm):
        raise ValueError("transpose permutation must cover each input axis")
    perm = tuple(n % rank for n in perm)
    if len(set(perm)) != rank:
        raise ValueError("transpose permutation has duplicate axes")
    shape = tuple(value.shape[n] for n in perm)
    if result_type.dtype != value.dtype or result_type.shape != shape:
        raise ValueError("transpose result shape differs from its declaration")
    strides = tuple(math.prod(value.shape[n + 1:]) for n in range(rank))
    result = []
    for flat in range(math.prod(shape)):
        source, remainder = 0, flat
        for n, axis in enumerate(perm):
            coordinate, remainder = divmod(remainder, math.prod(shape[n + 1:]))
            source += coordinate * strides[axis]
        result.append(value.values[source])
    return Tensor(value.dtype, shape, tuple(result))


def _conv1d(value, weight, bias, strides, dilations, pad, pad_type, groups, result_type):
    """CPU rank-3 convolution, including the exact Whisper custom pad spell."""
    value, weight = _tensor(value), _tensor(weight)
    if value.dtype != "fp16" or weight.dtype != "fp16" or len(value.shape) != 3 or len(weight.shape) != 3:
        raise ValueError("conv1d requires rank-3 fp16 tensors")
    if len(strides) != 1 or strides[0] <= 0 or dilations != (1,):
        raise ValueError("conv1d reference requires a positive stride and unit dilation")
    batches, channels, length = value.shape
    outputs, group_channels, kernel = weight.shape
    if groups <= 0 or channels % groups or outputs % groups or group_channels != channels // groups:
        raise ValueError("conv1d groups must divide both channel counts")
    if pad_type == "custom":
        if len(pad) != 2 or any(n < 0 for n in pad):
            raise ValueError("conv1d custom pad must be two nonnegative values")
        left = pad[0]
        out_length = (length + sum(pad) - kernel) // strides[0] + 1
    else:
        left, out_length = _conv_padding(length, kernel, strides[0], pad_type)
    shape = (batches, outputs, out_length)
    if out_length <= 0 or result_type != MILType("fp16", shape):
        raise ValueError("conv1d result shape differs from its declaration")
    if bias is not None and (_tensor(bias).dtype != "fp16" or bias.shape != (outputs,)):
        raise ValueError("conv1d bias must hold one value per output channel")
    result = []
    for batch in range(batches):
        for channel in range(outputs):
            base_channel = (channel // (outputs // groups)) * group_channels
            for column in range(out_length):
                total = 0.0
                for offset in range(group_channels):
                    for tap in range(kernel):
                        source = column * strides[0] + tap - left
                        if 0 <= source < length:
                            sample = value.values[(batch * channels + base_channel + offset) * length + source]
                            w = weight.values[(channel * group_channels + offset) * kernel + tap]
                            total = fp32(total + fp32(float(sample) * float(w)))
                if bias is not None:
                    total = fp32(total + float(bias.values[channel]))
                result.append(fp16(total))
    return Tensor("fp16", shape, tuple(result))


def _shape_op(name, value, parameter, result_type):
    value = _tensor(value)
    requested = (tuple(index for index, dimension in enumerate(value.shape) if dimension == 1)
                 if name == "squeeze" and parameter is None else
                 tuple(int(item) for item in _tensor(parameter).values))
    if name == "reshape":
        if requested.count(-1) > 1 or any(n == 0 or n < -1 for n in requested):
            raise ValueError("reshape requires positive dimensions and at most one -1")
        if -1 in requested:
            known = math.prod(n for n in requested if n != -1)
            count = math.prod(value.shape)
            if count % known:
                raise ValueError("reshape inferred dimension is not integral")
            requested = tuple(count // known if n == -1 else n for n in requested)
        if requested != result_type.shape:
            raise ValueError("reshape shape differs from its declaration")
    elif name == "squeeze":
        axes = tuple(index if index >= 0 else index + len(value.shape) for index in requested)
        expected = tuple(dimension for index, dimension in enumerate(value.shape) if index not in axes)
        if any(value.shape[index] != 1 for index in axes) or expected != result_type.shape:
            raise ValueError("squeeze axes differ from its declaration")
    else:
        rank = len(value.shape) + len(requested)
        axes = tuple(index if index >= 0 else index + rank for index in requested)
        shape = list(value.shape)
        for index in sorted(axes):
            shape.insert(index, 1)
        if tuple(shape) != result_type.shape:
            raise ValueError("expand_dims axes differ from its declaration")
    if math.prod(value.shape) != math.prod(result_type.shape):
        raise ValueError(f"{name} changes the element count")
    return Tensor(value.dtype, result_type.shape, value.values)


def _reduction_groups(shape, axes):
    """The flat input indices contributing to each flat output element, and the
    keep-dims output shape, for a reduction over `axes`."""
    rank = len(shape)
    resolved = set()
    for axis in axes:
        index = axis if axis >= 0 else axis + rank
        if not 0 <= index < rank or index in resolved:
            raise ValueError("reduction axes must be distinct and in range")
        resolved.add(index)
    if not resolved:
        raise ValueError("reduction requires at least one axis")
    kept_shape = tuple(1 if index in resolved else dimension
                       for index, dimension in enumerate(shape))
    strides = []
    stride = 1
    for dimension in reversed(shape):
        strides.append(stride)
        stride *= dimension
    strides.reverse()
    kept_strides = []
    stride = 1
    for dimension in reversed(kept_shape):
        kept_strides.append(stride)
        stride *= dimension
    kept_strides.reverse()
    groups = [[] for _ in range(math.prod(kept_shape))]
    for flat in range(math.prod(shape)):
        target = 0
        for index, dimension in enumerate(shape):
            if index in resolved:
                continue
            target += (flat // strides[index]) % dimension * kept_strides[index]
        groups[target].append(flat)
    return kept_shape, groups


def _sum_fp32(values):
    total = 0.0
    for value in values:
        total = fp32(total + float(value))
    return total


def _reduce(name, value, axes, keep_dims, result_type):
    value = _tensor(value)
    if value.dtype != "fp16":
        raise ValueError(f"{name} requires an fp16 tensor")
    kept_shape, groups = _reduction_groups(value.shape, axes)
    expected = kept_shape if keep_dims else tuple(
        dimension for dimension, kept in zip(value.shape, kept_shape)
        if dimension == kept)
    if result_type.shape != expected:
        raise ValueError(f"{name} result shape differs from its declaration")
    results = []
    for group in groups:
        elements = [value.values[index] for index in group]
        if name == "reduce_max":
            results.append(fp16(max(elements)))
        elif name == "reduce_sum":
            results.append(fp16(_sum_fp32(elements)))
        else:
            results.append(fp16(fp32(_sum_fp32(elements) / len(elements))))
    return Tensor("fp16", result_type.shape, tuple(results))


def _softmax(value, axis, result_type):
    """Max-subtracted softmax accumulated in float32.

    Apple's H13 program evaluates the exponential and the reciprocal through
    fp16 lookup tables, so device results differ from this reference by those
    tables' interpolation error. Parity is asserted on the emitted bytes;
    numerical agreement is a tolerance, documented in README.
    """
    value = _tensor(value)
    if value.dtype != "fp16":
        raise ValueError("softmax requires an fp16 tensor")
    if result_type.shape != value.shape:
        raise ValueError("softmax result shape differs from its declaration")
    _, groups = _reduction_groups(value.shape, (axis,))
    results = [0.0] * len(value.values)
    for group in groups:
        elements = [float(value.values[index]) for index in group]
        peak = max(elements)
        shifted = [fp32(math.exp(fp32(element - peak))) for element in elements]
        total = _sum_fp32(shifted)
        for index, exponential in zip(group, shifted):
            results[index] = fp16(fp32(exponential / total))
    return Tensor("fp16", result_type.shape, tuple(results))


def _layer_norm(value, axes, epsilon, result_type, gamma=None, beta=None):
    """Zero-mean unit-variance normalization with float32 statistics.

    Optional affine parameters are applied in fp32 before the final fp16
    rounding. Reference support does not authorize hardware normalization.
    """
    value = _tensor(value)
    if value.dtype != "fp16":
        raise ValueError("layer_norm requires an fp16 tensor")
    if result_type.shape != value.shape:
        raise ValueError("layer_norm result shape differs from its declaration")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("layer_norm epsilon must be finite and positive")
    if not value.shape:
        raise ValueError("layer_norm requires a positive-rank tensor")
    resolved = tuple(axis % len(value.shape) for axis in axes)
    affine_shape = tuple(value.shape[axis] for axis in sorted(resolved))
    for parameter in (gamma, beta):
        if parameter is not None and (_tensor(parameter).dtype != "fp16" or
                                      parameter.shape != affine_shape):
            raise ValueError("layer_norm affine shape differs from reduced axes")
    _, groups = _reduction_groups(value.shape, axes)
    results = [0.0] * len(value.values)
    for group in groups:
        elements = [float(value.values[index]) for index in group]
        mean = fp32(_sum_fp32(elements) / len(elements))
        variance = fp32(_sum_fp32([fp32((element - mean) ** 2)
                                   for element in elements]) / len(elements))
        scale = fp32(1.0 / math.sqrt(fp32(variance + epsilon)))
        for offset, (index, element) in enumerate(zip(group, elements)):
            normalized = fp32(fp32(element - mean) * scale)
            if gamma is not None:
                normalized = fp32(normalized * gamma.values[offset])
            if beta is not None:
                normalized = fp32(normalized + beta.values[offset])
            results[index] = fp16(normalized)
    return Tensor("fp16", result_type.shape, tuple(results))


def _conv_padding(extent, kernel, stride, pad_type):
    """The leading pad and output extent of one spatial axis.

    coremltools 9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4,
    `coremltools/converters/mil/mil/ops/defs/iOS15/conv.py`: `same` pads so the
    output covers `ceil(D / stride)` and splits the total padding with the
    smaller half in front, `valid` pads nothing and yields
    `floor((D - kernel) / stride) + 1`. Dilations stay at one here, which is
    the only form the parity encoders cover.
    """
    if pad_type == "valid":
        if extent < kernel:
            raise ValueError("conv kernel does not fit a valid pad_type input")
        return 0, (extent - kernel) // stride + 1
    if pad_type != "same":
        raise ValueError(f"unsupported conv pad_type {pad_type}")
    output = (extent + stride - 1) // stride
    total = max(0, (output - 1) * stride + kernel - extent)
    return total // 2, output


def _conv(value, weight, bias, strides, dilations, pad, pad_type, groups,
          result_type):
    """`conv` over a [1, Cin, H, W] fp16 input with a [Cout, Cin/groups, kh, kw]
    weight, accumulating each output in fp32 exactly as `_matmul` does."""
    value = _tensor(value)
    weight = _tensor(weight)
    if value.dtype != "fp16" or weight.dtype != "fp16":
        raise ValueError("conv requires fp16 tensors")
    if len(value.shape) != 4 or value.shape[0] != 1:
        raise ValueError("conv requires a [1, Cin, H, W] input")
    if len(weight.shape) != 4:
        raise ValueError("conv requires a [Cout, Cin/groups, kh, kw] weight")
    if any(dilation != 1 for dilation in dilations):
        raise ValueError("conv reference covers unit dilations")
    if any(amount != 0 for amount in pad):
        raise ValueError("conv reference covers zero explicit padding")
    _, channels, height, width = value.shape
    outputs, group_channels, kernel_height, kernel_width = weight.shape
    if groups < 1 or channels % groups or outputs % groups or \
            group_channels != channels // groups:
        raise ValueError("conv groups must divide both channel counts")
    top, out_height = _conv_padding(height, kernel_height, strides[0], pad_type)
    left, out_width = _conv_padding(width, kernel_width, strides[1], pad_type)
    expected = (1, outputs, out_height, out_width)
    if tuple(result_type.shape) != expected:
        raise ValueError("conv result shape differs from its declaration")
    biases = _tensor(bias).values if bias is not None else None
    if biases is not None and len(biases) != outputs:
        raise ValueError("conv bias must hold one value per output channel")
    per_group = outputs // groups
    results = []
    for channel in range(outputs):
        group = channel // per_group
        for row in range(out_height):
            for column in range(out_width):
                total = 0.0
                for offset in range(group_channels):
                    source = group * group_channels + offset
                    for tap_row in range(kernel_height):
                        position = row * strides[0] + tap_row - top
                        if position < 0 or position >= height:
                            continue
                        for tap_column in range(kernel_width):
                            span = column * strides[1] + tap_column - left
                            if span < 0 or span >= width:
                                continue
                            sample = value.values[
                                ((source * height) + position) * width + span]
                            tap = weight.values[
                                (((channel * group_channels + offset) *
                                  kernel_height) + tap_row) * kernel_width +
                                tap_column]
                            total = fp32(total + fp32(float(sample) * float(tap)))
                if biases is not None:
                    total = fp32(total + float(biases[channel]))
                results.append(fp16(total))
    return Tensor("fp16", expected, tuple(results))


def conv_tiled(value, weight, bias=None, groups=1, patch_size=64):
    """CPU-only halo reference for stride-one SAME square k1/k3 conv.

    No programs are emitted. Accumulation order within each output is the
    same as the dense reference, making seam and border comparisons exact.
    """
    from spatial_tiles import conv_tiles
    value, weight = _tensor(value), _tensor(weight)
    if len(value.shape) != 4 or value.shape[0] != 1 or len(weight.shape) != 4:
        raise ValueError('spatial.rank-outside-reference-envelope')
    _, channels, height, width = value.shape
    outputs, _, kh, kw = weight.shape
    if not all(map(math.isfinite, value.values)) or not all(map(math.isfinite, weight.values)):
        raise ValueError('spatial.nonfinite-outside-reference-envelope')
    if kh != kw:
        raise ValueError('spatial.kernel-outside-reference-envelope')
    tiles = conv_tiles(height, width, kh, patch_size)
    result = [0.0] * (outputs * height * width)
    for t in tiles:
        patch = []
        for c in range(channels):
            for y in range(t.patch_size):
                for x in range(t.patch_size):
                    sy, sx = t.input_top + y, t.input_left + x
                    patch.append(value.values[(c * height + sy) * width + sx]
                                 if 0 <= sy < height and 0 <= sx < width else 0.0)
        shape = (1, channels, t.patch_size, t.patch_size)
        output = _conv(Tensor('fp16', shape, tuple(patch)), weight, bias,
                       (1, 1), (1, 1), (0, 0, 0, 0), 'same', groups,
                       MILType('fp16', (1, outputs, t.patch_size, t.patch_size)))
        for c in range(outputs):
            for y in range(t.output_height):
                dst = (c * height + t.output_top + y) * width + t.output_left
                src = (c * t.patch_size + t.crop_top + y) * t.patch_size + t.crop_left
                result[dst:dst + t.output_width] = output.values[src:src + t.output_width]
    return Tensor('fp16', (1, outputs, height, width), tuple(result))


def _scalar(value):
    if isinstance(value, Tensor) and value.shape == () and len(value.values) == 1:
        return value.values[0]
    return value


def _concat(values, axis, interleave, result_type):
    if interleave:
        raise ValueError('reference.concat-interleave-outside-envelope')
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError('reference.concat-inputs')
    tensors = [_tensor(v) for v in values]
    rank = len(tensors[0].shape)
    if type(axis) is not int or not -rank <= axis < rank:
        raise ValueError('reference.concat-axis')
    axis %= rank
    shape = list(tensors[0].shape)
    shape[axis] = sum(t.shape[axis] for t in tensors if len(t.shape) == rank)
    if any(t.dtype != 'fp16' or len(t.shape) != rank or
           any(d != t.shape[i] for i, d in enumerate(shape) if i != axis) for t in tensors):
        raise ValueError('reference.concat-shape')
    if result_type.dtype != 'fp16' or tuple(shape) != result_type.shape:
        raise ValueError('reference.concat-result-shape')
    inner, outer = math.prod(shape[axis + 1:]), math.prod(shape[:axis])
    result = []
    for o in range(outer):
        for t in tensors:
            span = t.shape[axis] * inner
            result.extend(t.values[o * span:(o + 1) * span])
    return Tensor('fp16', tuple(shape), tuple(result))


def _resize(value, height, width, mode, result_type):
    value = _tensor(value)
    if value.dtype != 'fp16' or len(value.shape) < 3 or len(value.shape) > 4:
        raise ValueError('reference.resize-type')
    if any(type(v) is not int or v <= 0 for v in (height, width)):
        raise ValueError('reference.resize-size')
    ih, iw = value.shape[-2:]
    shape = value.shape[:-2] + (height, width)
    if result_type.dtype != 'fp16' or result_type.shape != shape:
        raise ValueError('reference.resize-result-shape')
    if mode not in ('nearest', 'DEFAULT', 'STRICT_ALIGN_CORNERS', 'UNALIGN_CORNERS'):
        raise ValueError('reference.resize-sampling-mode-outside-envelope')
    if mode == 'nearest' and (height % ih or width % iw):
        raise ValueError('reference.nearest-fractional-scale-outside-envelope')

    def coordinate(i, n, out):
        if mode == 'STRICT_ALIGN_CORNERS':
            return i * (n - 1) / (out - 1) if out > 1 else 0.0
        x = (i + .5) * n / out - .5 if mode == 'UNALIGN_CORNERS' else i * n / out
        return min(n - 1, max(0.0, x))

    result = []
    for plane in range(math.prod(value.shape[:-2])):
        for y in range(height):
            fy = y * ih // height if mode == 'nearest' else coordinate(y, ih, height)
            y0, y1 = math.floor(fy), min(ih - 1, math.floor(fy) + 1)
            dy = fy - y0
            for x in range(width):
                fx = x * iw // width if mode == 'nearest' else coordinate(x, iw, width)
                x0, x1 = math.floor(fx), min(iw - 1, math.floor(fx) + 1)

                def sample(sy, sx):
                    return value.values[(plane * ih + sy) * iw + sx]

                if mode == 'nearest':
                    result.append(sample(y0, x0))
                else:
                    dx = fx - x0
                    a = fp32(sample(y0, x0) * (1 - dx) + sample(y0, x1) * dx)
                    b = fp32(sample(y1, x0) * (1 - dx) + sample(y1, x1) * dx)
                    result.append(fp16(fp32(a * (1 - dy) + b * dy)))
    return Tensor('fp16', shape, tuple(result))


def _execute(operation, environment, model_root):
    arguments = {name: _resolve(value, environment, model_root)
                 for name, value in operation.arguments.items()}
    attributes = {name: _resolve(value, environment, model_root)
                  for name, value in operation.attributes.items()}
    name = operation.name
    if name == "const":
        if "val" not in attributes:
            raise ValueError("const lacks val")
        value = attributes["val"]
        if operation.result_type.shape:
            return _tensor(value, operation.result_type)
        return value
    if name in ("add", "mul", "maximum", "minimum", "sub", "real_div"):
        return _binary(name, arguments["x"], arguments["y"], operation.result_type)
    if name == "relu":
        return _binary("maximum", arguments["x"], 0.0, operation.result_type)
    if name == 'concat':
        return _concat(arguments['values'], _scalar(arguments['axis']),
                       _scalar(arguments.get('interleave', False)), operation.result_type)
    if name in ('upsample_nearest_neighbor', 'upsample_bilinear', 'resize_nearest_neighbor', 'resize_bilinear'):
        value = _tensor(arguments['x'])
        if name.startswith('upsample'):
            factors = (_scalar(arguments.get('scale_factor_height', 1)),
                       _scalar(arguments.get('scale_factor_width', 1)))
            if any(not math.isfinite(v) or v <= 0 for v in factors):
                raise ValueError('reference.resize-scale')
            if 'nearest' in name and any(int(v) != v for v in factors):
                raise ValueError('reference.nearest-fractional-scale-outside-envelope')
            height, width = (math.floor(n * f) for n, f in zip(value.shape[-2:], factors))
            align = _scalar(arguments.get('align_corners', True))
            mode = 'STRICT_ALIGN_CORNERS' if align else 'UNALIGN_CORNERS'
            if 'half_pixel_centers' in arguments:
                half = bool(_scalar(arguments['half_pixel_centers']))
                if align and half:
                    raise ValueError('reference.resize-coordinate-conflict')
                if not align and not half:
                    mode = 'DEFAULT'
        else:
            height, width = _scalar(arguments['target_size_height']), _scalar(arguments['target_size_width'])
            mode = _scalar(arguments.get('sampling_mode', 'DEFAULT'))
        return _resize(value, height, width, 'nearest' if 'nearest' in name else mode, operation.result_type)
    if name == "clip":
        lower = _binary("maximum", arguments["x"], arguments["alpha"], operation.result_type)
        return _binary("minimum", lower, arguments["beta"], operation.result_type)
    if name == "matmul":
        return _matmul(arguments["x"], arguments["y"], bool(arguments.get("transpose_x", False)),
                       bool(arguments.get("transpose_y", False)), operation.result_type)
    if name == "linear":
        projected_type = MILType("fp16", operation.result_type.shape)
        projected = _matmul(arguments["x"], arguments["weight"], False, True, projected_type)
        return projected if "bias" not in arguments else _binary(
            "add", projected, arguments["bias"], operation.result_type)
    if name in ("reshape", "squeeze", "expand_dims"):
        parameter = arguments.get("shape" if name == "reshape" else "axes")
        return _shape_op(name, arguments["x"], parameter, operation.result_type)
    if name in ("reduce_sum", "reduce_max", "reduce_mean"):
        axes = _tensor(arguments["axes"]).values
        return _reduce(name, arguments["x"], axes,
                       bool(arguments.get("keep_dims", True)),
                       operation.result_type)
    if name == "softmax":
        return _softmax(arguments["x"], int(arguments["axis"]),
                        operation.result_type)
    if name == "layer_norm":
        axes = _tensor(arguments["axes"]).values
        return _layer_norm(arguments["x"], axes,
                           fp32(arguments.get("epsilon", 1e-5)),
                           operation.result_type, arguments.get("gamma"),
                           arguments.get("beta"))
    if name == "transpose":
        return _transpose(arguments["x"], _tensor(arguments["perm"]).values,
                          operation.result_type)
    if name == "gelu":
        if arguments.get("mode", "EXACT") != "EXACT":
            raise ValueError("reference.gelu-mode-outside-envelope")
        value = _tensor(arguments["x"])
        if value.dtype != "fp16" or operation.result_type != MILType("fp16", value.shape):
            raise ValueError("gelu result shape differs from its declaration")
        return Tensor("fp16", value.shape, tuple(fp16(fp32(
            0.5 * float(x) * (1.0 + math.erf(float(x) / math.sqrt(2.0)))))
            for x in value.values))
    if name == "conv":
        if len(_tensor(arguments["x"]).shape) == 3:
            return _conv1d(arguments["x"], arguments["weight"], arguments.get("bias"),
                tuple(_tensor(arguments["strides"]).values),
                tuple(_tensor(arguments["dilations"]).values),
                tuple(_tensor(arguments["pad"]).values), str(arguments["pad_type"]),
                int(arguments["groups"]), operation.result_type)
        return _conv(arguments["x"], arguments["weight"],
                     arguments.get("bias"),
                     tuple(_tensor(arguments["strides"]).values),
                     tuple(_tensor(arguments["dilations"]).values),
                     tuple(_tensor(arguments["pad"]).values),
                     str(arguments["pad_type"]), int(arguments["groups"]),
                     operation.result_type)
    raise ValueError(f"unsupported H13 MIL operation {name}")


def evaluate(mil_text, model_root, inputs):
    """Evaluate one MIL function and return its named outputs as dense fp16 bytes."""
    parameters, operations, returns = Parser(mil_text).parse()
    expected_names = {name for _, name in parameters}
    if set(inputs) != expected_names:
        raise ValueError(f"inputs must be exactly {', '.join(sorted(expected_names))}")
    environment = {name: _tensor(inputs[name], value_type)
                   for value_type, name in parameters}
    for operation in operations:
        if operation.result in environment:
            raise ValueError(f"duplicate MIL value {operation.result}")
        environment[operation.result] = _execute(operation, environment, Path(model_root))
    outputs = {}
    for name in returns:
        value = environment.get(name)
        if not isinstance(value, Tensor) or value.dtype != "fp16":
            raise ValueError("H13 reference outputs must be fp16 tensors")
        outputs[name] = encode_fp16(value.values)
    return outputs


def _bindings(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("bindings must use NAME=PATH")
        name, path = item.split("=", 1)
        if not name or name in result:
            raise ValueError("binding names must be unique and non-empty")
        result[name] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mil", type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--input", action="append", default=[], metavar="NAME=DENSE_FP16")
    parser.add_argument("--output", action="append", default=[], metavar="NAME=DENSE_FP16")
    args = parser.parse_args()
    try:
        input_paths = _bindings(args.input)
        output_paths = _bindings(args.output)
        outputs = evaluate(args.mil.read_text(), args.model_root,
                           {name: path.read_bytes() for name, path in input_paths.items()})
        if set(output_paths) != set(outputs):
            raise ValueError(f"outputs must be exactly {', '.join(sorted(outputs))}")
        for name, path in output_paths.items():
            path.write_bytes(outputs[name])
            print(f"wrote {len(outputs[name])} fp16 bytes to {path}")
    except (OSError, ValueError, struct.error) as error:
        parser.exit(1, f"H13 reference error: {error}\n")


if __name__ == "__main__":
    main()
