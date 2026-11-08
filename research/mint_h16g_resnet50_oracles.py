#!/usr/bin/env python3
"""Mint the public-clean H16G ResNet-50 field campaign.

The Apple compiler output exists only inside ``mint_oracles.run_case``'s
temporary directory.  Persisted output is decoded facts, hashes, source MIL,
and compiler metadata; no HWX bytes are retained.  The 23 convolution rows are
the unique geometries of ``tools/resnet_fp16_reference.py``'s exported model.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402


# Cin, Cout, input spatial, kernel, stride.  All exported ResNet convolutions
# are dense, biased, batch-one fp16 and SAME padded.
CONV_GEOMETRIES = (
    (3, 64, 224, 7, 2),
    (64, 64, 56, 1, 1),
    (64, 64, 56, 3, 1),
    (64, 256, 56, 1, 1),
    (256, 64, 56, 1, 1),
    (256, 128, 56, 1, 1),
    (128, 128, 56, 3, 2),
    (128, 512, 28, 1, 1),
    (256, 512, 56, 1, 2),
    (512, 128, 28, 1, 1),
    (128, 128, 28, 3, 1),
    (512, 256, 28, 1, 1),
    (256, 256, 28, 3, 2),
    (256, 1024, 14, 1, 1),
    (512, 1024, 28, 1, 2),
    (1024, 256, 14, 1, 1),
    (256, 256, 14, 3, 1),
    (1024, 512, 14, 1, 1),
    (512, 512, 14, 3, 2),
    (512, 2048, 7, 1, 1),
    (1024, 2048, 14, 1, 2),
    (2048, 512, 7, 1, 1),
    (512, 512, 7, 3, 1),
)


def name_for_conv(cin: int, cout: int, spatial: int, kernel: int,
                  stride: int, relu: bool) -> str:
    return (f"resnet_conv_k{kernel}_c{cin}_n{cout}_s{spatial}_st{stride}"
            f"_bias1_relu{int(relu)}")


def conv_case(cin: int, cout: int, spatial: int, kernel: int, stride: int,
              relu: bool = False) -> dict:
    output_spatial = (spatial + stride - 1) // stride
    x_shape = (1, cin, spatial, spatial)
    y_shape = (1, cout, output_spatial, output_spatial)
    w_shape = (cout, cin, kernel, kernel)
    weight, weight_elements = om.blobfile("w", w_shape)
    bias, bias_elements = om.blobfile("b", (cout,))
    convolution = "c" if relu else "y"
    body = [
        'string pt = const()[name = string("pt"), val = string("same")];',
        'tensor<int32, [2]> st = const()[name = string("st"), '
        f'val = tensor<int32, [2]>([{stride}, {stride}])];',
        'tensor<int32, [4]> pd = const()[name = string("pd"), '
        'val = tensor<int32, [4]>([0, 0, 0, 0])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), '
        'val = tensor<int32, [2]>([1, 1])];',
        'int32 gp = const()[name = string("gp"), val = int32(1)];',
        weight, bias,
        f'{om.tensor_type(y_shape)} {convolution} = conv(bias = b, '
        'dilations = dl, groups = gp, pad = pd, pad_type = pt, '
        'strides = st, weight = w, x = x)'
        f'[name = string("{convolution}")];',
    ]
    if relu:
        body.append(f'{om.tensor_type(y_shape)} y = relu(x = c)'
                    '[name = string("y")];')
    return om.env_case(name_for_conv(cin, cout, spatial, kernel, stride, relu),
        "resnet_conv", {
            "kernel": kernel, "input_channels": cin,
            "output_channels": cout, "spatial": spatial,
            "output_spatial": output_spatial, "stride": stride,
            "groups": 1, "bias": True, "relu": relu,
            "input_shape": x_shape, "output_shape": y_shape,
            "weight_shape": w_shape,
        }, f"{om.tensor_type(x_shape)} x", body, "y",
        max(weight_elements, bias_elements), [w_shape, (cout,)])


def max_pool_case() -> dict:
    x_shape, y_shape = (1, 64, 112, 112), (1, 64, 56, 56)
    body = [
        'tensor<int32, [2]> ks = const()[name = string("ks"), '
        'val = tensor<int32, [2]>([3, 3])];',
        'tensor<int32, [2]> st = const()[name = string("st"), '
        'val = tensor<int32, [2]>([2, 2])];',
        'string pt = const()[name = string("pt"), val = string("same")];',
        f'{om.tensor_type(y_shape)} y = max_pool(x = x, kernel_sizes = ks, '
        'strides = st, pad_type = pt)[name = string("y")];',
    ]
    return om.env_case("resnet_max_pool_c64_s112_k3_st2_same", "resnet_pool", {
        "operation": "max_pool", "input_shape": x_shape,
        "output_shape": y_shape, "kernel": (3, 3), "stride": (2, 2),
        "padding": "same",
    }, f"{om.tensor_type(x_shape)} x", body, "y")


def add_relu_case(channels: int, spatial: int) -> dict:
    shape = (1, channels, spatial, spatial)
    kind = om.tensor_type(shape)
    body = [
        f'{kind} sum = add(x = body, y = skip)[name = string("sum")];',
        f'{kind} y = relu(x = sum)[name = string("y")];',
    ]
    return om.env_case(f"resnet_add_relu_c{channels}_s{spatial}",
        "resnet_residual", {"operations": ["add", "relu"], "shape": shape},
        f"{kind} body, {kind} skip", body, "y")


def bottleneck_case(projection: bool) -> dict:
    """One complete 56x56 residual block, with or without projection."""
    input_channels = 64 if projection else 256
    x_shape = (1, input_channels, 56, 56)
    mid_shape = (1, 64, 56, 56)
    out_shape = (1, 256, 56, 56)
    body: list[str] = []
    shapes: list[tuple[int, ...]] = []
    largest = 0

    def stage(tag: str, source: str, cin: int, cout: int, kernel: int,
              result: str, relu: bool) -> None:
        nonlocal largest
        w_shape = (cout, cin, kernel, kernel)
        weight, weight_elements = om.blobfile(f"{tag}_w", w_shape)
        bias, bias_elements = om.blobfile(f"{tag}_b", (cout,))
        shapes.extend((w_shape, (cout,)))
        largest = max(largest, weight_elements, bias_elements)
        conv_result = f"{result}_conv" if relu else result
        result_shape = mid_shape if cout == 64 else out_shape
        body.extend((weight, bias,
            f'{om.tensor_type(result_shape)} {conv_result} = conv('
            f'bias = {tag}_b, dilations = dl, groups = gp, pad = pd, '
            f'pad_type = pt, strides = st, weight = {tag}_w, x = {source})'
            f'[name = string("{conv_result}")];'))
        if relu:
            body.append(f'{om.tensor_type(result_shape)} {result} = relu('
                        f'x = {conv_result})[name = string("{result}")];')

    body.extend([
        'string pt = const()[name = string("pt"), val = string("same")];',
        'tensor<int32, [2]> st = const()[name = string("st"), '
        'val = tensor<int32, [2]>([1, 1])];',
        'tensor<int32, [4]> pd = const()[name = string("pd"), '
        'val = tensor<int32, [4]>([0, 0, 0, 0])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), '
        'val = tensor<int32, [2]>([1, 1])];',
        'int32 gp = const()[name = string("gp"), val = int32(1)];',
    ])
    stage("a", "x", input_channels, 64, 1, "a", True)
    stage("b", "a", 64, 64, 3, "b", True)
    stage("c", "b", 64, 256, 1, "c", False)
    skip = "x"
    if projection:
        stage("skip", "x", 64, 256, 1, "skip", False)
    body.extend([
        f'{om.tensor_type(out_shape)} sum = add(x = c, y = {skip})'
        '[name = string("sum")];',
        f'{om.tensor_type(out_shape)} y = relu(x = sum)[name = string("y")];',
    ])
    kind = "projection" if projection else "identity"
    operations = ["conv", "relu", "conv", "relu", "conv"]
    if projection:
        operations.append("conv")
    operations.extend(("add", "relu"))
    return om.env_case(f"resnet_bottleneck_{kind}_c256_s56",
        "resnet_bottleneck", {
            "operations": operations, "projection": projection,
            "input_shape": x_shape, "output_shape": out_shape,
        }, f"{om.tensor_type(x_shape)} x", body, "y", largest, shapes)


def reduce_mean_case() -> dict:
    x_shape, y_shape = (1, 2048, 7, 7), (1, 2048)
    body = [
        'tensor<int32, [2]> axes = const()[name = string("axes"), '
        'val = tensor<int32, [2]>([2, 3])];',
        f'{om.tensor_type(y_shape)} y = reduce_mean(x = x, axes = axes, '
        'keep_dims = bool(false))[name = string("y")];',
    ]
    return om.env_case("resnet_reduce_mean_c2048_s7_axes23", "resnet_head", {
        "operation": "reduce_mean", "input_shape": x_shape,
        "output_shape": y_shape, "axes": (2, 3), "keep_dims": False,
    }, f"{om.tensor_type(x_shape)} x", body, "y")


def linear_case() -> dict:
    x_shape, y_shape, w_shape = (1, 2048), (1, 1000), (1000, 2048)
    weight, weight_elements = om.blobfile("w", w_shape)
    bias, bias_elements = om.blobfile("b", (1000,))
    body = [weight, bias,
        f'{om.tensor_type(y_shape)} y = linear(x = x, weight = w, bias = b)'
        '[name = string("y")];']
    return om.env_case("resnet_linear_k2048_n1000_bias1", "resnet_head", {
        "operation": "linear", "input_shape": x_shape,
        "output_shape": y_shape, "weight_shape": w_shape, "bias": True,
    }, f"{om.tensor_type(x_shape)} x", body, "y",
        max(weight_elements, bias_elements), [w_shape, (1000,)])


def campaign() -> list[dict]:
    # Both activation states are required: the first two bottleneck
    # convolutions fuse ReLU, while projections and bottleneck outputs do not.
    cases = [conv_case(*geometry, relu=relu)
             for geometry in CONV_GEOMETRIES for relu in (False, True)]
    cases.append(max_pool_case())
    cases.extend(add_relu_case(channels, spatial) for channels, spatial in
                 ((256, 56), (512, 28), (1024, 14), (2048, 7)))
    cases.extend((bottleneck_case(True), bottleneck_case(False)))
    cases.extend((reduce_mean_case(), linear_case()))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles/resnet50")
    parser.add_argument("--target", choices=("h16",), default="h16")
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    parser.add_argument("--host-label", default="macos-oracle-host")
    args = parser.parse_args()
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        print("\n".join(item["name"] for item in selected))
        print(f"cases={len(selected)}")
        return 0
    output, tool = Path(args.output), Path(args.oracle_tool)
    decoded = rejected = 0
    for item in selected:
        destination = output / args.target / f'{item["name"]}.json'
        if destination.exists() and not args.force:
            continue
        status = om.run_case(item, args.target, output, tool, args.source_commit)
        record = json.loads(destination.read_text())
        record["compiler"]["host"] = args.host_label
        record["compiler"]["tool"] = "LOCAL_ORACLE_TOOL"
        record["compiler"]["command"][0] = "LOCAL_ORACLE_TOOL"
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        decoded += status == "decoded"
        rejected += status == "rejected"
        print(f'{args.target} {item["name"]} {status}', flush=True)
    print(f"SUMMARY decoded={decoded} rejected={rejected}")
    return 0 if rejected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
