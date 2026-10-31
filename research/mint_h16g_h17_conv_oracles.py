#!/usr/bin/env python3
"""Mint the focused H13/H16G/H17 MobileNetV1 convolution campaign.

The compiler-produced HWX object only exists in a temporary directory.  The
output is the same decoded-facts JSON used by ``mint_oracles.py``: hashes,
container metadata, task headers and register writes, never Apple bytes.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402


def conv_stage(name: str, input_name: str, output_name: str, channels_in: int,
               channels_out: int, spatial: int, kernel: int, stride: int,
               groups: int, relu: bool, bias: bool = False
               ) -> tuple[list[str], tuple[int, ...], int]:
    output_spatial = (spatial + stride - 1) // stride
    input_shape = (1, channels_in, spatial, spatial)
    output_shape = (1, channels_out, output_spatial, output_spatial)
    weight_shape = (channels_out, channels_in // groups, kernel, kernel)
    weight, elements = om.blobfile(f"{name}_w", weight_shape)
    bias_arg = ""
    intermediate = f"{name}_conv" if relu else output_name
    lines = [
        f'string {name}_pt = const()[name = string("{name}_pt"), '
        'val = string("same")];',
        f'tensor<int32, [2]> {name}_st = const()[name = string("{name}_st"), '
        f'val = tensor<int32, [2]>([{stride}, {stride}])];',
        f'tensor<int32, [4]> {name}_pd = const()[name = string("{name}_pd"), '
        'val = tensor<int32, [4]>([0, 0, 0, 0])];',
        f'tensor<int32, [2]> {name}_dl = const()[name = string("{name}_dl"), '
        'val = tensor<int32, [2]>([1, 1])];',
        f'int32 {name}_gp = const()[name = string("{name}_gp"), '
        f'val = int32({groups})];',
        weight,
    ]
    if bias:
        bias_shape = (channels_out,)
        bias_source, bias_elements = om.blobfile(f"{name}_b", bias_shape)
        lines.append(bias_source)
        bias_arg = f", bias = {name}_b"
        elements = max(elements, bias_elements)
    lines.append(
        f'{om.tensor_type(output_shape)} {intermediate} = conv('
        f'dilations = {name}_dl, groups = {name}_gp, pad = {name}_pd, '
        f'pad_type = {name}_pt, strides = {name}_st, weight = {name}_w, '
        f'x = {input_name}{bias_arg})[name = string("{intermediate}")];')
    if relu:
        lines.append(
            f'{om.tensor_type(output_shape)} {output_name} = relu('
            f'x = {intermediate})[name = string("{output_name}")];')
    return lines, weight_shape, elements


def single(name: str, channels_in: int, channels_out: int, spatial: int,
           kernel: int, stride: int = 1, groups: int = 1,
           relu: bool = False, bias: bool = False) -> dict:
    lines, weight_shape, elements = conv_stage(
        "s0", "x", "y", channels_in, channels_out, spatial, kernel, stride,
        groups, relu, bias)
    input_shape = (1, channels_in, spatial, spatial)
    output_spatial = (spatial + stride - 1) // stride
    output_shape = (1, channels_out, output_spatial, output_spatial)
    return om.env_case(name, "conv_descriptor", {
        "operations": ["conv", *( ["relu"] if relu else [])],
        "kernel": kernel, "input_channels": channels_in,
        "output_channels": channels_out, "spatial": spatial,
        "output_spatial": output_spatial, "stride": stride,
        "groups": groups, "relu": relu, "bias": bias,
        "input_shape": input_shape,
        "output_shape": output_shape, "weight_shape": weight_shape,
    }, f"{om.tensor_type(input_shape)} x", lines, "y", elements,
       [weight_shape])


def chain2() -> dict:
    input_shape = (1, 64, 112, 112)
    first, w0, n0 = conv_stage(
        "s0", "x", "a", 64, 64, 112, 3, 2, 64, True)
    second, w1, n1 = conv_stage(
        "s1", "a", "y", 64, 128, 56, 1, 1, 1, True)
    return om.env_case("conv_desc_chain2_dw64s2_pw128_s112_relu",
                       "conv_descriptor_chain", {
        "operations": ["depthwise_conv", "relu", "pointwise_conv", "relu"],
        "input_shape": input_shape, "output_shape": (1, 128, 56, 56),
        "weight_shapes": [w0, w1],
    }, f"{om.tensor_type(input_shape)} x", first + second, "y", max(n0, n1),
       [w0, w1])


def chain3() -> dict:
    input_shape = (1, 32, 112, 112)
    first, w0, n0 = conv_stage(
        "s0", "x", "a", 32, 64, 112, 1, 1, 1, True)
    second, w1, n1 = conv_stage(
        "s1", "a", "b", 64, 64, 112, 3, 2, 64, True)
    third, w2, n2 = conv_stage(
        "s2", "b", "y", 64, 128, 56, 1, 1, 1, True)
    return om.env_case("conv_desc_chain3_pw64_dw64s2_pw128_s112_relu",
                       "conv_descriptor_chain", {
        "operations": ["pointwise_conv", "relu", "depthwise_conv", "relu",
                       "pointwise_conv", "relu"],
        "input_shape": input_shape, "output_shape": (1, 128, 56, 56),
        "weight_shapes": [w0, w1, w2],
    }, f"{om.tensor_type(input_shape)} x", first + second + third, "y",
       max(n0, n1, n2), [w0, w1, w2])


def chain(name: str, input_channels: int, spatial: int,
          stages: list[tuple[int, int, int, int, bool]]) -> dict:
    """Build a linked MobileNet chain.

    Each stage is ``(Cout, kernel, stride, groups, bias)``.  ReLU follows
    every convolution, matching the measured MobileNetV1 body/tail forms.
    A single uniform BLOBFILE payload is deliberately reused by every stage;
    only decoded facts and section hashes are retained by the minting path.
    """
    lines: list[str] = []
    weight_shapes: list[tuple[int, ...]] = []
    operations: list[str] = []
    elements = 0
    cin = input_channels
    current_spatial = spatial
    input_name = "x"
    for index, (cout, kernel, stride, groups, bias) in enumerate(stages):
        output_name = "y" if index == len(stages) - 1 else f"v{index}"
        stage_lines, weight_shape, stage_elements = conv_stage(
            f"s{index}", input_name, output_name, cin, cout,
            current_spatial, kernel, stride, groups, True, bias)
        lines.extend(stage_lines)
        weight_shapes.append(weight_shape)
        operations.extend([
            "depthwise_conv" if groups == cin == cout else "pointwise_conv",
            "relu",
        ])
        elements = max(elements, stage_elements)
        current_spatial = (current_spatial + stride - 1) // stride
        cin = cout
        input_name = output_name
    input_shape = (1, input_channels, spatial, spatial)
    output_shape = (1, cin, current_spatial, current_spatial)
    return om.env_case(name, "conv_descriptor_chain", {
        "operations": operations,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "weight_shapes": weight_shapes,
        "stage_count": len(stages),
    }, f"{om.tensor_type(input_shape)} x", lines, "y", elements,
       weight_shapes)


def campaign() -> list[dict]:
    return [
        single("conv_desc_cmp_pw_c64_n64_s64", 64, 64, 64, 1),
        single("conv_desc_pw_c32_n64_s112", 32, 64, 112, 1),
        single("conv_desc_pw_c64_n128_s112", 64, 128, 112, 1),
        single("conv_desc_reg_k3_c3_n32_s224_st2", 3, 32, 224, 3, 2),
        single("conv_desc_reg_k3_c64_n64_s112", 64, 64, 112, 3),
        single("conv_desc_dw_k3_c32_s112", 32, 32, 112, 3, groups=32),
        single("conv_desc_dw_k3_c64_s112_st2", 64, 64, 112, 3, 2, 64),
        single("conv_desc_dw_k3_c128_s56_st2", 128, 128, 56, 3, 2, 128),
        single("conv_desc_pw_c64_n128_s56_relu", 64, 128, 56, 1, relu=True),
        single("conv_desc_stem_k3_c3_n32_s224_st2_relu", 3, 32, 224, 3, 2,
               relu=True),
        chain2(),
        chain3(),
        # MobileNetV1 224 body/tail forms omitted by the first campaign.
        single("conv_desc_pw_c128_n256_s28", 128, 256, 28, 1),
        single("conv_desc_pw_c256_n256_s28", 256, 256, 28, 1),
        single("conv_desc_pw_c256_n512_s14", 256, 512, 14, 1),
        single("conv_desc_pw_c512_n512_s14", 512, 512, 14, 1),
        single("conv_desc_pw_c512_n1024_s7", 512, 1024, 7, 1),
        single("conv_desc_pw_c1024_n1024_s7", 1024, 1024, 7, 1),
        single("conv_desc_dw_k3_c256_s28", 256, 256, 28, 3,
               groups=256),
        single("conv_desc_dw_k3_c256_s28_st2", 256, 256, 28, 3, 2, 256),
        single("conv_desc_dw_k3_c512_s14", 512, 512, 14, 3,
               groups=512),
        single("conv_desc_dw_k3_c512_s14_st2", 512, 512, 14, 3, 2, 512),
        single("conv_desc_dw_k3_c1024_s7", 1024, 1024, 7, 3,
               groups=1024),
        single("conv_desc_pw_c512_n1024_s7_bias", 512, 1024, 7, 1,
               bias=True),
        chain("conv_desc_chain6_mob_c128_s28", 128, 28, [
            (256, 1, 1, 1, False),
            (256, 3, 1, 256, False),
            (256, 1, 1, 1, False),
            (256, 3, 2, 256, False),
            (512, 1, 1, 1, False),
            (512, 3, 1, 512, False),
        ]),
        chain("conv_desc_chain_tail_c512_s14", 512, 14, [
            (512, 3, 1, 512, False),
            (512, 1, 1, 1, False),
            (512, 3, 2, 512, False),
            (1024, 1, 1, 1, False),
            (1024, 3, 1, 1024, False),
            (1024, 1, 1, 1, True),
        ]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--targets", nargs="+", choices=("h13", "h16", "h17"),
                        default=("h13", "h16", "h17"))
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    parser.add_argument("--host-label", default="macos-oracle-host",
                        help="non-identifying host label retained in public JSON")
    args = parser.parse_args()
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        print("\n".join(item["name"] for item in selected))
        return 0
    output = Path(args.output)
    tool = Path(args.oracle_tool)
    decoded = rejected = 0
    for target in args.targets:
        for item in selected:
            destination = output / target / f'{item["name"]}.json'
            if destination.exists() and not args.force:
                continue
            status = om.run_case(item, target, output, tool, args.source_commit)
            # The tool hash and platform are reproducibility facts.  Local
            # machine names and temporary paths are not public-source facts.
            record = json.loads(destination.read_text())
            record["compiler"]["host"] = args.host_label
            record["compiler"]["tool"] = "LOCAL_ORACLE_TOOL"
            record["compiler"]["command"][0] = "LOCAL_ORACLE_TOOL"
            destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            decoded += status == "decoded"
            rejected += status == "rejected"
            print(f'{target} {item["name"]} {status}', flush=True)
    print(f"SUMMARY decoded={decoded} rejected={rejected}")
    return 0 if rejected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
