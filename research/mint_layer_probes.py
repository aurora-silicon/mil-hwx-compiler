#!/usr/bin/env python3
"""Compile-only H13/H14 layer probes; retain decoded facts and hashes only.

Uses the temporary-container lifecycle of mint_oracles.run_case. No ANE work
is submitted. Supply an existing Apple compile tool with --tool.
MIL resize semantics: apple/coremltools, iOS15/image_resizing.py (BSD-3-Clause).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path

import mint_oracles as om
from mint_conv_probes import conv_case, finite_index_payload
from layer_graph_cases import graph_case


def resize_case(op, c, h, w, ho, wo, mode=None):
    inp, out = (1, c, h, w), (1, c, ho, wo)
    body = []
    if op.startswith("upsample"):
        args = f"scale_factor_height = fp32({ho / h}), scale_factor_width = fp32({wo / w})"
        if op == "upsample_bilinear":
            args += f", align_corners = bool({str(mode == 'align').lower()})"
    else:
        args = f"target_size_height = int32({ho}), target_size_width = int32({wo})"
        if op == "resize_bilinear":
            args += f', sampling_mode = string("{mode}")'
    body.append(f'{om.tensor_type(out)} y = {op}(x = x, {args})[name = string("y")];')
    name = f"layer_{op}_c{c}_{h}x{w}_{ho}x{wo}" + (f"_{mode}" if mode else "")
    return om.case(name, "layer_resize", dict(operation=op, input_shape=inp,
                   output_shape=out, mode=mode),
                   om.program(f"{om.tensor_type(inp)} x", body, "y"))


def unary_case(op, c, h, w):
    shape = (1, c, h, w)
    return om.case(f"layer_{op}_c{c}_{h}x{w}", "layer_activation",
                   dict(operation=op, input_shape=shape, output_shape=shape),
                   om.program(f"{om.tensor_type(shape)} x", [
                       f'{om.tensor_type(shape)} y = {op}(x = x)[name = string("y")];'], "y"))


def hardswish_case(c, h, w):
    shape = (1, c, h, w)
    kind = om.tensor_type(shape)
    return om.case(f"layer_hardswish_explicit_c{c}_{h}x{w}", "layer_activation",
                   dict(operation="hardswish", input_shape=shape, output_shape=shape),
                   om.program(f"{kind} x", [
                       'fp16 alpha = const()[name = string("alpha"), val = fp16(0x1.554p-3)];',
                       'fp16 beta = const()[name = string("beta"), val = fp16(0x1p-1)];',
                       f'{kind} h = sigmoid_hard(x = x, alpha = alpha, beta = beta)[name = string("h")];',
                       f'{kind} y = mul(x = x, y = h)[name = string("y")];'], "y"))


def concat_case(c0, c1, h, w):
    a, b, out = (1, c0, h, w), (1, c1, h, w), (1, c0 + c1, h, w)
    return om.case(f"layer_concat_c{c0}_{c1}_{h}x{w}", "layer_concat",
                   dict(input_shapes=[a, b], output_shape=out, axis=1),
                   om.program(f"{om.tensor_type(a)} x, {om.tensor_type(b)} z", [
                       f'{om.tensor_type(out)} y = concat(values = (x, z), axis = int32(1), interleave = bool(false))[name = string("y")];'], "y"))


def hardswish_arithmetic_case(c, h, w):
    shape = (1, c, h, w)
    kind = om.tensor_type(shape)
    return om.case(f"layer_hardswish_arithmetic_c{c}_{h}x{w}", "layer_activation",
                   dict(operation="hardswish", input_shape=shape, output_shape=shape),
                   om.program(f"{kind} x", [
                       'fp16 three = const()[name = string("three"), val = fp16(0x1.8p1)];',
                       'fp16 six = const()[name = string("six"), val = fp16(0x1.8p2)];',
                       f'{kind} a = add(x = x, y = three)[name = string("a")];',
                       f'{kind} b = relu6(x = a)[name = string("b")];',
                       f'{kind} h = real_div(x = b, y = six)[name = string("h")];',
                       f'{kind} y = mul(x = x, y = h)[name = string("y")];'], "y"))


def linked_dw_pw_case(spatial=112):
    c = 64
    shape = (1, c, spatial, spatial)
    w0, n0 = om.blobfile("w0", (c, 1, 3, 3))
    w1, n1 = om.blobfile("w1", (c, c, 1, 1))
    body = [
        'string pt = const()[name = string("pt"), val = string("same")];',
        'tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1,1])];',
        'tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0,0,0,0])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1,1])];',
        'int32 dw = const()[name = string("dw"), val = int32(64)];',
        'int32 gp = const()[name = string("gp"), val = int32(1)];', w0, w1,
        f'{om.tensor_type(shape)} h = conv(x = x, weight = w0, strides = st, dilations = dl, pad = pd, pad_type = pt, groups = dw)[name = string("h")];',
        f'{om.tensor_type(shape)} y = conv(x = h, weight = w1, strides = st, dilations = dl, pad = pd, pad_type = pt, groups = gp)[name = string("y")];',
    ]
    return om.case(f"layer_link_dw_pw_c64_s{spatial}", "layer_link", dict(spatial=spatial),
                   om.program(f"{om.tensor_type(shape)} x", body, "y"), max(n0,n1))


def cases():
    for c in (3,8,16,24,32,96):
        for spatial in (16,32):
            for bias in (False,True):
                item=conv_case(5,c,c,spatial,2,c,bias,weights=finite_index_payload(c*25),name_prefix="layer_conv_dw5s2")
                item["weights_description"].update(value="indexed_finite",seed=3991)
                yield item
    for bias in (False,True):
        yield conv_case(5,96,96,32,1,96,bias,name_prefix="layer_conv")
        for co in (24,32,128):
            item=conv_case(1,96,co,16,1,1,bias,weights=finite_index_payload(co*96),name_prefix="layer_conv_indexed")
            item["weights_description"].update(value="indexed_finite",seed=3991)
            yield item
    for shared,consumer in ((False,False),(True,False),(False,True)):
        mil,weights=graph_case(shared,consumer)
        yield om.case(f"layer_graph_concat_shared{int(shared)}_consumer{int(consumer)}",
            "layer_graph",dict(channels=16,spatial=16,shared=shared,consumer=consumer),mil,weights,
            dict(storage="BLOBFILE",pattern="finite_index_payload with seed 3991+entry, aligned CoreML chunks",
                 blob_sha256=hashlib.sha256(weights).hexdigest()))
    for ci in (11,56):
        for k in (1,3,5):
            for bias in (False,True):
                yield conv_case(k,ci,32,16,1,1,bias,name_prefix="layer_conv")
    for ci in (11,24,56):
        for bias in (False,True):
            item=conv_case(5,ci,32,16,1,1,bias,weights=finite_index_payload(32*ci*25),name_prefix="layer_conv_indexed")
            item["weights_description"].update(value="indexed_finite",seed=3991)
            yield item
    for bias in (False,True):
        yield conv_case(3,512,128,35,1,1,bias,name_prefix="layer_conv")
        item = conv_case(3,512,128,35,1,1,bias,weights=finite_index_payload(128*512*9),name_prefix="layer_conv_indexed")
        item["weights_description"].update(value="indexed_finite",seed=3991)
        yield item
    yield linked_dw_pw_case()
    yield conv_case(3, 64, 64, 112, 1, 64, False, name_prefix="layer_conv")
    yield conv_case(1, 64, 64, 112, 1, 1, False, name_prefix="layer_conv")
    for k, ci, co, g in ((5,24,24,24),(5,3,32,1),(3,8,32,1),(5,32,32,1),(5,8,8,8),
                         (5,16,3,1),(5,16,8,1),(5,16,24,1)):
        for bias in (False,True):
            count = co * (ci // g) * k * k
            item = conv_case(k,ci,co,16,1,g,bias,weights=finite_index_payload(count),name_prefix="layer_conv_indexed")
            item["weights_description"].update(value="indexed_finite",seed=3991)
            yield item
    for bias in (False,True):
        item = conv_case(3,128,512,35,1,1,bias,weights=finite_index_payload(512*128*9),name_prefix="layer_conv_indexed")
        item["weights_description"].update(value="indexed_finite",seed=3991)
        yield item
    # Failure reported on H13: padded 96->384 gather stage, 128->512 at 35x35.
    for s in (16, 32, 33, 34, 35, 36, 40, 48, 64):
        for co in (256, 512):
            yield conv_case(3, 128, co, s, 1, 1, False, name_prefix="layer_conv")
            if s == 35:
                yield conv_case(3, 128, co, s, 1, 1, True, name_prefix="layer_conv")
    for c in (3, 8, 16, 24, 32, 48, 64):
        for k in (1, 3, 5):
            yield conv_case(k, c, 32, 16, 1, 1, False, name_prefix="layer_conv")
            yield conv_case(k, c, c, 16, 1, c, False, name_prefix="layer_conv")
    for s in (8, 16, 32, 64):
        for c in (3, 8, 16, 24, 32):
            for k in (1, 3, 5):
                for bias in (False, True):
                    yield conv_case(k, c, 32, s, 1, 1, bias, name_prefix="layer_conv")
                    yield conv_case(k, c, c, s, 1, c, bias, name_prefix="layer_conv")
        for co in (3,8,16,24):
            for k in (1,3,5):
                for bias in (False,True):
                    yield conv_case(k,16,co,s,1,1,bias,name_prefix="layer_conv")
    for c, h, w, ho, wo in ((1, 8, 8, 16, 16), (3, 8, 8, 16, 16), (16, 16, 16, 32, 32),
                             (32, 35, 35, 70, 70), (64, 16, 16, 32, 32),
                             # Camera shapes exported by the public-model campaign.
                             (128,16,16,32,32), (24,32,32,64,64), (16,64,64,128,128),
                             (64,10,10,20,20), (64,20,20,40,40), (1,128,128,256,256)):
        yield resize_case("upsample_nearest_neighbor", c, h, w, ho, wo)
        yield resize_case("resize_nearest_neighbor", c, h, w, ho, wo)
        yield resize_case("upsample_bilinear", c, h, w, ho, wo, "half")
        yield resize_case("resize_bilinear", c, h, w, ho, wo, "UNALIGN_CORNERS")
    for c in (1, 3, 16, 32, 64):
        for op in ("sigmoid", "hard_swish"):
            yield unary_case(op, c, 16, 16)
        yield hardswish_case(c, 16, 16)
        yield hardswish_arithmetic_case(c, 16, 16)
    for c, h, w in ((16,1,1), (24,1,1), (96,1,1), (128,1,1), (1,256,256)):
        yield unary_case("sigmoid",c,h,w)
    for a, b in ((3, 8), (16, 16), (24, 32), (64, 64)):
        yield concat_case(a, b, 16, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=("h13", "h14"), default=["h13", "h14"])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "oracles")
    parser.add_argument("--private-failures", type=Path, default=Path(".git/oracle-failures"))
    args = parser.parse_args()
    for item in cases():
        if args.case and not any(fnmatch.fnmatchcase(item["name"], p) for p in args.case):
            continue
        for target in args.targets:
            dest = args.output / target / f"{item['name']}.json"
            if dest.exists():
                continue
            status = om.run_case(item, target, args.output, args.tool, om.source_commit())
            record = json.loads(dest.read_text())
            # Tool identity is its hash; local account/host paths are not public evidence.
            record["compiler"].pop("host", None)
            record["compiler"]["tool"] = "ane-compile-hwx"
            record["compiler"]["command"][0] = "ane-compile-hwx"
            record["compiler"]["probe_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            if record["error"]:
                args.private_failures.mkdir(parents=True, exist_ok=True)
                (args.private_failures / f"{target}-{item['name']}.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n")
                # Apple diagnostics include temporary paths; keep the failure locally,
                # export a bounded classification rather than private compiler logs.
                record["error"] = "Apple compiler rejected probe; no decoded task stream"
            dest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(target, item["name"], status, flush=True)


if __name__ == "__main__":
    main()
