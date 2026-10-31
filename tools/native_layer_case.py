#!/usr/bin/env python3
"""Prepare native H13/H14 device fixtures or compare a returned output.

No hardware is accessed. The lead runs the package through the approved
device path, then passes the returned surface to compare.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from mint_layer_probes import resize_case, unary_case, hardswish_arithmetic_case, concat_case
from layer_graph_cases import graph_case
import layer_reference as reference
from surf import pack, unpack, stats
from milgen import conv_mil


def build(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("fixture output must be empty; preserve previous results")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "src"
    source.mkdir()
    rng = np.random.default_rng(args.seed)
    c, h, w = args.channels, args.height, args.width
    extent = 8 if args.op in ("sigmoid", "hardswish") else 1
    inputs = [rng.uniform(-extent, extent, (c,h,w)).astype(np.float16)]
    extra = {}
    linked=False
    if args.op=="branch-concat":
        if args.two_inputs:
            inputs.append(rng.uniform(-1,1,(c,h,w)).astype(np.float16))
        wa=(rng.standard_normal((c,c,1,1))/np.sqrt(c)).astype(np.float16)
        wb=(rng.standard_normal((c,c,5,5))/np.sqrt(c*25)).astype(np.float16)
        ba=(rng.standard_normal(c)*0.1).astype(np.float16)
        bb=(rng.standard_normal(c)*0.1).astype(np.float16)
        text,blob=graph_case(payloads=[a.tobytes() for a in (wa,ba,wb,bb)],channels=c,spatial=h,two_inputs=args.two_inputs)
        if h!=w: raise ValueError("branch-concat fixture requires square spatial dimensions")
        (source/"model.mil").write_text(text);(source/"weights.bin").write_bytes(blob)
        mil=source/"model.mil";item=None;linked=True
        a=reference.f16_ties_away(reference.conv(inputs[0],wa,ba))
        b=reference.f16_ties_away(reference.conv(inputs[-1] if args.two_inputs else inputs[0],wb,bb,pad=2))
        y=reference.concat_channels(a,b)
        extra=dict(weights_a=wa,weights_b=wb,bias_a=ba,bias_b=bb)
    elif args.op in ("nearest", "bilinear"):
        operation = "resize_nearest_neighbor" if args.op == "nearest" else "resize_bilinear"
        item = resize_case(operation,c,h,w,h*2,w*2,"UNALIGN_CORNERS" if args.op == "bilinear" else None)
        y = (reference.nearest2x if args.op == "nearest" else reference.bilinear_half2x)(inputs[0])
    elif args.op == "sigmoid":
        item = unary_case("sigmoid",c,h,w)
        y = reference.sigmoid(inputs[0])
    elif args.op == "hardswish":
        item = hardswish_arithmetic_case(c,h,w)
        y = reference.hardswish(inputs[0])
    elif args.op == "concat":
        inputs.append(rng.uniform(-1,1,(args.other_channels,h,w)).astype(np.float16))
        item = concat_case(c,args.other_channels,h,w)
        y = reference.concat_channels(*inputs)
    else:
        groups = c if args.op == "depthwise5" else 1
        co = c if args.op == "depthwise5" else args.output_channels
        weights = (rng.standard_normal((co,c//groups,5,5))/np.sqrt(c//groups*25)).astype(np.float16)
        bias = (rng.standard_normal(co)*0.1).astype(np.float16)
        layer = dict(cout=co,k=5,s=1,g=groups,bias=True,relu=False)
        mil,_,_ = conv_mil([layer],(c,h,w),str(source),weights=[dict(w=weights,b=bias)])
        item = None
        extra = dict(weights=weights,bias=bias)
        y = reference.conv(inputs[0],weights,bias,groups=groups,pad=2)
    if item:
        (source/"model.mil").write_text(item["mil"])
        mil = source/"model.mil"
    command=[str(args.compiler.resolve()),"--target",args.target,"--mil",str(mil),
             "--model-root",str(source),"--output",str(args.output/"pkg")]
    if linked: command += ["--schedule","linked"]
    subprocess.run(command,check=True)
    manifest = json.loads((args.output/"pkg/manifest.json").read_text())
    assert len(manifest["programs"])==1
    program = manifest["programs"][0]
    expected = reference.f16_ties_away(y)
    for index,(x,binding) in enumerate(zip(inputs,program["inputs"])):
        (args.output/f"input{index}.bin").write_bytes(pack(x,binding["nchw"],binding["allocationBytes"]))
    output = program["outputs"][0]
    (args.output/"expected.bin").write_bytes(pack(np.ascontiguousarray(expected),output["nchw"],output["allocationBytes"]))
    np.savez(args.output/"reference.npz",expected=expected,
             **{f"input{i}":x for i,x in enumerate(inputs)},**extra)
    files = [args.output/f"input{i}.bin" for i in range(len(inputs))]
    files += [args.output/"expected.bin", args.output/"pkg"/program["file"]]
    fixture = dict(op=args.op,target=args.target,seed=args.seed,
                   compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                   reference="float64, fp16 surface rounding ties away",
                   sha256={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.output/"fixture.json").write_text(json.dumps(fixture,indent=2)+"\n")
    print(args.output/"fixture.json")


def compare(args):
    fixture = json.loads((args.package/"fixture.json").read_text())
    manifest = json.loads((args.package/"pkg/manifest.json").read_text())
    output = manifest["programs"][0]["outputs"][0]
    raw = args.actual.read_bytes()
    if len(raw) != output["allocationBytes"]:
        print(json.dumps({"pass":False,"error":"returned surface allocation length differs"}))
        return 1
    actual = unpack(raw,output["nchw"])
    with np.load(args.package/"reference.npz",allow_pickle=False) as data:
        expected = data["expected"]
    result = stats(actual,expected)
    exact = fixture["op"] in ("nearest","concat")
    if result["nonfinite"]:
        result.update(max_abs=None,rel_l2=None)
        passed = False
    else:
        passed = result["max_abs"]==0 if exact else result["rel_l2"]<=0.005 and result["max_abs"]<=0.02
    result["pass"] = passed
    result["op"] = fixture["op"]
    result["target"] = fixture["target"]
    print(json.dumps(result,allow_nan=False))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    make = commands.add_parser("build")
    make.add_argument("--compiler",type=Path,default=Path("build/mil-hwxc"))
    make.add_argument("--target",choices=("H13","H14"),default="H13")
    make.add_argument("--op",choices=("nearest","bilinear","concat","sigmoid","hardswish","conv5","depthwise5","branch-concat"),required=True)
    make.add_argument("--two-inputs",action="store_true")
    make.add_argument("--channels",type=int,default=16)
    make.add_argument("--other-channels",type=int,default=16)
    make.add_argument("--output-channels",type=int,default=32)
    make.add_argument("--height",type=int,default=16)
    make.add_argument("--width",type=int,default=16)
    make.add_argument("--seed",type=int,default=3991)
    make.add_argument("--output",type=Path,required=True)
    check = commands.add_parser("compare")
    check.add_argument("package",type=Path)
    check.add_argument("actual",type=Path)
    args = parser.parse_args()
    return build(args) if args.command=="build" else compare(args)


if __name__=="__main__": raise SystemExit(main())
