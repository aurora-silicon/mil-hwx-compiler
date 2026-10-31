#!/usr/bin/env python3
"""Prepare or compare either camera island-046 dense stage on H13/H14.

build writes a standalone 128->512, 3x3 s1 p1, 35x35 fp16 package, input
surface, and float64 reference rounded to fp16 with ties away. It never
executes a device. gather exercises the exact 96->384 padded channel layout;
dense exercises every one of the 128 input and 512 output channels. mix
exercises the padded 384->96 second-stage layout in a 512->128 surface.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from milgen import conv_mil
from run_layer import conv_ref
from surf import f16_ties_away, pack, stats, unpack


def parameters(kind, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (128, 35, 35)).astype(np.float16)
    if kind == "gather":
        # Four corner shifts, block-major over the 96 live input channels.
        w = np.zeros((512, 128, 3, 3), dtype=np.float16)
        live = np.arange(96)
        for block, (dy, dx) in enumerate(((0, 0), (0, 2), (2, 0), (2, 2))):
            w[block * 96 + live, live, dy, dx] = 1
        x[96:] = 0
        b = np.zeros(512, dtype=np.float16)
    elif kind == "mix":
        x = rng.uniform(-1,1,(512,35,35)).astype(np.float16)
        x[384:] = 0
        w = np.zeros((128,512,3,3),dtype=np.float16)
        taps = (rng.standard_normal((96,5,5))/5).astype(np.float16)
        for c in range(96):
            for dy in range(5):
                for dx in range(5):
                    by,bx = int(dy>2),int(dx>2)
                    w[c,(by*2+bx)*96+c,dy-2*by,dx-2*bx] = taps[c,dy,dx]
        b = np.zeros(128,dtype=np.float16)
        b[:96] = (rng.standard_normal(96)*0.1).astype(np.float16)
    else:
        w = (rng.standard_normal((512, 128, 3, 3)) / np.sqrt(128 * 9)).astype(np.float16)
        b = (rng.standard_normal(512) * 0.1).astype(np.float16)
    return x, w, b


def build(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("fixture output must be empty; preserve previous results")
    args.output.mkdir(parents=True, exist_ok=True)
    x, w, b = parameters(args.kind, args.seed)
    layer = dict(cout=w.shape[0], k=3, s=1, g=1, bias=True, relu=False)
    mil, _, _ = conv_mil([layer], x.shape, str(args.output / "src"),
                         weights=[dict(w=w, b=b)])
    package = args.output / "pkg"
    subprocess.run([str(args.compiler.resolve()), "--target", args.target, "--mil", mil,
                    "--model-root", str(args.output / "src"), "--output", str(package)],
                   env=dict(os.environ, MILHWXC_H13_CONV_GENERIC="1"), check=True)
    manifest = json.loads((package / "manifest.json").read_text())
    assert len(manifest["programs"]) == 1
    program = manifest["programs"][0]
    assert program["taskDescriptors"] == 2
    reference = f16_ties_away(conv_ref(x, w, b, 1, 1, 1, False))
    (args.output / "input.bin").write_bytes(pack(x, program["inputs"][0]["nchw"],
                                               program["inputs"][0]["allocationBytes"]))
    (args.output / "expected.bin").write_bytes(pack(reference, program["outputs"][0]["nchw"],
                                                  program["outputs"][0]["allocationBytes"]))
    np.savez(args.output / "reference.npz", x=x, w=w, b=b, y=reference)
    hashes = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (args.output / "input.bin", args.output / "expected.bin",
                        package / program["file"])}
    (args.output / "reproducer.json").write_text(json.dumps(dict(
        geometry=f"{x.shape[0]}->{w.shape[0]} k3 s1 p1 35x35 fp16", target=args.target, kind=args.kind, seed=args.seed,
        active_channels=[96,384] if args.kind=="gather" else ([384,96] if args.kind=="mix" else [128,512]),
        reference="float64, fp16 ties away", sha256=hashes), indent=2) + "\n")
    print(args.output / "reproducer.json")


def compare(args):
    manifest = json.loads((args.package / "pkg/manifest.json").read_text())
    output = manifest["programs"][0]["outputs"][0]
    actual = unpack(args.actual.read_bytes(), output["nchw"])
    with np.load(args.package / "reference.npz", allow_pickle=False) as ref:
        expected = ref["y"]
    result = stats(actual, expected)
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    result["max_error_chw"] = list(map(int, np.unravel_index(np.argmax(error), error.shape)))
    kind = json.loads((args.package / "reproducer.json").read_text())["kind"]
    result["pass"] = result["nonfinite"] == 0 and (
        result["max_abs"] == 0 if kind == "gather" else result["rel_l2"] <= 0.002)
    print(json.dumps(result))
    return 0 if result["pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("--compiler", type=Path, default=Path("build/mil-hwxc"))
    make.add_argument("--target", choices=("H13","H14"), default="H13")
    make.add_argument("--kind", choices=("gather", "dense", "mix"), default="gather")
    make.add_argument("--seed", type=int, default=3991)
    make.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("compare")
    check.add_argument("package", type=Path)
    check.add_argument("actual", type=Path)
    args = parser.parse_args()
    return build(args) if args.command == "build" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
