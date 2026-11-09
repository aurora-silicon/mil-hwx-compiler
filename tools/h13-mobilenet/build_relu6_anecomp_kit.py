#!/usr/bin/env python3
"""Build the H13 ANECOMP fused-ReLU6 qualification kit.

The output contains four single convolution cases and a full linked-L2
MobileNetV1.  Every case carries deterministic fp16 input and float64-derived
fp16 reference files; no device access is performed.

usage: build_relu6_anecomp_kit.py COMPILER OUTDIR
"""
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chain import chain_ref, compile_chain, conv_mil, save_params  # noqa: E402
import mobilenet as mobile  # noqa: E402


def write_case(compiler, root, name, layer, inshape, seed):
    case = root / name
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(inshape).astype(np.float16)
    mil, params, outshape = conv_mil([layer], inshape, case / "src", seed=seed)
    manifest = compile_chain(compiler, mil, case / "src", case / "package", None)
    reference = chain_ref(x, [layer], params)[-1]
    (case / "input.fp16").write_bytes(x.tobytes())
    (case / "reference.fp16").write_bytes(np.asarray(reference, dtype=np.float16).tobytes())
    return {"name": name, "package": f"{name}/package",
            "input": f"{name}/input.fp16", "reference": f"{name}/reference.fp16",
            "inputShape": list(inshape), "outputShape": list(outshape),
            "tasks": manifest["programs"][0]["taskDescriptors"]}


def main():
    compiler = str(Path(sys.argv[1]).resolve())
    root = Path(sys.argv[2]).resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    os.environ["MILHWXC_TARGET"] = "H13"
    cases = []
    for bias in (False, True):
        cases.append(write_case(compiler, root,
            f"pw1x1_64x64_bias{int(bias)}",
            dict(cout=64, k=1, s=1, g=1, bias=bias,
                 relu=False, relu6=True), (64, 16, 16), 100 + bias))
        cases.append(write_case(compiler, root,
            f"dw3x3_c32_bias{int(bias)}",
            dict(cout=32, k=3, s=1, g=32, bias=bias,
                 relu=False, relu6=True), (32, 16, 16), 200 + bias))

    name = "mobilenetv1_relu6_linked_l2"
    case = root / name
    layers = mobile.ane_layers(relu6=True)
    tf_weights = mobile.random_tf_weights(5962)
    params = mobile.to_ane_params(tf_weights)
    mil, params, outshape = conv_mil(layers, (16, 112, 112), case / "src",
                                     weights=params)
    manifest = compile_chain(compiler, mil, case / "src", case / "package",
                             "linked-l2")
    if manifest["programs"][0]["taskDescriptors"] != 45:
        raise RuntimeError("ReLU6 MobileNet linked-l2 must contain 45 tasks")
    rng = np.random.default_rng(5962)
    image = np.clip(rng.standard_normal((3, 224, 224)) * 0.35, -1, 1)
    x = mobile.preprocess(image)
    reference64 = chain_ref(x, layers, params)[-1].astype(np.float64)
    (case / "input.fp16").write_bytes(x.tobytes())
    (case / "reference.fp16").write_bytes(reference64.astype(np.float16).tobytes())
    logits64 = reference64.mean(axis=(1, 2))[:1001]
    (case / "logits.float64").write_bytes(logits64.tobytes())
    save_params(case / "params.npz", layers, params, (16, 112, 112))
    cases.append({"name": name, "package": f"{name}/package",
        "input": f"{name}/input.fp16", "reference": f"{name}/reference.fp16",
        "logitsFloat64": f"{name}/logits.float64",
        "inputShape": [16, 112, 112], "outputShape": list(outshape),
        "tasks": 45})
    (root / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (root / "README.md").write_text(
        "# H13 fused ReLU6 ANECOMP kit\n\n"
        "`cases.json` lists four single-task conv+clip cases and the 45-task "
        "MobileNetV1 linked-L2 package. Inputs and fp16 references are raw "
        "CHW tensors; MobileNet also includes float64 logits.\n")
    print(f"ANECOMP kit: {root} ({len(cases)} cases; MobileNet tasks=45)")


if __name__ == "__main__":
    main()
