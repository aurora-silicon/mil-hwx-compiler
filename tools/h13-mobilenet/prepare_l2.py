#!/usr/bin/env python3
"""Prepare a real-weight, cold-checkable MobileNet device package on the host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from l2_stats import package_stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL_SHA256 = "ecc3a67c47c5a609ec35f6a58a7d97532834e43df4cb7d3f1204a8164b7d20dd"
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--compiler", default="build/mil-hwxc", type=Path)
    parser.add_argument("--revision", help="source commit when building a controller snapshot without .git")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a fresh directory to preserve prior evidence")
    weights = args.weights.resolve()
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        parser.error("expected the pinned public MobileNetV1 quantized weights (SHA-256 mismatch)")
    out = args.out.resolve()
    env = dict(os.environ, MILC=str(args.compiler.resolve()), LINKED_SCHEDULE="linked-l2")
    subprocess.run([sys.executable, str(HERE / "mobilenet.py"), str(out), "--weights", str(weights)], env=env, check=True)
    subprocess.run([sys.executable, str(HERE / "mn_refs.py"), str(out), "16"], env=env, check=True)
    stats = dict(linked=package_stats(out / "linked"), perop=package_stats(out / "perop"))
    from chain import compile_chain, conv_mil, load_params, save_params
    _, params, _ = load_params(out / "params.npz")
    probe = out / "probes/wide128"
    probe_layers = [dict(cout=128, k=k, s=1, g=g, bias=True, relu=True)
                    for k, g in ((1, 1), (3, 128), (1, 1))]
    # This is a descriptor probe, not a MobileNet subnetwork: reuse the real
    # 128->128 pointwise weights for its last layer to match the proven form.
    probe_mil, probe_params, _ = conv_mil(probe_layers, (128, 56, 56), str(probe / "src"),
                                         weights=[params[6], params[7], params[6]])
    save_params(probe / "params.npz", probe_layers, probe_params, (128, 56, 56))
    for schedule, directory in (("linked-l2", "linked"), (None, "perop")):
        compile_chain(str(args.compiler.resolve()), probe_mil, str(probe / "src"), str(probe / directory), schedule)
    stats["wide128Probe"] = package_stats(probe / "linked")
    (out / "host-stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    # Only public source and the generated model package travel to the lead.
    for relative in ("h13-mobilenet/chain.py", "h13-mobilenet/run_l2.py", "h13-mobilenet/coldcheck.py", "h13-mobilenet/anechain.c",
                     "h13-conv-bench/surf.py", "h13-conv-bench/milgen.py"):
        destination = out / "tools" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "tools" / relative, destination)
    revision = args.revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_files = ["plugins/H13/H13Program.cpp", "plugins/H13/H13Program.h", "plugins/H13/ANEH13Compiler.mm"]
    provenance = dict(sourceRevision=revision, weightsSha256=digest, weightsSource=MODEL_URL,
                      weightsLicense="Apache-2.0", reference="ReLU (not ReLU6), fp16 weights and per-layer rounding; 16 procedural images",
                      compilerSha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                      sourceSha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in source_files})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copy2(ROOT / "LICENSE", out / "COMPILER-LICENSE")
    # Full license accompanies the redistributed model-derived weights.
    shutil.copy2(HERE / "APACHE-2.0.txt", out / "WEIGHTS-LICENSE")
    (out / "WEIGHTS-NOTICE.txt").write_text(
        "MobileNetV1 1.0 224 quantized, TensorFlow Authors, Apache-2.0.\n"
        "Source: " + MODEL_URL + "\n"
        "The package dequantizes the weights to fp16, mirrors spatial kernels,\n"
        "moves stride-2 sampling to pointwise layers, pads the classifier, and\n"
        "uses ReLU in place of ReLU6. See provenance.json for exact hashes.\n")
    (out / "RUN.txt").write_text("From this package directory, with libane available:\n"
        "gcc -O2 -I\"$LIBANE\" -o tools/h13-mobilenet/anechain tools/h13-mobilenet/anechain.c \"$LIBANE/libane.a\" -ldrm\n"
        "python3 tools/h13-mobilenet/run_l2.py . --passes 5000 --output timing-r3\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "programs"} for key, value in stats.items()}, indent=2))


if __name__ == "__main__":
    main()
