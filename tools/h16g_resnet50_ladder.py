#!/usr/bin/env python3
"""Build reproducible ResNet-50 prefix sources, CPU refs, and H16G objects.

Every prefix keeps a deterministic reduce-mean/linear probe head, giving the
runner one ordinary 1000-element output while the residual body grows from the
stem to the full network.  The full rung reuses the original public head.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


RUNGS = (
    ("01-stem", "relu_0", 64),
    ("02-stem-maxpool", "max_pool_0", 64),
    ("03-projection-block", "relu_3", 256),
    ("04-first-stage", "relu_9", 256),
    ("05-through-stage3", "relu_21", 512),
    ("06-through-stage4", "relu_39", 1024),
    ("07-full", "relu_48", 2048),
)
DUMP_RUNGS = RUNGS[:4]
BLOB_MAGIC = int.from_bytes(b"\xef\xbe\xad\xde", "little")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("h16g_resnet_cpu_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operation_line(lines: list[str], name: str) -> int:
    needle = f'[name = string("{name}")]'
    hits = [index for index, line in enumerate(lines) if needle in line]
    if len(hits) != 1:
        raise ValueError(f"MIL operation {name} is not unique")
    return hits[0]


def prefix_graph(graph: dict, arrays, target: str, channels: int,
                 head_weight: np.ndarray, head_bias: np.ndarray) -> tuple[dict, dict]:
    target_index = next(index for index, op in enumerate(graph["operations"])
                        if op["name"] == target)
    operations = [dict(op) for op in graph["operations"][:target_index + 1]]
    boundary = operations[-1]["result"]
    boundary_shape = operations[-1]["type"]["shape"]
    if boundary_shape[1] != channels or len(boundary_shape) != 4:
        raise ValueError(f"{target} does not expose the declared prefix channels")
    names = {"axes": f"rb3_{target}_axes", "keep": f"rb3_{target}_keep",
             "weight": f"rb3_{target}_weight", "bias": f"rb3_{target}_bias",
             "mean": f"rb3_{target}_mean", "logits": f"rb3_{target}_logits"}
    operations.extend([
        {"name": f"rb3_mean_{target}", "op": "reduce_mean", "result": names["mean"],
         "type": {"dtype": "fp16", "shape": [1, channels]},
         "inputs": {"x": boundary, "axes": names["axes"],
                    "keep_dims": names["keep"]}},
        {"name": "linear_0", "op": "linear", "result": names["logits"],
         "type": {"dtype": "fp16", "shape": [1, 1000]},
         "inputs": {"x": names["mean"], "weight": names["weight"],
                    "bias": names["bias"]}},
    ])
    used = {value for op in operations[:-2] for value in op["inputs"].values()}
    constants, payload = {}, {}
    for name in sorted(used & set(graph["constants"])):
        constants[name] = graph["constants"][name]
        entry = constants[name]
        if "array" in entry:
            payload[entry["array"]] = arrays[entry["array"]]
    constants.update({
        names["axes"]: {"array": names["axes"]},
        names["keep"]: {"value": False},
        names["weight"]: {"array": names["weight"]},
        names["bias"]: {"array": names["bias"]},
    })
    payload.update({names["axes"]: np.asarray([2, 3], dtype=np.int32),
                    names["weight"]: head_weight,
                    names["bias"]: head_bias})
    variables = {name: value for name, value in graph["variables"].items()
                 if name in {graph["inputs"][0]} |
                    {op["result"] for op in operations[:-2]} | used}
    variables.update({
        names["axes"]: {"dtype": "int32", "shape": [2]},
        names["keep"]: {"dtype": "bool", "shape": []},
        names["weight"]: {"dtype": "fp16", "shape": [1000, channels]},
        names["bias"]: {"dtype": "fp16", "shape": [1000]},
        names["mean"]: {"dtype": "fp16", "shape": [1, channels]},
        names["logits"]: {"dtype": "fp16", "shape": [1, 1000]},
    })
    return {"inputs": graph["inputs"], "outputs": [names["logits"]],
            "variables": variables, "constants": constants,
            "operations": operations}, payload


def prefix_mil(lines: list[str], target: str, channels: int,
               weight_offset: int, bias_offset: int) -> str:
    first_op = operation_line(lines, "conv_0")
    target_op = operation_line(lines, target)
    if target_op < first_op:
        raise ValueError("prefix target precedes the stem")
    boundary_line = lines[target_op]
    boundary = boundary_line.split(" = ", 1)[0].split()[-1]
    names = {"axes": f"rb3_{target}_axes", "keep": f"rb3_{target}_keep",
             "weight": f"rb3_{target}_weight", "bias": f"rb3_{target}_bias",
             "mean": f"rb3_{target}_mean", "logits": f"rb3_{target}_logits"}
    declarations = lines[:first_op]
    declarations.extend([
        f'    tensor<int32, [2]> {names["axes"]} = const()[name = string("{names["axes"]}"), val = tensor<int32, [2]>([2, 3])];',
        f'    bool {names["keep"]} = const()[name = string("{names["keep"]}"), val = bool(false)];',
        f'    tensor<fp16, [1000, {channels}]> {names["weight"]} = const()[name = string("{names["weight"]}"), val = tensor<fp16, [1000, {channels}]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({weight_offset})))];',
        f'    tensor<fp16, [1000]> {names["bias"]} = const()[name = string("{names["bias"]}"), val = tensor<fp16, [1000]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({bias_offset})))];',
    ])
    operations = lines[first_op:target_op + 1]
    operations.extend([
        f'    tensor<fp16, [1, {channels}]> {names["mean"]} = reduce_mean(axes = {names["axes"]}, keep_dims = {names["keep"]}, x = {boundary})[name = string("rb3_mean_{target}")];',
        f'    tensor<fp16, [1, 1000]> {names["logits"]} = linear(bias = {names["bias"]}, weight = {names["weight"]}, x = {names["mean"]})[name = string("linear_0")];',
        f'  }} -> ({names["logits"]});',
        '}',
    ])
    return "\n".join(declarations + operations) + "\n"


def boundary_graph(graph: dict, arrays, target: str) -> tuple[dict, dict]:
    target_index = next(index for index, op in enumerate(graph["operations"])
                        if op["name"] == target)
    operations = [dict(op) for op in graph["operations"][:target_index + 1]]
    boundary = operations[-1]["result"]
    used = {value for op in operations for value in op["inputs"].values()}
    constants = {name: graph["constants"][name]
                 for name in sorted(used & set(graph["constants"]))}
    payload = {entry["array"]: arrays[entry["array"]]
               for entry in constants.values() if "array" in entry}
    variables = {name: graph["variables"][name]
                 for name in {graph["inputs"][0], boundary} | used |
                              {op["result"] for op in operations}}
    return {"inputs": graph["inputs"], "outputs": [boundary],
            "variables": variables, "constants": constants,
            "operations": operations}, payload


def boundary_mil(lines: list[str], target: str) -> str:
    first_op = operation_line(lines, "conv_0")
    target_op = operation_line(lines, target)
    boundary = lines[target_op].split(" = ", 1)[0].split()[-1]
    return "\n".join(lines[:first_op] + lines[first_op:target_op + 1] +
                     [f"  }} -> ({boundary});", "}"]) + "\n"


def make_boundary_references(reference, model: Path, output: Path,
                             count: int, seed: int) -> dict:
    ref = reference.Reference(model)
    input_name = ref.graph["inputs"][0]
    output_name = ref.graph["outputs"][0]
    shape = ref.graph["variables"][input_name]["shape"]
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    traces = []
    output_shape = None
    for index in range(count):
        x = rng.uniform(-1, 1, shape).astype("<f2")
        trace = []
        y = ref.evaluate({input_name: x}, trace)[output_name]
        if y.ndim != 4 or y.shape[0] != 1:
            raise ValueError(f"boundary {output_name} is not batch-one NCHW")
        (output / f"input_{index:02d}.bin").write_bytes(reference.surface(x))
        (output / f"boundary_{index:02d}.surface.fp16").write_bytes(
            reference.surface(y))
        traces.append(trace)
        output_shape = list(y.shape)
    (output / "trace.json").write_text(json.dumps(traces, indent=2) + "\n")
    return {"output": output_name, "shape": output_shape,
            "reference": "boundary_00.surface.fp16"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_blob(path: Path, payload: bytes) -> int:
    """Append one BLOBFILE chunk and return its header offset."""
    with path.open("ab") as output:
        header_offset = output.tell()
        payload_offset = header_offset + 64
        header = bytearray(64)
        struct.pack_into("<IIQQ", header, 0, BLOB_MAGIC, 1,
                         len(payload), payload_offset)
        output.write(header)
        output.write(payload)
    return header_offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="rb2/full source directory")
    parser.add_argument("--cpu-reference", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--validator", type=Path)
    parser.add_argument("--images", type=int, default=1)
    args = parser.parse_args()
    if args.out_dir.exists() or args.images < 1:
        parser.error("--out-dir must not exist and --images must be positive")

    args.out_dir.mkdir(parents=True)
    shared = args.out_dir / "source"
    shutil.copytree(args.source, shared)
    weight_path = shared / "weights/weight.bin"
    graph = json.loads((args.source / "graph.json").read_text())
    lines = (args.source / "model.mil").read_text().splitlines()
    reference = load_module(args.cpu_reference)
    rng = np.random.default_rng(6486)
    manifest = {"seed": 6486, "probe_outputs": 1000, "rungs": [],
                "boundary_dumps": []}
    with np.load(args.source / "constants.npz", allow_pickle=False) as arrays:
        for rung_index, (name, target, channels) in enumerate(RUNGS):
            rung = args.out_dir / "ladder" / name
            model = rung / "source"
            model.mkdir(parents=True)
            os.symlink(Path("../../../source/weights"), model / "weights")
            if name == "07-full":
                shutil.copy2(args.source / "model.mil", model / "model.mil")
                shutil.copy2(args.source / "graph.json", model / "graph.json")
                shutil.copy2(args.source / "constants.npz", model / "constants.npz")
            else:
                head_weight = rng.uniform(-0.02, 0.02, (1000, channels)).astype("<f2")
                head_bias = rng.uniform(-0.02, 0.02, (1000,)).astype("<f2")
                weight_offset = append_blob(weight_path, head_weight.tobytes())
                bias_offset = append_blob(weight_path, head_bias.tobytes())
                built_graph, payload = prefix_graph(
                    graph, arrays, target, channels, head_weight, head_bias)
                (model / "model.mil").write_text(prefix_mil(
                    list(lines), target, channels, weight_offset, bias_offset))
                (model / "graph.json").write_text(json.dumps(built_graph, indent=2) + "\n")
                np.savez(model / "constants.npz", **payload)
            refs = rung / "refs"
            reference.make_references(model, refs, args.images, 1234)
            if args.compiler:
                package = rung / "program"
                subprocess.run([str(args.compiler), "--target", "H16G",
                                "--mil", str(model / "model.mil"),
                                "--model-root", str(model), "--output", str(package)],
                               check=True)
                if args.validator:
                    subprocess.run([sys.executable, str(args.validator),
                                    str(package / "program-0.hwx")], check=True)
            manifest["rungs"].append({"name": name, "target": target,
                                      "channels": channels,
                                      "model_sha256": sha256(model / "model.mil")})
        for name, target, channels in DUMP_RUNGS:
            dump = args.out_dir / "boundary-dumps" / name
            model = dump / "source"
            model.mkdir(parents=True)
            os.symlink(Path("../../../source/weights"), model / "weights")
            built_graph, payload = boundary_graph(graph, arrays, target)
            (model / "model.mil").write_text(boundary_mil(list(lines), target))
            (model / "graph.json").write_text(
                json.dumps(built_graph, indent=2) + "\n")
            np.savez(model / "constants.npz", **payload)
            reference_facts = make_boundary_references(
                reference, model, dump / "refs", args.images, 1234)
            if args.compiler:
                package = dump / "program"
                subprocess.run([str(args.compiler), "--target", "H16G",
                                "--mil", str(model / "model.mil"),
                                "--model-root", str(model), "--output", str(package)],
                               check=True)
                if args.validator:
                    subprocess.run([sys.executable, str(args.validator),
                                    str(package / "program-0.hwx")], check=True)
            manifest["boundary_dumps"].append({
                "name": name, "target": target, "channels": channels,
                "model_sha256": sha256(model / "model.mil"),
                **reference_facts})
    (args.out_dir / "ladder.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
