#!/usr/bin/env python3
"""Build the H16G ResNet task/head bisection kit.

The five primary cases deliberately cross complete compiler-built object
boundaries instead of deleting records from an HWX in place:

* T1 is a terminal one-task stem object;
* T2 is the accepted full-object control, copied byte-for-byte;
* T3 is the corrected stem plus resident mean/linear head;
* T4 is the pre-fix stem plus scratch-reading head, copied byte-for-byte;
* T5 is the corrected head alone, fed the CPU stem surface.

Value-only head switches start from T3's valid record framing.  The manifest
also records a semantic head/framing diff against the accepted full object.
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

from h16g_load_grammar import check, object_fields, walk


SWITCHES = (
    ("mean-common", 1, (1, 7), (0x14, 0x18, 0x34)),
    ("mean-source", 1, (), (0x4104, 0x4114, 0x4118)),
    ("linear-layout", 2, (1,), (0x40, 0x4104, 0x4114, 0x4148)),
    ("terminal-format", 2, (), (0x5138,)),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def op_line(lines: list[str], name: str) -> int:
    needle = f'[name = string("{name}")]'
    hits = [index for index, line in enumerate(lines) if needle in line]
    if len(hits) != 1:
        raise ValueError(f"MIL operation {name} is not unique")
    return hits[0]


def graph_slice(graph: dict, operations: list[dict], inputs: list[str],
                outputs: list[str]) -> tuple[dict, dict[str, np.ndarray]]:
    needed = {value for op in operations for value in op["inputs"].values()}
    constants = {name: graph["constants"][name]
                 for name in sorted(needed & set(graph["constants"]))}
    variables = {name: graph["variables"][name]
                 for name in set(inputs) | set(outputs) | needed |
                 {op["result"] for op in operations}}
    return {"inputs": inputs, "outputs": outputs, "variables": variables,
            "constants": constants, "operations": operations}, constants


def write_source(source: Path, model_text: str, graph: dict,
                 arrays: dict[str, np.ndarray]) -> None:
    source.mkdir(parents=True)
    os.symlink(Path("../../../source/weights"), source / "weights")
    (source / "model.mil").write_text(model_text)
    (source / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    np.savez(source / "constants.npz", **arrays)


def compile_case(compiler: Path, validator: Path, source: Path,
                 program: Path) -> Path:
    subprocess.run([str(compiler), "--target", "H16G", "--mil",
                    str(source / "model.mil"), "--model-root", str(source),
                    "--output", str(program)], check=True)
    obj = program / "program-0.hwx"
    subprocess.run([sys.executable, str(validator), str(obj)], check=True)
    return obj


def endpoint(path: Path) -> tuple[bytes, dict, list[dict], dict]:
    data = path.read_bytes()
    fields = object_fields(data)
    facts = walk(fields, include_tasks=True, enforce_residual=False)
    tasks = facts.pop("_tasks")
    return data, fields, tasks, facts


def checked(path: Path) -> dict:
    facts = check(path)
    facts.pop("path", None)
    return facts


def hex32(value: int | None) -> str | None:
    return None if value is None else f"0x{value:08x}"


def task_diff(left: dict, right: dict) -> dict:
    keys = sorted(set(left["registers"]) | set(right["registers"]))
    return {
        "words": [left["words"], right["words"]],
        "header": [{"word": index, "left": hex32(a), "right": hex32(b)}
                   for index, (a, b) in enumerate(zip(left["header"],
                                                       right["header"]))
                   if a != b],
        "registers": [{"address": f"0x{key:04x}",
                       "left": hex32(left["registers"].get(key)),
                       "right": hex32(right["registers"].get(key))}
                      for key in keys if left["registers"].get(key) !=
                      right["registers"].get(key)],
        "record_headers": {
            "left": [hex32(record["header"]) for record in left["records"]],
            "right": [hex32(record["header"]) for record in right["records"]],
        },
    }


def framing(fields: dict, facts: dict) -> dict:
    program = fields["program"]
    scratch = struct.unpack_from("<I", program, 0x858)[0]
    return {"tasks": facts["tasks"], "records": facts["records"],
            "record_code": facts["record_code"],
            "format_code": facts["format_code"],
            "bar_slots": facts["bar_slots"],
            "kernel_relocations": facts["kernel_relocations"],
            "terminal_control": facts["terminal_control"],
            "text_bytes": fields["text_size"],
            "kernel_bytes": fields["kernel_size"],
            "scratch_bytes": scratch}


def patch_switch(fixed: tuple, old: tuple, switch: tuple) -> bytes:
    name, index, header_words, registers = switch
    data, fields, tasks, _ = fixed
    source = old[2][index]
    target = tasks[index]
    image = bytearray(data)
    for word in header_words:
        struct.pack_into("<I", image,
                         fields["text_off"] + target["offset"] + 4 * word,
                         source["header"][word])
    for address in registers:
        if address not in target["register_offsets"] or address not in source["registers"]:
            raise ValueError(f"{name} lacks shared register 0x{address:x}")
        struct.pack_into("<I", image,
                         fields["text_off"] + target["register_offsets"][address],
                         source["registers"][address])
    return bytes(image)


def external_surface(array: np.ndarray, pad_marker: int | None = None) -> bytes:
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError("surface needs batch-one NCHW")
    _, channels, height, width = array.shape
    row_elements = (2 * width + 63) // 64 * 32
    fill = np.uint16(0xffff if pad_marker is not None else 0)
    result = np.full((channels, height, row_elements), fill, dtype="<u2")
    result[:, :, :width] = array[0].astype("<f2").view("<u2")
    return result.tobytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stem-source", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path, required=True)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--precompiled-t1", type=Path)
    parser.add_argument("--precompiled-t5", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--accepted-main", type=Path, required=True)
    parser.add_argument("--accepted-refs", type=Path, required=True)
    parser.add_argument("--old-stem-head", type=Path, required=True)
    parser.add_argument("--old-full", type=Path, required=True)
    parser.add_argument("--fixed-stem-head", type=Path, required=True)
    parser.add_argument("--fixed-full", type=Path, required=True)
    parser.add_argument("--fixed-stem-refs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir must not exist")
    args.out_dir.mkdir(parents=True)

    graph = json.loads((args.stem_source / "graph.json").read_text())
    lines = (args.stem_source / "model.mil").read_text().splitlines()
    conv_at, relu_at = op_line(lines, "conv_0"), op_line(lines, "relu_0")
    mean_at = op_line(lines, "rb3_mean_relu_0")
    linear_at = op_line(lines, "linear_0")
    declarations = list(lines[:conv_at])
    stem_ops = [dict(op) for op in graph["operations"][:2]]
    stem_output = stem_ops[-1]["result"]
    head_ops = [dict(op) for op in graph["operations"][-2:]]
    logits = head_ops[-1]["result"]

    stem_graph, stem_constants = graph_slice(
        graph, stem_ops, list(graph["inputs"]), [stem_output])
    head_graph, head_constants = graph_slice(
        graph, head_ops, [stem_output], [logits])
    with np.load(args.stem_source / "constants.npz", allow_pickle=False) as payload:
        stem_arrays = {entry["array"]: payload[entry["array"]]
                       for entry in stem_constants.values() if "array" in entry}
        head_arrays = {entry["array"]: payload[entry["array"]]
                       for entry in head_constants.values() if "array" in entry}

    t1 = args.out_dir / "T1-stem-alone"
    stem_mil = "\n".join(declarations + lines[conv_at:relu_at + 1] +
                           [f"  }} -> ({stem_output});", "}"]) + "\n"
    write_source(t1 / "source", stem_mil, stem_graph, stem_arrays)

    t5 = args.out_dir / "T5-head-alone"
    for index, line in enumerate(declarations):
        if "func main<" in line:
            declarations[index] = (f"  func main<ios18>(tensor<fp16, "
                f"[1, 64, 112, 112]> {stem_output}) {{")
            break
    head_mil = "\n".join(declarations + lines[mean_at:linear_at + 1] +
                           [f"  }} -> ({logits});", "}"]) + "\n"
    write_source(t5 / "source", head_mil, head_graph, head_arrays)

    reference = load_module(args.cpu_reference, "h16g_resnet_reference")
    stem_ref = reference.Reference(t1 / "source")
    head_ref = reference.Reference(t5 / "source")
    input_raw = (args.fixed_stem_refs / "input_00.bin").read_bytes()
    x = np.frombuffer(input_raw[:3 * 224 * 224 * 2], dtype="<f2").reshape(1, 3, 224, 224)
    stem_value = stem_ref.evaluate({graph["inputs"][0]: x})[stem_output]
    head_value = head_ref.evaluate({stem_output: stem_value})[logits]
    for case in (t1, t5):
        (case / "refs").mkdir()
    shutil.copy2(args.fixed_stem_refs / "input_00.bin", t1 / "refs/input_00.bin")
    (t1 / "refs/stem_00.surface.fp16").write_bytes(
        external_surface(stem_value, pad_marker=0xffff))
    (t5 / "refs/input_00.bin").write_bytes(external_surface(stem_value))
    (t5 / "refs/logits_00.fp16").write_bytes(head_value.astype("<f2").tobytes())
    if args.prepare_only:
        print(f"prepared {t1 / 'source'} and {t5 / 'source'}")
        return 0

    def obtain(precompiled: Path | None, case: Path) -> Path:
        target = case / "program" / "program-0.hwx"
        if precompiled:
            target.parent.mkdir(parents=True)
            shutil.copy2(precompiled, target)
            subprocess.run([sys.executable, str(args.validator), str(target)],
                           check=True)
            return target
        if not args.compiler:
            parser.error("assembly needs --compiler or both --precompiled-t1/--precompiled-t5")
        return compile_case(args.compiler, args.validator,
                            case / "source", case / "program")

    t1_object = obtain(args.precompiled_t1, t1)
    t5_object = obtain(args.precompiled_t5, t5)

    cases = {
        "T1-stem-alone": t1_object,
        "T2-accepted-full": args.accepted_main,
        "T3-fixed-stem-head": args.fixed_stem_head,
        "T4-old-stem-head": args.old_stem_head,
        "T5-head-alone": t5_object,
    }
    for name, source in list(cases.items())[1:4]:
        target = args.out_dir / name / "program" / "program-0.hwx"
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        cases[name] = target

    for name, refs in (("T2-accepted-full", args.accepted_refs),
                       ("T3-fixed-stem-head", args.fixed_stem_refs),
                       ("T4-old-stem-head", args.fixed_stem_refs)):
        target = args.out_dir / name / "refs"
        target.mkdir()
        shutil.copy2(refs / "input_00.bin", target / "input_00.bin")
        shutil.copy2(refs / "logits_00.fp16", target / "logits_00.fp16")

    accepted = endpoint(cases["T2-accepted-full"])
    old_stem = endpoint(cases["T4-old-stem-head"])
    fixed_stem = endpoint(cases["T3-fixed-stem-head"])
    old_full = endpoint(args.old_full)
    fixed_full = endpoint(args.fixed_full)
    if len(old_stem[2]) != 3 or len(fixed_stem[2]) != 3:
        raise ValueError("stem/head endpoints must have three tasks")
    if old_stem[0] == fixed_stem[0]:
        raise ValueError("fixed stem/head object did not change")

    switches = args.out_dir / "head-field-switches"
    switches.mkdir()
    switch_entries = []
    for index, switch in enumerate(SWITCHES, 1):
        payload = patch_switch(fixed_stem, old_stem, switch)
        name = f"S{index}-{switch[0]}.hwx"
        path = switches / name
        path.write_bytes(payload)
        switch_entries.append({"name": switch[0], "file": name,
                               "sha256": sha256(path), "facts": checked(path)})

    diff = {
        "schema": "mil-hwxc.h16g-resnet-head-framing-diff.v1",
        "left": "accepted full main object",
        "right_old": "pre-fix rule-built object",
        "right_fixed": "corrected rule-built object",
        "framing": {"accepted": framing(accepted[1], accepted[3]),
                    "old_full": framing(old_full[1], old_full[3]),
                    "fixed_full": framing(fixed_full[1], fixed_full[3])},
        "head": {
            "mean_old": task_diff(accepted[2][-2], old_full[2][-2]),
            "mean_fixed": task_diff(accepted[2][-2], fixed_full[2][-2]),
            "linear_old": task_diff(accepted[2][-1], old_full[2][-1]),
            "linear_fixed": task_diff(accepted[2][-1], fixed_full[2][-1]),
        },
    }
    (args.out_dir / "head-framing-diff.json").write_text(
        json.dumps(diff, indent=2) + "\n")

    manifest = {"purpose": "least-likely-to-wedge-first task bisection",
                "order": list(cases), "cases": [], "switches": switch_entries}
    for name, path in cases.items():
        facts = checked(path)
        refs = args.out_dir / name / "refs"
        expected = ("stem_00.surface.fp16" if name == "T1-stem-alone"
                    else "logits_00.fp16")
        manifest["cases"].append({"name": name,
            "file": str(path.relative_to(args.out_dir)), "sha256": sha256(path),
            "input": str((refs / "input_00.bin").relative_to(args.out_dir)),
            "cpu_reference": str((refs / expected).relative_to(args.out_dir)),
            "facts": facts})
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
