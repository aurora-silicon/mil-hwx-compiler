#!/usr/bin/env python3
"""Build the device-free H16G LUT/reduction qualification matrix.

The current compiler role writes deterministic fp16 inputs, exact float64
references, one program per case, and the rb16-style cases.json.  The
old-main role writes the comparison program beside every shape main accepts
and a named refusal file beside held-out reductions that main cannot compile.
No Apple program or private comparison data is read by this script.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


LUT_OPERATIONS = (
    "sigmoid", "tanh", "gelu", "silu", "exp", "log", "sqrt",
    "rsqrt", "reciprocal", "relu",
)
REDUCTIONS = ("reduce_sum", "reduce_mean", "reduce_max")


def tensor(shape: tuple[int, ...]) -> str:
    return f'tensor<fp16, [{", ".join(map(str, shape))}]>'


def unary_mil(operation: str, size: int) -> str:
    kind = tensor((1, 1, size, size))
    extra = ', mode = string("EXACT")' if operation == "gelu" else ""
    return (
        "program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
        f"  func main<ios18>({kind} x) {{\n"
        f"    {kind} y = {operation}(x = x{extra})"
        '[name = string("unary")];\n  } -> (y);\n}\n'
    )


def reduction_mil(operation: str, shape: tuple[int, int, int, int],
                  axis: int) -> str:
    output = list(shape)
    output[axis] = 1
    return (
        "program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
        f"  func main<ios18>({tensor(shape)} x) {{\n"
        f"    tensor<int32, [1]> axes = const()[name = string(\"axes\"), "
        f"val = tensor<int32, [1]>([{axis}])];\n"
        f"    {tensor(tuple(output))} y = {operation}(x = x, axes = axes, "
        'keep_dims = bool(true))[name = string("reduce")];\n'
        "  } -> (y);\n}\n"
    )


def unary_reference(operation: str, values):
    import numpy as np
    x = values.astype(np.float64)
    if operation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if operation == "tanh":
        return np.tanh(x)
    if operation == "gelu":
        flat = [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
                for v in x.reshape(-1)]
        return np.asarray(flat, dtype=np.float64).reshape(x.shape)
    if operation == "silu":
        return x / (1.0 + np.exp(-x))
    if operation == "exp":
        return np.exp(x)
    if operation == "log":
        return np.log(x)
    if operation == "sqrt":
        return np.sqrt(x)
    if operation == "rsqrt":
        return 1.0 / np.sqrt(x)
    if operation == "reciprocal":
        return 1.0 / x
    if operation == "relu":
        return np.maximum(x, 0.0)
    raise ValueError(operation)


def unary_input(operation: str, size: int, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    if operation == "log":
        values = rng.uniform(0.5, 1.0, size * size)
    elif operation in ("sqrt",):
        values = rng.uniform(0.0, 4.0, size * size)
    elif operation in ("rsqrt", "reciprocal"):
        values = rng.uniform(1.0, 2.0, size * size)
    elif operation == "exp":
        values = rng.uniform(0.0, 1.0, size * size)
    else:
        values = rng.uniform(-4.0, 4.0, size * size)
    return values.astype(np.float16).reshape(1, 1, size, size)


def program_record(manifest: dict) -> dict:
    records = manifest.get("artifacts") or manifest.get("programs") or []
    if len(records) != 1:
        raise RuntimeError("qualification case must contain exactly one program")
    return records[0]


def compile_case(compiler: Path, validator: Path, case: Path, mil: str,
                 role: str) -> tuple[Path | None, str | None, dict | None]:
    source = case / "source.mil"
    source.write_text(mil)
    package = case / f".{role}-package"
    result = subprocess.run([
        str(compiler), "--target", "H16G", "--mil", str(source),
        "--model-root", str(case), "--output", str(package),
    ], text=True, capture_output=True, timeout=180)
    if result.returncode:
        if package.exists():
            shutil.rmtree(package)
        return None, (result.stderr or result.stdout).strip(), None
    manifest = json.loads((package / "manifest.json").read_text())
    record = program_record(manifest)
    raw_name = record["file"]
    raw = package / raw_name
    validation = subprocess.run([
        sys.executable, str(validator), str(raw), "--manifest",
        str(package / "manifest.json"),
    ], text=True, capture_output=True, timeout=60)
    if validation.returncode:
        raise RuntimeError(f"bind-extent validation failed: "
                           f"{validation.stderr or validation.stdout}")
    record = dict(record)
    record["validatedTaskCount"] = json.loads(validation.stdout)["tasks"]
    name = "program.anec" if role == "current" else "old-main.anec"
    output = case / name
    output.write_bytes(raw.read_bytes())
    shutil.rmtree(package)
    return output, None, record


def case_specs() -> list[dict]:
    specs: list[dict] = []
    for index, operation in enumerate(LUT_OPERATIONS):
        covered = 256 if operation == "log" else 128
        held = 192 if index % 2 == 0 else 320
        for coverage, size in (("main-covered", covered), ("held-out", held)):
            specs.append(dict(kind="lut", operation=operation, size=size,
                              coverage=coverage, bytes=2 * size * size))
    reduced = {
        1: ((1, 64, 8, 8), (1, 96, 12, 20)),
        2: ((1, 32, 64, 16), (1, 7, 33, 19)),
        3: ((1, 1, 64, 64), (1, 5, 11, 37)),
    }
    for operation in REDUCTIONS:
        for axis, (covered, held) in reduced.items():
            for coverage, shape in (("main-covered", covered),
                                    ("held-out", held)):
                row_bytes = ((shape[3] * 2 + 63) // 64) * 64
                specs.append(dict(kind="reduce", operation=operation,
                    axis=axis, shape=shape, coverage=coverage,
                    bytes=shape[1] * shape[2] * row_bytes))
    return sorted(specs, key=lambda item: (item["bytes"], item["operation"],
                                           item.get("axis", 0)))


def case_name(spec: dict) -> str:
    if spec["kind"] == "lut":
        return f'{spec["operation"]}-{spec["size"]}x{spec["size"]}-{spec["coverage"]}'
    shape = spec["shape"]
    return (f'{spec["operation"]}-c{shape[1]}h{shape[2]}w{shape[3]}-'
            f'a{spec["axis"]}-{spec["coverage"]}')


def write_reference(case: Path, spec: dict, seed: int) -> tuple[list[int], list[int]]:
    import numpy as np
    if spec["kind"] == "lut":
        values = unary_input(spec["operation"], spec["size"], seed)
        reference = unary_reference(spec["operation"], values)
        input_bytes = values.tobytes()
        input_shape = output_shape = list(values.shape)
    else:
        shape = tuple(spec["shape"])
        rng = np.random.default_rng(seed)
        logical = rng.uniform(-1.0, 1.0, np.prod(shape)).astype(np.float16).reshape(shape)
        row_halfwords = ((shape[3] * 2 + 63) // 64) * 32
        storage = np.zeros((shape[1], shape[2], row_halfwords), np.float16)
        storage[:, :, :shape[3]] = logical[0]
        input_bytes = storage.tobytes()
        axis = spec["axis"]
        if spec["operation"] == "reduce_sum":
            reference = np.sum(logical.astype(np.float64), axis=axis, keepdims=True)
        elif spec["operation"] == "reduce_mean":
            reference = np.mean(logical.astype(np.float64), axis=axis, keepdims=True)
        else:
            reference = np.max(logical.astype(np.float64), axis=axis, keepdims=True)
        input_shape = list(shape)
        output_shape = list(reference.shape)
    (case / "input.bin").write_bytes(input_bytes)
    (case / "reference.float64.bin").write_bytes(
        np.asarray(reference, dtype=np.float64).tobytes())
    return input_shape, output_shape


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--role", choices=("current", "old-main"), default="current")
    parser.add_argument("--validator", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--programs-only", action="store_true")
    mode.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, spec in enumerate(case_specs()):
        name = case_name(spec)
        case = args.output / name
        case.mkdir()
        mil = (unary_mil(spec["operation"], spec["size"])
               if spec["kind"] == "lut" else
               reduction_mil(spec["operation"], tuple(spec["shape"]), spec["axis"]))
        program=refusal=record=None
        if not args.references_only:
            program, refusal, record = compile_case(
                args.compiler.resolve(), args.validator.resolve(), case, mil, args.role)
        if not args.programs_only:
            input_shape, output_shape = write_reference(case, spec, 610600 + index)
        elif spec["kind"] == "lut":
            input_shape=output_shape=[1,1,spec["size"],spec["size"]]
        else:
            input_shape=list(spec["shape"]);output_shape=list(spec["shape"])
            output_shape[spec["axis"]]=1
        tolerance = ({"relL2": 0.02, "maxAbs": 0.08}
                     if spec["kind"] == "lut" else
                     {"relL2": 0.005, "maxAbs": 0.02})
        entry = dict(name=name,
            program=f"{name}/" + ("program.anec" if args.role == "current"
                                  else "old-main.anec"),
            input=f"{name}/input.bin",
            reference=f"{name}/reference.float64.bin",
            tolerance=tolerance, inputShape=input_shape,
            outputShape=output_shape, coverage=spec["coverage"])
        if record:
            entry["tasks"] = record.get("taskDescriptors",
                record.get("taskCount",record["validatedTaskCount"]))
        if refusal:
            (case / "old-main-unavailable.txt").write_text(refusal + "\n")
            entry["program"] = None
            entry["oldMainRefusal"] = f"{name}/old-main-unavailable.txt"
        cases.append(entry)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    action=("programs" if args.programs_only else
            "references" if args.references_only else "complete")
    print(f"H16G LUT/reduce kit: {len(cases)} cases ({args.role}, {action})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
