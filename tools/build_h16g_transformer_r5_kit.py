#!/usr/bin/env python3
"""Build the r5 layer_norm task-prefix bisect and old-main controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from build_h16g_transformer_r4_kit import (checksum_ledger,
    patch_task_stream)


def run(arguments: list[str]) -> bytes:
    completed = subprocess.run(arguments, capture_output=True)
    if completed.returncode:
        raise RuntimeError(" ".join(arguments) + "\n" +
                           completed.stdout.decode(errors="replace") +
                           completed.stderr.decode(errors="replace"))
    return completed.stdout


def reduce_mean_mil() -> str:
    tokens, channels = 128, 128
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, 1, {tokens}, {channels}]> x) {{
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([3])];
    tensor<fp16, [1, 1, {tokens}, 1]> y = reduce_mean(x = x, axes = axes, keep_dims = bool(true))[name = string("reduce")];
  }} -> (y);
}}
'''


def write_judge(root: Path) -> None:
    target = root / "judge-layer-norm-stages.py"
    target.write_text(r'''#!/usr/bin/env python3
import json
import math
from pathlib import Path
import struct
import sys
root = Path(__file__).resolve().parent
case, output = sys.argv[1], Path(sys.argv[2])
record = json.loads((root / "cases.json").read_text())["runnable"][case]

def halves(path):
    raw = path.read_bytes()
    if len(raw) % 2:
        raise ValueError(f"{path}: odd fp16 byte count")
    return [item[0] for item in struct.iter_unpack("<e", raw)]

def metrics(actual, expected):
    finite = [(a, e) for a, e in zip(actual, expected)
              if math.isfinite(a) and math.isfinite(e)]
    denominator = math.sqrt(math.fsum(e * e for e in expected))
    numerator = math.sqrt(math.fsum((a - e) ** 2 for a, e in zip(actual, expected)))
    correlation = float("nan")
    if finite:
        aa, ee = zip(*finite)
        am, em = math.fsum(aa) / len(aa), math.fsum(ee) / len(ee)
        av = math.fsum((a - am) ** 2 for a in aa)
        ev = math.fsum((e - em) ** 2 for e in ee)
        if av and ev:
            correlation = math.fsum((a - am) * (e - em)
                                    for a, e in finite) / math.sqrt(av * ev)
    return {"rel_l2": numerator / denominator if denominator else numerator,
            "correlation": correlation,
            "nonfinite": sum(not math.isfinite(a) for a in actual)}

tokens, channels = int(record["tokens"]), int(record["channels"])
flat = halves(root / record["inputs"][0])
rows = [flat[start:start + channels] for start in range(0, len(flat), channels)]
means = [math.fsum(row) / channels for row in rows]
variances = [math.fsum((value - means[index]) ** 2 for value in row) / channels
             for index, row in enumerate(rows)]
epsilon = struct.unpack("<e", struct.pack("<e", 1e-5))[0]
results, first = {}, None
for item in record["prefixes"]:
    task = int(item["task_count"])
    name = f"prefix-{task:02d}"
    paths = (output / name / "output_00.bin", output / f"{name}.bin")
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        results[name] = {"missing": True}
        first = task if first is None else first
        continue
    count = int(item["result_rows"]) * int(item["result_row_bytes"]) // 2
    actual = halves(path)
    if len(actual) < count:
        results[name] = {"elements": len(actual), "expected_minimum_elements": count}
        first = task if first is None else first
        continue
    actual = actual[:count]
    stage = item.get("expected_stage")
    if stage is None:
        result = {"expected_stage": None,
                  "nonfinite": sum(not math.isfinite(value) for value in actual),
                  "nonzero": sum(value != 0 for value in actual)}
        failed = bool(result["nonfinite"])
    else:
        start, height = int(item["token_start"]), int(item["result_rows"])
        width = int(item["result_row_bytes"]) // 2
        if stage == "mean":
            scalars = means[start:start + height]
        elif stage == "variance":
            scalars = variances[start:start + height]
        else:
            scalars = None
        if scalars is not None:
            expected = [value for value in scalars for _ in range(width)]
            packed = metrics(actual, expected)
            lanes = [metrics(actual[lane::width], scalars) for lane in range(width)]
            best = min(range(width), key=lambda lane: lanes[lane]["rel_l2"])
            result = {"expected_stage": stage, "packed": packed,
                      "best_lane": {"index": best, **lanes[best]},
                      "nonfinite": packed["nonfinite"],
                      "rel_l2": lanes[best]["rel_l2"]}
        else:
            expected = []
            for row_index in range(start, start + height):
                scale = 1.0 / math.sqrt(variances[row_index] + epsilon)
                expected.extend((value - means[row_index]) * scale
                                for value in rows[row_index][:width])
            result = {"expected_stage": stage, **metrics(actual, expected)}
        failed = bool(result["nonfinite"] or result["rel_l2"] > 1e-2)
    results[name] = result
    if first is None and failed:
        first = task
print(json.dumps({"prefixes": results, "first_failing_task": first,
                  "threshold": 1e-2}, indent=2, sort_keys=True))
''')
    target.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("control_compiler", type=Path)
    parser.add_argument("validator", type=Path)
    parser.add_argument("layer_norm_dump", type=Path)
    parser.add_argument("base_kit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", default=
        "t-6676-mil-hwx-compiler-h16g-layer-norm-round-5-t-6568-")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    shutil.copytree(args.base_kit, args.output)
    shutil.rmtree(args.output / "__pycache__", ignore_errors=True)
    cases_path = args.output / "cases.json"
    cases = json.loads(cases_path.read_text())
    cases["task"] = args.task

    order = []
    for name, record in cases["runnable"].items():
        program = args.output / "cases" / name / "program"
        if record["kind"] == "chain":
            before = [(args.output / item).read_bytes()
                      for item in record["programs"]]
            source = args.output / "cases" / name / "model.mil"
            shutil.rmtree(program)
            run([str(args.candidate), "--mil", str(source), "--model-root",
                 str(source.parent), "--target", "H16G", "--output", str(program)])
            after = [(program / f"program-{index}.hwx").read_bytes()
                     for index in range(3)]
            if after != before:
                raise AssertionError(f"{name}: attention changed from r4/r3")
            order.append({"case": name, "kind": "attention"})
            continue

        tokens, channels = int(record["tokens"]), int(record["channels"])
        source = args.output / "cases" / name / "model.mil"
        shutil.rmtree(program)
        run([str(args.candidate), "--mil", str(source), "--model-root",
             str(source.parent), "--target", "H16G", "--output", str(program)])
        template = program / "program-0.hwx"
        run([sys.executable, str(args.validator), str(template),
             "--manifest", str(program / "manifest.json")])

        stale = args.output / "cases" / name / "stages"
        shutil.rmtree(stale, ignore_errors=True)
        record.pop("stage_programs", None)
        info = json.loads(run([str(args.layer_norm_dump), str(tokens),
                               str(channels), "prefix-info"]))
        prefixes = []
        for item in info:
            task_count = int(item["task_count"])
            image = args.output / "cases" / name / "prefix" / \
                f"prefix-{task_count:02d}.hwx"
            stream = run([str(args.layer_norm_dump), str(tokens), str(channels),
                          f"prefix-{task_count}"])
            patch_task_stream(template, stream, image)
            run([sys.executable, str(args.validator), str(image)])
            prefixes.append({**item, "program": str(image.relative_to(args.output))})
        record["prefixes"] = prefixes
        # Full-width producer views are least likely to stress the compact
        # writer. The task order remains in `prefixes` for first-failure judgment.
        run_order = sorted(prefixes,
            key=lambda item: (-int(item["result_row_bytes"]),
                              int(item["task_count"])))
        record["prefix_run_order"] = [int(item["task_count"])
                                      for item in run_order]

        control = args.output / "cases" / name / "old-main-reduce-mean"
        control.mkdir(parents=True, exist_ok=True)
        control_source = control / "source.mil"
        control_source.write_text(reduce_mean_mil())
        control_program = control / "program"
        run([str(args.control_compiler), "--mil", str(control_source),
             "--model-root", str(control), "--target", "H16G", "--output",
             str(control_program)])
        control_image = control_program / "program-0.hwx"
        run([sys.executable, str(args.validator), str(control_image),
             "--manifest", str(control_program / "manifest.json")])
        value = np.fromfile(args.output / record["inputs"][0], dtype="<f2").reshape(
            tokens, channels)
        controls = []
        for row_start in range(0, tokens, 128):
            for channel_start in range(0, channels, 128):
                tile = np.zeros((128, 128), dtype=np.float16)
                rows = min(128, tokens - row_start)
                columns = min(128, channels - channel_start)
                tile[:rows, :columns] = value[
                    row_start:row_start + rows,
                    channel_start:channel_start + columns]
                label = f"r{row_start:04d}-c{channel_start:04d}"
                tile_input = control / f"input-{label}.fp16"
                expected = control / f"expected-{label}.fp16"
                tile.astype("<f2").tofile(tile_input)
                tile.astype(np.float64).mean(axis=1).astype("<f2").tofile(expected)
                controls.append({
                    "program": str(control_image.relative_to(args.output)),
                    "input": str(tile_input.relative_to(args.output)),
                    "expected": str(expected.relative_to(args.output)),
                    "row_start": row_start,
                    "channel_start": channel_start,
                })
        record["old_main_controls"] = controls
        order.append({"case": name, "kind": "old-main-reduce-mean-tiles",
                      "tiles": len(controls)})
        order.extend({"case": name, "kind": "prefix", "task_count": count}
                     for count in record["prefix_run_order"])
        order.append({"case": name, "kind": "final"})

    cases_path.write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    shutil.copy2(Path(__file__).with_name("h16g_layer_norm_stage_check.py"),
                 args.output)
    shutil.copy2(Path(__file__).with_name("h16g_transformer_r5_run_case.py"),
                 args.output / "run-case.py")
    write_judge(args.output)
    (args.output / "ORDER.txt").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in order) + "\n")
    (args.output / "README.md").write_text(
        "# 20261114-h16g-layernorm-r5\n\n"
        "Run-one kit for t-6676. Each layer_norm geometry first covers its "
        "complete [S,C] input with padded 128x128 known-good main reduce_mean "
        "tiles (direct wide last-axis reduction is not an admitted form), then runs compact, "
        "producer-derived task-prefix dumps. Prefix metadata records the named "
        "L2 result base and row stride; no centered or reciprocal-stddev dump is "
        "claimed because the measured prefix does not materialize either as a "
        "standalone boundary. Attention H=6/H=8 is byte-identical to r3/r4. "
        "Layer_norm remains gated. Run judge-layer-norm-stages.py with python3 "
        "-I after execution to name the first missing or numerically bad task.\n")
    checksum_ledger(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
