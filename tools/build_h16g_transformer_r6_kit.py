#!/usr/bin/env python3
"""Build the r6 layer_norm prefix kit from the qualified r5 kit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h13_td import split_h14_tasks  # noqa: E402


def run(arguments: list[str]) -> bytes:
    completed = subprocess.run(arguments, capture_output=True)
    if completed.returncode:
        raise RuntimeError(" ".join(arguments) + "\n" +
                           completed.stdout.decode(errors="replace") +
                           completed.stderr.decode(errors="replace"))
    return completed.stdout


def cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode()


def patch_task_stream(template: Path, stream: bytes, output: Path) -> None:
    data = bytearray(template.read_bytes())
    command_count, command_bytes = struct.unpack_from("<2I", data, 16)
    cursor, command_end = 32, 32 + command_bytes
    text_entry = descriptor = None
    old_size = text_offset = None
    for _ in range(command_count):
        command, size = struct.unpack_from("<2I", data, cursor)
        kind = struct.unpack_from("<I", data, cursor + 8)[0] if size >= 12 else None
        if command == 0x19:
            section = cursor + 72
            count = struct.unpack_from("<I", data, cursor + 64)[0]
            segment = cstring(data[cursor + 8:cursor + 24])
            for _ in range(count):
                name = cstring(data[section:section + 16])
                if segment == "__TEXT" and name == "__text":
                    text_entry = section
                    old_size = struct.unpack_from("<Q", data, section + 40)[0]
                    text_offset = struct.unpack_from("<I", data, section + 48)[0]
                section += 80
        elif command == 4 and kind == 4:
            descriptor = cursor
        cursor += size
    if cursor != command_end or None in (text_entry, descriptor, old_size,
                                         text_offset):
        raise ValueError("template lacks one H16G task section/descriptor")
    if len(stream) > old_size or len(stream) % 4 or \
            stream[:16] != b"\x01" + bytes(15):
        raise ValueError("stage task stream does not fit template")
    tasks = split_h14_tasks(stream[16:])
    data[text_offset:text_offset + old_size] = stream + bytes(old_size - len(stream))
    struct.pack_into("<Q", data, text_entry + 40, len(stream))
    struct.pack_into("<I", data, descriptor + 0x824, len(stream) // 4)
    struct.pack_into("<I", data, descriptor + 0x830, len(tasks))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def checksum_ledger(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and
                       item.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                     f"{path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("validator", type=Path)
    parser.add_argument("layer_norm_dump", type=Path)
    parser.add_argument("base_kit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", default=
        "t-6707-mil-hwx-compiler-h16g-layer-norm-round-6-t-6676-")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    shutil.copytree(args.base_kit, args.output)
    shutil.rmtree(args.output / "__pycache__", ignore_errors=True)
    cases_path = args.output / "cases.json"
    cases = json.loads(cases_path.read_text())
    cases["task"] = args.task
    cases["cold_runs"] = 3

    order = []
    for name, record in cases["runnable"].items():
        program = args.output / "cases" / name / "program"
        source = args.output / "cases" / name / "model.mil"
        if record["kind"] == "chain":
            before = [(args.output / item).read_bytes()
                      for item in record["programs"]]
            shutil.rmtree(program)
            run([str(args.candidate), "--mil", str(source), "--model-root",
                 str(source.parent), "--target", "H16G", "--output",
                 str(program)])
            after = [(program / f"program-{index}.hwx").read_bytes()
                     for index in range(3)]
            if after != before:
                raise AssertionError(f"{name}: attention changed from r5/#44")
            order.append({"case": name, "kind": "attention"})
            continue

        tokens, channels = int(record["tokens"]), int(record["channels"])
        shutil.rmtree(program)
        run([str(args.candidate), "--mil", str(source), "--model-root",
             str(source.parent), "--target", "H16G", "--output",
             str(program)])
        template = program / "program-0.hwx"
        run([sys.executable, str(args.validator), str(template),
             "--manifest", str(program / "manifest.json")])

        info = json.loads(run([str(args.layer_norm_dump), str(tokens),
                               str(channels), "prefix-info"]))
        prefix_dir = args.output / "cases" / name / "prefix"
        shutil.rmtree(prefix_dir, ignore_errors=True)
        prefixes = []
        for item in info:
            task_count = int(item["task_count"])
            image = prefix_dir / f"prefix-{task_count:02d}.hwx"
            stream = run([str(args.layer_norm_dump), str(tokens),
                          str(channels), f"prefix-{task_count}"])
            patch_task_stream(template, stream, image)
            run([sys.executable, str(args.validator), str(image)])
            prefixes.append({**item,
                "program": str(image.relative_to(args.output))})
        record["prefixes"] = prefixes
        run_order = sorted(prefixes,
            key=lambda item: (-int(item["result_row_bytes"]),
                              int(item["task_count"])))
        record["prefix_run_order"] = [int(item["task_count"])
                                      for item in run_order]

        control_image = prefix_dir / "prefix-01-dead-control.hwx"
        control_stream = run([str(args.layer_norm_dump), str(tokens),
                              str(channels), "prefix-1-dead-control"])
        patch_task_stream(template, control_stream, control_image)
        run([sys.executable, str(args.validator), str(control_image)])
        record["dead_field_control"] = {
            "kind": "repeat-mean-divisor-same-value",
            "task_count": 1,
            "program": str(control_image.relative_to(args.output)),
        }

        order.extend({"case": name, "kind": "prefix", "task_count": count,
                      "cold_runs": 3}
                     for count in record["prefix_run_order"])
        order.append({"case": name, "kind": "dead-field-control",
                      "task_count": 1, "cold_runs": 3})
        order.append({"case": name, "kind": "final", "cold_runs": 3})

    cases_path.write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    shutil.copy2(Path(__file__).with_name("h16g_layer_norm_stage_check.py"),
                 args.output)
    shutil.copy2(Path(__file__).with_name("h16g_transformer_r6_run_case.py"),
                 args.output / "run-case.py")
    shutil.copy2(Path(__file__).with_name("h16g_transformer_r6_judge.py"),
                 args.output / "judge-layer-norm-r6.py")
    (args.output / "run-case.py").chmod(0o755)
    (args.output / "judge-layer-norm-r6.py").chmod(0o755)
    (args.output / "run-one.sh").write_text(
        "#!/bin/sh\nset -eu\n"
        "kit=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
        "name=${1:?usage: run-one.sh CASE OUT-DIR}\n"
        "out=${2:?usage: run-one.sh CASE OUT-DIR}\n"
        "exec timeout 7200 python3 -I \"$kit/run-case.py\" \"$name\" \"$out\"\n")
    (args.output / "run-one.sh").chmod(0o755)
    (args.output / "ORDER.txt").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in order) + "\n")
    (args.output / "README.md").write_text(
        "# 20261114-h16g-layernorm-r6\n\n"
        "Reviewer kit for t-6707. The compact prefix writer now uses the "
        "original model width for the zero-beta BAR offset and names the "
        "complete result-DMA plane span. Each geometry retains every r5 "
        "producer prefix, adds a byte-distinct same-value mean-divisor "
        "negative control, and runs every layer_norm object in three fresh "
        "loads. CPU references are retained from r5. Attention H=6/H=8 "
        "objects are byte-identical to r5 and production gate #44. Run "
        "judge-layer-norm-r6.py after the case runner; acceptance requires "
        "finite prefix-01 output, final relL2 <= 0.02, and bit-identical "
        "cold outputs.\n")
    checksum_ledger(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
