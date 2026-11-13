#!/usr/bin/env python3
"""Build the r4 transformer kit with layer_norm boundary dump objects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h16g_attention_stage_check import (cpu_references as attention_references,
    load_inputs, write_references as write_attention_references)
from h16g_layer_norm_stage_check import (STAGES as LAYER_NORM_STAGES,
    cpu_references as layer_norm_references,
    write_references as write_layer_norm_references)
from h13_td import split_h14_tasks


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
    """Replace __TEXT/__text and its descriptor counts in a built H16G HWX."""
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
    if cursor != command_end or None in (text_entry, descriptor, old_size, text_offset):
        raise ValueError("template lacks one H16G task section/descriptor")
    if len(stream) > old_size or len(stream) % 4 or stream[:16] != b"\x01" + bytes(15):
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


def write_judges(root: Path) -> None:
    (root / "judge-layer-norm-stages.py").write_text(r'''#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import h16g_layer_norm_stage_check as check
root = Path(__file__).resolve().parent
case, output = sys.argv[1], Path(sys.argv[2])
record = json.loads((root / "cases.json").read_text())["runnable"][case]
value = check.load_input(root / record["inputs"][0], record["tokens"], record["channels"])
print(json.dumps(check.analyze_stage_directory(value, output), indent=2, sort_keys=True))
''')
    (root / "judge-layer-norm-stages.py").chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("validator", type=Path)
    parser.add_argument("layer_norm_dump", type=Path)
    parser.add_argument("base_kit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task", default="t-6568-mil-hwx-compiler-h16g-layer-norm-round-4-t-6544-")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    shutil.copytree(args.base_kit, args.output)
    shutil.rmtree(args.output / "__pycache__", ignore_errors=True)
    cases_path = args.output / "cases.json"
    cases = json.loads(cases_path.read_text())
    cases["task"] = args.task
    cases["qualification_gates"]["attention"] = "qualified by t-6549"

    for name, record in cases["runnable"].items():
        old_attention = []
        if record["kind"] == "chain":
            old_attention = [(args.output / item).read_bytes()
                             for item in record["programs"]]
        program = args.output / "cases" / name / "program"
        shutil.rmtree(program)
        source = args.output / "cases" / name / "model.mil"
        run([str(args.candidate), "--mil", str(source), "--model-root",
             str(source.parent), "--target", "H16G", "--output", str(program)])
        for image in sorted(program.glob("program-*.hwx")):
            run([sys.executable, str(args.validator), str(image),
                 "--manifest", str(program / "manifest.json")])

        if record["kind"] == "chain":
            current = [(program / f"program-{index}.hwx").read_bytes()
                       for index in range(3)]
            if current != old_attention:
                raise AssertionError(f"{name}: attention programs changed from r3")
            heads,tokens,head_dim = int(record["heads"]),1500,64
            physical_heads = int(record["physical_score_heads"])
            score_columns = int(record["score_plane_bytes"] // (tokens * 2))
            scale = 1.0 / head_dim
            inputs = [args.output / item for item in record["inputs"]]
            q,k,v = load_inputs(*inputs,heads,tokens,head_dim,score_columns)
            stage_dir = args.output / "refs" / name / "stages"
            write_attention_references(stage_dir,*attention_references(
                q,k,v,scale,physical_heads,score_columns))
            continue

        tokens = 128 if name.endswith("bert") else 1500
        channels = {"layer-norm-bert": 768, "layer-norm-wtiny": 384,
                    "layer-norm-wbase": 512}[name]
        value = np.fromfile(args.output / record["inputs"][0], dtype="<f2").reshape(tokens,channels)
        stage_dir = args.output / "refs" / name / "stages"
        write_layer_norm_references(stage_dir,layer_norm_references(value))
        stage_programs = {}
        template = program / "program-0.hwx"
        for stage in LAYER_NORM_STAGES:
            raw = run([str(args.layer_norm_dump),str(tokens),str(channels),stage])
            image = args.output / "cases" / name / "stages" / f"{stage}.hwx"
            patch_task_stream(template,raw,image)
            run([sys.executable,str(args.validator),str(image)])
            stage_programs[stage] = str(image.relative_to(args.output))
        record.update({"tokens":tokens,"channels":channels,
            "stage_programs":stage_programs,
            "stage_expected":{stage:str((stage_dir/f"{stage}.fp16").relative_to(args.output))
                              for stage in LAYER_NORM_STAGES}})

    cases_path.write_text(json.dumps(cases,indent=2,sort_keys=True)+"\n")
    shutil.copy2(Path(__file__).with_name("h16g_attention_stage_check.py"),
                 args.output / "h16g-attention-stage-check.py")
    shutil.copy2(Path(__file__).with_name("h16g_layer_norm_stage_check.py"),args.output)
    shutil.copy2(Path(__file__).with_name("h16g_transformer_r4_run_case.py"),
                 args.output / "run-case.py")
    write_judges(args.output)
    (args.output / "README.md").write_text(
        "# 20261113-h16g-layernorm-r4\n\n"
        "Run-one reviewer kit for t-6568. Attention H=6/H=8 objects are "
        "byte-identical to r3 and qualified by t-6549. Layer norm remains "
        "gated; each case runs mean, centered, variance, rsqrt, normalized "
        "boundary dumps before the final candidate. Assumed external dump "
        "layout is contiguous row-major fp16 [S,C]. Accept layer norm relL2 "
        "<=1e-2, attention <=5e-3, no hang and zero DART faults.\n")
    checksum_ledger(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
