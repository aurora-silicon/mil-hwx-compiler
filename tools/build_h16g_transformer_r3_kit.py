#!/usr/bin/env python3
"""Rebuild the isolated H16G transformer reviewer kit with stage references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from h16g_attention_stage_check import cpu_references, load_inputs, write_references


def run(arguments: list[str]) -> None:
    completed = subprocess.run(arguments, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(" ".join(arguments) + "\n" +
                           completed.stdout + completed.stderr)


def write_stage_judge(path: Path) -> None:
    path.write_text(r'''#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
if len(sys.argv) != 3:
    raise SystemExit("usage: judge-stages.py attention-h6|attention-h8 OUT-DIR")
case, output = sys.argv[1], Path(sys.argv[2])
record = json.loads((root / "cases.json").read_text())["runnable"][case]
if record["kind"] != "chain":
    raise SystemExit(f"{case} is not an attention chain")
spec = importlib.util.spec_from_file_location(
    "stage_check", root / "h16g-attention-stage-check.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
inputs = [root / name for name in record["inputs"]]
q, k, v = module.load_inputs(
    *inputs, record["heads"], record["tokens"], record["head_dim"],
    record["score_columns"])
print(json.dumps(module.analyze_device(
    output, q, k, v, record["score_scale"],
    record["physical_score_heads"], record["score_columns"]),
    indent=2, sort_keys=True))
''')
    path.chmod(0o755)


def checksum_ledger(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and
                       item.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("validator", type=Path)
    parser.add_argument("base_kit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage-checker", type=Path, required=True)
    parser.add_argument("--task", default="t-6544")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    shutil.copytree(args.base_kit, args.output)
    cases_path = args.output / "cases.json"
    cases = json.loads(cases_path.read_text())
    cases["task"] = args.task

    for name, record in cases["runnable"].items():
        program = args.output / "cases" / name / "program"
        shutil.rmtree(program)
        source = args.output / "cases" / name / "model.mil"
        run([str(args.candidate), "--mil", str(source), "--model-root",
             str(source.parent), "--target", "H16G", "--output", str(program)])
        for image in sorted(program.glob("program-*.hwx")):
            run([sys.executable, str(args.validator), str(image),
                 "--manifest", str(program / "manifest.json")])

        if record["kind"] != "chain":
            continue
        heads = int(record["heads"])
        tokens = 1500
        head_dim = 64
        physical_heads = int(record["physical_score_heads"])
        score_columns = int(record["score_plane_bytes"] // (tokens * 2))
        scale = 1.0 / head_dim
        inputs = [args.output / item for item in record["inputs"]]
        q, k, v = load_inputs(*inputs, heads, tokens, head_dim, score_columns)
        stage_dir = args.output / "refs" / name / "stages"
        write_references(stage_dir, *cpu_references(
            q, k, v, scale, physical_heads, score_columns))
        record.update({
            "tokens": tokens,
            "head_dim": head_dim,
            "score_columns": score_columns,
            "score_scale": scale,
            "stage_expected": {
                "qk": str((stage_dir / "qk.fp16").relative_to(args.output)),
                "softmax": str((stage_dir / "softmax.fp16").relative_to(args.output)),
                "pv": str((stage_dir / "pv.fp16").relative_to(args.output)),
            },
        })

    cases_path.write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
    shutil.copy2(args.stage_checker,
                 args.output / "h16g-attention-stage-check.py")
    write_stage_judge(args.output / "judge-stages.py")
    checksum_ledger(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
