#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run(arguments: list[str], *, check: bool = True) -> int:
    completed = subprocess.run(arguments)
    if check and completed.returncode:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    record = json.loads((root / "cases.json").read_text())["runnable"][args.case]
    args.out.mkdir(parents=True, exist_ok=True)
    programs = [root / path for path in record["programs"]]
    inputs = [root / path for path in record["inputs"]]
    expected = root / record["expected"]
    runner = ["timeout", "300", sys.executable, "-I", str(root / "ane-run.py")]
    if record["kind"] == "single":
        failed = False
        for stage in ("mean", "centered", "variance", "rsqrt", "normalized"):
            relative = record.get("stage_programs", {}).get(stage)
            if relative is None:
                continue
            stage_out = args.out / stage
            rc = run(runner + ["--hwx", str(root / relative), "--case",
                str(inputs[0]), "--runs", "1", "--out-dir", str(stage_out)],
                check=False)
            failed |= rc != 0
        rc = run(runner + ["--hwx", str(programs[0]), "--case",
            f"{inputs[0]}:{expected}", "--runs", "3", "--max-ulp",
            str(record["max_ulp"]), "--out-dir", str(args.out / "final")],
            check=False)
        raise SystemExit(1 if failed or rc else 0)
    qk_out = args.out / "qk"
    run(runner + ["--hwx", str(programs[0]), "--case",
        f"{inputs[0]},{inputs[1]}", "--runs", "1", "--out-dir", str(qk_out)])
    score = bytearray((qk_out / "output_00.bin").read_bytes())
    logical = record["heads"] * record["score_plane_bytes"]
    physical = record["physical_score_heads"] * record["score_plane_bytes"]
    if len(score) < physical:
        score.extend(bytes(physical - len(score)))
    score[logical:physical] = bytes(physical - logical)
    score_path = args.out / "scores-padded.fp16"
    score_path.write_bytes(score[:physical])
    softmax_out = args.out / "softmax"
    run(runner + ["--hwx", str(programs[1]), "--case", str(score_path),
        "--runs", "1", "--out-dir", str(softmax_out)])
    probabilities = softmax_out / "output_00.bin"
    final_out = args.out / "pv"
    run(runner + ["--hwx", str(programs[2]), "--case",
        f"{probabilities},{inputs[2]}:{expected}", "--runs", "3",
        "--max-ulp", str(record["max_ulp"]), "--out-dir", str(final_out)])


if __name__ == "__main__":
    main()
