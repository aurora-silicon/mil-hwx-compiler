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
    cases = json.loads((root / "cases.json").read_text())
    record = cases["runnable"][args.case]
    cold_runs = int(cases.get("cold_runs", 3))
    args.out.mkdir(parents=True, exist_ok=True)
    programs = [root / path for path in record["programs"]]
    inputs = [root / path for path in record["inputs"]]
    runner = ["timeout", "60", sys.executable, "-I", str(root / "ane-run.py")]

    def cold(program: Path, case: str, destination: Path) -> bool:
        failed = False
        for index in range(cold_runs):
            failed |= run(runner + ["--hwx", str(program), "--case", case,
                "--runs", "1", "--out-dir",
                str(destination / f"cold-{index:02d}")], check=False) != 0
        return failed

    if record["kind"] == "single":
        failed = False
        prefixes = {int(item["task_count"]): item
                    for item in record["prefixes"]}
        for task_count in record["prefix_run_order"]:
            item = prefixes[int(task_count)]
            name = f"prefix-{int(task_count):02d}"
            failed |= cold(root / item["program"], str(inputs[0]),
                           args.out / name)
        control = record["dead_field_control"]
        failed |= cold(root / control["program"], str(inputs[0]),
                       args.out / "mean-dead-control")
        failed |= cold(programs[0], str(inputs[0]), args.out / "final")
        raise SystemExit(1 if failed else 0)

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
    expected = root / record["expected"]
    final_out = args.out / "pv"
    run(runner + ["--hwx", str(programs[2]), "--case",
        f"{softmax_out / 'output_00.bin'},{inputs[2]}:{expected}",
        "--runs", "3", "--max-ulp", str(record["max_ulp"]),
        "--out-dir", str(final_out)])


if __name__ == "__main__":
    main()
