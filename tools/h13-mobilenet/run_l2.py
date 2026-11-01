#!/usr/bin/env python3
"""On the designated M1 Pro: 16 cold comparisons, top-1, then exec-only soak.

Invocation is a hardware action; the ANE lead submits it through the device
controller. This script never transports code or contacts another machine.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from chain import pack, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--passes", type=int, default=5000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("passes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    package = args.package.resolve()
    exe = Path(__file__).resolve().with_name("anechain")
    lm = json.loads((package / "linked/manifest.json").read_text())
    pm = json.loads((package / "perop/manifest.json").read_text())
    lp = lm["programs"][0]
    linked = [str(package / "linked" / lp["file"])]
    perop = [str(package / "perop" / pm["programs"][i]["file"]) for i in pm["dispatchPlan"]]
    references = np.load(package / "refs.npz", allow_pickle=False)
    assert len(references["inputs"]) == 16
    device = Path("/sys/class/accel/accel0/device")

    def snapshot(name):
        counters = {}
        for attribute in ("ane_stats", "wedged"):
            text = (device / attribute).read_text()
            (args.output / f"{name}-{attribute}.txt").write_text(text)
            if attribute == "ane_stats":
                for line in text.splitlines():
                    fields = line.split()
                    if len(fields) == 2:
                        try:
                            counters[fields[0].rstrip(':')] = int(fields[1], 0)
                        except ValueError:
                            pass
        return counters

    def execute(name, programs, repeat, warmup=0, exec_only=False):
        output = args.output / (name + ".bin")
        command = [str(exe), "-w", str(warmup), "-r", str(repeat), "-i", str(args.output / "input.bin"),
                   "-o", str(output), *(["-n"] if exec_only else []), *programs]
        run = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (args.output / (name + ".stdout")).write_text(run.stdout)
        (args.output / (name + ".stderr")).write_text(run.stderr)
        (args.output / (name + ".command.json")).write_text(json.dumps(command) + "\n")
        if run.returncode:
            raise RuntimeError(f"{name}: anechain exited {run.returncode}; see preserved stderr")
        return unpack(output.read_bytes(), lp["outputs"][0]["nchw"]), json.loads(run.stdout.strip().splitlines()[-1])

    before = snapshot("before")
    rows = []
    try:
        probe_command = [sys.executable, str(exe.with_name("coldcheck.py")),
                         str(package / "probes/wide128"), "6"]
        probe = subprocess.run(probe_command, capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, ANECHAIN=str(exe),
                                        ANEBENCH_TMP=str(args.output / "wide128")))
        (args.output / "wide128.stdout").write_text(probe.stdout)
        (args.output / "wide128.stderr").write_text(probe.stderr)
        (args.output / "wide128.command.json").write_text(json.dumps(probe_command) + "\n")
        if probe.returncode:
            raise RuntimeError(f"wide128 cold probe exited {probe.returncode}; see preserved logs")
        for i, x in enumerate(references["inputs"]):
            (args.output / "input.bin").write_bytes(pack(x, lp["inputs"][0]["nchw"], lp["inputs"][0]["allocationBytes"]))
            a, _ = execute(f"cold-{i}-linked", linked, 1)
            b, _ = execute(f"cold-{i}-perop", perop, 1)
            logits = a.astype(np.float64).mean(axis=(1, 2))[:1001]
            ref = references["logits_relu"][i]
            row = dict(image=i, exact=bool(np.array_equal(a.view(np.uint16), b.view(np.uint16))),
                       finite=bool(np.isfinite(a).all()), top1=int(logits.argmax()), referenceTop1=int(ref.argmax()),
                       relL2=float(np.linalg.norm(logits - ref) / np.linalg.norm(ref)))
            rows.append(row)
            (args.output / "cold.json").write_text(json.dumps(rows, indent=2) + "\n")
            assert row["exact"] and row["finite"] and row["top1"] == row["referenceTop1"], row
        _, timing = execute("exec-only", linked, args.passes, 20, True)
        assert (device / "wedged").read_text().strip() == "0"
        after = snapshot("after")
        errors = {key: value - before[key] for key, value in after.items()
                  if key in before and ("error" in key.lower() or "recover" in key.lower())}
        assert all(delta == 0 for delta in errors.values()), errors
        result = dict(wide128ColdExact=6, coldExact=16, top1Agree=16, timing=timing, successfulTimedPasses=args.passes,
                      taskExecFailures=0, driverErrorCounterDeltas=errors,
                      driverErrorCountersAvailable=bool(errors),
                      baselineP50Us=1641.0, p50Ratio=timing["p50_us"] / 1641.0)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        snapshot("after")


if __name__ == "__main__":
    main()
