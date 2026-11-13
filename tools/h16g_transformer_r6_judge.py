#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def halves(path: Path) -> list[float]:
    raw = path.read_bytes()
    if len(raw) % 2:
        raise ValueError(f"{path}: odd fp16 byte count")
    return [item[0] for item in struct.iter_unpack("<e", raw)]


def metrics(actual: list[float], expected: list[float]) -> dict:
    finite = [(a, e) for a, e in zip(actual, expected)
              if math.isfinite(a) and math.isfinite(e)]
    denominator = math.sqrt(math.fsum(e * e for e in expected))
    numerator = math.sqrt(math.fsum((a - e) ** 2
                                    for a, e in zip(actual, expected)))
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


def main() -> int:
    root = Path(__file__).resolve().parent
    case, output = sys.argv[1], Path(sys.argv[2])
    cases = json.loads((root / "cases.json").read_text())
    record = cases["runnable"][case]
    cold_runs = int(cases.get("cold_runs", 3))
    tokens, channels = int(record["tokens"]), int(record["channels"])
    flat = halves(root / record["inputs"][0])
    rows = [flat[start:start + channels]
            for start in range(0, len(flat), channels)]
    means = [math.fsum(row) / channels for row in rows]
    variances = [math.fsum((value - means[index]) ** 2 for value in row) /
                 channels for index, row in enumerate(rows)]
    epsilon = struct.unpack("<e", struct.pack("<e", 1e-5))[0]
    normalized = [(value - means[index]) /
                  math.sqrt(variances[index] + epsilon)
                  for index, row in enumerate(rows) for value in row]

    results, first = {}, None
    prefix_bytes = {}
    for item in record["prefixes"]:
        task = int(item["task_count"])
        name = f"prefix-{task:02d}"
        count = int(item["result_rows"]) * int(item["result_row_bytes"]) // 2
        paths = [output / name / f"cold-{index:02d}" / "output_00.bin"
                 for index in range(cold_runs)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            results[name] = {"missing": missing}
            first = task if first is None else first
            continue
        raw = [path.read_bytes()[:count * 2] for path in paths]
        prefix_bytes[task] = raw
        actual = halves(paths[0])[:count]
        stage = item.get("expected_stage")
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
        elif stage == "normalized":
            expected = []
            for row_index in range(start, start + height):
                expected.extend(normalized[
                    row_index * channels:row_index * channels + width])
        else:
            expected = actual
        result = {"expected_stage": stage, **metrics(actual, expected),
                  "bit_identical_cold": len(set(raw)) == 1,
                  "sha256": [hashlib.sha256(value).hexdigest()
                             for value in raw]}
        failed = (result["nonfinite"] or not result["bit_identical_cold"] or
                  (stage is not None and result["rel_l2"] > 1e-2))
        results[name] = result
        if first is None and failed:
            first = task

    control_paths = [output / "mean-dead-control" / f"cold-{index:02d}" /
                     "output_00.bin" for index in range(cold_runs)]
    control = {"missing": [str(path) for path in control_paths
                           if not path.is_file()]}
    if not control["missing"] and 1 in prefix_bytes:
        count = int(record["prefixes"][0]["result_rows"]) * \
            int(record["prefixes"][0]["result_row_bytes"])
        raw = [path.read_bytes()[:count] for path in control_paths]
        control = {"bit_identical_cold": len(set(raw)) == 1,
                   "matches_prefix_01": raw == prefix_bytes[1],
                   "sha256": [hashlib.sha256(value).hexdigest()
                              for value in raw]}

    final_paths = [output / "final" / f"cold-{index:02d}" / "output_00.bin"
                   for index in range(cold_runs)]
    final = {"missing": [str(path) for path in final_paths
                         if not path.is_file()]}
    if not final["missing"]:
        raw = [path.read_bytes()[:tokens * channels * 2]
               for path in final_paths]
        final = {**metrics(halves(final_paths[0])[:tokens * channels], normalized),
                 "bit_identical_cold": len(set(raw)) == 1,
                 "sha256": [hashlib.sha256(value).hexdigest()
                            for value in raw]}
    control_pass = (not control.get("missing") and
                    control.get("bit_identical_cold") and
                    control.get("matches_prefix_01"))
    final_pass = (not final.get("missing") and not final.get("nonfinite") and
                  final.get("bit_identical_cold") and
                  final.get("rel_l2", math.inf) <= 0.02)
    passed = first is None and control_pass and final_pass
    print(json.dumps({"prefixes": results, "first_failing_task": first,
                      "dead_field_control": control, "final": final,
                      "thresholds": {"prefix_rel_l2": 0.01,
                                     "final_rel_l2": 0.02},
                      "pass": passed}, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
