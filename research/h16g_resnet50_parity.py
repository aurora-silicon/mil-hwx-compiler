#!/usr/bin/env python3
"""Compare a rule-built H16G program with a private descriptor-fact include.

The reference file is an explicit command-line input and is never copied or
summarized into the repository.  Output names every differing task header and
register field so a private parity run remains auditable.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h13_td import register_words  # noqa: E402
from mint_oracles import parse_hwx  # noqa: E402

RECORD_BASE_MASK = (1 << 15) - 1
MASKED_RECORD = 1 << 31


def array_body(text: str, name: str) -> str:
    match = re.search(rf"{name}\[\]\s*=\s*\{{(.*?)\n\}};", text, re.S)
    if not match:
        raise ValueError(f"reference has no {name} array")
    return match.group(1)


def record_addresses(header: int) -> list[int]:
    base = (header & RECORD_BASE_MASK) * 4
    if header & MASKED_RECORD:
        return [base] + [base + 4 * delta for delta in range(1, 17)
                         if header & (1 << (delta + 14))]
    return [base + 4 * index
            for index in range(((header >> 15) & 0x3F) + 1)]


def reference_tasks(path: Path):
    text = path.read_text()
    records = [tuple(int(value.rstrip("uU"), 0) for value in item)
               for item in re.findall(
                   r"\{(0x[0-9a-f]+u?),\s*(\d+)u,\s*(\d+)u\}",
                   array_body(text, "resnetRecordFacts"), re.I)]
    values = [int(value, 0) for value in re.findall(
        r"0x[0-9a-f]+", array_body(text, "resnetRecordValues"), re.I)]
    tasks = []
    pattern = re.compile(r"\{\{([^}]+)\},\s*(\d+)u,\s*(\d+)u\}")
    for words, start, count in pattern.findall(
            array_body(text, "resnetTaskFacts")):
        header = [int(value, 0) for value in
                  re.findall(r"0x[0-9a-f]+", words, re.I)]
        registers = {}
        for packed, value_start, value_count in records[
                int(start):int(start) + int(count)]:
            addresses = record_addresses(packed)
            if len(addresses) != value_count:
                raise ValueError("reference record value count is inconsistent")
            registers.update(zip(
                addresses, values[value_start:value_start + value_count]))
        tasks.append((header, registers))
    return tasks


def hex32(value):
    return None if value is None else f"0x{value:08x}"


def compare(actual_path: Path, reference_path: Path):
    reference = reference_tasks(reference_path)
    decoded = parse_hwx(actual_path.read_bytes(), "h16")
    actual = decoded["task_descriptors"]
    result = {
        "actual_tasks": len(actual),
        "reference_tasks": len(reference),
        "header_differences": 0,
        "field_differences": 0,
        "missing_fields": 0,
        "extra_fields": 0,
        "tasks": [],
    }
    for index in range(max(len(actual), len(reference))):
        if index >= len(actual) or index >= len(reference):
            result["tasks"].append({
                "task": index,
                "status": "missing-actual" if index >= len(actual)
                          else "extra-actual",
            })
            continue
        reference_header, reference_registers = reference[index]
        actual_header = [int(value, 16)
                         for value in actual[index]["header_words"]]
        actual_registers = register_words(actual[index])
        headers = []
        for word, (want, got) in enumerate(zip(reference_header, actual_header)):
            if word == 1 or want == got:  # execution-cost hint is non-semantic
                continue
            headers.append({"word": word, "reference": hex32(want),
                            "actual": hex32(got)})
        fields = []
        for address in sorted(set(reference_registers) | set(actual_registers)):
            want, got = reference_registers.get(address), actual_registers.get(address)
            if want == got:
                continue
            status = "different"
            if got is None:
                status = "missing"
                result["missing_fields"] += 1
            elif want is None:
                status = "extra"
                result["extra_fields"] += 1
            fields.append({"address": f"0x{address:04x}", "status": status,
                           "reference": hex32(want), "actual": hex32(got)})
        if headers or fields:
            result["tasks"].append({"task": index, "headers": headers,
                                    "fields": fields})
        result["header_differences"] += len(headers)
        result["field_differences"] += len(fields)
    result["parity"] = (len(actual) == len(reference) and
                        not result["header_differences"] and
                        not result["field_differences"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual_hwx", type=Path)
    parser.add_argument("private_reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.actual_hwx, args.private_reference)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    print("parity", "PASS" if result["parity"] else "DIFF",
          f"headers={result['header_differences']}",
          f"fields={result['field_differences']}", file=sys.stderr)
    return 0 if result["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
