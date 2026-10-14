#!/usr/bin/env python3
"""Generate the H14-format elementwise parity templates from decoded oracles.

Writes `plugins/<T>/<T>ElementwiseTemplates.inc` from
`research/oracles/<t>/*.json` for every target in `ELEMENTWISE_TARGETS`, and
`plugins/H14/H14MatvecTemplates.inc` and `plugins/H14/H14ExpChainTemplates.inc`. Only decoded task words,
program-descriptor metadata, and constant-section contents are emitted; no HWX
container bytes are read or stored.

Regenerate and compare with:

```sh
python3 research/generate_h14_templates.py --check
```
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h13_td import decode_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MATVEC_OUTPUT = ROOT / "plugins/H14/H14MatvecTemplates.inc"
EXP_CHAIN_OUTPUT = ROOT / "plugins/H14/H14ExpChainTemplates.inc"
# Generations with an elementwise parity target, and the program-descriptor
# offset of the per-program word no formula resolves: 0x880 in H14's
# 0x890-byte descriptor, 0x890 in the 0x8a0-byte H17 and H18 descriptor.
ELEMENTWISE_TARGETS = {"h14": 0x880, "h17": 0x890, "h18": 0x890}
TRAILER_OFFSET = 0x810
CONSTANT_ALIGNMENT = 0x40

# Task-stream framing: a 16-byte zero prefix that decodes as a zero-size task,
# then every task 16-byte aligned with the last task left unpadded.
TASK_ALIGNMENT = 16
STREAM_PREFIX_BYTES = 16

BINARY_OPERATIONS = {"add": 0, "mul": 1, "maximum": 2, "minimum": 3, "sub": 4,
                     "real_div": 5}
UNARY_OPERATIONS = {"abs": 0, "exp": 1, "gelu": 2, "leaky_relu": 3, "relu": 4,
                    "rsqrt": 5, "sigmoid": 6, "silu": 7, "sqrt": 8, "tanh": 9,
                    # Apple emits a different program per gelu mode, so each
                    # approximation is its own encoder operation.
                    "gelu_sigmoid_approximation": 10,
                    "gelu_tanh_approximation": 11}
GELU_MODES = {"EXACT": "gelu",
              "SIGMOID_APPROXIMATION": "gelu_sigmoid_approximation",
              "TANH_APPROXIMATION": "gelu_tanh_approximation"}
RUNTIME_BINARY = {"add", "mul", "maximum", "minimum", "sub"}
UNARY = set(UNARY_OPERATIONS)
KINDS = {"binary_runtime": "BinaryRuntime", "binary_constant": "BinaryScalar",
         "unary": "Unary"}


def classify(oracle: dict[str, Any]) -> tuple[str, str, tuple[int, int, int],
                                              tuple[int, int, int]] | None:
    """The encoder key of a decoded elementwise oracle — kind, operation, the
    CHW result surface, and the CHW surface of the second operand — or None
    when the case sits outside what the H14 encoder models.

    Two envelope forms stay outside: a batched surface (`N > 1`), whose
    descriptor records one batch's bytes against a rank-4 shape the encoder's
    surface formula does not spell, and a `BLOBFILE` broadcast operand, whose
    constant section is not the operand payload (see research/h14-td-fields.md).
    """
    if oracle.get("error") is not None:
        return None
    parameters = oracle.get("parameters", {})
    operation = parameters.get("operation")
    family = oracle.get("family")
    shape = parameters.get("shape")
    if family not in ("binary_runtime", "binary_constant", "unary",
                      "env_activation", "env_broadcast") or not shape:
        return None
    if shape[0] != 1:
        return None
    chw = (shape[1], shape[2], shape[3])
    if family in ("unary", "env_activation"):
        if operation == "gelu":
            operation = GELU_MODES.get(parameters.get("mode") or "EXACT")
        return ("Unary", operation, chw, chw) if operation in UNARY else None
    if family == "binary_constant":
        return ("BinaryScalar", operation, chw, chw) \
            if parameters.get("constant") == "scalar" and \
            operation in BINARY_OPERATIONS else None
    if operation not in RUNTIME_BINARY:
        return None
    if family == "binary_runtime":
        return ("BinaryRuntime", operation, chw, chw)
    if parameters["output_shape"] != shape:
        raise ValueError(f"{oracle['case']}: broadcast result is not the x shape")
    operand = parameters.get("operand")
    if operand == "scalar":
        return ("BinaryScalar", operation, chw, chw)
    if operand != "runtime":
        return None
    operand_shape = parameters["operand_shape"]
    if operand_shape[0] != 1:
        return None
    return ("BinaryRuntime", operation, chw,
            (operand_shape[1], operand_shape[2], operand_shape[3]))


def selected_oracles(target: str) -> list[dict[str, Any]]:
    """Every decoded case of `target` the elementwise encoder covers: the
    shipped binary, scalar-constant, and unary campaign plus the envelope
    campaign's activation shapes and runtime broadcast forms."""
    return [oracle for oracle in
            (json.loads(path.read_text()) for path in
             sorted((ROOT / "research/oracles" / target).glob("*.json")))
            if classify(oracle) is not None]


def task_words(task: dict[str, Any], target: str) -> list[int]:
    """Rebuild one task's exact word stream from its decoded records."""
    words = [int(word, 16) for word in task["header_words"]]
    registers = {int(address, 16): int(value, 16)
                 for block in task["blocks"].values()
                 for address, value in block["words"].items()}
    for record in task["records"]:
        header = int(record["header"], 16)
        base = (header & 0x7FFF) * 4
        if header & 0x80000000:
            mask = (header >> 15) & 0xFFFF
            addresses = (base,) + tuple(
                base + (bit + 1) * 4 for bit in range(16) if mask & (1 << bit))
        else:
            addresses = tuple(
                base + index * 4 for index in range(((header >> 15) & 0x3F) + 1))
        if len(addresses) != record["count"]:
            raise ValueError(f"record {record['header']} address count differs")
        words.append(header)
        words.extend(registers[address] for address in addresses)
    stream = struct.pack(f"<{len(words)}I", *words)
    if len(stream) != task["size_bytes"] or decode_task(stream, target) != task:
        raise ValueError("rebuilt task does not decode back to the oracle")
    return words


def task_stream(oracle: dict[str, Any]) -> list[int]:
    """Assemble the __TEXT/__text word stream for one oracle case."""
    words = [0] * (STREAM_PREFIX_BYTES // 4)
    tasks = oracle["task_descriptors"]
    for index, task in enumerate(tasks):
        words.extend(task_words(task, oracle["target"]))
        if index + 1 != len(tasks):
            while len(words) % (TASK_ALIGNMENT // 4):
                words.append(0)
    declared = oracle["program_descriptor"]["text_words"]
    if len(words) != declared:
        raise ValueError(
            f"{oracle['case']}: assembled {len(words)} text words, program "
            f"descriptor declares {declared}")
    return words


def section_runs(oracle: dict[str, Any],
                 key: str = "constant_section") -> list[tuple[int, int, int]]:
    """Nonzero fp16 halfword runs (index, bits, count) of the constant section,
    or of the H17/H18 `kernel_section`."""
    section = oracle[key]
    halfwords: dict[int, int] = {}
    if section["nonzero_bytes"]:
        words = section["nonzero_fp16_words"]
        if words is None:
            # Oracles above 256 bytes keep only hashes; the scalar real_div
            # divisor reciprocal splat is the one such parity case.
            elements = oracle["parameters"]["shape"][1]
            for index in range(elements):
                halfwords[elements + index] = 0x4000
        else:
            for word in words:
                value = word["value"]
                if value == "nan":
                    bits = 0x7E00
                elif value == "+inf":
                    bits = 0x7C00
                elif value == "-inf":
                    bits = 0xFC00
                else:
                    bits = struct.unpack("<H", struct.pack("<e", value))[0]
                halfwords[word["index"]] = bits
    data = bytearray(section["size"])
    for index, bits in halfwords.items():
        struct.pack_into("<H", data, index * 2, bits)
    digest = hashlib.sha256(data).hexdigest()
    if digest != section["sha256"]:
        raise ValueError(
            f"{oracle['case']}: rebuilt constants hash {digest} but the oracle "
            f"records {section['sha256']}")
    runs: list[tuple[int, int, int]] = []
    for index in sorted(halfwords):
        bits = halfwords[index]
        if runs and runs[-1][1] == bits and runs[-1][0] + runs[-1][2] == index:
            runs[-1] = (runs[-1][0], bits, runs[-1][2] + 1)
        else:
            runs.append((index, bits, 1))
    return runs


def word_rows(words: list[int], indent: str = "    ") -> str:
    rows = []
    for start in range(0, len(words), 8):
        row = ", ".join(f"0x{word:08x}" for word in words[start:start + 8])
        rows.append(f"{indent}{row},")
    return "\n".join(rows)


def template_fields(oracle: dict[str, Any]) -> tuple:
    """Everything the emitted template carries, so two oracles that share an
    encoder key are proven to carry identical Apple output."""
    trailer = oracle["program_descriptor"]["trailing_words"]
    if int(trailer[18], 16):
        raise ValueError(
            f"{oracle['case']}: descriptor word 0x858 is "
            f"{trailer[18]}, which the elementwise encoder does not carry")
    unresolved = (ELEMENTWISE_TARGETS[oracle["target"]] - TRAILER_OFFSET) // 4
    kernel = oracle.get("kernel_section")
    return (tuple(task_stream(oracle)), tuple(section_runs(oracle)),
            oracle["constant_section"]["size"],
            oracle["program_descriptor"]["task_count"],
            int(trailer[20], 16), int(trailer[unresolved], 16),
            tuple(section_runs(oracle, "kernel_section")) if kernel else (),
            kernel["size"] if kernel else 0)


def generate(target: str) -> str:
    templates: dict[tuple, tuple[dict[str, Any], tuple]] = {}
    for oracle in selected_oracles(target):
        key = classify(oracle)
        fields = template_fields(oracle)
        previous = templates.setdefault(key, (oracle, fields))
        if previous[1] != fields:
            raise ValueError(
                f"{oracle['case']} and {previous[0]['case']} share the encoder "
                f"key {key} but Apple emitted different programs")
    name = target.upper()
    lines = [f"// Generated from decoded {name} oracle task words by",
             "// research/generate_h14_templates.py. No HWX container bytes.",
             ""]
    entries = []
    for index, key in enumerate(templates):
        oracle, (stream, runs, constantBytes, taskCount, records,
                 unresolved, kernelRuns, kernelBytes) = templates[key]
        kind, operation, shape, operand = key
        code = (UNARY_OPERATIONS if kind == "Unary"
                else BINARY_OPERATIONS)[operation]
        lines.append(f"// {oracle['case']}")
        lines.append(f"static constexpr std::uint32_t k{name}Text{index}[] = {{")
        lines.append(word_rows(list(stream)))
        lines.append("};")
        constants = "nullptr, 0"
        if runs:
            lines.append(
                f"static constexpr ConstantRun k{name}Constants{index}[] = {{")
            for start, bits, count in runs:
                lines.append(f"    {{{start}, 0x{bits:04x}, {count}}},")
            lines.append("};")
            constants = (f"k{name}Constants{index}, "
                         f"std::size(k{name}Constants{index})")
        kernel = ""
        if kernelBytes:
            lines.append(
                f"static constexpr ConstantRun k{name}Kernel{index}[] = {{")
            for start, bits, count in kernelRuns:
                lines.append(f"    {{{start}, 0x{bits:04x}, {count}}},")
            lines.append("};")
            kernel = (f", k{name}Kernel{index}, "
                      f"std::size(k{name}Kernel{index}), {kernelBytes}")
        lines.append("")
        entries.append(
            f"    {{ElementwiseKind::{kind}, {code}, "
            f"{{{shape[0]}, {shape[1]}, {shape[2]}}}, "
            f"{{{operand[0]}, {operand[1]}, {operand[2]}}}, k{name}Text{index}, "
            f"std::size(k{name}Text{index}), {taskCount}, "
            f"{constants}, {constantBytes}, "
            f"0x{records:08x}, 0x{unresolved:08x}{kernel}}},")
    lines.append(
        f"static constexpr OracleTaskTemplate k{name}ElementwiseTasks[] = {{")
    lines.extend(entries)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def selected_matvec_oracles() -> list[dict[str, Any]]:
    """The H14 `transpose_y=true` matmul geometries Apple decoded, ordered by
    rows, reduction, then columns."""
    selected = []
    for path in sorted((ROOT / "research/oracles/h14").glob("matmul_m*_ty1.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is None and oracle["family"] == "matmul":
            selected.append(oracle)
    selected.sort(key=lambda item: (item["parameters"]["rows"],
                                    item["parameters"]["reduction"],
                                    item["parameters"]["columns"]))
    return selected


def generate_matvec() -> str:
    oracles = selected_matvec_oracles()
    lines = ["// Generated from decoded H14 matmul oracle task words by",
             "// research/generate_h14_templates.py. No HWX container bytes.",
             ""]
    entries = []
    for index, oracle in enumerate(oracles):
        parameters = oracle["parameters"]
        descriptor = oracle["program_descriptor"]
        rows = parameters["rows"]
        reduction = parameters["reduction"]
        columns = parameters["columns"]
        stream = task_stream(oracle)
        constant_offset = (int(descriptor["constant_address"], 16) -
                           int(descriptor["text_address"], 16))
        aligned = (len(stream) * 4 + CONSTANT_ALIGNMENT - 1) & \
            ~(CONSTANT_ALIGNMENT - 1)
        if constant_offset != aligned:
            raise ValueError(
                f"{oracle['case']}: constants sit at {constant_offset}, not "
                f"the {aligned} the aligned text stream implies")
        if oracle["constant_section"]["size"] != reduction * columns * 2:
            raise ValueError(
                f"{oracle['case']}: constant section is not K*N halfwords")
        lines.append(f"// {oracle['case']}")
        lines.append(f"static constexpr std::uint32_t kH14MatvecText{index}[] = {{")
        lines.append(word_rows(stream))
        lines.append("};")
        lines.append("")
        trailer = descriptor["trailing_words"]
        if int(trailer[18], 16):
            raise ValueError(
                f"{oracle['case']}: descriptor word 0x858 is {trailer[18]}, "
                "which the matvec encoder does not carry")
        entries.append(
            f"    {{{rows}, {reduction}, {columns}, kH14MatvecText{index}, "
            f"std::size(kH14MatvecText{index}), {descriptor['task_count']}, "
            f"0x{int(trailer[20], 16):08x}, 0x{int(trailer[28], 16):08x}}},")
    lines.append("static constexpr OracleMatvecTemplate kMatvecTasks[] = {")
    lines.extend(entries)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def selected_exp_chain_oracles() -> list[dict[str, Any]]:
    """The decoded H14 programs of the scaled exp chain (exp, mul, add, mul, mul as one MIL program), ordered by channels."""
    selected = []
    for path in sorted((ROOT / "research/oracles/h14").glob("chain_exp_d4_c*.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is None:
            selected.append(oracle)
    selected.sort(key=lambda item: item["parameters"]["shape"][1])
    return selected


def generate_exp_chain() -> str:
    """One fixed three-task program per decoded size. The constant section is the H13 exponential table, proven by its recorded
    SHA-256; the scalar constants of the MIL live in the task words, so the section holds nothing else."""
    from mint_norm_probes import constant_kind, kernel_tables  # noqa: PLC0415
    tables = kernel_tables()
    lines = ["// Generated from decoded H14 oracles of the scaled exp chain by",
             "// research/generate_h14_templates.py. No HWX container bytes.",
             ""]
    entries = []
    for index, oracle in enumerate(selected_exp_chain_oracles()):
        channels = oracle["parameters"]["shape"][1]
        descriptor = oracle["program_descriptor"]
        stream = task_stream(oracle)
        constant_offset = (int(descriptor["constant_address"], 16) -
                           int(descriptor["text_address"], 16))
        aligned = (len(stream) * 4 + CONSTANT_ALIGNMENT - 1) & ~(CONSTANT_ALIGNMENT - 1)
        if constant_offset != aligned:
            raise ValueError(f"{oracle['case']}: constants sit at {constant_offset}, not the {aligned} the aligned text stream implies")
        if constant_kind(oracle, tables) != "Exponential":
            raise ValueError(f"{oracle['case']}: the constant section is not the exponential table")
        if descriptor["task_count"] != 3 or oracle["constant_section"]["size"] != 128:
            raise ValueError(f"{oracle['case']}: expected three tasks and a 128-byte constant section")
        trailer = descriptor["trailing_words"]
        lines.append(f"// {oracle['case']}")
        lines.append(f"static constexpr std::uint32_t kH14ExpChainText{index}[] = {{")
        lines.append(word_rows(stream))
        lines.append("};")
        lines.append("")
        entries.append(
            f"    {{{channels}, kH14ExpChainText{index}, std::size(kH14ExpChainText{index}), {descriptor['task_count']}, "
            f"{oracle['constant_section']['size']}, 0x{int(trailer[20], 16):08x}, 0x{int(trailer[28], 16):08x}, "
            f"0x{int(trailer[18], 16):08x}}},")
    lines.append("static constexpr OracleExpChainTemplate kExpChainTasks[] = {")
    lines.extend(entries)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail when a checked-in file is stale")
    arguments = parser.parse_args()
    outputs = [(ROOT / f"plugins/{target.upper()}/{target.upper()}"
                       "ElementwiseTemplates.inc", generate(target))
               for target in ELEMENTWISE_TARGETS]
    outputs.append((MATVEC_OUTPUT, generate_matvec()))
    outputs.append((EXP_CHAIN_OUTPUT, generate_exp_chain()))
    for path, generated in outputs:
        if arguments.check:
            if (path.read_text() if path.exists() else "") != generated:
                raise SystemExit(f"{path} is stale; regenerate it")
            print(f"parity templates: up to date ({path})")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated)
        print(f"wrote {path} ({len(generated)} bytes)")


if __name__ == "__main__":
    main()
