#!/usr/bin/env python3
"""Compares emitted H14 task streams word-for-word with decoded Apple oracles."""
import collections
import contextlib
import hashlib
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h13_td import decode_task, split_h14_tasks  # noqa: E402
import inspect_hwx  # noqa: E402
import mint_h14_matvec_probes as probes  # noqa: E402
import mint_h14_norm_probes as norm_templates  # noqa: E402
import mint_norm_probes as norm_probes  # noqa: E402
import mint_conv_probes  # noqa: E402
import mint_chain_probes  # noqa: E402
import mint_oracles  # noqa: E402
from generate_h14_templates import (selected_matvec_oracles,  # noqa: E402
                                    selected_oracles)
from timeouts import scaled_timeout  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
TILE_BYTES = 0x4000
MATVEC_ENCODER = "apple-parity-matvec"
NORM_ENCODER = "apple-parity-norm"
ELEMENTWISE_ENCODER = "h14-oracle-parity"
CONV_ENCODER = "apple-parity-conv"
EXP_CHAIN_ENCODER = "apple-parity-expchain"
CONV_FAMILIES = {"env_conv", "conv_probe"}


def encoder(oracle):
    if oracle["case"].startswith("chain_exp_d4_"):
        return EXP_CHAIN_ENCODER
    if oracle["family"] in CONV_FAMILIES:
        return CONV_ENCODER
    if oracle["family"] in {"binary_runtime", "binary_constant", "unary",
                            "env_activation", "env_broadcast"}:
        return ELEMENTWISE_ENCODER
    if oracle["family"] in {"normalization", "reduction"}:
        return NORM_ENCODER
    return MATVEC_ENCODER


def write_weights(oracle, root):
    """Recreates the exact weights.bin Apple's compiler saw for this case."""
    description = oracle.get("weights", {})
    if description.get("storage") != "BLOBFILE":
        return
    parameters = oracle["parameters"]
    if description.get("value") == \
            "fp16 bits 0x3400 + index, one value per constant":
        shapes = [tuple(shape) for shape in parameters["constant_matrices"]]
        (root / "weights.bin").write_bytes(mint_chain_probes.weight_blob(shapes))
        return
    if oracle["family"] == "matvec_probe":
        payload = probes.payload(parameters["reduction"], parameters["columns"],
                                 parameters["pattern"])
        assert hashlib.sha256(payload).hexdigest() == \
            description["payload_sha256"], oracle["case"]
    elif description.get("value") == "fp16 signed period-5, scale 0.03125, offset 2":
        pattern = b"".join(struct.pack("<e", 0.03125 * (i - 2)) for i in range(5))
        size = description["payload_bytes"]
        payload = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
        assert hashlib.sha256(payload).hexdigest() == description["dense_sha256"]
    elif description.get("value") == "indexed_finite":
        payload = mint_conv_probes.finite_index_payload(description["payload_bytes"] // 2, description["seed"])
    elif description.get("value") == "distinct":
        # The known-weight convolution probes carry one distinct fp16 pattern
        # per element, which is what proves the packing permutation.
        payload = mint_conv_probes.known_weights(
            description["payload_bytes"] // 2)
    else:
        assert description["value"] == "fp16(0x1p-1)", oracle["case"]
        payload = mint_oracles.half_payload(description["payload_bytes"] // 2)
    (root / "weights.bin").write_bytes(mint_oracles.blob(payload))


def grid_probe_oracles():
    """The known-weight probes whose geometry the parity encoder covers, so the
    packed constant section is proven against a nonuniform Apple weight."""
    covered = {(oracle["parameters"]["rows"], oracle["parameters"]["reduction"],
                oracle["parameters"]["columns"])
               for oracle in selected_matvec_oracles()}
    selected = []
    for path in sorted((ROOT / "research/oracles/h14").glob("h14mv_*.json")):
        oracle = json.loads(path.read_text())
        parameters = oracle["parameters"]
        if oracle.get("error") is None and \
                (parameters["rows"], parameters["reduction"],
                 parameters["columns"]) in covered:
            selected.append(oracle)
    return selected


def norm_oracles():
    """Every decoded H14 softmax, layer_norm, and reduction oracle, including
    the shipped campaign's two shapes and the `h14norm_` probe grid. The
    encoder covers each one: `mint_h14_norm_probes.py` builds its template
    table from exactly this predicate."""
    selected = []
    for path in sorted((ROOT / "research/oracles/h14").glob("*.json")):
        oracle = json.loads(path.read_text())
        if oracle["family"] in ("normalization", "reduction") and \
                norm_probes.covered(oracle):
            selected.append(oracle)
    return selected


def check_norm_templates():
    """The checked-in norm table must be what the decoded oracles generate, the
    way `generate_h14_templates.py --check` guards the elementwise table."""
    generated = io.StringIO()
    records = norm_templates.selected("h14")
    norm_templates.emit(records, generated)
    checked_in = (ROOT / "plugins/H14/H14NormTemplates.inc").read_text()
    assert checked_in == generated.getvalue(), \
        "plugins/H14/H14NormTemplates.inc is stale; regenerate it with " \
        "research/mint_h14_norm_probes.py --emit-templates"
    return records


def compile_oracle(oracle, output, artifact_format):
    mil = output.parent / f"{oracle['case']}-{artifact_format}.mil"
    mil.write_text(oracle["mil"])
    write_weights(oracle, output.parent)
    result = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(output.parent),
         "--target", "H14", "--format", artifact_format, "--output", str(output)],
        capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
    assert result.returncode == 0, \
        f"{oracle['case']} {artifact_format}: {result.stdout}{result.stderr}"
    return json.loads((output / "manifest.json").read_text())


def aligned_blob(payload):
    """A weights.bin whose sub-header fields sit exactly where
    `ANEBlobResolver` reads them, so the resolved constant is `payload` alone.
    `mint_oracles.blob` packs the length and payload offset four bytes lower,
    which is why Apple's campaign sections start with the blob header."""
    data = bytearray(128 + len(payload))
    struct.pack_into("<II", data, 0, 1, 2)
    struct.pack_into("<I", data, 64, 0xDEADBEEF)
    struct.pack_into("<Q", data, 72, len(payload))
    struct.pack_into("<Q", data, 80, 128)
    data[128:] = payload
    return bytes(data)


def check_transposed_weights(root):
    """Apple rejects `transpose_y=false`, so no oracle covers it. Compiling the
    same non-square matrix in both forms must give one identical artifact, and
    its constants must be the packing the [N, K] weight implies."""
    rows, reduction, columns = 2, 512, 256
    weight = bytearray(columns * reduction * 2)
    transposed = bytearray(len(weight))
    for column in range(columns):
        for index in range(reduction):
            bits = 0x3000 | ((column * 131 + index * 17) & 0x3FF)
            struct.pack_into("<H", weight, (column * reduction + index) * 2, bits)
            struct.pack_into("<H", transposed,
                             (index * columns + column) * 2, bits)
    artifacts = {}
    for flag, payload, weight_shape in (
            ("true", weight, (columns, reduction)),
            ("false", transposed, (reduction, columns))):
        model = root / f"transpose-y-{flag}"
        model.mkdir()
        (model / "weights.bin").write_bytes(aligned_blob(bytes(payload)))
        x_kind = f"tensor<fp16, [{rows}, {reduction}]>"
        y_kind = f"tensor<fp16, [{rows}, {columns}]>"
        w_kind = f"tensor<fp16, [{weight_shape[0]}, {weight_shape[1]}]>"
        (model / "model.mil").write_text(mint_oracles.program(f"{x_kind} x", [
            'bool f = const()[name = string("f"), val = bool(false)];',
            f'bool ty = const()[name = string("ty"), val = bool({flag})];',
            f'{w_kind} w = const()[name = string("w"), val = {w_kind}'
            '(BLOBFILE(path = string("@model_path/weights.bin"), '
            'offset = uint64(64)))];',
            f'{y_kind} product = matmul(transpose_x = f, transpose_y = ty, '
            'x = x, y = w)[name = string("product")];'], "product"))
        result = subprocess.run(
            [str(COMPILER), "--mil", str(model / "model.mil"),
             "--model-root", str(model), "--target", "H14", "--format", "anec",
             "--output", str(model / "out")],
            capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
        assert result.returncode == 0, \
            f"transpose_y={flag}: {result.stdout}{result.stderr}"
        artifacts[flag] = (model / "out/program-0.anec").read_bytes()
    assert artifacts["true"] == artifacts["false"], \
        "transpose_y=false does not reproduce the transpose_y=true artifact"
    _, constants, _, _ = anec_contents(artifacts["true"])
    assert constants == probes.packed_section(reduction, columns, bytes(weight)), \
        "the emitted constants are not the modelled packing of the [N, K] weight"


def unpack_commands(hwx):
    """Decode the H14 fields research/mint_oracles.py records from Apple HWX."""
    magic, _, subtype, _, count, command_bytes, _, _ = struct.unpack_from("<8I", hwx)
    assert magic == 0xBEEFFACE and subtype == 5, (hex(magic), subtype)
    cursor = 32
    end = cursor + command_bytes
    sections = {}
    program = None
    tensors = []
    for _ in range(count):
        command, size = struct.unpack_from("<2I", hwx, cursor)
        kind = struct.unpack_from("<I", hwx, cursor + 8)[0] if size >= 12 else None
        if command == 0x19:
            fields = struct.unpack_from("<2I16s4Q4I", hwx, cursor)
            segment = fields[2].split(b"\0", 1)[0].decode()
            section_cursor = cursor + 72
            for _ in range(fields[-2]):
                values = struct.unpack_from("<16s16s2Q8I", hwx, section_cursor)
                section = values[0].split(b"\0", 1)[0].decode()
                sections[(segment, section)] = {
                    "address": values[2], "size": values[3], "offset": values[4]}
                section_cursor += 80
        elif command == 4 and kind == 4 and size >= 0x838:
            program = {
                "kind": kind, "command_size": size,
                "text_address": hex(struct.unpack_from("<Q", hwx, cursor + 0x10)[0]),
                "constant_address": hex(
                    struct.unpack_from("<Q", hwx, cursor + 0x20)[0]),
                "resource_addresses": [
                    hex(value) for value in
                    struct.unpack_from("<5Q", hwx, cursor + 0x30)],
                "text_words": struct.unpack_from("<I", hwx, cursor + 0x824)[0],
                "task_count": struct.unpack_from("<I", hwx, cursor + 0x830)[0],
                "trailing_words": [
                    f"0x{value:08x}" for value in struct.unpack_from(
                        f"<{(size - 0x810) // 4}I", hwx, cursor + 0x810)],
            }
        elif command == 4 and kind == 3 and size >= 0x78:
            tensors.append({
                "binding": struct.unpack_from("<I", hwx, cursor + 0x14)[0],
                "element_code": struct.unpack_from("<I", hwx, cursor + 0x24)[0],
                "shape": list(struct.unpack_from("<4I", hwx, cursor + 0x28)),
                "strides": list(struct.unpack_from("<4Q", hwx, cursor + 0x50)),
                "total_bytes": struct.unpack_from("<Q", hwx, cursor + 0x70)[0],
            })
        cursor += size
    assert cursor == end
    return sections, program, tensors


def anec_contents(payload):
    content_bytes, first_task_bytes, task_count, stream_bytes, constant_bytes, \
        inputs, outputs = struct.unpack_from("<QIIQQII", payload)
    stream = payload[0x1000:0x1000 + stream_bytes]
    tasks = split_h14_tasks(stream)
    constant_offset = (stream_bytes + 0x3F) & ~0x3F
    constants = payload[0x1000 + constant_offset:
                        0x1000 + constant_offset + constant_bytes]
    header = {"first_task_bytes": first_task_bytes, "task_count": task_count,
              "stream_bytes": stream_bytes, "inputs": inputs, "outputs": outputs,
              "content_bytes": content_bytes}
    return tasks, constants, constant_offset, header


def assert_tasks(actual, oracle):
    expected = oracle["task_descriptors"]
    assert len(actual) == len(expected), \
        f"{oracle['case']}: {len(actual)} tasks, oracle has {len(expected)}"
    for index, task in enumerate(actual):
        assert decode_task(task, "h14") == expected[index], \
            f"{oracle['case']} task {index} words differ from the oracle"


def assert_constants(constants, oracle):
    expected = oracle["constant_section"]
    assert len(constants) == expected["size"], oracle["case"]
    assert hashlib.sha256(constants).hexdigest() == expected["sha256"], \
        f"{oracle['case']} constant section differs from the oracle"
def descriptor_words(descriptor):
    return {address: value
            for block in descriptor["blocks"].values()
            for address, value in block["words"].items()}


def task_word_differences(left, right):
    left_words = descriptor_words(left)
    right_words = descriptor_words(right)
    assert left_words.keys() == right_words.keys()
    return {address: (left_words[address], right_words[address])
            for address in left_words if left_words[address] != right_words[address]}


def fp16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def check_anec(oracle, output, manifest):
    record = manifest["programs"][0]
    assert record["encoder"] == encoder(oracle), oracle["case"]
    payload = (output / record["file"]).read_bytes()
    tasks, constants, constant_offset, header = anec_contents(payload)
    assert_tasks(tasks, oracle)
    assert_constants(constants, oracle)
    text_bytes = oracle["program_descriptor"]["text_words"] * 4
    assert header["stream_bytes"] == text_bytes, oracle["case"]
    assert header["task_count"] == oracle["program_descriptor"]["task_count"], \
        oracle["case"]
    assert header["first_task_bytes"] == oracle["task_descriptors"][0]["size_bytes"], \
        oracle["case"]
    assert header["inputs"] == len(oracle["tensor_descriptors"]) - 1, oracle["case"]
    assert header["outputs"] == 1, oracle["case"]
    assert record["constantOffset"] == constant_offset == \
        (text_bytes + 0x3F) & ~0x3F, oracle["case"]
    assert header["content_bytes"] == constant_offset + len(constants), \
        oracle["case"]


def check_hwx(oracle, output, manifest):
    record = manifest["programs"][0]
    assert record["encoder"] == encoder(oracle), oracle["case"]
    path = output / record["file"]
    payload = path.read_bytes()
    sections, program, tensors = unpack_commands(payload)
    # The recorder learned `task_section` after the first H14 campaign, so the
    # older elementwise and matmul records carry one descriptor key fewer.
    expected = oracle["program_descriptor"]
    if "task_section" in expected:
        program["task_section"] = {
            "offset": 0, "size": sections[("__TEXT", "__text")]["size"]}
    assert program == expected, oracle["case"]
    assert tensors == oracle["tensor_descriptors"], oracle["case"]
    text = sections[("__TEXT", "__text")]
    constants = sections[("__TEXT", "__const")]
    assert text["size"] == program["text_words"] * 4, oracle["case"]
    assert constants["size"] == oracle["constant_section"]["size"], oracle["case"]
    assert constants["address"] - text["address"] == record["constantOffset"], \
        oracle["case"]
    # The surface allocations Apple's text address implies: 16 KiB tiles per
    # tensor, starting at the scratch base the descriptor records in slot 4.
    allocated = sum((descriptor["total_bytes"] + TILE_BYTES - 1) // TILE_BYTES
                    * TILE_BYTES for descriptor in tensors)
    assert text["address"] == int(program["resource_addresses"][4], 16) + allocated, \
        oracle["case"]
    assert_tasks(split_h14_tasks(payload[text["offset"]:
                                        text["offset"] + text["size"]]), oracle)
    assert_constants(payload[constants["offset"]:
                             constants["offset"] + constants["size"]], oracle)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        inspect_hwx.main(str(path))
    text_report = report.getvalue()
    assert "name=H14 isa=11" in text_report, oracle["case"]
    assert f"h14_tasks count={len(oracle['task_descriptors'])}" in text_report, \
        oracle["case"]


def conv_oracles():
    """Every decoded H14 convolution the parity encoder reproduces. The grid
    reaches past the encoder: a convolution Apple partitions into several
    tasks, and a strided grouped, multi-tap or 16-lane packing, stay outside
    the envelope and the compiler rejects them by name instead."""
    selected = []
    for path in sorted((ROOT / "research/oracles/h14").glob("*conv*.json")):
        oracle = json.loads(path.read_text())
        if oracle["family"] in CONV_FAMILIES and \
                mint_conv_probes.covered(dict(oracle, target="h14"),
                    allow_multi_task=mint_conv_probes.native_camera_multi(oracle)):
            selected.append(oracle)
    return selected


def check_chain_package(root):
    cases = (
        ("chain_add_relu_c512", "binary_add_1x512x1x1", 0, "0x00900",
         ("0x00080000", "0x00080020")),
        ("chain_pair_mm_relu_d256_s64", "chain_base_matmul_d256_s64", 1,
         "0x00d04", ("0x00101c00", "0x00111c00")),
    )
    for fused_name, base_name, task_index, address, expected_delta in cases:
        fused_oracle = json.loads((ROOT / "research/oracles/h14" /
                                   f"{fused_name}.json").read_text())
        base_oracle = json.loads((ROOT / "research/oracles/h14" /
                                  f"{base_name}.json").read_text())
        case_root = root / fused_name
        case_root.mkdir()
        model = case_root / "model.mil"
        model.write_text(fused_oracle["mil"])
        write_weights(fused_oracle, case_root)
        output = case_root / "out"
        result = subprocess.run(
            [str(COMPILER), "--mil", str(model), "--model-root", str(case_root),
             "--target", "H14", "--schedule", "chain", "--output", str(output)],
            capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
        assert result.returncode == 0, \
            f"{fused_name}: {result.stdout}{result.stderr}"
        manifest = json.loads((output / "manifest.json").read_text())
        program = manifest["programs"][0]
        assert manifest["schedule"] == "chain"
        assert manifest["dispatchPlan"] == [0]
        assert len(manifest["programs"]) == 1
        assert program["encoder"] == "composed-chain"
        assert program["operation"] == "chain"
        assert len(manifest["tasks"]) == program["taskDescriptors"]
        assert {item["operation"] for item in manifest["tasks"]} == \
            {fused_oracle["parameters"]["operations"][0]}
        assert manifest["scratch"] == {"bytes": 0, "regions": {}}
        payload = (output / program["file"]).read_bytes()
        tasks, _, _, header = anec_contents(payload)
        assert len(tasks) == header["task_count"] == program["taskDescriptors"]
        assert_tasks(tasks, fused_oracle)
        decoded = [decode_task(task, "h14") for task in tasks]
        differences = {}
        for index, (base, fused) in enumerate(zip(
                base_oracle["task_descriptors"], decoded)):
            for word, values in task_word_differences(base, fused).items():
                differences[(index, word)] = values
        assert differences == {(task_index, address): expected_delta}

        activation = int(descriptor_words(decoded[task_index])[address], 16)
        if fused_name == "chain_add_relu_c512":
            samples = ((-2.0, 0.5), (1.0, -0.25), (2.0, 3.0))
            actual = [fp16(max(fp16(a + b), 0.0)) if activation & 0x20
                      else fp16(a + b) for a, b in samples]
            assert actual == [0.0, 0.75, 5.0]
            invalid_mil = fused_oracle["mil"].replace(
                "relu(x = added)[", "relu(x = added, extra = added)[")
            assert invalid_mil != fused_oracle["mil"]
            invalid_model = case_root / "invalid-relu.mil"
            invalid_model.write_text(invalid_mil)
            invalid_result = subprocess.run(
                [str(COMPILER), "--mil", str(invalid_model),
                 "--model-root", str(case_root), "--target", "H14",
                 "--schedule", "chain", "--output",
                 str(case_root / "invalid-out")],
                capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
            assert invalid_result.returncode != 0
            assert "h14.chain-unrepresentable-edge" in (
                invalid_result.stdout + invalid_result.stderr)
            model.write_text(fused_oracle["mil"].replace(" = add(", " = mul("))
            rejected = subprocess.run(
                [str(COMPILER), "--mil", str(model), "--model-root", str(case_root),
                 "--target", "H14", "--schedule", "chain",
                 "--output", str(case_root / "unsupported-producer")],
                capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
            assert rejected.returncode > 0, rejected.stdout + rejected.stderr
            assert "h14.chain-unrepresentable-edge" in rejected.stdout + rejected.stderr
        else:
            assert not (int(descriptor_words(decoded[0]).get(address, "0"), 16)
                        & 0x00010000)
            row = [-1.0] * 128 + [1.0] * 128
            accumulation = sum(value * fp16(0.25) for value in row)
            actual = (fp16(max(accumulation, 0.0))
                      if activation & 0x00010000 else fp16(accumulation))
            assert actual == 0.0


def exp_chain_oracles():
    """The scaled exp chain as one MIL program, which Apple compiles to three tasks."""
    found = []
    for path in sorted((ROOT / "research/oracles/h14").glob("chain_exp_d4_c*.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is None:
            found.append(oracle)
    return found


def main():
    elementwise = selected_oracles("h14")
    matvec = selected_matvec_oracles()
    probe = grid_probe_oracles()
    norm = norm_oracles()
    families = collections.Counter(oracle["family"] for oracle in norm)
    templates = check_norm_templates()
    assert len(elementwise) == 165, \
        f"expected 165 decoded elementwise oracles, found {len(elementwise)}"
    assert len(matvec) == 38, \
        f"expected 38 decoded matvec oracles, found {len(matvec)}"
    assert families == collections.Counter(
        {"normalization": 105, "reduction": 114}), \
        f"decoded H14 norm oracles per family: {dict(families)}"
    grid = {(oracle["parameters"]["reduction"], oracle["parameters"]["columns"])
            for oracle in probe}
    assert grid == {(reduction, columns)
                    for reduction in probes.GRID_SIDES
                    for columns in probes.GRID_SIDES}, sorted(grid)
    conv = conv_oracles()
    original_conv = [oracle for oracle in conv if not oracle["case"].startswith("layer_")]
    assert len(original_conv) == 284, \
        f"expected 284 original covered H14 convolution oracles, found {len(original_conv)}"
    native_conv = [oracle for oracle in conv if oracle["case"].startswith("layer_")]
    assert len(native_conv) >= 250, f"native convolution campaign lost coverage: {len(native_conv)}"
    exp_chain = exp_chain_oracles()
    assert sorted(oracle["parameters"]["shape"][1] for oracle in exp_chain) == [64, 512], \
        f"expected the 64 and 512 channel scaled exp chains, found {len(exp_chain)}"
    oracles = elementwise + matvec + probe + norm + conv + exp_chain
    with tempfile.TemporaryDirectory(prefix="h14-parity-") as directory:
        root = Path(directory)
        check_chain_package(root)
        for oracle in oracles:
            for artifact_format, check in (("anec", check_anec), ("hwx", check_hwx)):
                output = root / f"{oracle['case']}-{artifact_format}"
                manifest = compile_oracle(oracle, output, artifact_format)
                assert len(manifest["programs"]) == 1, \
                    f"{oracle['case']}: {len(manifest['programs'])} programs"
                check(oracle, output, manifest)
        check_transposed_weights(root)
    probes.verify()
    print(f"H14 oracle parity: PASS ({len(oracles)} cases, "
          f"{len(elementwise)} elementwise, {len(matvec)} matvec, "
          f"{len(probe)} known-weight matvec probes over {len(grid)} (K, N) "
          f"grid points, {families['normalization']} softmax/layer_norm, "
          f"{families['reduction']} reduction over {len(templates)} norm "
          f"templates, {len(conv)} convolution, {len(exp_chain)} scaled exp chain, "
          f"{len(oracles) * 2} artifacts)")


if __name__ == "__main__":
    main()
