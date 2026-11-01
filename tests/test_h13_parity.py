#!/usr/bin/env python3
"""Compares emitted H13 task streams word-for-word with decoded Apple oracles."""
import collections
import contextlib
import hashlib
import io
import json
import math
import struct
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h13_td import decode_task, split_h13_tasks  # noqa: E402
from inspect_hwx import h13_anec  # noqa: E402
import mint_oracles  # noqa: E402
import mint_conv_probes  # noqa: E402
from timeouts import scaled_timeout  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
ORACLES = ROOT / "research/oracles/h13"
RUNTIME_BINARY = {"add", "mul", "maximum", "minimum", "sub"}
UNARY = {"abs", "exp", "gelu", "leaky_relu", "relu", "rsqrt", "sigmoid", "silu",
         "sqrt", "tanh"}
BROADCAST_FAMILIES = {"env_broadcast", "rrmm_broadcast"}
MATMUL_FAMILIES = {"env_matmul", "rrmm_matmul", "rrmm_matvec"}
CONV_FAMILIES = {"env_conv", "conv_probe"}
DEFAULT_ENCODER = "h13-oracle-parity"

# Commit 4849a0e ("feat(compiler): add decoded chains and native M1
# operation selection") prefers the source-qualified binary encoder for
# per-operation schedules -- nativeBinaryPlan takes every broadcastable
# binary it covers, and elementwise binaries whose final shape carries at
# most 64 channels -- and prefers single-row matvec encoders for the
# transposed-y matmul shapes below. These cases no longer emit the decoded
# Apple task stream, so Apple byte parity is not their contract; their
# native emission is exercised by the CLI, encoding, and simulation tests.
# The set is exact: a case leaving it (or joining it) changes the family
# counts this suite asserts, so any further selection drift stays visible.
NATIVE_SELECTION_CASES = frozenset({
    "binary_add_1x64x1x1", "binary_maximum_1x64x1x1", "binary_minimum_1x64x1x1",
    "binary_mul_1x64x1x1", "binary_mul_c64_constant_scalar",
    "env_bcast_add_1x16384x1x1_runtime_1x16384x1x1",
    "env_bcast_add_1x200x1x1_runtime_1x200x1x1", "env_bcast_add_1x300x1x1_runtime_1x300x1x1",
    "env_bcast_add_1x3072x1x1_runtime_1x3072x1x1", "env_bcast_add_1x768x1x1_runtime_1x768x1x1",
    "env_bcast_add_1x8192x1x1_runtime_1x8192x1x1", "env_bcast_add_1x96x1x1_runtime_1x96x1x1",
    "env_bcast_add_2x1024x1x1_runtime_2x1024x1x1", "env_bcast_add_2x512x1x1_runtime_2x512x1x1",
    "env_bcast_add_2x64x1x1_runtime_2x64x1x1", "env_bcast_add_2x64x8x8_runtime_2x64x8x8",
    "env_bcast_add_8x1024x1x1_runtime_8x1024x1x1", "env_bcast_add_8x512x1x1_runtime_8x512x1x1",
    "env_bcast_add_8x64x1x1_runtime_8x64x1x1", "env_bcast_add_8x64x8x8_runtime_8x64x8x8",
    "env_bcast_mul_1x16384x1x1_runtime_1x16384x1x1", "env_bcast_mul_1x200x1x1_runtime_1x200x1x1",
    "env_bcast_mul_1x300x1x1_runtime_1x300x1x1", "env_bcast_mul_1x3072x1x1_runtime_1x3072x1x1",
    "env_bcast_mul_1x64x16x16_scalar", "env_bcast_mul_1x64x8x8_scalar",
    "env_bcast_mul_1x768x16x16_scalar", "env_bcast_mul_1x768x1x1_runtime_1x768x1x1",
    "env_bcast_mul_1x768x8x8_scalar", "env_bcast_mul_1x8192x1x1_runtime_1x8192x1x1",
    "env_bcast_mul_1x96x1x1_runtime_1x96x1x1", "env_bcast_mul_2x1024x1x1_runtime_2x1024x1x1",
    "env_bcast_mul_2x512x1x1_runtime_2x512x1x1", "env_bcast_mul_2x64x1x1_runtime_2x64x1x1",
    "env_bcast_mul_2x64x8x8_runtime_2x64x8x8", "env_bcast_mul_8x1024x1x1_runtime_8x1024x1x1",
    "env_bcast_mul_8x512x1x1_runtime_8x512x1x1", "env_bcast_mul_8x64x1x1_runtime_8x64x1x1",
    "env_bcast_mul_8x64x8x8_runtime_8x64x8x8", "env_mm_r2rb_m1_k2048_n2048_tx0_ty1",
    "env_mm_r2rb_m1_k2048_n2048_tx1_ty1", "env_mm_r2rb_m1_k2048_n4096_tx0_ty1",
    "env_mm_r2rb_m1_k2048_n8192_tx0_ty1", "env_mm_r2rb_m1_k4096_n2048_tx0_ty1",
    "env_mm_r2rb_m1_k4096_n4096_tx0_ty1", "env_mm_r2rb_m1_k4096_n4096_tx1_ty1",
    "env_mm_r2rb_m1_k4096_n8192_tx0_ty1", "env_mm_r2rb_m1_k8192_n2048_tx0_ty1",
    "env_mm_r2rb_m1_k8192_n4096_tx0_ty1", "env_mm_r2rb_m1_k8192_n8192_tx0_ty1",
    "env_mm_r3rb_m1_k2048_n2048_tx0_ty1_b1", "env_mm_r3rb_m1_k2048_n4096_tx0_ty1_b1",
    "env_mm_r3rb_m1_k2048_n8192_tx0_ty1_b1", "env_mm_r3rb_m1_k4096_n2048_tx0_ty1_b1",
    "env_mm_r3rb_m1_k4096_n4096_tx0_ty1_b1", "env_mm_r3rb_m1_k4096_n8192_tx0_ty1_b1",
    "env_mm_r3rb_m1_k8192_n2048_tx0_ty1_b1", "env_mm_r3rb_m1_k8192_n4096_tx0_ty1_b1",
    "env_mm_r3rb_m1_k8192_n8192_tx0_ty1_b1", "matmul_m1_k1024_n1024_ty1",
    "matmul_m1_k1024_n256_ty1", "matmul_m1_k1024_n512_ty1", "matmul_m1_k256_n1024_ty1",
    "matmul_m1_k256_n256_ty1", "matmul_m1_k256_n512_ty1", "matmul_m1_k512_n1024_ty1",
    "matmul_m1_k512_n256_ty1", "matmul_m1_k512_n512_ty1", "rrmm_r2rb_m1_k256_n128_tx0_ty1",
    "rrmm_r2rb_m1_k256_n256_tx0_ty1", "rrmm_r2rb_m1_k256_n512_tx1_ty1",
    "rrmm_r2rb_m1_k256_n64_tx0_ty1", "rrmm_r2rb_m1_k512_n512_tx1_ty1",
})

# Commit ea903c4 ("fix(h13): qualify Linux ANEC execution") rewrites every
# emitted task for the Linux driver ABI before the artifact leaves the
# compiler: header word 0 bits 16..23 become the driver-derived kernel
# window 0x40, and the three 5-bit surface-selector fields in header word 8
# (bit shifts 0, 6, 12) rebind each declared channel to role order 4, 5, 6.
# The parity encoders (elementwise, broadcast, norm, conv) declare channels
# {output=5, input0=4, input1=6}; the matvec and matmul envelope encoders
# keep the default {4, 5, 6}, for which the rebinding is the identity.
# Applying the same fixed transform to each oracle keeps the comparison
# byte-exact for everything else the encoder emits.
PARITY_CHANNEL_BINDING = {5: 4, 4: 5, 6: 6}
ENVELOPE_CHANNEL_BINDING = {4: 4, 5: 5, 6: 6}


def bound_task_descriptor(task, binding):
    header = [int(word, 16) for word in task["header_words"]]
    for shift in (0, 6, 12):
        channel = (header[8] >> shift) & 31
        if channel >= 4:
            header[8] = (header[8] & ~(31 << shift)) | (binding[channel] << shift)
    header[0] = (header[0] & ~0x00FF0000) | 0x00400000
    bound = dict(task)
    bound["header_words"] = [f"0x{word:08x}" for word in header]
    return bound


def channel_binding(oracle):
    identity = oracle["family"] in MATMUL_FAMILIES or oracle["family"] in (
        "matmul", "encoder_linear", "encoder_bmm", "chain")
    return ENVELOPE_CHANNEL_BINDING if identity else PARITY_CHANNEL_BINDING


def selected_oracles():
    selected = []
    for path in sorted(ORACLES.glob("*.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is not None:
            continue
        if oracle["case"] in NATIVE_SELECTION_CASES:
            continue
        parameters = oracle.get("parameters", {})
        operation = parameters.get("operation")
        family = oracle.get("family")
        if family == "binary_runtime" and operation in RUNTIME_BINARY:
            selected.append(oracle)
        elif family == "binary_constant" and parameters.get("constant") == "scalar":
            selected.append(oracle)
        elif family == "unary" and operation in UNARY:
            selected.append(oracle)
        elif family == "matmul":
            selected.append(oracle)
        elif family in ("normalization", "reduction"):
            selected.append(oracle)
        elif family in BROADCAST_FAMILIES:
            selected.append(oracle)
        elif family in MATMUL_FAMILIES and parameters["x_storage"] == "runtime":
            # A constant `x` leaves `y` as the runtime operand; no encoder
            # lowers that form, so it stays outside the envelope.
            selected.append(oracle)
        elif family in CONV_FAMILIES and conv_covered(oracle, "h13"):
            selected.append(oracle)
        elif family == "chain" and parameters.get("probe") == "ffn_matmul":
            selected.append(oracle)
        elif family == "encoder_linear" or family == "encoder_bmm":
            selected.append(oracle)
        elif family == "encoder_structure" and \
                parameters.get("probe", "").startswith(("transpose", "slice")):
            selected.append(oracle)
    return selected


def conv_covered(oracle, target):
    """Whether the convolution encoder reproduces this oracle.

    The campaign grid reaches past the encoder: Apple partitions a large
    convolution into several tasks, and the strided packing is only derived
    for a groups-1 1x1 kernel below 16 interleaved lanes. Those cases stay
    outside the parity set and the compiler rejects them by name instead.
    The named encoder captures lower as linked multi-task programs and ride
    the same opt-in the template emitter uses.
    """
    oracle = dict(oracle, target=target)
    return mint_conv_probes.covered(
        oracle,
        allow_multi_task=oracle["case"] in mint_conv_probes.MULTI_TASK_CASES or mint_conv_probes.native_camera_multi(oracle))


def encoder(oracle):
    family = oracle["family"]
    if family == "encoder_linear":
        return "apple-parity-linear"
    if family == "encoder_bmm":
        return "apple-parity-batched-matvec"
    if family == "encoder_structure":
        return "apple-parity-slice" \
            if oracle["parameters"]["probe"].startswith("slice") \
            else "apple-parity-transpose"
    if family == "chain":
        return "apple-parity-ffn-chain"
    if family in MATMUL_FAMILIES:
        return "apple-parity-matmul" \
            if oracle["parameters"]["w_storage"] == "runtime" \
            else "apple-parity-matvec"
    if family in BROADCAST_FAMILIES:
        return "apple-parity-broadcast"
    if family in CONV_FAMILIES:
        return "apple-parity-conv"
    return {"matmul": "apple-parity-matvec",
            "normalization": "apple-parity-norm",
            "reduction": "apple-parity-norm"}.get(family, DEFAULT_ENCODER)


def write_weights(oracle, root):
    """Recreates the exact weights.bin the campaign compiled against."""
    description = oracle.get("weights", {})
    if description.get("storage") != "BLOBFILE":
        return
    shapes = description.get("shapes")
    if description.get("pattern") == "uint16_le_index_plus_one_wrapping":
        # One payload spanning every constant: uint16(index + 1), wrapping,
        # with the campaign's 24-byte record table and data region.
        data_start = (64 + 4096 * 24 + 63) & ~63
        payloads = []
        value = 0
        for shape in shapes:
            count = math.prod(shape)
            payloads.append(struct.pack(
                f"<{count}H", *[(value + index + 1) & 0xFFFF
                                for index in range(count)]))
            value += count
        blob = bytearray(data_start + sum(len(p) for p in payloads))
        struct.pack_into("<II", blob, 0, len(shapes), 2)
        offset = data_start
        for index, payload in enumerate(payloads):
            struct.pack_into("<IIQQ", blob, 64 + index * 24, 0xDEADBEEF, 1,
                             len(payload), offset)
            blob[offset:offset + len(payload)] = payload
            offset += len(payload)
        (root / "weights.bin").write_bytes(bytes(blob))
        return
    elements = description["payload_bytes"] // 2
    if description.get("value") == \
            "fp16 bits 0x3400 + index, one value per constant":
        # One repeated value per constant: fp16 bits 0x3400 + ordinal.
        data_start = (64 + 4096 * 24 + 63) & ~63
        payloads = [struct.pack("<H", 0x3400 + index) * math.prod(shape)
                    for index, shape in enumerate(shapes)]
        blob = bytearray(data_start + sum(len(p) for p in payloads))
        struct.pack_into("<II", blob, 0, len(shapes), 2)
        offset = data_start
        for index, payload in enumerate(payloads):
            struct.pack_into("<IIQQ", blob, 64 + index * 24, 0xDEADBEEF, 1,
                             len(payload), offset)
            blob[offset:offset + len(payload)] = payload
            offset += len(payload)
        (root / "weights.bin").write_bytes(bytes(blob))
        return
    if description["value"] == "indexed_finite":
        payload = mint_conv_probes.finite_index_payload(elements, description["seed"])
    elif description["value"] == "distinct":
        # The known-weight convolution probes carry one distinct fp16 pattern
        # per element, which is what proves the packing permutation.
        payload = mint_conv_probes.known_weights(elements)
    else:
        assert description["value"] == "fp16(0x1p-1)", oracle["case"]
        payload = mint_oracles.half_payload(elements)
    (root / "weights.bin").write_bytes(mint_oracles.blob(payload))


def compile_oracle(oracle, output, artifact_format):
    mil = output.parent / f"{oracle['case']}-{artifact_format}.mil"
    mil.write_text(oracle["mil"])
    write_weights(oracle, output.parent)
    result = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(output.parent),
         "--target", "H13", "--format", artifact_format, "--output", str(output)],
        capture_output=True, text=True, timeout=scaled_timeout(600), check=False)
    assert result.returncode == 0, \
        f"{oracle['case']} {artifact_format}: {result.stdout}{result.stderr}"
    return json.loads((output / "manifest.json").read_text())


def expected_task_stream_bytes(oracle):
    tasks = oracle["task_descriptors"]
    offset = 0
    for index, task in enumerate(tasks[:-1]):
        offset = int(task["header_words"][7], 16)
        assert offset >= 0, index
    return offset + tasks[-1]["size_bytes"]


def unpack_commands(hwx):
    _, _, _, _, count, command_bytes, _, _ = struct.unpack_from("<8I", hwx)
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
        elif command == 4 and kind == 1:
            program = {
                "code": struct.unpack_from("<I", hwx, cursor + 0x0C)[0],
                "command_size": size,
                "text_address": hex(struct.unpack_from("<Q", hwx, cursor + 0x10)[0]),
                "constant_address": hex(struct.unpack_from("<Q", hwx, cursor + 0x18)[0]),
                "resource_addresses": [
                    hex(struct.unpack_from("<Q", hwx, cursor + 0x30 + index * 8)[0])
                    for index in range(5)],
                "kind": kind,
                "task_words_minus_one": struct.unpack_from("<I", hwx, cursor + 0x818)[0],
                "task_count": struct.unpack_from("<I", hwx, cursor + 0x81C)[0],
            }
        elif command == 4 and kind == 3:
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
    _, first_task_bytes, task_count, task_stream_bytes, constant_bytes, _, _ = \
        struct.unpack_from("<QIIQQII", payload)
    stream = payload[0x1000:0x1000 + task_stream_bytes]
    tasks = split_h13_tasks(stream, first_task_bytes // 4 - 1, task_count)
    constant_offset = (task_stream_bytes + 0x3F) & ~0x3F
    constants = payload[0x1000 + constant_offset:
                        0x1000 + constant_offset + constant_bytes]
    return tasks, constants, constant_offset


def assert_tasks(actual, oracle):
    binding = channel_binding(oracle)
    expected = [bound_task_descriptor(task, binding)
                for task in oracle["task_descriptors"]]
    assert len(actual) == len(expected), \
        f"{oracle['case']}: {len(actual)} tasks, oracle has {len(expected)}"
    for index, task in enumerate(actual):
        assert decode_task(task, "h13") == expected[index], \
            f"{oracle['case']} task {index} words differ from the oracle"


def assert_constants(constants, oracle):
    expected = oracle["constant_section"]
    assert len(constants) == expected["size"], oracle["case"]
    assert hashlib.sha256(constants).hexdigest() == expected["sha256"], \
        f"{oracle['case']} constant section differs from the oracle"


def check_anec(oracle, output, manifest):
    assert len(manifest["programs"]) == 1, \
        f"{oracle['case']}: {len(manifest['programs'])} programs, expected 1"
    record = manifest["programs"][0]
    assert record["encoder"] == encoder(oracle), oracle["case"]
    payload = (output / record["file"]).read_bytes()
    tasks, constants, constant_offset = anec_contents(payload)
    assert_tasks(tasks, oracle)
    assert_constants(constants, oracle)
    assert record["constantOffset"] == constant_offset, oracle["case"]
    assert constant_offset == (expected_task_stream_bytes(oracle) + 0x3F) & ~0x3F, \
        oracle["case"]


def check_hwx(oracle, output, manifest):
    assert len(manifest["programs"]) == 1, \
        f"{oracle['case']}: {len(manifest['programs'])} programs, expected 1"
    record = manifest["programs"][0]
    assert record["encoder"] == encoder(oracle), oracle["case"]
    payload = (output / record["file"]).read_bytes()
    sections, program, tensors = unpack_commands(payload)
    expected = dict(oracle["program_descriptor"])
    # The envelope campaign records which slice of __TEXT/__text each program
    # descriptor owns; a single-program object owns all of it.
    section = expected.pop("task_section", None)
    assert program == expected, oracle["case"]
    assert section in (None, {"offset": 0,
                              "size": expected_task_stream_bytes(oracle)}), \
        oracle["case"]
    assert tensors == oracle["tensor_descriptors"], oracle["case"]
    assert sections[("__TEXT", "__text")]["size"] == \
        expected_task_stream_bytes(oracle), oracle["case"]
    assert sections[("__TEXT", "__const")]["size"] == \
        oracle["constant_section"]["size"], oracle["case"]
    with contextlib.redirect_stdout(io.StringIO()):
        extracted, _ = h13_anec(payload)
    tasks, constants, _ = anec_contents(extracted)
    assert_tasks(tasks, oracle)
    assert_constants(constants, oracle)


def main():
    oracles = selected_oracles()
    families = collections.Counter(oracle["family"] for oracle in oracles)
    # Counts exclude the 73 NATIVE_SELECTION_CASES (4 binary_runtime,
    # 1 binary_constant, 34 env_broadcast, 20 env_matmul, 9 matmul,
    # 5 rrmm_matvec) that commit 4849a0e moved to native encoders.
    expected = {"binary_runtime": 46, "binary_constant": 11, "unary": 28,
                "matmul": 27, "normalization": 108, "reduction": 114,
                "env_broadcast": 68, "env_matmul": 90, "env_conv": 15,
                "encoder_linear": 11, "encoder_bmm": 1, "chain": 1,
                "encoder_structure": 5}
    for family in ("rrmm_broadcast", "rrmm_matmul", "rrmm_matvec",
                   "conv_probe"):
        if families[family]:
            expected[family] = families[family]
    assert families == collections.Counter(expected), \
        f"decoded parity oracles per family: {dict(families)}"
    with tempfile.TemporaryDirectory(prefix="h13-parity-") as directory:
        root = Path(directory)
        for oracle in oracles:
            for artifact_format, check in (("anec", check_anec), ("hwx", check_hwx)):
                output = root / f"{oracle['case']}-{artifact_format}"
                check(oracle, output, compile_oracle(oracle, output, artifact_format))
                # The constant-weight grid reaches 134 MiB per artifact, so a
                # case is discarded as soon as it has been compared.
                shutil.rmtree(output)
            for stale in root.glob(f"{oracle['case']}-*.mil"):
                stale.unlink()
    matmul = families["matmul"] + families["env_matmul"] + \
        families["rrmm_matmul"] + families["rrmm_matvec"]
    broadcast = families["env_broadcast"] + families["rrmm_broadcast"]
    convolution = families["env_conv"] + families["conv_probe"]
    print(f"H13 oracle parity: PASS ({len(oracles)} cases, "
          f"{matmul} matmul, {broadcast} broadcast, "
          f"{families['normalization']} softmax/layer_norm, "
          f"{families['reduction']} reduction, {convolution} convolution, "
          f"{len(oracles) * 2} artifacts)")


if __name__ == "__main__":
    main()
