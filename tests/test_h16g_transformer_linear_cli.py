#!/usr/bin/env python3
"""Exercise the formula-built H16G transformer-linear CLI path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from mint_oracles import parse_hwx  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
DUMP = Path(sys.argv[2] if len(sys.argv) > 2 else
            ROOT / "build/h16_transformer_linear_dump").resolve()
VALIDATOR = ROOT / "tools/h16g_validate.py"


def fp16_pattern(elements: int, seed: int) -> bytes:
    values = bytearray(2 * elements)
    for index in range(elements):
        # Finite, nonuniform binary16 values; their exact numerics are not
        # relevant to this byte-packing and descriptor-parity test.
        word = 0x3000 | ((index * 73 + seed * 29) & 0x07ff)
        struct.pack_into("<H", values, 2 * index, word)
    return bytes(values)


def blob(payload: bytes) -> bytes:
    result = bytearray(128 + len(payload))
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<IIQQ", result, 64, 0xDEADBEEF, 1,
                     len(payload), 128)
    result[128:] = payload
    return bytes(result)


def linear_bar_records(data: bytes) -> list[tuple[int, int, int]]:
    """Return (file offset, register address, BAR slot) for linear DMA records."""
    commands = struct.unpack_from("<I", data, 16)[0]
    cursor = 32
    text = text_offset = input_vm = output_vm = kernel_vm = scratch_vm = descriptor = None
    for _ in range(commands):
        command, length = struct.unpack_from("<II", data, cursor)
        if command == 0x19:
            segment = struct.unpack_from("<2I16s4Q4I", data, cursor)
            name = segment[2].split(b"\0", 1)[0]
            if name == b"__TEXT":
                section = struct.unpack_from("<16s16s2Q8I", data, cursor + 72)
                text_offset = section[4]
                text = data[section[4]:section[4] + section[3]]
            elif name == b"__FVMLIB" and segment[8] == 1:
                input_vm = segment[3]
            elif name == b"__FVMLIB" and segment[8] == 2:
                output_vm = segment[3]
            elif name == b"__KERN_0":
                kernel_vm = segment[3]
            elif name == b"__DATA":
                scratch_vm = segment[3]
        elif command == 4 and struct.unpack_from("<I", data, cursor + 8)[0] == 4:
            descriptor = cursor
        cursor += length
    assert (text is not None and text_offset is not None and input_vm is not None and
            output_vm is not None and kernel_vm is not None and descriptor is not None)
    records = []
    cursor = 16
    while cursor < len(text):
        words = (struct.unpack_from("<I", text, cursor)[0] >> 16) & 0x7ff
        assert words
        end = cursor + 4 * words
        header_words = 10 if struct.unpack_from("<I", text, cursor + 32)[0] & 3 == 3 else 9
        record = cursor + 4 * header_words
        while record < end:
            header_offset = record
            header = struct.unpack_from("<I", text, record)[0]
            record += 4
            if header & (1 << 31):
                count = 1 + ((header >> 15) & 0xffff).bit_count()
            else:
                count = ((header >> 15) & 0x3f) + 1
                if header & (1 << 29) and count == 2:
                    address = (header & 0x7fff) * 4
                    if address in (0x4d10, 0x5108, 0x5510):
                        records.append((text_offset + header_offset, address,
                                        (header >> 23) & 0x3f))
            record += 4 * count
        assert record == end
        cursor = (end + 15) & ~15
    expected = {
        0x4d10: {3: scratch_vm, 6: input_vm},
        0x5108: {3: scratch_vm, 7: output_vm},
        0x5510: {8: kernel_vm},
    }
    for _, address, slot in records:
        assert slot in expected[address], (hex(address), slot)
        target = expected[address][slot]
        assert target is not None
        assert struct.unpack_from("<Q", data,
                                  descriptor + 0x10 * (slot + 1))[0] == target
    assert {(address, slot) for _, address, slot in records}.issuperset(
        {(0x4d10, 6), (0x5108, 7), (0x5510, 8)})
    return records


def old_linear_bars(data: bytes) -> bytes:
    """Reconstruct the v2 source/result BAR bug for validator regression."""
    changed = bytearray(data)
    for offset, address, slot in linear_bar_records(data):
        if (address, slot) in ((0x4d10, 6), (0x5108, 7)):
            header = struct.unpack_from("<I", changed, offset)[0]
            struct.pack_into("<I", changed, offset, header - 0x00800000)
    return bytes(changed)


def mil(tokens: int, cin: int, cout: int, bias: bool, post: str) -> str:
    bias_decl = ""
    bias_arg = ""
    if bias:
        bias_decl = (
            f'    tensor<fp16, [{cout}]> b = const()[name = string("b"), '
            f'val = tensor<fp16, [{cout}]>(BLOBFILE(path = '
            'string("@model_path/bias.bin"), offset = uint64(64)))];\n')
        bias_arg = ", bias = b"
    post_op = ""
    result = "linear"
    if post == "gelu":
        post_op = (f'    tensor<fp16, [1, {tokens}, {cout}]> y = gelu('
                   'x = linear, mode = string("EXACT"))[name = string("y")];\n')
        result = "y"
    elif post == "scale":
        post_op = ('    fp16 scale = const()[name = string("scale"), '
                   'val = fp16(0x1p-3)];\n'
                   f'    tensor<fp16, [1, {tokens}, {cout}]> y = mul('
                   'x = linear, y = scale)[name = string("y")];\n')
        result = "y"
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {tokens}, {cin}]> x) {{
    tensor<fp16, [{cout}, {cin}]> w = const()[name = string("w"), val = tensor<fp16, [{cout}, {cin}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
{bias_decl}    tensor<fp16, [1, {tokens}, {cout}]> linear = linear(x = x, weight = w{bias_arg})[name = string("linear")];
{post_op}  }} -> ({result});
}}
'''


def run_case(root: Path, name: str, tokens: int, cin: int, cout: int,
             has_bias: bool, post: str = "") -> None:
    weights = fp16_pattern(cin * cout, len(name))
    bias = fp16_pattern(cout, len(name) + 1) if has_bias else b""
    (root / "weights.bin").write_bytes(blob(weights))
    if has_bias:
        (root / "bias.bin").write_bytes(blob(bias))
    source = root / f"{name}.mil"
    source.write_text(mil(tokens, cin, cout, has_bias, post))
    output = root / name
    result = subprocess.run([
        str(COMPILER), "--mil", str(source), "--model-root", str(root),
        "--target", "H16G", "--output", str(output),
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["artifacts"]) == 1
    image = output / "program-0.hwx"
    image_data = image.read_bytes()
    decoded = parse_hwx(image_data, "h16")
    linear_bar_records(image_data)
    flags = (1 if has_bias else 0) | (2 if post == "gelu" else 0)
    scale = 0x3000 if post == "scale" else 0x3c00
    table = b""
    if post == "gelu":
        data = image.read_bytes()
        count = struct.unpack_from("<I", data, 16)[0]
        cursor = 32
        for _ in range(count):
            command, length = struct.unpack_from("<II", data, cursor)
            if command == 0x19:
                segment = struct.unpack_from("<2I16s4Q4I", data, cursor)
                section_cursor = cursor + 72
                for _ in range(segment[-2]):
                    entry = struct.unpack_from("<16s16s2Q8I", data,
                                               section_cursor)
                    if entry[0].split(b"\0", 1)[0] == b"__kern_0":
                        table = data[entry[4]:entry[4] + 128]
                    section_cursor += 80
            cursor += length
    expected_task = subprocess.run([
        str(DUMP), "task", str(tokens), str(cin), str(cout), str(flags),
        hex(scale)], input=weights + bias + table, stdout=subprocess.PIPE,
        timeout=120, check=True).stdout
    expected_kernel = subprocess.run([
        str(DUMP), "kernel", str(tokens), str(cin), str(cout), str(flags),
        hex(scale)], input=weights + bias + table, stdout=subprocess.PIPE,
        timeout=120, check=True).stdout
    assert decoded["program_descriptor"]["task_count"] == \
        len(decoded["task_descriptors"])
    assert decoded["program_descriptor"]["text_words"] * 4 == len(expected_task)
    assert decoded["kernel_section"]["sha256"] == \
        hashlib.sha256(expected_kernel).hexdigest()
    validation = subprocess.run([
        sys.executable, str(VALIDATOR), str(image),
    ], capture_output=True, text=True, timeout=30)
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert json.loads(validation.stdout)["dma_extents"] == \
        len(decoded["task_descriptors"])
    old_image = output / "program-v2-bars.hwx"
    old_image.write_bytes(old_linear_bars(image_data))
    old_validation = subprocess.run([
        sys.executable, str(VALIDATOR), str(old_image),
    ], capture_output=True, text=True, timeout=30)
    assert old_validation.returncode != 0
    assert "h16g.dma-unbound-slot" in old_validation.stderr


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="h16g-transformer-linear-") as temp:
        root = Path(temp)
        run_case(root, "bert-up", 128, 768, 3072, False)
        run_case(root, "wtiny-bias", 1500, 384, 384, True)
        run_case(root, "bert-gelu", 128, 768, 3072, True, "gelu")
        run_case(root, "wbase-scale", 1500, 512, 512, True, "scale")
        # The logical 0x465000-byte scratch is intentionally not segment
        # aligned; the object writer must round its __DATA allocation.
        run_case(root, "wtiny-contract", 1500, 1536, 384, False)
        # Sealed capture rows were not provided for these two geometries. They
        # remain explicit unrevealed predictions while still exercising the
        # formula, object writer, and validator end to end.
        run_case(root, "held-s64-640x640", 64, 640, 640, False)
        run_case(root, "held-s256-1024x4096", 256, 1024, 4096, False)
    print("H16G transformer linear CLI PASS: formula tasks, nonuniform kernels, "
          "validator, 2 unrevealed held-out predictions")


if __name__ == "__main__":
    main()
