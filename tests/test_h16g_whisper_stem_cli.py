#!/usr/bin/env python3
"""Whisper 1x3 stride-1 descriptor parity and CLI qualification."""
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
from mint_oracles import decode_task, parse_hwx, split_h14_tasks  # noqa: E402

COMPILER = Path(sys.argv[1]).resolve()
DUMP = Path(sys.argv[2]).resolve()
VALIDATOR = ROOT / "tools/h16g_validate.py"

ROWS = {
    (384, False): (185344, "a27203e5de84404cb86c460fdc14aab3e87c57d88ca40312968c1e15b3c31af2"),
    (384, True): (187392, "44f87b4e0396dcdf1f0e820e7548daaaece3692b6bce07cc07b384be81126f24"),
    (512, False): (246784, "98c3ce248b3557f0f5aed61cc31c58d5bf17de6fc6abd782a367a0f163472963"),
    (512, True): (248832, "fcedba63e926f004950dde60348edcc94941fcc38de9bc77c561b6c61858507e"),
}


def pattern(elements: int, seed: int) -> bytes:
    data = bytearray(2 * elements)
    for index in range(elements):
        struct.pack_into("<H", data, 2 * index,
                         0x3000 | ((index * 37 + seed * 19) & 0x7ff))
    return bytes(data)


def blob(path: Path, payloads: list[bytes]) -> list[int]:
    data = bytearray(struct.pack("<II", 1, 2) + bytes(56))
    offsets = []
    for payload in payloads:
        offset = len(data); offsets.append(offset)
        data += struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload), offset + 64)
        data += bytes(40) + payload
        data += bytes((-len(data)) % 64)
    path.write_bytes(data)
    return offsets


def mil(tokens: int, channels: int, offsets: list[int], gelu: bool) -> str:
    result = "y" if gelu else "c"
    activation = (f'    tensor<fp16, [1, {channels}, {tokens}]> y = gelu('
                  'x = c, mode = string("EXACT"))[name = string("y")];\n'
                  if gelu else "")
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, 80, {tokens}]> x) {{
    string pt = const()[name = string("pt"), val = string("same")];
    tensor<int32, [1]> st = const()[name = string("st"), val = tensor<int32, [1]>([1])];
    tensor<int32, [2]> pd = const()[name = string("pd"), val = tensor<int32, [2]>([0, 0])];
    tensor<int32, [1]> dl = const()[name = string("dl"), val = tensor<int32, [1]>([1])];
    int32 gp = const()[name = string("gp"), val = int32(1)];
    tensor<fp16, [{channels}, 80, 3]> w = const()[name = string("w"), val = tensor<fp16, [{channels}, 80, 3]>(BLOBFILE(path = string("@model_path/parameters.bin"), offset = uint64({offsets[0]})))];
    tensor<fp16, [{channels}]> b = const()[name = string("b"), val = tensor<fp16, [{channels}]>(BLOBFILE(path = string("@model_path/parameters.bin"), offset = uint64({offsets[1]})))];
    tensor<fp16, [1, {channels}, {tokens}]> c = conv(bias = b, dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string("c")];
{activation}  }} -> ({result});
}}
'''


def mil_stem2(channels: int, offsets: list[int], gelu: bool) -> str:
    result = "y" if gelu else "c"
    activation = (f'    tensor<fp16, [1, {channels}, 1500]> y = gelu('
                  'x = c, mode = string("EXACT"))[name = string("y")];\n'
                  if gelu else "")
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {channels}, 3000]> x) {{
    string pt = const()[name = string("pt"), val = string("same")];
    tensor<int32, [1]> st = const()[name = string("st"), val = tensor<int32, [1]>([2])];
    tensor<int32, [2]> pd = const()[name = string("pd"), val = tensor<int32, [2]>([0, 0])];
    tensor<int32, [1]> dl = const()[name = string("dl"), val = tensor<int32, [1]>([1])];
    int32 gp = const()[name = string("gp"), val = int32(1)];
    tensor<fp16, [{channels}, {channels}, 3]> w = const()[name = string("w"), val = tensor<fp16, [{channels}, {channels}, 3]>(BLOBFILE(path = string("@model_path/parameters.bin"), offset = uint64({offsets[0]})))];
    tensor<fp16, [{channels}]> b = const()[name = string("b"), val = tensor<fp16, [{channels}]>(BLOBFILE(path = string("@model_path/parameters.bin"), offset = uint64({offsets[1]})))];
    tensor<fp16, [1, {channels}, 1500]> c = conv(bias = b, dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string("c")];
{activation}  }} -> ({result});
}}
'''


def transpose_mil(channels: int) -> str:
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {channels}, 1500]> x) {{
    tensor<int32, [3]> p = const()[name = string("p"), val = tensor<int32, [3]>([0, 2, 1])];
    tensor<fp16, [1, 1500, {channels}]> y = transpose(perm = p, x = x)[name = string("y")];
  }} -> (y);
}}
'''


def positional_mil(channels: int, offset: int) -> str:
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, 1500, {channels}]> x) {{
    tensor<fp16, [1, 1500, {channels}]> p = const()[name = string("p"), val = tensor<fp16, [1, 1500, {channels}]>(BLOBFILE(path = string("@model_path/parameters.bin"), offset = uint64({offset})))];
    tensor<fp16, [1, 1500, {channels}]> y = add(x = x, y = p)[name = string("y")];
  }} -> (y);
}}
'''


def canonical(task: bytes) -> dict:
    decoded = decode_task(task, "h16")
    # Captures name the kernel operand at BAR 7. Packaged objects use
    # input/output/kernel at slots 6/7/8, so normalize only that slot rebase
    # while retaining parity coverage for every other descriptor field.
    for record in decoded["records"]:
        if record["address"] == "0x05510":
            header = int(record["header"], 16)
            if (header & 0x3f800000) == 0x24000000:
                record["header"] = f"0x{header - 0x00800000:08x}"
    return {"header": decoded["header_words"][:1] + decoded["header_words"][2:],
            "blocks": decoded["blocks"], "records": decoded["records"]}


def section(data: bytes, wanted: bytes) -> bytes:
    count = struct.unpack_from("<I", data, 16)[0]; cursor = 32
    for _ in range(count):
        command, length = struct.unpack_from("<II", data, cursor)
        if command == 0x19:
            segment = struct.unpack_from("<2I16s4Q4I", data, cursor)
            at = cursor + 72
            for _ in range(segment[-2]):
                entry = struct.unpack_from("<16s16s2Q8I", data, at)
                if entry[0].split(b"\0", 1)[0] == wanted:
                    return data[entry[4]:entry[4] + entry[3]]
                at += 80
        cursor += length
    return b""


def assert_kernel_bar_bound(data: bytes) -> bool:
    """Require slot 8 to bind __KERN_0 whenever the object has that segment."""
    count = struct.unpack_from("<I", data, 16)[0]
    cursor = 32
    text = kernel_vm = descriptor = None
    for _ in range(count):
        command, length = struct.unpack_from("<II", data, cursor)
        if command == 0x19:
            segment = struct.unpack_from("<2I16s4Q4I", data, cursor)
            name = segment[2].split(b"\0", 1)[0]
            if name == b"__TEXT":
                entry = struct.unpack_from("<16s16s2Q8I", data, cursor + 72)
                text = data[entry[4]:entry[4] + entry[3]]
            elif name == b"__KERN_0":
                kernel_vm = segment[3]
        elif command == 4 and struct.unpack_from("<I", data, cursor + 8)[0] == 4:
            descriptor = cursor
        cursor += length
    if kernel_vm is None:
        return False
    assert text is not None and descriptor is not None
    slots: set[int] = set()
    cursor = 16
    while cursor < len(text):
        words = (struct.unpack_from("<I", text, cursor)[0] >> 16) & 0x7ff
        assert words
        end = cursor + 4 * words
        header_words = 10 if struct.unpack_from("<I", text, cursor + 32)[0] & 3 == 3 else 9
        record = cursor + 4 * header_words
        while record < end:
            header = struct.unpack_from("<I", text, record)[0]
            record += 4
            if header & (1 << 31):
                values = 1 + ((header >> 15) & 0xffff).bit_count()
            else:
                values = ((header >> 15) & 0x3f) + 1
                if header & (1 << 29):
                    slots.add((header >> 23) & 0x3f)
            record += 4 * values
        assert record == end
        cursor = (end + 15) & ~15
    assert 8 in slots, slots
    assert struct.unpack_from("<Q", data, descriptor + 0x10 * 9)[0] == kernel_vm
    return True


def run_case(root: Path, tokens: int, channels: int, gelu: bool) -> None:
    weights = pattern(channels * 80 * 3, channels + tokens)
    bias = pattern(channels, channels + tokens + 1)
    offsets = blob(root / "parameters.bin", [weights, bias])
    source = root / f"stem-{tokens}-{channels}-{int(gelu)}.mil"
    source.write_text(mil(tokens, channels, offsets, gelu))
    output = root / source.stem
    result = subprocess.run([str(COMPILER), "--mil", str(source),
        "--model-root", str(root), "--target", "H16G", "--output", str(output)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    image = output / "program-0.hwx"; data = image.read_bytes()
    assert assert_kernel_bar_bound(data)
    kernel = section(data, b"__kern_0")
    table = kernel[:128] if gelu else b""
    payload = weights + bias + table
    args = [str(DUMP), "task", str(tokens), str(channels), str(int(gelu))]
    task = subprocess.run(args, input=payload, stdout=subprocess.PIPE,
                          timeout=120, check=True).stdout
    expected_kernel = subprocess.run([str(DUMP), "kernel", *args[2:]],
        input=payload, stdout=subprocess.PIPE, timeout=120, check=True).stdout
    assert parse_hwx(data, "h16")["program_descriptor"]["text_words"] * 4 == len(task)
    assert hashlib.sha256(kernel).digest() == hashlib.sha256(expected_kernel).digest()
    validation = subprocess.run([sys.executable, str(VALIDATOR), str(image)],
        capture_output=True, text=True, timeout=30)
    assert validation.returncode == 0, validation.stdout + validation.stderr


def compile_and_validate(root: Path, name: str, source_text: str) -> Path:
    source = root / f"{name}.mil"
    source.write_text(source_text)
    output = root / name
    result = subprocess.run([str(COMPILER), "--mil", str(source),
        "--model-root", str(root), "--target", "H16G", "--output", str(output)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    image = output / "program-0.hwx"
    assert_kernel_bar_bound(image.read_bytes())
    validation = subprocess.run([sys.executable, str(VALIDATOR), str(image),
        "--manifest", str(output / "manifest.json")], capture_output=True,
        text=True, timeout=30)
    assert validation.returncode == 0, validation.stdout + validation.stderr
    return image


def run_stem2_case(root: Path, channels: int, gelu: bool) -> None:
    weights = pattern(channels * channels * 3, channels + int(gelu))
    bias = pattern(channels, channels + int(gelu) + 1)
    offsets = blob(root / "parameters.bin", [weights, bias])
    image = compile_and_validate(root, f"stem2-{channels}-{int(gelu)}",
                                 mil_stem2(channels, offsets, gelu))
    kernel = section(image.read_bytes(), b"__kern_0")
    table = kernel[:128] if gelu else b""
    payload = weights + bias + table
    expected = subprocess.run([str(DUMP), "kernel2", "3000", str(channels),
        str(int(gelu))], input=payload, stdout=subprocess.PIPE,
        timeout=120, check=True).stdout
    assert hashlib.sha256(kernel).digest() == hashlib.sha256(expected).digest()


def run_layout_tail(root: Path, channels: int) -> None:
    compile_and_validate(root, f"transpose-{channels}", transpose_mil(channels))
    position = pattern(1500 * channels, channels + 7)
    offset = blob(root / "parameters.bin", [position])[0]
    compile_and_validate(root, f"position-{channels}",
                         positional_mil(channels, offset))


def main() -> None:
    for (channels, gelu), (kernel_bytes, digest) in ROWS.items():
        payload = bytes(channels * 80 * 3 * 2 + channels * 2 +
                        (128 if gelu else 0))
        task = subprocess.run([str(DUMP), "task", "3000", str(channels),
                               str(int(gelu))], input=payload,
                              stdout=subprocess.PIPE, check=True).stdout
        tasks = split_h14_tasks(task[16:])
        encoded = json.dumps([canonical(value) for value in tasks],
                             sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == digest
        kernel = subprocess.run([str(DUMP), "kernel", "3000", str(channels),
                                 str(int(gelu))], input=payload,
                                stdout=subprocess.PIPE, check=True).stdout
        assert len(kernel) == kernel_bytes
    with tempfile.TemporaryDirectory(prefix="h16g-whisper-stem-") as temp:
        root = Path(temp)
        run_case(root, 3000, 384, False)
        run_case(root, 3000, 512, True)
        run_case(root, 1000, 384, False)
        run_case(root, 3001, 512, True)
        run_stem2_case(root, 384, False)
        run_stem2_case(root, 512, True)
        run_layout_tail(root, 384)
        run_layout_tail(root, 512)
    print("H16G Whisper stem PASS: stem1 parity/held-outs; stem2, transpose, "
          "positional-add CLI/validator")


if __name__ == "__main__":
    main()
