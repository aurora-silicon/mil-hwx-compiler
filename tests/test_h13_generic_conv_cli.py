#!/usr/bin/env python3
"""Host-side checks for the formula-built H13 convolution and the conv+relu
fusion (docs/ane/fused-conv.md).

1. Parity: every decoded single-task dense or depthwise template the generic
   encoder accepts is rebuilt from the formulas (MILHWXC_H13_CONV_GENERIC=1)
   and must equal Apple's own task words and constant section. The one field
   allowed to differ is the low half of header[1], a cost hint the decoded
   corpus does not explain; the M1 Pro runs the formula-built programs with
   the base template's value.
2. Coverage: MobileNet-class shapes outside the decoded grid compile to one
   program, and kernels over the per-task budget split into linked tasks.
3. Fusion: conv followed by relu is one program with the NE clamp bit set.
4. Refusals: shapes the generic encoder has not run on a device stay refused.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "tests"))

from timeouts import scaled_timeout  # noqa: E402

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())

HEADER_BYTES = 0x1000
TASK_WORDS_HEADER = 10
NE_MAC_CONFIG = 0x0C804
RELU_CLAMP = 0x00010000


def blob(entries):
    """A Core ML weight file: one 64-byte metadata record per entry."""
    out = bytearray(struct.pack("<II", 1, 2) + bytes(56))
    offsets = []
    for payload in entries:
        meta = len(out)
        out += struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload), meta + 64) + bytes(40)
        out += payload
        out += bytes((-len(out)) % 64)
        offsets.append(meta)
    return bytes(out), offsets


def halves(count, seed):
    """Deterministic nonzero fp16 payload (no zero, so a stride-2 section has
    a value-independent size)."""
    values = bytearray()
    state = seed * 2654435761 % (1 << 32) or 1
    for _ in range(count):
        state = (state * 1103515245 + 12345) % (1 << 31)
        mantissa = state % 1024
        exponent = 9 + (state >> 10) % 5          # 2^-6 .. 2^-2
        sign = (state >> 16) & 1
        values += struct.pack("<H", (sign << 15) | (exponent << 10) | mantissa)
    return bytes(values)


def conv_source(cin, h, w, cout, k, stride, groups, bias, relu, pad="same"):
    p = (k - 1) // 2 if pad == "same" else 0
    ho = (h + 2 * p - k) // stride + 1
    wo = (w + 2 * p - k) // stride + 1
    cig = cin // groups
    entries = [halves(cout * cig * k * k, cin * 7 + cout * 3 + k)]
    if bias:
        entries.append(halves(cout, cout + 1))
    weights, offsets = blob(entries)
    pad_type = "custom" if p else "valid"
    pads = [p] * 4
    lines = [
        f'string pt = const()[name = string("pt"), val = string("{pad_type}")];',
        f'tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([{stride}, {stride}])];',
        f'tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>({pads})];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];',
        f'int32 gp = const()[name = string("gp"), val = int32({groups})];',
        f'tensor<fp16, [{cout}, {cig}, {k}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{cout}, {cig}, {k}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offsets[0]})))];',
    ]
    bias_arg = ""
    if bias:
        lines.append(f'tensor<fp16, [{cout}]> b = const()[name = string("b"), val = tensor<fp16, [{cout}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offsets[1]})))];')
        bias_arg = "bias = b, "
    result = "c" if relu else "y"
    lines.append(f'tensor<fp16, [1, {cout}, {ho}, {wo}]> {result} = conv({bias_arg}dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string("conv")];')
    if relu:
        lines.append(f'tensor<fp16, [1, {cout}, {ho}, {wo}]> y = relu(x = c)[name = string("relu")];')
    body = "\n    ".join(lines)
    mil = f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {cin}, {h}, {w}]> x) {{
    {body}
  }} -> (y);
}}
"""
    return mil, weights


def compile_case(root, name, spec, generic=False, success=True, diagnostic=None):
    mil, weights = conv_source(**spec)
    case = root / name
    case.mkdir()
    (case / "model.mil").write_text(mil)
    (case / "weights.bin").write_bytes(weights)
    env = dict(os.environ)
    env.pop("MILHWXC_H13_CONV_GENERIC", None)
    if generic:
        env["MILHWXC_H13_CONV_GENERIC"] = "1"
    run = subprocess.run(
        [compiler, "--target", "H13", "--mil", str(case / "model.mil"),
         "--model-root", str(case), "--output", str(case / "out")],
        capture_output=True, text=True, env=env, timeout=scaled_timeout(120))
    if not success:
        assert run.returncode != 0, (name, run.stdout)
        if diagnostic:
            assert diagnostic in run.stderr, (name, run.stderr)
        return None
    assert run.returncode == 0, (name, run.stderr)
    return json.loads((case / "out" / "manifest.json").read_text()), case / "out"


def task_words(anec, offset, count):
    return struct.unpack_from(f"<{count}I", anec, HEADER_BYTES + offset)


def register(anec, task_offset, task_bytes, address):
    words = task_words(anec, task_offset, task_bytes // 4)
    cursor = TASK_WORDS_HEADER + (1 if words[9] & 3 == 3 else 0)
    while cursor < len(words):
        header = words[cursor]
        count = (header >> 26) + 1
        base = header & 0x03FFFFFF
        if base <= address < base + 4 * count:
            return words[cursor + 1 + (address - base) // 4]
        cursor += 1 + count
    raise KeyError(hex(address))


def tasks_of(anec, first_bytes, count):
    offset, size, out = 0, first_bytes, []
    for _ in range(count):
        out.append((offset, size))
        header = task_words(anec, offset, 10)
        if not header[7]:
            break
        size = (((header[1] >> 16) & 0x1FF) + 1) * 4
        offset = header[7]
    return out


# Decoded single-task templates the generic form must reproduce, as
# (Cin, H, W, Cout, k, stride, groups, bias). Dense 1x1/3x3 across channels
# and every decoded spatial size, depthwise 1x1/3x3, stride-2 1x1.
PARITY = [
    (64, 1, 1, 64, 1, 1, 1, False), (64, 8, 8, 64, 1, 1, 1, True),
    (64, 16, 16, 128, 1, 1, 1, False), (128, 16, 16, 512, 1, 1, 1, False),
    (512, 16, 16, 512, 1, 1, 1, False), (1024, 16, 16, 512, 1, 1, 1, False),
    (64, 32, 32, 64, 1, 1, 1, False), (64, 64, 64, 64, 1, 1, 1, True),
    (64, 8, 8, 64, 3, 1, 1, False), (64, 16, 16, 256, 3, 1, 1, True),
    (256, 16, 16, 128, 3, 1, 1, False), (64, 32, 32, 64, 3, 1, 1, True),
    (64, 64, 64, 64, 3, 1, 1, False),
    (64, 16, 16, 64, 3, 1, 64, False), (512, 16, 16, 512, 3, 1, 512, True),
    (1024, 16, 16, 1024, 1, 1, 1024, True),
]

# Stride-2 sections skip zero weights, so a decoded template's section size
# fits only the oracle's own payload; with any other weights the compiler
# falls back to the generic form, whose words equal the template's apart from
# the kernel-DMA table sized from the actual section.
STRIDED = [(64, 16, 16, 64, 1, 2, 1, False), (64, 64, 64, 64, 1, 2, 1, True),
           (256, 16, 16, 128, 1, 2, 1, False)]

# MobileNet-class geometry outside the decoded grid: (spec, tasks).
COVERAGE = [
    ((32, 112, 112, 64, 1, 1, 1, True), 1),
    ((32, 112, 112, 32, 3, 1, 32, True), 1),
    ((16, 112, 112, 32, 3, 1, 1, True), 1),
    ((128, 56, 56, 128, 1, 1, 1, True), 1),
    ((128, 56, 56, 128, 3, 1, 128, True), 1),
    ((64, 112, 112, 128, 1, 2, 1, True), 1),
    ((256, 28, 28, 256, 1, 1, 1, True), 1),
    ((512, 14, 14, 512, 3, 1, 512, True), 1),
    ((1024, 7, 7, 1024, 3, 1, 1024, True), 1),
    ((512, 7, 7, 1024, 1, 1, 1, True), 2),
    ((1024, 7, 7, 1024, 1, 1, 1, True), 4),
]


def spec_of(row, relu=False):
    cin, h, w, cout, k, stride, groups, bias = row
    return dict(cin=cin, h=h, w=w, cout=cout, k=k, stride=stride,
                groups=groups, bias=bias, relu=relu)


with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-generic-") as tmp:
    root = Path(tmp)
    for index, row in enumerate(PARITY):
        template = compile_case(root, f"t{index}", spec_of(row))
        generic = compile_case(root, f"g{index}", spec_of(row), generic=True)
        assert template[0]["programs"][0]["encoder"] == "apple-parity-conv", row
        assert generic[0]["programs"][0]["encoder"] == "h13-generic-conv", row
        a = bytearray((template[1] / "program-0.anec").read_bytes())
        b = bytearray((generic[1] / "program-0.anec").read_bytes())
        assert len(a) == len(b), row
        # header[1] bits 15:0 are the unexplained cost hint.
        a[HEADER_BYTES + 4:HEADER_BYTES + 6] = b"\0\0"
        b[HEADER_BYTES + 4:HEADER_BYTES + 6] = b"\0\0"
        assert a == b, row

    for index, row in enumerate(STRIDED):
        manifest, out = compile_case(root, f"s{index}", spec_of(row))
        program = manifest["programs"][0]
        assert program["encoder"] == "h13-generic-conv", row
        anec = (out / program["file"]).read_bytes()
        assert register(anec, 0, 628, 0x0C800) == 0x182, row   # sparse kernel
        sizes = [register(anec, 0, 628, 0x1F888 + 4 * i) for i in range(16)]
        offsets = [register(anec, 0, 628, 0x1F848 + 4 * i) for i in range(16)]
        assert offsets == [sum(sizes[:i]) for i in range(16)], row
        assert sum(sizes) == program["constantBytes"], row

    for index, (row, tasks) in enumerate(COVERAGE):
        manifest, out = compile_case(root, f"c{index}", spec_of(row))
        assert len(manifest["programs"]) == 1, row
        program = manifest["programs"][0]
        assert program["encoder"] == "h13-generic-conv", (row, program["encoder"])
        assert program["taskDescriptors"] == tasks, (row, program["taskDescriptors"])
        anec = (out / program["file"]).read_bytes()
        linked = tasks_of(anec, 628, tasks)
        assert len(linked) == tasks, row
        cout = row[3]
        if tasks > 1:
            # Output-channel tasks: end-of-stream flags only on the last task,
            # each task writing its own slice of the output surface.
            covered = 0
            plane = program["outputs"][0]["nchw"][4]
            for position, (offset, size) in enumerate(linked):
                header = task_words(anec, offset, 10)
                last = position == tasks - 1
                assert bool(header[2] & 0x22) == last, (row, position)
                assert bool(header[8] & (1 << 23)) == (not last), (row, position)
                assert register(anec, offset, size, 0x17804) == covered * plane, row
                covered += register(anec, offset, size, 0x10)
            assert covered == cout, row

    # conv -> relu is one program whose NE MAC word carries the clamp.
    for index, row in enumerate([(64, 64, 64, 64, 1, 1, 1, False),
                                 (128, 56, 56, 128, 3, 1, 128, True),
                                 (1024, 7, 7, 1024, 1, 1, 1, True)]):
        manifest, out = compile_case(root, f"f{index}", spec_of(row, relu=True))
        assert len(manifest["programs"]) == 1, row
        program = manifest["programs"][0]
        assert program["operation"] == "conv+relu", row
        assert program["outputs"][0]["name"] == "y", row
        assert "c" not in manifest["tensors"], row
        anec = (out / program["file"]).read_bytes()
        for offset, size in tasks_of(anec, 628, program["taskDescriptors"]):
            assert register(anec, offset, size, NE_MAC_CONFIG) & RELU_CLAMP, row

    # Still refused: forms the generic encoder has not run on the device.
    for name, row in (
            ("s2-16-lanes", (128, 56, 56, 256, 1, 2, 1, False)),
            ("s2-3x3", (64, 16, 16, 64, 3, 2, 1, False)),
            ("mixed-lanes", (64, 16, 16, 48, 1, 1, 1, False)),
            ("odd-channels", (24, 16, 16, 64, 1, 1, 1, False)),
            ("grouped-4", (64, 56, 56, 64, 3, 1, 4, False))):
        compile_case(root, name, spec_of(row), success=False)

print(f"h13 generic conv cli: PASS ({len(PARITY)} parity, {len(STRIDED)} strided, {len(COVERAGE)} coverage, 3 fusion, 5 refusals)")
