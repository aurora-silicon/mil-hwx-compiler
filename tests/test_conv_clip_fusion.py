#!/usr/bin/env python3
"""Clean-room host checks for the H13/H14 fused convolution clip epilogue.

The expected stream deltas and 128-byte lane descriptor are constructed from
the public SPEC rules.  Set MILHWXC_CLIP_PRIVATE_STORE to a directory of
`TARGET-GEOMETRY-biasN-LOW-HIGH.anec` files to additionally compare private
captures; absence is an ordinary skip and no captured bytes enter the tree.
"""
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "tests"))
from h13_td import decode_task, register_words, split_h13_tasks, split_h14_tasks  # noqa: E402
from timeouts import scaled_timeout  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
BOUNDS = ((0.0, 6.0), (0.0, 1.0), (-1.0, 1.0))


def fp16_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def descriptor(lower, upper):
    slots = [0] * 64
    slots[0:4] = [lower, upper, lower, upper]
    slots[37] = slots[39] = 0x3C00
    slots[41], slots[42] = 0x0040, 0x0001
    return struct.pack("<64H", *slots)


def blob(payloads):
    out = bytearray(struct.pack("<II", 1, 2) + bytes(56))
    offsets = []
    for payload in payloads:
        offsets.append(len(out))
        out += struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload), len(out) + 64)
        out += bytes(40) + payload
        out += bytes((-len(out)) % 64)
    return bytes(out), offsets


def source(depthwise, bias, bounds=None):
    channels, kernel = (32, 3) if depthwise else (64, 1)
    groups = channels if depthwise else 1
    weight_count = channels * (1 if depthwise else channels) * kernel * kernel
    weights = struct.pack(f"<{weight_count}H",
                          *(0x3000 + (i % 0x300) for i in range(weight_count)))
    payloads = [weights]
    if bias:
        payloads.append(struct.pack(f"<{channels}H",
                                    *(0x2800 + i for i in range(channels))))
    weight_blob, offsets = blob(payloads)
    constants = []
    if bounds is not None:
        constants += [
            f'fp16 lo = const()[name = string("lo"), val = fp16({bounds[0]})];',
            f'fp16 hi = const()[name = string("hi"), val = fp16({bounds[1]})];',
        ]
    constants += [
        f'string pt = const()[val = string("{"custom" if kernel > 1 else "valid"}")];',
        'tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1,1])];',
        f'tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([{(kernel-1)//2},{(kernel-1)//2},{(kernel-1)//2},{(kernel-1)//2}])];',
        'tensor<int32, [2]> dl = const()[val = tensor<int32, [2]>([1,1])];',
        f'int32 gp = const()[val = int32({groups})];',
        f'tensor<fp16, [{channels}, {1 if depthwise else channels}, {kernel}, {kernel}]> w = const()[val = tensor<fp16, [{channels}, {1 if depthwise else channels}, {kernel}, {kernel}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offsets[0]})))];',
    ]
    bias_arg = ""
    if bias:
        constants.append(
            f'tensor<fp16, [{channels}]> b = const()[val = tensor<fp16, [{channels}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offsets[1]})))];')
        bias_arg = "bias = b, "
    result = "c" if bounds is not None else "y"
    constants.append(
        f'tensor<fp16, [1,{channels},16,16]> {result} = conv({bias_arg}dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x);')
    if bounds is not None:
        constants.append(
            f'tensor<fp16, [1,{channels},16,16]> y = clip(x = c, alpha = lo, beta = hi);')
    body = "\n    ".join(constants)
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1,{channels},16,16]> x) {{
    {body}
  }} -> (y);
}}
''', weight_blob


def compile_case(root, target, geometry, bias, bounds):
    case = root / f"{target}-{geometry}-bias{int(bias)}-{bounds}"
    case.mkdir()
    mil, weights = source(geometry == "dw3x3", bias, bounds)
    (case / "model.mil").write_text(mil)
    (case / "weights.bin").write_bytes(weights)
    env = dict(os.environ)
    env[f"MILHWXC_{target}_CONV_GENERIC"] = "1"
    run = subprocess.run([str(COMPILER), "--target", target, "--mil",
        str(case / "model.mil"), "--model-root", str(case), "--output",
        str(case / "out")], env=env, text=True, capture_output=True,
        timeout=scaled_timeout(120))
    assert run.returncode == 0, run.stdout + run.stderr
    manifest = json.loads((case / "out/manifest.json").read_text())
    data = (case / "out/program-0.anec").read_bytes()
    stream_bytes = struct.unpack_from("<Q", data, 0x10)[0]
    raw = data[0x1000:0x1000 + stream_bytes]
    if target == "H13":
        tasks = split_h13_tasks(raw, manifest["firstTaskBytes"] // 4 - 1,
                                manifest["taskDescriptors"])
    else:
        tasks = split_h14_tasks(raw)
    assert len(tasks) == 1
    registers = register_words(decode_task(tasks[0], target.lower()))
    constants = data[-manifest["constantBytes"]:]
    return manifest, tasks[0], registers, constants, case


def float64_reference_check():
    # One output element, accumulated in Python's IEEE binary64, then clipped.
    x = [1.5, -2.0, 4.0]
    weight = [2.0, -0.5, 0.25]
    bias = -1.0
    convolution = sum(float(a) * float(b) for a, b in zip(x, weight)) + bias
    fused = min(max(convolution, 0.0), 6.0)
    assert math.isclose(convolution, 4.0, rel_tol=0, abs_tol=0)
    assert math.isclose(fused, 4.0, rel_tol=0, abs_tol=0)
    assert min(max(20.0, 0.0), 6.0) == 6.0
    assert min(max(-20.0, 0.0), 6.0) == 0.0


def main():
    private = Path(os.environ["MILHWXC_CLIP_PRIVATE_STORE"]) \
        if os.environ.get("MILHWXC_CLIP_PRIVATE_STORE") else None
    private_checked = 0
    with tempfile.TemporaryDirectory(prefix="mil-hwx-conv-clip-") as tmp:
        root = Path(tmp)
        for target in ("H13", "H14"):
            selector = 0x0C804 if target == "H13" else 0x00D04
            offset_reg = 0x1F848 if target == "H13" else 0x01958
            size_reg = 0x1F888 if target == "H13" else 0x01998
            for geometry in ("pw1x1", "dw3x3"):
                for bias in (False, True):
                    base = compile_case(root, target, geometry, bias, None)
                    base_regs, base_constants = base[2], base[3]
                    streams = []
                    for low, high in BOUNDS:
                        fused = compile_case(root, target, geometry, bias,
                                             (low, high))
                        manifest, task, regs, constants, case = fused
                        assert manifest["programs"][0]["operation"] == "conv+clip"
                        assert regs[selector] == 0x00121C00 | (0x10 if bias else 0)
                        expected = bytearray()
                        base0 = base_regs.get(0x1908, 0) if target == "H14" else 0
                        for lane in range(16):
                            old_offset = base_regs.get(offset_reg + 4 * lane, 0)
                            old_size = base_regs[size_reg + 4 * lane]
                            assert regs.get(offset_reg + 4 * lane, 0) == old_offset + 128 * lane
                            assert regs[size_reg + 4 * lane] == old_size + 128
                            expected += descriptor(fp16_bits(low), fp16_bits(high))
                            expected += base_constants[base0 + old_offset:
                                                       base0 + old_offset + old_size]
                        assert constants == expected
                        if target == "H13":
                            assert len(task) == 628
                            if geometry == "dw3x3":
                                assert struct.unpack_from("<I", task, 4)[0] & 0xffff == 2
                        elif geometry == "dw3x3":
                            assert len(task) == 408
                            assert struct.unpack_from("<I", task, 0)[0] == 0x00660000
                            assert struct.unpack_from("<I", task, 4)[0] & 0xffff == 2
                            assert struct.pack("<I", 0x000F0657) in task
                        streams.append(task)
                        if private:
                            name = f"{target}-{geometry}-bias{int(bias)}-{low:g}-{high:g}.anec"
                            capture = private / name
                            if capture.exists():
                                assert capture.read_bytes() == (case / "out/program-0.anec").read_bytes()
                                private_checked += 1
                    assert streams[0] == streams[1] == streams[2]
    float64_reference_check()
    print("fused conv+clip SPEC parity: PASS (24 generated variants; "
          f"private captures checked={private_checked})")


if __name__ == "__main__":
    main()
