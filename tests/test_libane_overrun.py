#!/usr/bin/env python3
"""The device-free libane overrun check: the port refuses a transfer larger than its surface, and no H13 matvec program overruns its surfaces.

Usage: test_libane_overrun.py [COMPILER]
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import libane_overrun as lo
from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
MATVEC_MIL = '''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {k}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(true)];
    tensor<fp16, [{n}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{n}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {n}]> product = matmul(transpose_x = f, transpose_y = ty, x = x, y = w)[name = string("product")];
  }} -> (product);
}}
'''


def synthetic(source_bytes, source_tiles):
    """One-task ANEC: source 1 on channel 5, destination on channel 4 (one tile), the source transfer moving `source_bytes`."""
    words = [0] * 10
    words[8] = 5 | (4 << 12)
    for base, values in ((0x13800, [1, 0x8880, 0, 0, 0, source_bytes, source_bytes]), (0x17800, [1, 0, 0, 0, 0x4000, 0x4000])):
        words.append(((len(values) - 1) << 26) | base)
        words.extend(values)
    task = struct.pack(f"<{len(words)}I", *words)
    tiles = [0] * 32
    tiles[4], tiles[5] = 1, source_tiles
    header = struct.pack("<QIIQQII32I", len(task), len(task), 1, len(task), 0, 1, 1, *tiles)
    return header.ljust(lo.HEADER, b"\0") + task


def matvec_programs(k, n, tmp):
    (tmp / "model").mkdir(parents=True)
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, k * n * 2, 128)
    (tmp / "model" / "weights.bin").write_bytes(bytes(header) + bytes(k * n * 2))
    (tmp / "matvec.mil").write_text(MATVEC_MIL.format(k=k, n=n))
    out = tmp / "package"
    subprocess.run([str(COMPILER), "--mil", str(tmp / "matvec.mil"), "--model-root", str(tmp / "model"), "--target", "H13", "--output", str(out)],
                   check=True, capture_output=True, text=True, timeout=scaled_timeout(120))
    return sorted(out.glob("program-*.anec"))


def main():
    failures = []
    if lo.overruns(synthetic(0x8000, 1)) != [(5, 0x8000, 0x4000)]:
        failures.append("the port did not report a 32768-byte transfer on a 16384-byte surface")
    if lo.overruns(synthetic(0x8000, 2)) or lo.overruns(synthetic(0x4000, 1)):
        failures.append("the port refused a transfer that fits its surface")
    for k, n in ((256, 256), (256, 512), (512, 256), (512, 512)):
        with tempfile.TemporaryDirectory() as directory:
            for program in matvec_programs(k, n, Path(directory)):
                found = lo.overruns(program.read_bytes())
                if found:
                    failures.append(f"matvec K={k} N={n} {program.name}: surface channel {found[0][0]} needs {found[0][1]} bytes but is allocated {found[0][2]}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        sys.exit(1)
    print("PASS libane overrun check: synthetic cases and H13 matvec K, N in {256, 512}")


if __name__ == "__main__":
    main()
