#!/usr/bin/env python3
"""H14 unary programs (abs, exp, gelu, leaky_relu, rsqrt, sigmoid, silu, sqrt, tanh) compile to fixed bytes with the layout of the programs that ran.

The MIL is the oracle record's own (research/oracles/h14/unary_<op>_c<C>.json, error null). The programs hold no weights, so the whole
program is pinned: its size and sha256. The test also checks the header layout that `ane-run` and the device runs rely on: one input and one
output, each declared (1, C, 1, 1, 64, 64) with the tile count the surface needs (C channels of 64 bytes, in whole 16 KiB tiles). Relu is not
here: it already ran. clip and log have oracle records without Apple output, so the compiler correctly refuses them.
Usage: tests/test_h14_unary_programs.py [path-to-mil-hwxc]
"""
import hashlib
import json
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()

# ANEC package header, as read by libane's struct anec (little endian). Offsets and sizes in bytes.
HEADER_IO_COUNTS_OFFSET = 32  # two uint32: input count, output count
HEADER_TILES_OFFSET = 40  # one uint32 tile count per channel
TILE_COUNT_BYTES = 4
HEADER_NCHW_OFFSET = 168  # six uint64 (n, c, h, w, plane bytes, row bytes) per channel
NCHW_BYTES = 48
INPUT_CHANNEL, OUTPUT_CHANNEL = 5, 4
CHANNEL_BYTES = 64  # one fp16 value per channel, padded to 64 bytes
TILE_BYTES = 16384

EXPECTED = {  # case: (program bytes, sha256 of the whole program)
    "unary_abs_c1024": (20736, "6ace4bd1a0ee66f3bfd6b786e90ef9c33c7767d1f20c5faeeb2570f96a5730a6"),
    "unary_abs_c128": (20736, "d63fdf800086b4294c7b2165511ed06ec35836bc28310ecf89d0546f41583617"),
    "unary_abs_c2048": (20736, "ffae3199bb1799a2b3b4ed3e390451a08ade8a272b4a98f30c043e86b89596d6"),
    "unary_abs_c256": (20736, "2d3314e471740ff75b41ea3ea91763a5ff2c3024d35b4238c01571288e9232e6"),
    "unary_abs_c4096": (20736, "5a26389a9e2df618be38dc053b032288ee4ba762b3f282cf489d25dc0a6df191"),
    "unary_abs_c512": (20736, "9327a678c3ae29c343abd2a23539897158e8c2c4e9a30b45ed903215eea0268b"),
    "unary_abs_c64": (20736, "bcbfd0a70370c026c0cf49faa443bbb0fc4d4ceb93a995b738d79de65762a332"),
    "unary_exp_c512": (4480, "8599331bcf5da53bbbe018045b8f2bbd9907c59d02a6a4d556336ccaa36b83db"),
    "unary_exp_c64": (4480, "e0465c7c74733e651b101fbb6e591be96f8e2193d82afc68ee4d2dddaad8baba"),
    "unary_gelu_c512": (4480, "0721ecf5664c34e625b5436e9ce12504ae2c8cd21c6be40e5d7d84255d51cff4"),
    "unary_gelu_c64": (4480, "ff592b23887cfe80123cbe14f3aa2466c6d16383504563594ddd906f04341745"),
    "unary_leaky_relu_c512": (4480, "a7b7266821a2a5101f47c7d2a1d42d2b77d4847f3056e5095a0dfb6d0dca204f"),
    "unary_leaky_relu_c64": (4480, "1a78e07cd66b7a3ba67cd8bed2cdabc450d155878da7afea17521561e2d70dbd"),
    "unary_rsqrt_c512": (4544, "a60f80a6a2d3f7134bc51baaaf552e9850870b699fad4b841deafae6650316af"),
    "unary_rsqrt_c64": (4544, "f4e706dbac78af309bebf5150e6ed7e1a88ae9e3009129f04403cd40dba5e1d7"),
    "unary_sigmoid_c512": (4480, "52f904988eae89b1145a79de8f9e30d5b3ce53d063a4512f7078eaeb0446bcac"),
    "unary_sigmoid_c64": (4480, "2ec95a9b3ae8165b1b23f47f8290f6b5ec50ca4b43bbde4e426132d8c8bff2e0"),
    "unary_silu_c512": (4480, "f5d62c9529a76cb6ab0da8805d01fef42956d5b2fb771a00112679a8f595761b"),
    "unary_silu_c64": (4480, "d52fa9cc7cf9201a7a5c9ff6befd42a3e1d6a9e2b2875639af45bab97c704b54"),
    "unary_sqrt_c512": (4480, "c39db681df469d4a43d8aaf3d49cc60bb7e4611787e2abe4c48273054fc931fe"),
    "unary_sqrt_c64": (4480, "caf97d4fa1b55aafedb2000741a7b9c2dcd78fe6de6dc65fc872b0e3d87ea89a"),
    "unary_tanh_c512": (4480, "52ccd9028698828caabd7552a88998c548f8765d7b975d1fb778a5fbb91339cf"),
    "unary_tanh_c64": (4480, "559f14f9624dcadda2f31877b3a899cc2191838d115828df94a6c4557049d9e9"),
}


def compile_case(case):
    record = json.loads((ROOT / "research/oracles/h14" / f"{case}.json").read_text())
    assert record.get("error") is None, f"{case}: oracle record has an error"
    with tempfile.TemporaryDirectory() as tmp:
        mil = Path(tmp) / "op.mil"
        mil.write_text(record["mil"])
        model = Path(tmp) / "model"
        model.mkdir()
        out = Path(tmp) / "package"
        subprocess.run([str(COMPILER), "--mil", str(mil), "--model-root", str(model), "--target", "H14", "--output", str(out)],
                       check=True, capture_output=True, text=True, timeout=scaled_timeout(120))
        programs = sorted(out.glob("program-*.anec"))
        assert len(programs) == 1, f"{case}: expected one program, got {len(programs)}"
        return programs[0].read_bytes()


def layout_problems(blob, channels):
    problems = []
    if struct.unpack_from("<II", blob, HEADER_IO_COUNTS_OFFSET) != (1, 1):
        problems.append("not one input and one output")
    want_tiles = -(-channels * CHANNEL_BYTES // TILE_BYTES)
    for name, channel in (("input", INPUT_CHANNEL), ("output", OUTPUT_CHANNEL)):
        nchw = struct.unpack_from("<6Q", blob, HEADER_NCHW_OFFSET + NCHW_BYTES * channel)
        tiles = struct.unpack_from("<I", blob, HEADER_TILES_OFFSET + TILE_COUNT_BYTES * channel)[0]
        if nchw != (1, channels, 1, 1, CHANNEL_BYTES, CHANNEL_BYTES):
            problems.append(f"{name} declared {nchw}")
        if tiles != want_tiles:
            problems.append(f"{name} has {tiles} tiles, not {want_tiles}")
    return problems


def main():
    bad = 0
    for case, (size, sha) in EXPECTED.items():
        blob = compile_case(case)
        problems = layout_problems(blob, int(re.search(r"_c(\d+)$", case).group(1)))
        if len(blob) != size:
            problems.append(f"size {len(blob)} is not {size}")
        got = hashlib.sha256(blob).hexdigest()
        if got != sha:
            problems.append(f"sha256 {got} is not {sha}")
        if problems:
            print(f"FAIL {case}: " + "; ".join(problems))
            bad += 1
        else:
            print(f"PASS {case}")
    print(f"{len(EXPECTED) - bad} of {len(EXPECTED)} H14 unary programs match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
