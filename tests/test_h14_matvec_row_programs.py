#!/usr/bin/env python3
"""H14 matvec with 2, 8 and 64 rows (M x K times K x N, fp16 weights, the 256/512/1024 grid plus decoded M64 K2048/4096 N2048) compiles to a fixed program structure.

Each program is a 4,672-byte header and task stream followed by the weight section. The first 4,672 bytes depend only on M, K and N,
not on the weights (checked for every shape: two weight sets give the same head and different weight sections). The test pins each
program's size and the sha256 of that head, and checks that the header declares dense rows: input nchw (1, 1, M, K) and output
(1, 1, M, N) with row bytes K * 2 and N * 2, and tile counts equal to the surface bytes rounded up to 16 KiB tiles. Those are the
layout rules `ane-run --check matvec` applies. Weights come from the standard library, so no numpy is needed.
Usage: tests/test_h14_matvec_row_programs.py [path-to-mil-hwxc]
"""
import hashlib
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
HEAD = 4672

# ANEC package header, as read by libane's struct anec (little endian). Offsets and sizes in bytes.
HEADER_IO_COUNTS_OFFSET = 32  # two uint32: input count, output count
HEADER_TILES_OFFSET = 40  # one uint32 tile count per channel
TILE_COUNT_BYTES = 4
HEADER_NCHW_OFFSET = 168  # six uint64 (n, c, h, w, plane bytes, row bytes) per channel
NCHW_BYTES = 48
CHANNELS = 32  # channels in the header tables
INPUT_CHANNEL, OUTPUT_CHANNEL = 5, 4  # the matvec input and output surfaces
TILE_BYTES = 16384  # one tile of a channel surface

EXPECTED = {  # (M, K, N): (program bytes, sha256 of the first 4,672 bytes)
    (2, 256, 256): (135744, "cece7d5e7235fb7c746814be2f97245d3a42aa44a2b5d5da1f4349e884671cd9"),
    (2, 256, 512): (266816, "f6a791df4e3a0ff1432ea855c61cb18ff462cd35c9d6f84456993a342db51dea"),
    (2, 256, 1024): (528960, "dbd5d5b315d82b82224b2fdbcbd634282423c0b1df01d47306b68d87ac146760"),
    (2, 512, 256): (266816, "a1a5bdbcd14ed12022152696ccf245bd4923c18f27bd0882d5b36d5981c7375d"),
    (2, 512, 512): (528960, "7c88a9b3a10d05a17ef2e8408b0f12db2149ad9cb45d7e5583f97479a7935f3a"),
    (2, 512, 1024): (1053248, "72eb533543028fc5155cc2a6c3547da248ce9bb9fd209d84074bf42d1f7c1f52"),
    (2, 1024, 256): (528960, "b5e0ad07bdf87e1e3c2abda6e1ac677336dc198318e35fd095f411b49223e62b"),
    (2, 1024, 512): (1053248, "a81056f3d173d73720cfda1c2172b0dd9b9063fe093d9668613c30981e56a397"),
    (2, 1024, 1024): (2101824, "84ef80c2346ef9e2d34892518eea82b49ffa68236ea74f44ea5e1f18b4b940e3"),
    (8, 256, 256): (135744, "66c56b6c2bfaf7ab6d059a314d3a1e22ce91e7fb8545e3eb459b123ea4064669"),
    (8, 256, 512): (266816, "eefeda419c9ba107e9626b4c7881becdbfb9caf84427ae50a6eb9c828356d653"),
    (8, 256, 1024): (528960, "6c9c3d94478a41e0e723c25e3355b51de1bd1f3dc550d97d73577dd2f96a82af"),
    (8, 512, 256): (266816, "505c06fcc5f624f6e8635d35a876fdbc694e3822efb589c8d66d622b6c2e8d98"),
    (8, 512, 512): (528960, "bb31cb029ebf0affacd0c7aff8adb333aca4022f355200336508ebfe61f27913"),
    (8, 512, 1024): (1053248, "e86ee4cb15527d25448f4564810667785273b382a75d1e23420615dcd40f0581"),
    (8, 1024, 256): (528960, "53524cdb189a61b64e9e3f3e44fbb0f78deaf4a15925d6bde67e7bee55c41fd1"),
    (8, 1024, 512): (1053248, "6a53d186742be90411c733565e3c656065830008e1ec1a709e063af4fe506974"),
    (8, 1024, 1024): (2101824, "a611d14947f32bfc34c1943aaa72fdbc326a1fe7d9d98817a16020a385d3a18b"),
    (64, 256, 256): (135744, "69552e89f06bd394b7d58413b0f03d8530b2072a5baa0dcecefc2cb7b13a7b50"),
    (64, 256, 512): (266816, "bf0d23abf12a761531da4351377e04af88824b71f45c9085fd5b8df7b72489f3"),
    (64, 256, 1024): (528960, "ee430235153ce9293f6b46aeb191e17277dd1329b28fd8d9ef2e6d9dfda13cb4"),
    (64, 512, 256): (266816, "895c7c038da892964920451831ed4707906f34e7f06d0e22e4fe6bd1897a69c9"),
    (64, 512, 512): (528960, "6abfde60d8a5c92b2fc837e35f62b44ac287057c2e956cfa1f9b89f003e6d7f4"),
    (64, 512, 1024): (1053248, "7217b060c9206ba824334565d1c291b2a61d40833f5a815247921646fcd492b2"),
    (64, 1024, 256): (528960, "aa2935075cc9b50f415a6f6400e6652cbf72d258d5e84dd81c7d3bbe28b8b54b"),
    (64, 1024, 512): (1053248, "fa6b50d6605ede4043321f1ba3755b7cbbdb372105a7484890f58e4bde3ae6f0"),
    (64, 1024, 1024): (2101824, "390dc1c09220127540768fb532d42193e632dfd900aeb97f5b41be1e48325ec7"),
    (64, 2048, 2048): (8393280, "94d5fdfd86466caff9beb7f88b688f56daeb1cf062fbbcae6ebfe5fbbb1c7de2"),
    (64, 4096, 2048): (16781888, "25c728ea81ea13d5892223d83a8cb5e749aeb1f95eeb85f554b344829b9d430c"),
}

MIL = """program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [{m}, {k}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(true)];
    tensor<fp16, [{n}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{n}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [{m}, {n}]> product = matmul(transpose_x = f, transpose_y = ty, x = x, y = w)[name = string("product")];
  }} -> (product);
}}
"""


def compile_matvec(m, k, n, seed):
    rnd = random.Random(seed)
    weights = b"".join(struct.pack("<e", rnd.uniform(-1, 1)) for _ in range(k * n))
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(weights), 128)
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        (model / "weights.bin").write_bytes(bytes(header) + weights)
        mil = Path(tmp) / "matvec.mil"
        mil.write_text(MIL.format(m=m, k=k, n=n))
        out = Path(tmp) / "package"
        subprocess.run(
            [str(COMPILER), "--mil", str(mil), "--model-root", str(model), "--target", "H14", "--output", str(out)],
            check=True, capture_output=True, text=True, timeout=scaled_timeout(120))
        programs = sorted(out.glob("program-*.anec"))
        assert len(programs) == 1, f"M={m} K={k} N={n}: expected one program, got {len(programs)}"
        return programs[0].read_bytes()


def layout_problems(blob, m, k, n):
    problems = []
    src, dst = struct.unpack_from("<II", blob, HEADER_IO_COUNTS_OFFSET)
    tiles = struct.unpack_from(f"<{CHANNELS}I", blob, HEADER_TILES_OFFSET)
    n4 = struct.unpack_from("<6Q", blob, HEADER_NCHW_OFFSET + NCHW_BYTES * OUTPUT_CHANNEL)
    n5 = struct.unpack_from("<6Q", blob, HEADER_NCHW_OFFSET + NCHW_BYTES * INPUT_CHANNEL)
    if (src, dst) != (1, 1):
        problems.append(f"io counts {src},{dst}")
    if n4[:4] != (1, 1, m, n) or n5[:4] != (1, 1, m, k):
        problems.append(f"declared shapes {n4[:4]} {n5[:4]}")
    if n4[5] != n * 2 or n5[5] != k * 2:
        problems.append(f"row bytes {n4[5]} {n5[5]}")
    for channel, nbytes in ((OUTPUT_CHANNEL, m * n * 2), (INPUT_CHANNEL, m * k * 2)):
        want = -(-nbytes // TILE_BYTES)
        if tiles[channel] != want:
            problems.append(f"channel {channel} has {tiles[channel]} tiles, not {want}")
    return problems


def main():
    # New decoded entries do not admit an unmeasured neighbour or other row count.
    for m, k, n in ((64, 2048, 1024), (8, 2048, 2048)):
        try:
            compile_matvec(m, k, n, 1)
        except subprocess.CalledProcessError as error:
            assert "h14.outside-parity-envelope" in error.stderr, error.stderr
        else:
            raise AssertionError(f"undecoded shape accepted: {m}, {k}, {n}")
    bad = 0
    for (m, k, n), (size, head_sha) in EXPECTED.items():
        a, b = compile_matvec(m, k, n, 1), compile_matvec(m, k, n, 2)
        got = hashlib.sha256(a[:HEAD]).hexdigest()
        problems = layout_problems(a, m, k, n)
        if len(a) != size:
            problems.append(f"size {len(a)} is not {size}")
        if got != head_sha:
            problems.append(f"head sha {got} is not {head_sha}")
        if a[:HEAD] != b[:HEAD]:
            problems.append("the 4,672-byte head changed with the weights")
        if a[HEAD:] == b[HEAD:]:
            problems.append("the weight section did not change with the weights")
        if problems:
            print(f"FAIL M={m} K={k} N={n}: " + "; ".join(problems))
            bad += 1
        else:
            print(f"PASS M={m} K={k} N={n}")
    print(f"{len(EXPECTED) - bad} of {len(EXPECTED)} H14 row-variant matvec shapes match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
