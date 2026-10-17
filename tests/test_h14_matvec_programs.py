#!/usr/bin/env python3
"""H14 matvec (1 x K times K x N, fp16 weights) at the three square shapes and the six non-square shapes of 256, 512 and 1024
compiles to the same program structure.

An H14 matvec program is a 4,672-byte header and task stream followed by the weight section. The first 4,672 bytes depend
only on K and N, not on the weights, so the test pins their sha256 and the total size for each shape. For K = N = 256 those
4,672 bytes equal the first 4,672 bytes of the omarchy-ane fixture fixtures/h14-anec/matvec/program-0.anec. The test also
checks that two different weight sets give the same head and different weight sections. Weights come from the standard
library, so no numpy is needed.

The three square shapes ran on an M2 Max (device receipt: receipts/2026-10-11-h14-matvec-m2/). The six non-square shapes are
pinned here before any device run; receipts/2026-10-13-h14-matvec-nonsquare-m2/ holds the prediction and, once it exists, the
result. The device runs used numpy weights, so their whole-program hashes differ from any program this test builds; the
4,672-byte heads are the same (checked for the six non-square programs).
Usage: tests/test_h14_matvec_programs.py [path-to-mil-hwxc]
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

EXPECTED = {  # (K, N): (program bytes, sha256 of the first 4,672 bytes)
    (256, 256): (135744, "e539e0cebce90180ffac54afe05e6cc45fc2bac41255c25f1ed997c77cc097a4"),
    (512, 512): (528960, "26f06785c1e14a54aaf918c6467be7f5176c59a580424a0253c5c7e331f9b7e5"),
    (1024, 1024): (2101824, "c15bd71d6ae3da4e92d1ff5b3bc039911c6c7505aaa6c62f9b13db4fb34b2f98"),
    (256, 512): (266816, "1b3e636b8c76b6a1df5cebce6c5867ea9b41925f8aa2b8f2de5b62b72e013329"),
    (256, 1024): (528960, "1f362af81ee7cf2622282ee3e9cdf0537f3d67a2bf1d694de5e8d57edde3c2cd"),
    (512, 256): (266816, "c7e6d4fffc805f3fde6f9c12c1c529b7305a1dbc650a0de750656636f002930a"),
    (512, 1024): (1053248, "5a3bed120b444aaf21bd5d1d681935dbfc0604b08ff62003195bae9e59fe168e"),
    (1024, 256): (528960, "b2c812d09ee7dde5c9a534f8963d280aabbf2f1f9982fc54c73ec2df4014def5"),
    (1024, 512): (1053248, "daf23195b05ae95ae916f7ae97f303a778debacf74370d1fa7e029341c89faf2"),
}

MIL = """program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {k}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(true)];
    tensor<fp16, [{n}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{n}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {n}]> product = matmul(transpose_x = f, transpose_y = ty, x = x, y = w)[name = string("product")];
  }} -> (product);
}}
"""


def compile_matvec(k, n, seed):
    rnd = random.Random(seed)
    weights = b"".join(struct.pack("<e", rnd.uniform(-1, 1)) for _ in range(k * n))
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(weights), 128)
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "model"
        model.mkdir()
        (model / "weights.bin").write_bytes(bytes(header) + weights)
        mil = Path(tmp) / "matvec.mil"
        mil.write_text(MIL.format(k=k, n=n))
        out = Path(tmp) / "package"
        subprocess.run(
            [str(COMPILER), "--mil", str(mil), "--model-root", str(model), "--target", "H14", "--output", str(out)],
            check=True, capture_output=True, text=True, timeout=scaled_timeout(120))
        programs = sorted(out.glob("program-*.anec"))
        assert len(programs) == 1, f"K={k} N={n}: expected one program, got {len(programs)}"
        return programs[0].read_bytes()


def main():
    bad = 0
    for (k, n), (size, head_sha) in EXPECTED.items():
        a, b = compile_matvec(k, n, 1), compile_matvec(k, n, 2)
        got = hashlib.sha256(a[:HEAD]).hexdigest()
        problems = []
        if len(a) != size:
            problems.append(f"size {len(a)} is not {size}")
        if got != head_sha:
            problems.append(f"head sha {got} is not {head_sha}")
        if a[:HEAD] != b[:HEAD]:
            problems.append("the 4,672-byte head changed with the weights")
        if a[HEAD:] == b[HEAD:]:
            problems.append("the weight section did not change with the weights")
        if problems:
            print(f"FAIL K={k} N={n}: " + "; ".join(problems))
            bad += 1
        else:
            print(f"PASS K={k} N={n}")
    print(f"{len(EXPECTED) - bad} of {len(EXPECTED)} H14 matvec shapes match")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
