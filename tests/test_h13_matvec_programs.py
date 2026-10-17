#!/usr/bin/env python3
"""Pin H13 matvec package structures at K=N=256, 512, and 1024 using stdlib only."""
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
PINNED = {
    256: ((529024, "b4720f876d3b0f47645eb6fa8dfa7fc9522cdbdee1dd36715a22c9f42358e2cc"),),
    512: ((529024, "619076bd76e1e7208b54d630cb217b7f2382b44249a8b052c33cdc005fe38542"),),
    1024: (
        (529024, "619076bd76e1e7208b54d630cb217b7f2382b44249a8b052c33cdc005fe38542"),
        (529024, "619076bd76e1e7208b54d630cb217b7f2382b44249a8b052c33cdc005fe38542"),
        (529024, "619076bd76e1e7208b54d630cb217b7f2382b44249a8b052c33cdc005fe38542"),
        (529024, "619076bd76e1e7208b54d630cb217b7f2382b44249a8b052c33cdc005fe38542"),
        (20992, "e3e304d6691ab77888d3e928c9d6493f47f0c1c00829ea4a706a5496ffb5f2f9"),
    ),
}
PREFIX_BYTES = 640
MIL = '''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {k}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(true)];
    tensor<fp16, [{k}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{k}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {k}]> product = matmul(transpose_x = f, transpose_y = ty, x = x, y = w)[name = string("product")];
  }} -> (product);
}}
'''


def compile_matvec(k, seed, tmp):
    rng = random.Random(seed)
    weights = b"".join(struct.pack("<e", rng.uniform(-1, 1)) for _ in range(k * k))
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(weights), 128)
    model = tmp / "model"
    model.mkdir(parents=True)
    (model / "weights.bin").write_bytes(bytes(header) + weights)
    mil = tmp / "matvec.mil"
    mil.write_text(MIL.format(k=k))
    out = tmp / "package"
    subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(model), "--target", "H13", "--output", str(out)],
        check=True, capture_output=True, text=True, timeout=scaled_timeout(120))
    programs = sorted(out.glob("program-*.anec"))
    return [p.read_bytes() for p in programs]


def main():
    failures = []
    for k, expected in PINNED.items():
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            first = compile_matvec(k, 1, tmp / "a")
            second = compile_matvec(k, 2, tmp / "b")
        if len(first) != len(expected) or len(second) != len(expected):
            failures.append(f"K=N={k}: expected {len(expected)} programs, got {len(first)} and {len(second)}")
            continue
        for index, (a, b, (size, prefix_sha)) in enumerate(zip(first, second, expected)):
            prefix = hashlib.sha256(a[:PREFIX_BYTES]).hexdigest()
            if len(a) != size:
                failures.append(f"K=N={k} program-{index}: size {len(a)} is not {size}")
            if prefix != prefix_sha:
                failures.append(f"K=N={k} program-{index}: prefix sha {prefix} is not {prefix_sha}")
            if a[:PREFIX_BYTES] != b[:PREFIX_BYTES]:
                failures.append(f"K=N={k} program-{index}: structure changed with weights")
            if index < len(expected) - 1 or k != 1024:
                if a[PREFIX_BYTES:] == b[PREFIX_BYTES:]:
                    failures.append(f"K=N={k} program-{index}: weight constants did not change")
        if not any(f"K=N={k} " in failure for failure in failures):
            print(f"PASS H13 matvec K=N={k}: {len(expected)} pinned program(s)")
    if failures:
        print("FAIL " + "; ".join(failures))
        return 1
    print(f"PASS {len(PINNED)} H13 matvec geometries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
