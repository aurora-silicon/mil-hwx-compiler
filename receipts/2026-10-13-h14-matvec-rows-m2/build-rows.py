#!/usr/bin/env python3
"""Build the 27 H14 matvec programs with 2, 8 and 64 rows (K and N each 256, 512, 1024) for a device run. No device is touched.

Usage: build-rows.py OUTDIR [path-to-mil-hwxc] [--only SHAPE,SHAPE]   (default compiler: build/mil-hwxc in this repository)
Each shape directory mMkKnN holds matvec.mil, model/weights.bin, w.raw.f16 (fp16 [N, K]), x.f16 (M * K dense halves, zero padded to
256 KiB) and out/program-0.anec. Weights and input come from numpy, seeded from M, K and N, so a rerun gives the same files.
"""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

args = sys.argv[1:]
only = None
if "--only" in args:
    i = args.index("--only")
    only = set(args[i + 1].split(","))
    del args[i:i + 2]
out_root = Path(args[0])
compiler = Path(args[1] if len(args) > 1 else Path(__file__).resolve().parents[2] / "build/mil-hwxc").resolve()

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

bad = 0
for m in (2, 8, 64):
    for k in (256, 512, 1024):
        for n in (256, 512, 1024):
            if only is not None and f"m{m}k{k}n{n}" not in only:
                continue
            d = out_root / f"m{m}k{k}n{n}"
            (d / "model").mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng((m * 1000 + k) * 100003 + n)
            w = rng.uniform(-1, 1, (n, k)).astype(np.float16)
            x = np.zeros(1 << 17, dtype=np.float16)
            x[:m * k] = rng.uniform(-1, 1, m * k).astype(np.float16)
            head = bytearray(128)
            struct.pack_into("<IIQQ", head, 64, 0xDEADBEEF, 1, w.nbytes, 128)
            (d / "model/weights.bin").write_bytes(bytes(head) + w.tobytes())
            w.tofile(d / "w.raw.f16")
            x.tofile(d / "x.f16")
            (d / "matvec.mil").write_text(MIL.format(m=m, k=k, n=n))
            r = subprocess.run([str(compiler), "--mil", str(d / "matvec.mil"), "--model-root", str(d / "model"),
                                "--target", "H14", "--output", str(d / "out")], capture_output=True, text=True, check=False)
            programs = sorted((d / "out").glob("program-*.anec"))
            if r.returncode or len(programs) != 1:
                print(f"M={m} K={k} N={n}: failed (exit {r.returncode}, {len(programs)} programs)")
                bad += 1
            else:
                print(f"M={m:<2} K={k:<4} N={n:<4} program bytes {programs[0].stat().st_size}")
sys.exit(1 if bad else 0)
