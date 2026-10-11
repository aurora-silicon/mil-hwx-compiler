"""Build a K x N fp16 matvec for H14: MIL (matmul with transpose_y, BLOBFILE weights), weights.bin, raw weights, input.
Usage: make_matvec.py K N OUTDIR   (writes OUTDIR/{matvec.mil,model/weights.bin,w.raw.f16,x.f16})"""
import os
import struct
import sys

import numpy as np

K, N, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(f"{out}/model", exist_ok=True)
rng = np.random.default_rng(K * 100003 + N)
w = rng.uniform(-1, 1, (N, K)).astype(np.float16)
x = np.zeros(1 << 20 >> 1, dtype=np.float16)  # 1 MB file; the K valid halves first, every padding lane zero
x[:K] = rng.uniform(-1, 1, K).astype(np.float16)
with open(f"{out}/model/weights.bin", "wb") as fh:
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, w.nbytes, 128)
    fh.write(bytes(header))
    fh.write(w.tobytes())
w.tofile(f"{out}/w.raw.f16")
x.tofile(f"{out}/x.f16")
mil = f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {K}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(true)];
    tensor<fp16, [{N}, {K}]> w = const()[name = string("w"), val = tensor<fp16, [{N}, {K}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {N}]> product = matmul(transpose_x = f, transpose_y = ty, x = x, y = w)[name = string("product")];
  }} -> (product);
}}
"""
with open(f"{out}/matvec.mil", "w") as fh:
    fh.write(mil)
print("wrote", out, "K", K, "N", N)
