#!/usr/bin/env python3
"""(GPTQ variant: the integers and scales come from the npz of gptq_s1.py, NPZ env or argv[2]) H13 INT8 per-row weight variants of the 17 large programs of h13-qwen38-control38-scaled-convffnexpdiv-L192-pkg-v1 (CPU only).
Every fp16 const named *_w that is 2-D with at least 2^20 elements (the matmul weights, all used as matmul y with transpose_y) becomes
constexpr_affine_dequantize(int8 [N,K], fp16 scale [N], axis 0, zero_point 0), per-output-row symmetric, the same encoding as ANEForge int8 compress
and as the weight-dtype probe that ran on a 13-inch machine. All other consts (norm gains, biases, small tensors) are copied as fp16.
Reads the MIL of the shipped program (same sha as the package manifest) and its weights.bin (same sha as the manifest), writes <out>/pNN/{model.mil,weights.bin}
and <out>/report.json (relative Frobenius error of each quantised weight)."""
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ["ANEFORGE_DIR"])
from aneforge._blob import FP16, INT8, BlobWriter, quantize_per_row  # noqa: E402

PKG_A = Path(os.environ["GPTQ_PKG_MANIFEST"])   # manifest.json of the fp16 package the programs come from
EXPORT = Path(os.environ["GPTQ_EXPORT"])
MIL_ROOT = Path(os.environ["GPTQ_MIL_ROOT"])
MIL_DIR = {"attn": MIL_ROOT / "mil192", "mid": MIL_ROOT / "mil50"}
WANT = [4, 6, 8, 10, 12, 14, 16, 18, 23, 25, 27, 29, 31, 33, 35]   # the S1 set; 2 and 37 stay fp16
ATTN = {6, 12, 18, 25, 31, 37}
MIN_ELEMS = 1 << 20
NPZ = Path(os.environ["GPTQ_NPZ"])   # output directory of gptq_s1.py
CONST = re.compile(r'^(\s*)tensor<fp16, \[([\d, ]+)\]> (\w+) = const\(\)\[name = string\("(\w+)"\), val = tensor<fp16, \[[\d, ]+\]>\(BLOBFILE\(path = string\("@model_path/weights.bin"\), offset = uint64\((\d+)\)\)\)\];$')
sha = lambda b: hashlib.sha256(b).hexdigest()


def blob_at(w: bytes, off: int, nbytes: int) -> bytes:
  magic, code, size = struct.unpack_from("<IIQ", w, off)
  assert magic == 0xDEADBEEF and code == FP16 and size == nbytes, (off, hex(magic), code, size, nbytes)
  data_off = struct.unpack_from("<Q", w, off + 16)[0]
  return w[data_off:data_off + nbytes]


def transform(n: int, out: Path, manifest: dict) -> dict:
  prog = manifest["programs"][n]
  mil_path = (MIL_DIR["attn"] / f"p{n:02d}" if n in ATTN else MIL_DIR["mid"] / "p14") / "model.mil"
  if n == 37:
    mil_path = MIL_DIR["attn"] / "p37/model.mil"
  text = mil_path.read_text()
  assert sha(text.encode()) == prog["mil_sha256"], (n, "mil sha")
  wpath = EXPORT / f"prog{n:02d}/weights.bin"
  w = wpath.read_bytes()
  assert sha(w) == prog["weights_sha256"], (n, "weights sha")
  bw = BlobWriter()
  lines, quant = [], []
  for line in text.split("\n"):
    m = CONST.match(line)
    if not m:
      lines.append(line)
      continue
    indent, dims, name, name2, off = m.groups()
    assert name == name2
    shape = [int(x) for x in dims.split(",")]
    count = int(np.prod(shape))
    arr = np.frombuffer(blob_at(w, int(off), count * 2), dtype=np.float16).reshape(shape)
    if name.endswith("_w") and len(shape) == 2 and count >= MIN_ELEMS:
      assert re.search(r"matmul\(transpose_x = bool\(false\), transpose_y = bool\(true\), x = \w+, y = %s\)" % name, text), (n, name, "not a transpose_y matmul weight")
      z = np.load(NPZ / f"p{n}-{name}.npz")   # GPTQ integers and the fp16-exact row scales (gptq_s1.py)
      qa, sa = z["q"], z["scale"].astype(np.float16)
      assert qa.shape == tuple(shape) and qa.dtype == np.int8 and sa.shape == (shape[0],) and np.array_equal(sa.astype(np.float32), z["scale"]), (n, name)
      q, scale = np.ascontiguousarray(qa).tobytes(), np.ascontiguousarray(sa).tobytes()
      qoff, soff = bw.add(q, INT8), bw.add(scale, FP16)
      a32 = arr.astype(np.float32)
      rec = np.frombuffer(q, dtype=np.int8).reshape(shape).astype(np.float32) * np.frombuffer(scale, dtype=np.float16).astype(np.float32)[:, None]
      quant.append({"name": name, "shape": shape, "rel_fro_err": float(np.linalg.norm(rec - a32) / np.linalg.norm(a32))})
      N, K = shape
      lines.append(f'{indent}tensor<fp16, [{N}, {K}]> {name} = constexpr_affine_dequantize()[axis = int32(0), name = string("{name}"), quantized_data = tensor<int8, [{N}, {K}]>'
                   f'(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({qoff}))), scale = tensor<fp16, [{N}]>'
                   f'(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({soff}))), zero_point = int8(0)];')
    else:
      noff = bw.add(arr.tobytes(), FP16)
      lines.append(line.replace(f"uint64({off})", f"uint64({noff})"))
  new_text = "\n".join(lines)
  assert new_text.count("constexpr_affine_dequantize") == len(quant) and quant, n
  d = out / f"p{n:02d}"
  d.mkdir(parents=True)
  (d / "model.mil").write_text(new_text)
  (d / "weights.bin").write_bytes(bw.build())
  return {"index": n, "mil_sha256_src": prog["mil_sha256"], "weights_sha256_src": prog["weights_sha256"], "mil_sha256": sha(new_text.encode()),
          "weights_sha256": sha((d / "weights.bin").read_bytes()), "quantised": len(quant), "max_rel_fro_err": max(q["rel_fro_err"] for q in quant),
          "mean_rel_fro_err": float(np.mean([q["rel_fro_err"] for q in quant])), "weights_bytes_src": len(w), "weights_bytes_int8": (d / "weights.bin").stat().st_size, "tensors": quant}


if __name__ == "__main__":
  out = Path(sys.argv[1])
  assert not out.exists()
  manifest = json.loads(PKG_A.read_text())
  rep = []
  for n in WANT:
    r = transform(n, out, manifest)
    print(n, r["quantised"], "max err %.4f mean %.4f" % (r["max_rel_fro_err"], r["mean_rel_fro_err"]), r["weights_bytes_src"], "->", r["weights_bytes_int8"], flush=True)
    rep.append(r)
  (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
