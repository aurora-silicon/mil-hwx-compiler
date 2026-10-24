#!/usr/bin/env python3
"""Activation-aware per-output-row int8 for the large matmul weights of h13 package A programs (CPU only, torch).
For each *_w const of at least 2^20 elements of a program: W [N, K] fp16 (the shipped weights), X [T, K] fp16 (calibration activations at that matmul input).
Per-row symmetric int8 with an fp16 scale (the format of the package: constexpr_affine_dequantize, axis 0). The scale is fixed first (plain max/127 or the MSE clip), then the
integers are chosen column by column with the GPTQ error feedback so that the OUTPUT error ||X (W - Wq)^T|| is small, not the weight error:
  H = X^T X / T + damp * mean(diag) * I ; U = chol(inv(H)) upper ; for each column j: q_j = clamp(round(w_j / s)), e = (w_j - q_j s) / U[j,j], w[:, j+1:] -= e (x) U[j, j+1:]
Held-out check: --holdout F keeps the last F of the rows of X out of H and reports the output error on them (the numbers go in report.json).
usage: gptq_int8.py PROG OUTDIR [--scale max|mse] [--damp 0.01] [--holdout 0.2] [--only const,const]
Writes OUTDIR/p<PROG>-<const>.npz (q int8 [N,K], scale fp32 [N] holding the fp16 value) and OUTDIR/p<PROG>-report.json."""
import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

import os

# Paths come from the environment; there are no defaults that point at one machine.
sys.path.insert(0, os.environ["ANEFORGE_DIR"])   # a checkout of ANEForge (the weight blob writer)
MIL_ROOT = Path(os.environ["GPTQ_MIL_ROOT"])      # holds mil192/ and mil50/ of the compiled MIL programs
EXPORT = Path(os.environ["GPTQ_EXPORT"])          # holds prog<NN>/weights.bin of the package
ACT = Path(os.environ["GPTQ_ACT"])                # calibration activations: p<prog>-<const>.npy, [T, K] fp16
CONST = re.compile(r'^(\s*)tensor<fp16, \[([\d, ]+)\]> (\w+) = const\(\)\[name = string\("(\w+)"\), val = tensor<fp16, \[[\d, ]+\]>\(BLOBFILE\(path = string\("@model_path/weights.bin"\), offset = uint64\((\d+)\)\)\)\];$')
ATTN = {6, 12, 18, 25, 31, 37}
MIL_DIR = {"attn": MIL_ROOT / "mil192", "mid": MIL_ROOT / "mil50"}
CLIPS = [1.0, 0.98, 0.96, 0.94, 0.92, 0.9, 0.87, 0.84, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]


def blob_at(w, off, nbytes):
  magic, code, size = struct.unpack_from("<IIQ", w, off)
  assert magic == 0xDEADBEEF and size == nbytes, (off, hex(magic), size, nbytes)
  data_off = struct.unpack_from("<Q", w, off + 16)[0]
  return w[data_off:data_off + nbytes]


def mil_path(n):
  if n == 37:
    return MIL_DIR["attn"] / "p37/model.mil"
  return (MIL_DIR["attn"] / f"p{n:02d}" if n in ATTN else MIL_DIR["mid"] / "p14") / "model.mil"


def weights_of(n):
  text = mil_path(n).read_text()
  w = (EXPORT / f"prog{n:02d}/weights.bin").read_bytes()
  out = {}
  for line in text.split("\n"):
    m = CONST.match(line)
    if not m:
      continue
    _, dims, name, _, off = m.groups()
    shape = [int(x) for x in dims.split(",")]
    if name.endswith("_w") and len(shape) == 2 and shape[0] * shape[1] >= 1 << 20:
      out[name] = np.frombuffer(blob_at(w, int(off), shape[0] * shape[1] * 2), dtype=np.float16).reshape(shape)
  return out


def row_scales(W, mode):
  """fp16 per-row scales as the package stores them. W float32 [N, K] torch."""
  mx = W.abs().amax(dim=1).clamp_min(1e-8)
  if mode == "max":
    return (mx / 127.0).half().float()
  best_err = torch.full_like(mx, float("inf"))
  best = mx / 127.0
  for c in CLIPS:
    sc = (mx * c / 127.0)[:, None]
    q = torch.clamp(torch.round(W / sc), -127, 127)
    err = ((q * sc - W) ** 2).sum(dim=1)
    better = err < best_err
    best_err = torch.where(better, err, best_err)
    best = torch.where(better, sc[:, 0], best)
  return best.half().float()


def rtn(W, sc):
  return torch.clamp(torch.round(W / sc[:, None]), -127, 127)


def gptq(W, X, sc, damp, block=128):
  """W [N,K] float32, X [T,K] float32, sc [N]. Returns integer matrix [N,K] float32 (values -127..127)."""
  N, K = W.shape
  H = (X.T.double() @ X.double()) / X.shape[0]
  dead = torch.diag(H) == 0
  H[dead, dead] = 1.0
  W = W.clone()
  W[:, dead] = 0
  H += damp * torch.diag(H).mean() * torch.eye(K, dtype=torch.float64)
  L = torch.linalg.cholesky(H)
  Hinv = torch.cholesky_inverse(L)
  U = torch.linalg.cholesky(Hinv, upper=True).float()
  Q = torch.zeros_like(W)
  for i1 in range(0, K, block):
    i2 = min(i1 + block, K)
    W1 = W[:, i1:i2].clone()
    E = torch.zeros_like(W1)
    U1 = U[i1:i2, i1:i2]
    for i in range(i2 - i1):
      w = W1[:, i]
      d = U1[i, i]
      q = torch.clamp(torch.round(w / sc), -127, 127)
      Q[:, i1 + i] = q
      err = (w - q * sc) / d
      W1[:, i:] -= err[:, None] * U1[i, i:][None, :]
      E[:, i] = err
    W[:, i2:] -= E @ U[i1:i2, i2:]
  return Q


def out_err(W, Wq, X):
  d = (X @ (W - Wq).T)
  return float(torch.linalg.norm(d) / torch.linalg.norm(X @ W.T))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("prog", type=int)
  ap.add_argument("out")
  ap.add_argument("--scale", default="mse", choices=["max", "mse"])
  ap.add_argument("--damp", type=float, default=0.01)
  ap.add_argument("--holdout", type=float, default=0.2)
  ap.add_argument("--only", default="")
  ap.add_argument("--no-write", action="store_true")
  a = ap.parse_args()
  torch.set_num_threads(8)
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  ws = weights_of(a.prog)
  rep = []
  for name, W16 in ws.items():
    if a.only and name not in a.only.split(","):
      continue
    X16 = np.load(ACT / f"p{a.prog}-{name}.npy")
    T = X16.shape[0]
    nh = int(T * a.holdout)
    W = torch.from_numpy(W16.astype(np.float32))
    X = torch.from_numpy(X16.astype(np.float32))
    assert X.shape[1] == W.shape[1], (name, X.shape, W.shape)
    Xc, Xh = (X[:T - nh], X[T - nh:]) if nh else (X, X)
    t0 = time.time()
    sc = row_scales(W, a.scale)
    Qr = rtn(W, sc)
    Qg = gptq(W, Xc, sc, a.damp)
    e = {
        "const": name, "shape": list(W.shape), "scale": a.scale, "seconds": round(time.time() - t0, 1),
        "weight_err_rtn": float(torch.linalg.norm(Qr * sc[:, None] - W) / torch.linalg.norm(W)),
        "weight_err_gptq": float(torch.linalg.norm(Qg * sc[:, None] - W) / torch.linalg.norm(W)),
        "out_err_rtn_calib": out_err(W, Qr * sc[:, None], Xc), "out_err_gptq_calib": out_err(W, Qg * sc[:, None], Xc),
        "out_err_rtn_heldout": out_err(W, Qr * sc[:, None], Xh), "out_err_gptq_heldout": out_err(W, Qg * sc[:, None], Xh),
    }
    print(json.dumps(e), flush=True)
    rep.append(e)
    if not a.no_write:
      np.savez(out / f"p{a.prog}-{name}.npz", q=Qg.numpy().astype(np.int8), scale=sc.numpy().astype(np.float32))
  (out / f"p{a.prog}-report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
  main()
