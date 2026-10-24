#!/usr/bin/env python3
"""GPTQ per-row int8 for the 15 mid and attention programs of the package (CPU, torch), calibrated on captured matmul inputs (GPTQ_ACT).
Per tensor: baseline = plain per-row int8 (scale max/127, the S1 package), candidates = MSE-clip scale with GPTQ at damping 1, 3, 10 (Hessian from the first 80 percent of the rows);
the candidate with the lowest output error on the last 20 percent wins; if none beats the baseline the plain int8 is written (so the result is never worse on the held-out rows).
The final integers use ALL rows with the chosen damping. Writes OUT/p<prog>-<const>.npz (q int8 [N,K], scale fp32 [N] = the fp16 value stored) and OUT/p<prog>-report.json.
usage: gptq_s1.py PROG OUT"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import gptq_int8 as G

ACT = Path(os.environ["GPTQ_ACT"])
G.ACT = ACT
DAMPS = [float(x) for x in os.environ.get("GPTQ_DAMPS", "1,3,10").split(",")]


def main():
  prog, out = int(sys.argv[1]), Path(sys.argv[2])
  out.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 6)
  rep = []
  for name, W16 in G.weights_of(prog).items():
    f = ACT / f"p{prog}-{name}.npy"
    if not f.exists():
      if os.environ.get("GPTQ_PLAIN_FALLBACK") == "1":
        W0 = torch.from_numpy(W16.astype(np.float32))
        sc0 = G.row_scales(W0, "max")
        np.savez(out / f"p{prog}-{name}.npz", q=G.rtn(W0, sc0).numpy().astype(np.int8), scale=sc0.numpy().astype(np.float32))
        print(json.dumps({"prog": prog, "const": name, "shape": list(W0.shape), "choice": "rtn-max (no capture)", "damp": None}), flush=True)
        rep.append({"prog": prog, "const": name, "shape": list(W0.shape), "choice": "rtn-max (no capture)", "damp": None, "rows": 0,
                    "heldout_baseline": None, "heldout_chosen": None, "heldout_ratio": None, "all_rows_baseline": None, "all_rows_chosen": None, "seconds": 0})
        continue
      print(f"p{prog} {name}: no capture, skipped", flush=True)
      continue
    t0 = time.time()
    W = torch.from_numpy(W16.astype(np.float32))
    X = torch.from_numpy(np.load(f).astype(np.float32))
    T = X.shape[0]
    nh = int(T * 0.2)
    Xc, Xh = X[:T - nh], X[T - nh:]
    sc_max = G.row_scales(W, "max")
    base_h = G.out_err(W, G.rtn(W, sc_max) * sc_max[:, None], Xh)
    sc = G.row_scales(W, "mse")
    best = ("rtn-max", None, base_h)
    for d in DAMPS:
      Q = G.gptq(W, Xc, sc, d)
      e = G.out_err(W, Q * sc[:, None], Xh)
      if e < best[2]:
        best = ("gptq-mse", d, e)
    if best[0] == "rtn-max":
      q, scale = G.rtn(W, sc_max), sc_max
    else:
      q, scale = G.gptq(W, X, sc, best[1]), sc
    full_err = G.out_err(W, q * scale[:, None], X)
    base_full = G.out_err(W, G.rtn(W, sc_max) * sc_max[:, None], X)
    np.savez(out / f"p{prog}-{name}.npz", q=q.numpy().astype(np.int8), scale=scale.numpy().astype(np.float32))
    e = {"prog": prog, "const": name, "shape": list(W.shape), "choice": best[0], "damp": best[1], "rows": T,
         "heldout_baseline": base_h, "heldout_chosen": best[2], "heldout_ratio": best[2] / base_h,
         "all_rows_baseline": base_full, "all_rows_chosen": full_err, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(e), flush=True)
    rep.append(e)
  (out / f"p{prog}-report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
  main()
