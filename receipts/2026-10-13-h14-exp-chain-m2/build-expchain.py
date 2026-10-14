#!/usr/bin/env python3
"""Build the programs and inputs of the exp chain cell. Host only.

Usage: build-expchain.py COMPILER OUTDIR
Writes OUTDIR/prog/{exp,mul,add}_c<C>.anec for C in 64 and 512, each compiled by mil-hwxc --target H14 from the MIL of an oracle record
(research/oracles/h14/unary_exp_c<C>, binary_mul_1x<C>x1x1, binary_add_1x<C>x1x1). All three are programs that already exist in the
compiler's tests; none is new. It also writes OUTDIR/in/x_c<C>.f16 (the saved input of the failed run, copied byte for byte) and
OUTDIR/in/k_c<C>.f16 (the constant K in every lane, zero padding). K = fp16(log2(e) / fp16(log2(e)) - 1), the relative error of the
scale word 0x3dc5. A rerun gives the same bytes. SHA256SUMS lists every file."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "receipts/2026-10-13-h14-unary-m2/inputs"
LANE_STRIDE = 32  # one fp16 per 64-byte channel
LOG2E = 1.4426950408889634
K = float(np.float16(LOG2E / float(np.float16(LOG2E)) - 1.0))
SOURCES = {"exp": "unary_exp_c{c}", "mul": "binary_mul_1x{c}x1x1", "add": "binary_add_1x{c}x1x1"}


def compile_case(compiler, case):
    record = json.loads((ROOT / "research/oracles/h14" / f"{case}.json").read_text())
    assert record.get("error") is None, f"{case}: oracle record has an error"
    with tempfile.TemporaryDirectory() as tmp:
        mil = Path(tmp) / "op.mil"
        mil.write_text(record["mil"])
        (Path(tmp) / "model").mkdir()
        out = Path(tmp) / "package"
        subprocess.run([compiler, "--mil", str(mil), "--model-root", str(Path(tmp) / "model"), "--target", "H14", "--output", str(out)],
                       check=True, capture_output=True, text=True, timeout=120)
        programs = sorted(out.glob("program-*.anec"))
        assert len(programs) == 1, f"{case}: expected one program, got {len(programs)}"
        return programs[0].read_bytes()


def k_surface(channels):
    tile_halves = -(-channels * 64 // 16384) * 16384 // 2
    surface = np.zeros(tile_halves, dtype=np.float16)
    surface[: channels * LANE_STRIDE : LANE_STRIDE] = np.float16(K)
    return surface.tobytes()


def main(compiler, out):
    out = Path(out)
    (out / "prog").mkdir(parents=True, exist_ok=True)
    (out / "in").mkdir(exist_ok=True)
    sums = []
    for c in (64, 512):
        for name, pattern in SOURCES.items():
            path = out / "prog" / f"{name}_c{c}.anec"
            path.write_bytes(compile_case(compiler, pattern.format(c=c)))
            sums.append(path)
        x = (SAVED / f"unary_exp_c{c}" / "x.f16").read_bytes()
        (out / "in" / f"x_c{c}.f16").write_bytes(x)
        (out / "in" / f"k_c{c}.f16").write_bytes(k_surface(c))
        assert len(x) == len(k_surface(c)), (c, len(x))
        sums += [out / "in" / f"x_c{c}.f16", out / "in" / f"k_c{c}.f16"]
    (out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n" for p in sorted(sums)))
    print(f"K = {K!r} (fp16 bits 0x{np.float16(K).view(np.uint16):04x}); wrote {len(sums)} files to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
