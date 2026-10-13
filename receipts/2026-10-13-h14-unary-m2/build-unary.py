#!/usr/bin/env python3
"""Build the H14 unary programs for a device run. No device is touched.

Usage: build-unary.py OUTDIR [path-to-mil-hwxc]   (default compiler: build/mil-hwxc in this repository)
Every oracle record research/oracles/h14/unary_<op>_c<C>.json that holds Apple-compiled output (error null) is built, except relu, which
already ran on a device. The oracle's MIL is used unchanged. Per case directory unary_<op>_c<C>: op.mil, out/program-0.anec, and x.f16,
the input surface: lane j is one fp16 at half index j * 32 (one value per 64-byte channel), zero elsewhere, padded to the whole 16 KiB
tiles of the input channel. Inputs are seeded from the case name, so a rerun gives the same files."""
import json
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
out_root = Path(sys.argv[1])
compiler = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "build/mil-hwxc").resolve()

LANE_STRIDE = 32
TILE_BYTES = 16384
HEADER_TILES_OFFSET = 40  # one uint32 tile count per channel
INPUT_CHANNEL = 5
# input domain per operation, and values that are always included
DOMAIN = {"sqrt": (0.0, 64.0), "rsqrt": (0.0625, 64.0)}
DEFAULT_DOMAIN = (-8.0, 8.0)
ALWAYS = {"sqrt": [0.0, 1.0, 4.0, 16.0, 64.0], "rsqrt": [0.0625, 1.0, 4.0, 16.0, 64.0]}
DEFAULT_ALWAYS = [0.0, 1.0, -1.0, 0.5, -0.5, 8.0, -8.0]


def inputs(op, count, case):
    lo, hi = DOMAIN.get(op, DEFAULT_DOMAIN)
    rng = np.random.default_rng(zlib.crc32(case.encode()))
    values = rng.uniform(lo, hi, count).astype(np.float16)
    fixed = np.array(ALWAYS.get(op, DEFAULT_ALWAYS), dtype=np.float16)
    values[:min(len(fixed), count)] = fixed[:count]
    return values


bad = 0
oracles = sorted((ROOT / "research/oracles/h14").glob("unary_*_c*.json"))
for oracle in oracles:
    m = re.fullmatch(r"unary_([a-z_]+?)_c(\d+)", oracle.stem)
    if not m or m.group(1) == "relu":
        continue
    record = json.loads(oracle.read_text())
    if record.get("error") is not None:
        continue
    op, count, case = m.group(1), int(m.group(2)), oracle.stem
    d = out_root / case
    (d / "model").mkdir(parents=True, exist_ok=True)
    (d / "op.mil").write_text(record["mil"])
    r = subprocess.run([str(compiler), "--mil", str(d / "op.mil"), "--model-root", str(d / "model"), "--target", "H14",
                        "--output", str(d / "out")], capture_output=True, text=True, check=False)
    programs = sorted((d / "out").glob("program-*.anec"))
    if r.returncode or len(programs) != 1:
        print(f"{case}: failed (exit {r.returncode}, {len(programs)} programs)")
        bad += 1
        continue
    tiles = struct.unpack_from("<I", programs[0].read_bytes(), HEADER_TILES_OFFSET + 4 * INPUT_CHANNEL)[0]
    surface = np.zeros(tiles * TILE_BYTES // 2, dtype=np.float16)
    surface[::LANE_STRIDE][:count] = inputs(op, count, case)
    surface.tofile(d / "x.f16")
    print(f"{case:24} program bytes {programs[0].stat().st_size:<6} input surface {surface.nbytes} bytes")
sys.exit(1 if bad else 0)
