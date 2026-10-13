#!/usr/bin/env python3
"""K=N=512 H13 correctness gate; no timing claim or multi-program execution.

Usage: gate_h13_matvec.py CASE_DIR RUNNER NEW_OUTDIR
CASE_DIR contains out/{manifest.json,program-0.anec} and w.raw.f16 from
receipts/2026-10-12-h13-matvec-m1's seeded generator. Run via idle-guard.
--self-test checks packing, numeric rejection and padding without hardware.
"""
import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "receipts/2026-10-11-h13-elementwise-m1"))
from gate_h13 import to_half_away

K = N = 512
BYTES = 32768
STRIDE = 32  # fp16 words, one value per 64-byte channel


def pack(values):
    values = np.asarray(values, dtype="<f2")
    if values.shape != (K,) or not np.isfinite(values).all():
        raise ValueError("expected 512 finite fp16 input values")
    out = np.zeros(BYTES // 2, dtype="<f2")
    out[:K * STRIDE:STRIDE] = values
    return out.tobytes()


def compare(raw, x, w):
    if len(raw) != BYTES:
        raise ValueError(f"output length {len(raw)}, expected {BYTES}")
    surface = np.frombuffer(raw, dtype="<f2")
    got = surface[:N * STRIDE:STRIDE].astype(np.float64)
    terms = w.astype(np.float64) * x.astype(np.float64)
    exact, sumabs = terms.sum(axis=1), np.abs(terms).sum(axis=1)
    want = np.array([to_half_away(float(v)) for v in exact], dtype="<f2")
    # Same band as ane-run's matvec checker: 2 ulp OR 4 * 2^-11 * sum(abs(products)).
    _, exponent = np.frexp(np.abs(want.astype(np.float64)))
    ulp = np.maximum(np.ldexp(np.ones(N), exponent - 11), 2.0 ** -24)
    ulp[want == 0] = 2.0 ** -24
    in_band = np.isfinite(got) & ((np.abs(got - want) <= 2 * ulp) |
                                 (np.abs(got - exact) <= 4 * 2.0 ** -11 * sumabs))
    padding = surface.reshape(N, STRIDE)[:, 1:]
    padbad = int(np.count_nonzero(padding.view(np.uint16)))
    err = float(np.max(np.abs(got - exact)))
    return {"pass": bool(in_band.all() and padbad == 0),
            "lanes_in_band": int(in_band.sum()), "lanes": N,
            "padding_nonzero": padbad, "max_abs_error": err if np.isfinite(err) else None}


def validate(manifest, blob):
    if manifest.get("target") != "H13" or manifest.get("operation") != "matmul":
        raise ValueError("requires H13 matmul")
    if len(manifest.get("programs", [])) != 1 or manifest.get("dispatchPlan") != [0]:
        raise ValueError("only complete single-program packages supported")
    if len(blob) != manifest.get("bytes") or len(blob) < 1704:
        raise ValueError("program size disagrees with manifest or header is truncated")
    if struct.unpack_from("<II", blob, 32) != (1, 1):
        raise ValueError("requires exactly one runtime input and output")
    for kind, channel in (("inputs", 5), ("outputs", 4)):
        specs = manifest[kind]
        if len(specs) != 1:
            raise ValueError(f"expected one {kind} port")
        spec = specs[0]
        layout = [1, 512, 1, 1, 64, 64]
        if (spec["shape"] != [1, 512] or spec["dtype"] != "float16" or
                spec["index"] != channel or spec["allocationBytes"] != BYTES or
                spec["nchw"] != layout):
            raise ValueError(f"unsupported {kind} layout (K=256 and packages excluded)")
        tiles = struct.unpack_from("<I", blob, 40 + 4 * channel)[0]
        nchw = list(struct.unpack_from("<6Q", blob, 168 + 48 * channel))
        if tiles * 16384 != BYTES or nchw != layout:
            raise ValueError(f"header/manifest {kind} layout disagreement")


def self_test():
    x = np.ones(K, dtype="<f2")
    w = np.eye(N, K, dtype="<f2")
    raw = pack(x)
    assert compare(raw, x, w)["pass"]
    assert not compare(pack(-x), x, w)["pass"]
    assert not compare(pack(np.zeros(K)), x, w)["pass"]
    bad = bytearray(raw); struct.pack_into("<H", bad, 2, 1)
    assert not compare(bad, x, w)["pass"]
    bad = bytearray(raw); struct.pack_into("<H", bad, 0, 0x7e00)
    assert not compare(bad, x, w)["pass"]
    # a failing device output with NaN or infinity must still give a strictly valid JSON report
    for word in (0x7e00, 0x7c00, 0xfc00):
        bad = bytearray(raw); struct.pack_into("<H", bad, 0, word)
        report = compare(bad, x, w)
        assert not report["pass"]
        json.loads(json.dumps(report, allow_nan=False))
    assert compare(pack(np.zeros(K)), np.zeros(K), w)["pass"]
    try:
        compare(raw[:-2], x, w)
    except ValueError:
        pass
    else:
        raise AssertionError("truncated output accepted")
    print("PASS: packing, identity, wrong sign, zero output, padding, NaN, zero reference, truncation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if len(args.paths) != 3:
        parser.error("CASE_DIR RUNNER NEW_OUTDIR required")
    case, runner, out = map(Path, args.paths)
    program = case / "out/program-0.anec"
    blob = program.read_bytes()
    validate(json.loads((case / "out/manifest.json").read_text()), blob)
    rng = np.random.default_rng(K * 100003 + N)
    w = rng.uniform(-1, 1, (N, K)).astype("<f2")
    x = rng.uniform(-1, 1, K).astype("<f2")
    if w.tobytes() != (case / "w.raw.f16").read_bytes():
        raise ValueError("seeded weights do not match the package's raw weights")
    out.mkdir(parents=True, exist_ok=False)  # never accept stale output from an earlier run
    infile, outfile = out / "x.channel.f16", out / "y.channel.f16"
    infile.write_bytes(pack(x))
    cmd = ["flock", "-w", "60", "/var/tmp/ane-run.lock", "timeout", "60", str(runner),
           "--anec", str(program), "--in", f"0={infile}", "--out", f"0={outfile}", "--repeat", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=125, check=False)
    (out / "runner.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        print(f"runner failed rc={result.returncode}; see {out / 'runner.log'}"); return 1
    report = compare(outfile.read_bytes(), x, w)
    report["program_sha256"] = hashlib.sha256(blob).hexdigest()
    (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}", file=sys.stderr); sys.exit(1)
