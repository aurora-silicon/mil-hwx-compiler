#!/usr/bin/env python3
"""Build the programs and inputs of the one-ANEC scaled exp cell. Host only.

Usage: build-expchain.py COMPILER OUTDIR
Writes OUTDIR/prog/expone_c<C>.anec for C in 64 and 512: ONE ANEC program per size, compiled by mil-hwxc --target H14 from the MIL of the oracle record
research/oracles/h14/chain_exp_d4_c<C>.json (exp, mul by fp16 0x1b18, add 16, mul, mul by 0.0625). The compiler emits Apple's three-task program for it. It
also writes OUTDIR/in/x_c<C>.f16, the saved input of the failed exp run, copied byte for byte. A rerun gives the same bytes. SHA256SUMS lists every file."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / "receipts/2026-10-13-h14-unary-m2/inputs"


def compile_case(compiler, case):
    record = json.loads((ROOT / "research/oracles/h14" / f"{case}.json").read_text())
    assert record.get("error") is None, f"{case}: oracle record has an error"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "op.mil").write_text(record["mil"])
        (Path(tmp) / "model").mkdir()
        out = Path(tmp) / "package"
        subprocess.run([compiler, "--mil", str(Path(tmp) / "op.mil"), "--model-root", str(Path(tmp) / "model"), "--target", "H14", "--output", str(out)],
                       check=True, capture_output=True, text=True, timeout=120)
        manifest = json.loads((out / "manifest.json").read_text())
        programs = sorted(out.glob("program-*.anec"))
        assert len(programs) == 1 and len(manifest["programs"]) == 1, f"{case}: expected one program, got {len(programs)}"
        assert manifest["programs"][0]["encoder"] == "apple-parity-expchain" and manifest["programs"][0]["taskDescriptors"] == 3, manifest["programs"]
        return programs[0].read_bytes()


def main(compiler, out):
    out = Path(out)
    (out / "prog").mkdir(parents=True, exist_ok=True)
    (out / "in").mkdir(exist_ok=True)
    files = []
    for c in (64, 512):
        path = out / "prog" / f"expone_c{c}.anec"
        path.write_bytes(compile_case(compiler, f"chain_exp_d4_c{c}"))
        x = out / "in" / f"x_c{c}.f16"
        x.write_bytes((SAVED / f"unary_exp_c{c}" / "x.f16").read_bytes())
        files += [path, x]
    (out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n" for p in sorted(files)))
    print(f"wrote {len(files)} files to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
