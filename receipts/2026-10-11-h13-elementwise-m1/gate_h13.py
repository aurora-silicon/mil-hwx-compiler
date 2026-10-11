#!/usr/bin/env python3
"""H13 device gate for mil-hwx-compiler programs: run each <op>/program-0.anec through ane-run and check all 64 lanes.

Layout (soak_h13.py, proven on this chip): a 16384-byte surface per tensor, lane j is one fp16 at byte offset j*64, every
other byte zero. Inputs come from the package manifest (runtime inputs get seeded random lanes, a constant input uses the
surface hex the manifest carries). Reference semantics: fp16 results round to nearest with ties away from zero (the add rule
proven on the device); relu and the clips are exact compares.
Usage: gate_h13.py ROOT RUNNER OUTDIR [--repeat N]   (ROOT holds <op>/program-0.anec and <op>/manifest.json)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SURFACE, PLANE, LANES = 16384, 64, 64
OPS = ["add", "mul", "relu", "add-scalar", "mul-scalar", "real-div-scalar", "clip-low", "clip-high"]
HALF = 0.5


def to_half_away(d):
    """Round an exactly representable double to fp16, nearest, ties away from zero."""
    h = np.float16(d)
    if float(h) == d:
        return h
    lo, hi = (h, np.nextafter(h, np.float16(np.inf))) if float(h) < d else (np.nextafter(h, np.float16(-np.inf)), h)
    if (float(lo) + float(hi)) / 2 == d:
        return hi if d > 0 else lo
    return h


def surface(values):
    buf = np.zeros(SURFACE // 2, dtype=np.float16)
    buf[0::PLANE // 2][:LANES] = values
    return buf.tobytes()


def lanes_of(raw):
    arr = np.frombuffer(raw[:SURFACE], dtype=np.float16)
    return arr[0::PLANE // 2][:LANES], np.delete(arr, np.arange(0, arr.size, PLANE // 2)[:LANES])


def reference(op, a, b):
    f = lambda v: float(v)  # noqa: E731
    out = []
    for i in range(LANES):
        x = f(a[i])
        if op == "add":
            out.append(to_half_away(x + f(b[i])))
        elif op == "mul":
            out.append(to_half_away(x * f(b[i])))
        elif op == "add-scalar":
            out.append(to_half_away(x + HALF))
        elif op == "mul-scalar":
            out.append(to_half_away(x * HALF))
        elif op == "real-div-scalar":
            out.append(to_half_away(x / HALF))
        elif op == "relu":
            out.append(np.float16(x if x > 0 else 0.0))
        elif op == "clip-low":
            out.append(np.float16(x if x >= HALF else HALF))
        elif op == "clip-high":
            out.append(np.float16(x if x <= HALF else HALF))
    return np.array(out, dtype=np.float16)


def main():
    root, runner, outdir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 5
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261011)
    passed = failed = 0
    for op in OPS:
        prog = root / op / "program-0.anec"
        if not prog.exists():
            print(f"{op}: program missing")
            failed += 1
            continue
        manifest = json.loads((root / op / "manifest.json").read_text())
        runtime = [i for i in manifest["inputs"] if i.get("binding") != "constant"]
        consts = [i for i in manifest["inputs"] if i.get("binding") == "constant"]
        a = rng.uniform(-8, 8, LANES).astype(np.float16)
        b = rng.uniform(-8, 8, LANES).astype(np.float16)
        files, cmd = [], []
        vals = [a, b]
        n = 0
        for spec in manifest["inputs"]:
            path = outdir / f"{op}.in{n}.f16"
            if spec.get("binding") == "constant":
                logical = np.frombuffer(bytes.fromhex(manifest["constantInputs"][spec["name"]]), dtype=np.float16)
                assert logical.size == LANES, f"constant {spec['name']} has {logical.size} values"
                path.write_bytes(surface(logical))
            else:
                path.write_bytes(surface(vals[len(files)]))
                files.append(path)
            cmd += ["--in", f"{n}={path}"]
            n += 1
        yfile = outdir / f"{op}.y.f16"
        r = subprocess.run(["flock", "-w", "600", "/var/tmp/ane-run.lock", "timeout", "60", runner, "--anec", str(prog)] + cmd +
                           ["--out", f"0={yfile}", "--repeat", str(repeat)], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            print(f"{op}: ane-run rc={r.returncode}: {(r.stderr or r.stdout).strip().splitlines()[-1][:140] if (r.stderr or r.stdout).strip() else ''}")
            failed += 1
            continue
        got, pad = lanes_of(yfile.read_bytes())
        want = reference(op, a, b)
        mism = int((got.view(np.uint16) != want.view(np.uint16)).sum())
        padbad = int((pad.view(np.uint16) != 0).sum())
        ok = mism == 0 and padbad == 0
        print(f"{op:16} {manifest['encoder']:22} {int(manifest['bytes']):6} B  {LANES - mism}/{LANES} lanes bit-exact, padding nonzero {padbad}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            idx = [i for i in range(LANES) if got[i].view(np.uint16) != want[i].view(np.uint16)][:3]
            for i in idx:
                print(f"    lane {i}: a={float(a[i]):.4g} b={float(b[i]):.4g} got={float(got[i]):.6g} want={float(want[i]):.6g}")
        passed += ok
        failed += not ok
    print(f"SUMMARY pass={passed} fail={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
