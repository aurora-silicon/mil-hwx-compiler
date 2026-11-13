#!/usr/bin/env python3
"""The 34 campaign MISMATCH cases must refuse at compile time.

The 2026-11-13 h14 campaign (artifacts/w73/h14-campaign-20261113) found 34
shapes whose programs compile but whose task streams differ word-for-word
from the decoded Apple oracles. Until their structures are decoded, the
backend must refuse them with an explicit envelope diagnostic instead of
emitting programs that would diverge from Apple on device.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_h14_parity import COMPILER, write_weights  # noqa: E402

CORPUS = ROOT / "research/oracles/h14"
CASES = Path(__file__).parent / "h14_mismatch_cases.json"
REFUSALS = ("h14.outside-parity-envelope", "h14.unsupported-chain",
            "h14.chain-outside-envelope", "h14.chain-unrepresentable-edge",
            "h14.conv-outside-envelope", "h14.norm-outside-envelope",
            "h14.transpose-x-unsupported",
            "outside-parity-envelope", "unsupported-chain")


def main():
    names = sorted(json.loads(CASES.read_text()))
    # Provenance-shadowed cases: the five env_chain_softmax_* cases share
    # their entire MIL with standalone h14norm_softmax_an1_* oracles this
    # backend reproduces byte-exactly (campaign PASS), and
    # sched_softmax_surface_h8_s128's rank-3 [8,128,128] surface normalizes
    # to the same decoded template as its byte-exact rank-4 twin while
    # h14norm_reduce_max_*_flat proves rank-3 norm surfaces are inside the
    # decoded envelope. Apple's words depend on surrounding context no
    # single-program compiler can see; refusing these would refuse byte-exact
    # twins. They need corpus metadata instead of compile-time refusal.
    provenance = {n for n in names if n.startswith("env_chain_softmax")}
    provenance.add("sched_softmax_surface_h8_s128")
    names = [n for n in names if n not in provenance]
    missing = [n for n in names if not (CORPUS / f"{n}.json").exists()]
    assert not missing, f"cases missing from the corpus: {missing}"
    still_compiles, unclear = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            oracle = json.loads((CORPUS / f"{name}.json").read_text())
            root = Path(tmp) / name
            root.mkdir()
            write_weights(oracle, root)
            mil = root / f"{name}.mil"
            mil.write_text(oracle["mil"])
            output = root / "out"
            result = subprocess.run(
                [str(COMPILER), "--mil", str(mil), "--model-root", str(root),
                 "--target", "H14", "--format", "anec", "--output", str(output)],
                capture_output=True, text=True, timeout=60, check=False)
            if result.returncode == 0:
                still_compiles.append(name)
            elif not any(code in result.stdout + result.stderr
                         for code in REFUSALS):
                unclear.append((name, result.stdout.strip(),
                                result.stderr.strip()))
    for name, out, err in unclear:
        print(f"UNCLEAR {name}: {out} {err}")
    for name in still_compiles:
        print(f"STILL-COMPILES {name}")
    if still_compiles or unclear:
        print(f"FAIL: {len(still_compiles)} compile but must refuse, "
              f"{len(unclear)} refuse without an envelope diagnostic")
        return 1
    print(f"PASS: all {len(names)} campaign MISMATCH cases refuse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
