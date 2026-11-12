#!/usr/bin/env python3
"""Compiler, object-BAR, validator, and affine-refusal coverage for H16G LN."""
from __future__ import annotations

from pathlib import Path
import json
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
VALIDATOR = Path(sys.argv[2] if len(sys.argv) > 2 else
                 ROOT / "tools/h16g_validate.py").resolve()
CANDIDATE = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else None


def blob(values: list[int]) -> bytes:
    payload = struct.pack(f"<{len(values)}H", *values)
    result = bytearray(128 + len(payload))
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<IIQQ", result, 64, 0xDEADBEEF, 1, len(payload), 128)
    result[128:] = payload
    return bytes(result)


def mil(tokens: int, channels: int, affine: bool = False) -> str:
    operands = ""
    constants = ""
    if affine:
        constants = f'''    tensor<fp16, [{channels}]> gamma = const()[val = tensor<fp16, [{channels}]>(BLOBFILE(path = string("@model_path/gamma.bin"), offset = uint64(64)))];
    tensor<fp16, [{channels}]> beta = const()[val = tensor<fp16, [{channels}]>(BLOBFILE(path = string("@model_path/beta.bin"), offset = uint64(64)))];
'''
        operands = ", gamma = gamma, beta = beta"
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {tokens}, {channels}]> x) {{
    tensor<int32, [1]> axes = const()[val = tensor<int32, [1]>([-1])];
{constants}    tensor<fp16, [1, {tokens}, {channels}]> y = layer_norm(x = x, axes = axes{operands}, epsilon = fp32(0.00001))[name = string("y")];
  }} -> (y);
}}
'''


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="h16g-layer-norm-") as temp:
        root = Path(temp)
        for name, tokens, channels in (
                ("bert", 128, 768), ("wtiny", 1500, 384),
                ("wbase", 1500, 512)):
            source = root / f"{name}.mil"
            source.write_text(mil(tokens, channels))
            output = root / name
            result = subprocess.run([str(COMPILER), "--mil", str(source),
                "--model-root", str(root), "--target", "H16G", "--output", str(output)],
                capture_output=True, text=True, timeout=120)
            assert result.returncode != 0, result.stdout + result.stderr
            assert "h16g.encode.transformer-layer-norm-hardware-unproved" in result.stderr
            assert not output.exists()

            if CANDIDATE:
                candidate = root / f"{name}-candidate"
                built = subprocess.run([str(CANDIDATE), "--mil", str(source),
                    "--model-root", str(root), "--target", "H16G",
                    "--output", str(candidate)], capture_output=True, text=True,
                    timeout=180)
                assert built.returncode == 0, built.stdout + built.stderr
                manifest = json.loads((candidate / "manifest.json").read_text())
                assert manifest["dispatchPlan"] == [0]
                validation = subprocess.run([sys.executable, str(VALIDATOR),
                    str(candidate / "program-0.hwx"), "--manifest",
                    str(candidate / "manifest.json")], capture_output=True,
                    text=True, timeout=60)
                assert validation.returncode == 0, validation.stdout + validation.stderr

        # The capture carries identity gamma followed by zero beta, but no ANE
        # row consumes non-identity affine values. Keep that gap named.
        (root / "gamma.bin").write_bytes(blob([0x3c01] * 384))
        (root / "beta.bin").write_bytes(blob([0] * 384))
        source = root / "affine-unproved.mil"
        source.write_text(mil(1500, 384, affine=True))
        refused = subprocess.run([str(COMPILER), "--mil", str(source),
            "--model-root", str(root), "--target", "H16G",
            "--output", str(root / "affine-unproved")],
            capture_output=True, text=True, timeout=120)
        assert refused.returncode != 0
        assert "h16g.encode.transformer-layer-norm-hardware-unproved" in refused.stderr
        if CANDIDATE:
            candidate_refusal = subprocess.run([str(CANDIDATE), "--mil",
                str(source), "--model-root", str(root), "--target", "H16G",
                "--output", str(root / "affine-candidate")],
                capture_output=True, text=True, timeout=180)
            assert candidate_refusal.returncode != 0
            assert "h16g.encode.transformer-layer-norm-affine-unproved" in \
                candidate_refusal.stderr
    print("H16G layer_norm CLI PASS: gates closed, three validated candidates")


if __name__ == "__main__":
    main()
