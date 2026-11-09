#!/usr/bin/env python3
"""Compiler/container coverage for native H16G transformer attention."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1]).resolve()
VALIDATOR = Path(sys.argv[2]).resolve()


def typ(shape: tuple[int, ...]) -> str:
    return f'tensor<fp16, [{", ".join(map(str, shape))}]>'


def program(arguments: str, body: list[str], result: str) -> str:
    return "\n".join(["program(1.3)", "[buildInfo = dict<string, string>({})]", "{",
        f"  func main<ios18>({arguments}) {{", *(f"    {line}" for line in body),
        f"  }} -> ({result});", "}", ""])


def layout(operation: str, sequence: int, heads: int, dim: int) -> str:
    flat, split = (1, sequence, heads * dim), (1, heads, sequence, dim)
    if operation in ("split", "splitT"):
        out = split if operation == "split" else (1, heads, dim, sequence)
        perm = (0, 2, 1, 3) if operation == "split" else (0, 2, 3, 1)
        body = [
            f'tensor<int32, [4]> shape = const()[val = tensor<int32, [4]>([1, {sequence}, {heads}, {dim}])];',
            f'{typ((1, sequence, heads, dim))} r = reshape(shape = shape, x = x);',
            f'tensor<int32, [4]> perm = const()[val = tensor<int32, [4]>([{", ".join(map(str, perm))}])];',
            f'{typ(out)} y = transpose(perm = perm, x = r)[name = string("y")];',
        ]
        return program(f"{typ(flat)} x", body, "y")
    body = [
        'tensor<int32, [4]> perm = const()[val = tensor<int32, [4]>([0, 2, 1, 3])];',
        f'{typ((1, sequence, heads, dim))} t = transpose(perm = perm, x = x);',
        f'tensor<int32, [3]> shape = const()[val = tensor<int32, [3]>([1, {sequence}, {heads * dim}])];',
        f'{typ(flat)} y = reshape(shape = shape, x = t)[name = string("y")];',
    ]
    return program(f"{typ(split)} x", body, "y")


def matmul(operation: str, sequence: int, heads: int, dim: int) -> str:
    q, kt = (1, heads, sequence, dim), (1, heads, dim, sequence)
    scores = (1, heads, sequence, sequence)
    if operation == "pv":
        left, right, out, transpose = scores, q, q, "f"
    else:
        left, right, out = q, (q if operation == "qkT" else kt), scores
        transpose = "t" if operation == "qkT" else "f"
    body = ['bool f = const()[val = bool(false)];',
            'bool t = const()[val = bool(true)];',
            f'{typ(out)} z = matmul(transpose_x = f, transpose_y = {transpose}, x = x, y = y)[name = string("z")];']
    return program(f"{typ(left)} x, {typ(right)} y", body, "z")


def softmax(sequence: int, heads: int) -> str:
    shape = (1, heads, sequence, sequence)
    return program(f"{typ(shape)} x", [
        'int32 axis = const()[val = int32(-1)];',
        f'{typ(shape)} y = softmax(axis = axis, x = x)[name = string("y")];'], "y")


def fused(sequence: int, heads: int, dim: int) -> str:
    q, kt = (1, heads, sequence, dim), (1, heads, dim, sequence)
    scores = (1, heads, sequence, sequence)
    return program(f"{typ(q)} q, {typ(kt)} k, {typ(q)} v", [
        'bool f = const()[val = bool(false)];',
        f'{typ(scores)} scores = matmul(transpose_x = f, transpose_y = f, x = q, y = k);',
        f'fp16 scale = const()[val = fp16({1 / dim})];',
        f'{typ(scores)} scaled = mul(x = scores, y = scale);',
        'int32 axis = const()[val = int32(-1)];',
        f'{typ(scores)} probabilities = softmax(axis = axis, x = scaled);',
        f'{typ(q)} y = matmul(transpose_x = f, transpose_y = f, x = probabilities, y = v)[name = string("y")];'], "y")


def main() -> None:
    cases = {
        "split_wtiny": layout("split", 1500, 6, 64),
        "splitT_wbase": layout("splitT", 1500, 8, 64),
        "merge_wtiny": layout("merge", 1500, 6, 64),
        "qk_wtiny": matmul("qk", 1500, 6, 64),
        "qkT_wbase": matmul("qkT", 1500, 8, 64),
        "pv_wtiny": matmul("pv", 1500, 6, 64),
        "softmax_wbase": softmax(1500, 8),
        "split_held_prediction": layout("split", 64, 2, 80),
        "qk_held_prediction": matmul("qk", 64, 2, 80),
        "pv_held_prediction": matmul("pv", 64, 2, 80),
    }
    with tempfile.TemporaryDirectory(prefix="h16g-attention-") as temp:
        root = Path(temp)
        for name, source in cases.items():
            mil = root / f"{name}.mil"; mil.write_text(source)
            output = root / name
            result = subprocess.run([str(COMPILER), "--mil", str(mil),
                "--model-root", str(root), "--target", "H16G", "--output", str(output)],
                capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, result.stdout + result.stderr
            image = output / "program-0.hwx"
            validation = subprocess.run([sys.executable, str(VALIDATOR), str(image),
                "--manifest", str(output / "manifest.json")],
                capture_output=True, text=True, timeout=30)
            assert validation.returncode == 0, name + ": " + validation.stdout + validation.stderr
            manifest = json.loads((output / "manifest.json").read_text())
            assert manifest["target"] == "H16G"
        refused_cases = {
            "softmax_wtiny": (softmax(1500, 6),
                "h16g.encode.transformer-softmax-hardware-unproved"),
            "softmax_bert": (softmax(128, 12),
                "h16g.encode.transformer-softmax-hardware-unproved"),
            "attn_wtiny": (fused(1500, 6, 64),
                "h16g.encode.fused-attention-accuracy-unproved"),
            "attn_wbase": (fused(1500, 8, 64),
                "h16g.encode.fused-attention-accuracy-unproved"),
            "attn_held": (fused(64, 2, 80),
                "h16g.encode.fused-attention-accuracy-unproved"),
        }
        for name, (source, diagnostic) in refused_cases.items():
            mil = root / f"{name}.mil"
            mil.write_text(source)
            output = root / name
            refused = subprocess.run([str(COMPILER), "--mil", str(mil),
                "--model-root", str(root), "--target", "H16G",
                "--output", str(output)], capture_output=True, text=True,
                timeout=120)
            assert refused.returncode != 0, name
            assert diagnostic in refused.stderr, (name, refused.stderr)
            assert not output.exists(), name
    print(f"H16G transformer attention CLI PASS: {len(cases)} validated, "
          f"{len(refused_cases)} named refusals")


if __name__ == "__main__":
    main()
