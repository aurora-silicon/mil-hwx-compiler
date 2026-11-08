#!/usr/bin/env python3
"""Sanitized parity for native H16G transformer movement packets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from mint_oracles import decode_task, split_h14_tasks  # noqa: E402

DUMP = str(Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / "build/h16_transformer_attention_dump").resolve())

CASES = [
    ("split", 6, "6a5a6e9d40a56e6b547c4e38299d59338cb0f51f4bc9ba0c5c61cecc8042b23d"),
    ("split", 8, "cfac8c10e239b9ae6b01fe44b15971c6d73554651a63246c8829590003ff6d2b"),
    ("splitT", 6, "5e9507335ead1406514e253954d9e01a34df5c95c3e89324353c21cb60be9d1a"),
    ("splitT", 8, "5eefa7e91dcdc617c09de174d54fe6c14cba8fd2625e8bdeed5c4af820afd809"),
    ("merge", 6, "00dcb44a84ab3c290f335ee234b505f1f05e20e74b59538f04af691a5fd57b87"),
    ("merge", 8, "965e0baf2041a6a341503ba444c71011ce27e092cdb0371be24903ce3f451fde"),
]
MATMUL_CASES = [
    ("qk", 6, [208, 504, 532],
     "8a3b3976537fbb3a7989d1127685975d2bdfbc9f2c3375a7f75a8a997323e208"),
    ("qk", 8, [208, 504, 532],
     "c7d89e38cae587d3772f30df807e57247ba645533c9cde2b1ae630a806817be8"),
    ("qkT", 6, [208, 188, 456, 472],
     "4d8b92f48e4677224ab9d04ebbb3d6cb127d9eb14fc0319b5d1070134a3eb836"),
    ("qkT", 8, [208, 188, 456, 472],
     "c6566c67e5f725efbe75691e4a1ae4666f9028ec4ba863d9534f58d5a68973b7"),
    ("pv", 6, [224, 488, 488, 488, 488, 488, 516],
     "1568e576c6385d67924fadaad1a0f884beeb9558abfab175ac6aafe4ab33f73a"),
    ("pv", 8, [224, 488, 488, 488, 488, 488, 488, 488, 516],
     "0fb0d9c850f07f0501fd8efd9dc6052fcc314ab5a649840a8b8fb733032095d9"),
]
SOFTMAX_CASES = [
    (1500, 8, [192, 212, 276, 176, 260],
     "10f61e3b00cd95ec53ecd5556eab73639a302d75e356183869638c1e2c089e66"),
]
FUSED_CASES = [
    (6, [216, 504, 504, 192, 188, 188, 184, 212, 300, 180, 236,
         488, 488, 488, 488, 488, 516],
     "9f7788742237b0599b55ce2edfb78b8a45e281daad9681a5d86380b270e0e1a4"),
    (8, [216, 504, 504, 192, 188, 188, 184, 212, 300, 180, 236,
         488, 488, 488, 488, 488, 488, 488, 516],
     "b188ebf66aa204d05259b7938c2bbcec8b7e298c06ab2c938321dec5d428e5e5"),
]


def canonical(task: bytes) -> bytes:
    decoded = decode_task(task, "h16")
    value = {
        "header": decoded["header_words"][:1] + decoded["header_words"][2:],
        "blocks": decoded["blocks"],
        "records": decoded["records"],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def emit(operation: str, sequence: int, heads: int, head_dim: int) -> list[bytes]:
    stream = subprocess.check_output([
        DUMP, "layout", operation, str(sequence), str(heads), str(head_dim)])
    assert stream[:16] == b"\x01" + bytes(15)
    return split_h14_tasks(stream[16:])


def emit_matmul(operation: str, sequence: int, heads: int,
                head_dim: int) -> list[bytes]:
    stream = subprocess.check_output([
        DUMP, "matmul", operation, str(sequence), str(heads), str(head_dim)])
    assert stream[:16] == b"\x01" + bytes(15)
    return split_h14_tasks(stream[16:])


def main() -> None:
    for operation, heads, digest in CASES:
        tasks = emit(operation, 1500, heads, 64)
        assert len(tasks) == 1
        assert hashlib.sha256(canonical(tasks[0])).hexdigest() == digest
    for operation, heads, sizes, digest in MATMUL_CASES:
        tasks = emit_matmul(operation, 1500, heads, 64)
        assert [len(task) for task in tasks] == sizes
        encoded = json.dumps([
            json.loads(canonical(task)) for task in tasks], sort_keys=True,
            separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == digest
    for sequence, heads, sizes, digest in SOFTMAX_CASES:
        stream = subprocess.check_output([DUMP, "softmax", str(sequence), str(heads)])
        tasks = split_h14_tasks(stream[16:])
        assert [len(task) for task in tasks] == sizes
        encoded = json.dumps([json.loads(canonical(task)) for task in tasks],
                             sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == digest
    for sequence, heads in ((1500, 1), (1500, 6), (128, 1), (128, 12)):
        stream = subprocess.check_output([DUMP, "softmax", str(sequence), str(heads)])
        assert len(split_h14_tasks(stream[16:])) == 5
    for heads, sizes, digest in FUSED_CASES:
        stream = subprocess.check_output([DUMP, "fused", "1500", str(heads), "64"])
        tasks = split_h14_tasks(stream[16:])
        assert [len(task) for task in tasks] == sizes
        encoded = json.dumps([json.loads(canonical(task)) for task in tasks],
                             sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == digest
    for sequence, heads, head_dim in ((64, 2, 80), (256, 3, 128),
                                      (512, 5, 64)):
        for operation in ("qk", "qkT", "pv"):
            tasks = emit_matmul(operation, sequence, heads, head_dim)
            assert tasks
        stream = subprocess.check_output([
            DUMP, "fused", str(sequence), str(heads), str(head_dim)])
        assert len(split_h14_tasks(stream[16:])) == 11 + heads
    # Prediction rows are deliberately unrevealed: exercise only construction,
    # task grammar and exact EOF, never claim descriptor parity.
    for sequence, heads, head_dim in ((128, 12, 64), (64, 2, 80),
                                      (256, 3, 128), (512, 5, 64)):
        for operation in ("split", "splitT", "merge"):
            tasks = emit(operation, sequence, heads, head_dim)
            assert len(tasks) == 1 and tasks[0]
    print("H16G attention/softmax parity PASS: 15 measured rows, "
          "27 unrevealed predictions")


if __name__ == "__main__":
    main()
