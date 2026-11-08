#!/usr/bin/env python3
"""Whisper stride-2 stem and encoder transpose measured parity.

Only sanitized descriptor hashes and kernel hashes of public generated inputs
are retained here. ExeCycles is the sole excluded descriptor field.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from mint_oracles import decode_task, split_h14_tasks  # noqa: E402

DUMP = str(Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / "build/h16_whisper_stem_dump").resolve())

# C, GELU, task sizes, measured kernel bytes, sanitized measured descriptor
# SHA-256, and kernel SHA-256 for the deterministic public pattern below.
STEMS = [
    (384, False, [456], 959488,
     "e1184310f34744d63b64745b0e9d016493c1d75d216fcd39b352c70f71aeac06",
     "908624f6ff506ae5ee1c8437d9891588c93f36baa1dc435fd0789ecce803ebfb"),
    (384, True, [428, 456], 964608,
     "35e8db1332942038c7633158a7a5e666f4ab0fd83a733b6ac01961219dfbcb25",
     "f7c966a2fa026f4e92fc76c8936d8e9ee84e97c25eb6479e83907e504c05f455"),
    (512, False, [432, 428, 452], 1715200,
     "bcf7cdffa6e690f39a1a7d7b537fcdbfbacbcc331a02a483a8546549dde2f84c",
     "c9b7da34c88f9ae4d22d73332a2901d8af3621b69a2e80e0b7042a310c35369e"),
    (512, True, [432, 428, 456], 1713152,
     "0c30f18c3e0d7866e5808b805a4fd0b3778f003ac41dbac48b92b7f3b5ebc3c3",
     "8b73235bedad60a3ef52ef35a22482f6ab67cfebac7c13659ee9fc57babebdf4"),
]

TRANSPOSES = [
    (384, "6a5b2df915473acebdaa13122c07de365c17258c0dbf95236cbcb8ba5df1440c"),
    (512, "dbdf6f32faa9a37f5d01bfaf380cbde7f84e3826dbb2f90d5faa9e640c39a870"),
]


def pattern(elements: int, seed: int) -> bytes:
    data = bytearray(2 * elements)
    for index in range(elements):
        struct.pack_into("<H", data, 2 * index,
                         0x3000 | ((index * 37 + seed * 19) & 0x7ff))
    return bytes(data)


def canonical(task: bytes) -> dict:
    decoded = decode_task(task, "h16")
    # Captures name external source/result/kernel operands one slot before
    # the packaged object's 6/7/8 apertures. Normalize only those corrections.
    packaged = {"0x04d10": 0x23000000,
                "0x05108": 0x23800000,
                "0x05510": 0x24000000}
    for record in decoded["records"]:
        header = int(record["header"], 16)
        if (header & 0x3f800000) == packaged.get(record["address"]):
            record["header"] = f"0x{header - 0x00800000:08x}"
    return {"header": decoded["header_words"][:1] + decoded["header_words"][2:],
            "blocks": decoded["blocks"], "records": decoded["records"]}


def descriptor_digest(stream: bytes) -> tuple[list[int], str]:
    tasks = split_h14_tasks(stream[16:])
    encoded = json.dumps([canonical(task) for task in tasks], sort_keys=True,
                         separators=(",", ":")).encode()
    return [len(task) for task in tasks], hashlib.sha256(encoded).hexdigest()


def main() -> None:
    for channels, gelu, sizes, kernel_bytes, descriptor_sha, kernel_sha in STEMS:
        payload = pattern(channels * channels * 3, channels + int(gelu))
        payload += pattern(channels, channels + int(gelu) + 1)
        if gelu:
            payload += bytes(range(128))
        args = [DUMP, "task2", "3000", str(channels), str(int(gelu))]
        task = subprocess.run(args, input=payload, stdout=subprocess.PIPE,
                              timeout=120, check=True).stdout
        assert descriptor_digest(task) == (sizes, descriptor_sha), channels
        kernel = subprocess.run([DUMP, "kernel2", *args[2:]], input=payload,
                                stdout=subprocess.PIPE, timeout=120,
                                check=True).stdout
        assert len(kernel) == kernel_bytes, channels
        assert hashlib.sha256(kernel).hexdigest() == kernel_sha, channels

    # Both fp16 signs of zero clear their mask bits and consume no value bytes.
    channels = 384
    weights = bytearray(pattern(channels * channels * 3, channels))
    struct.pack_into("<H", weights, 0, 0x0000)
    struct.pack_into("<H", weights, 2 * 12345, 0x8000)
    payload = bytes(weights) + pattern(channels, channels + 1)
    kernel = subprocess.run([DUMP, "kernel2", "3000", "384", "0"],
                            input=payload, stdout=subprocess.PIPE,
                            timeout=120, check=True).stdout
    assert hashlib.sha256(kernel).hexdigest() == \
        "32f07b69cf1880ad72c16de48fb8a4e7a8a7c066a4d64706f5bc49c4db1f616b"

    for channels, descriptor_sha in TRANSPOSES:
        task = subprocess.run([DUMP, "transpose", str(channels)],
                              stdout=subprocess.PIPE, timeout=30,
                              check=True).stdout
        assert descriptor_digest(task) == ([240], descriptor_sha), channels
    print("H16G Whisper stem2/transpose PASS: 4 measured stem rows, "
          "2 measured transposes, signed-zero narrow-tail writer")


if __name__ == "__main__":
    main()
