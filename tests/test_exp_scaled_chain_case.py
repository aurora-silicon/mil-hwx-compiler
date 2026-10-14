#!/usr/bin/env python3
"""The MIL of the chain_exp_d4 cases carries the constants of the D4 cell: 16 K = fp16 0x1b18, 16.0 and 0.0625, in the five-operation order exp, mul, add, mul, mul.

research/mint_oracles.py builds the case. A Mac with the Apple compiler mints it. This test needs neither and uses only the standard library.
Usage: tests/test_exp_scaled_chain_case.py"""
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import mint_oracles  # noqa: E402


def fp16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def main():
    want = {"k16": fp16(0x1B18), "sixteen": 16.0, "sixteenth": 0.0625}
    for channels in (64, 512):
        item = mint_oracles.exp_scaled_chain(channels)
        assert item["name"] == f"chain_exp_d4_c{channels}" and item["family"] == "chain", item["name"]
        mil = item["mil"]
        got = {name: float.fromhex(value) for name, value in re.findall(r'fp16 (\w+) = const\(\)\[name = string\("\w+"\), val = fp16\((0x[0-9a-fp.+-]+)\)\]', mil)}
        assert got == want, (got, want)
        ops = re.findall(r"= (\w+)\(x = (\w+)(?:, y = (\w+))?\)", mil)
        assert ops == [("exp", "x", ""), ("mul", "x", "k16"), ("add", "c", "sixteen"), ("mul", "e", "g"), ("mul", "p", "sixteenth")], ops
        assert f"tensor<fp16, [1, {channels}, 1, 1]> x" in mil
    assert {"chain_exp_d4_c64", "chain_exp_d4_c512"} <= {item["name"] for item in mint_oracles.campaign()}
    print("chain_exp_d4 cases: constants and operation order as registered")


if __name__ == "__main__":
    sys.exit(main())
