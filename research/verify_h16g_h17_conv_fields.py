#!/usr/bin/env python3
"""Re-derive H16G/H17 convolution fields from the focused oracle corpus."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def align(value: int, amount: int) -> int:
    return (value + amount - 1) // amount * amount


def registers(task: dict) -> dict[int, int]:
    return {
        int(address, 16): int(value, 16)
        for block in task["blocks"].values()
        for address, value in block["words"].items()
    }


def oracle(target: str, name: str) -> dict:
    return json.loads((ROOT / "research/oracles" / target /
                       f"{name}.json").read_text())


@dataclass(frozen=True)
class Geometry:
    name: str
    cin: int
    cout: int
    width: int
    output_width: int
    kernel: int
    stride: int
    tile_height: int
    internal_width: int
    source_skew: int
    result_mode: int
    result_skew: int
    common_cfg_h16: int
    common_cfg_h17: int


CASES = (
    Geometry("conv_desc_cmp_pw_c64_n64_s64", 64, 64, 64, 64, 1, 1,
             64, 64, 0, 0x0050017a, 0, 0x00210000, 0x00210000),
    Geometry("conv_desc_pw_c32_n64_s112", 32, 64, 112, 112, 1, 1,
             112, 64, 0x50, 0x00500172, 0x80, 0x00231100, 0x00231100),
    Geometry("conv_desc_reg_k3_c64_n64_s112", 64, 64, 112, 112, 3, 1,
             112, 112, 0, 0x0050017a, 0, 0x00200000, 0x08200000),
)


def expected_conv_cfg(g: Geometry) -> int:
    if g.kernel == 1:
        return 0
    padding = 1 if g.kernel == 3 and g.stride == 1 else 0
    return (g.kernel | (g.kernel << 6) | (g.stride << 13) |
            (g.stride << 15) | (padding << 17) | (padding << 22) |
            (1 << 28) | (1 << 30))


def check_geometry(target: str, g: Geometry) -> int:
    record = oracle(target, g.name)
    task = record["task_descriptors"][0]
    words = registers(task)
    output_height = g.output_width
    input_row = 2 * g.internal_width
    source_row = g.cin * input_row + g.source_skew
    rows_in = min(g.width, g.tile_height * g.stride + g.kernel - 1)
    result_base = rows_in * source_row
    dma_input_row = align(2 * g.width, 64)
    dma_output_row = align(2 * g.output_width, 64)
    lanes = g.cout // 16
    chunk = align(2 * g.cin * g.kernel * g.kernel * lanes, 64)
    expected = {
        0x0004: g.width, 0x0008: g.width, 0x000c: g.cin,
        0x0014: g.output_width, 0x0018: output_height, 0x001c: g.cout,
        0x0034: g.tile_height,
        0x003c: g.common_cfg_h17 if target == "h17" else g.common_cfg_h16,
        0x0040: lanes.bit_length() - 1,
        0x004c: 0x00200000 if target == "h17" else 0x00100000,
        0x4104: 0x00500172, 0x4108: 0x00500130,
        0x4114: input_row, 0x4118: source_row,
        0x411c: g.cin * input_row, 0x4120: g.cin * input_row,
        0x4148: g.result_mode, 0x414c: result_base,
        0x4d18: dma_input_row, 0x4d1c: dma_input_row * g.width,
        0x5110: dma_output_row, 0x5114: dma_output_row * output_height,
        0x5500: 0x000102c0,
        0x4904: 0x00103c00, 0x4910: 0x00003c00,
    }
    cfg = expected_conv_cfg(g)
    if cfg:
        expected[0x0028] = cfg
    else:
        assert 0x0028 not in words
    if g.result_mode == 0x00500172:
        result_row = 2 * g.internal_width
        plain = g.cout * result_row
        expected.update({0x4150: result_row, 0x4154: plain + g.result_skew,
                         0x4158: plain, 0x415c: plain})
    for index in range(16):
        if index:
            expected[0x5560 + 4 * index] = index * chunk
        expected[0x55a0 + 4 * index] = chunk
        expected[0x5520 + 4 * index] = 0x21
    if target == "h17":
        expected[0x4900] = 0x10000082
        expected[0x5100] = (0x01000021 if g.result_mode == 0x0050017a
                            else 0x00000021)
    else:
        assert 0x4900 not in words
        expected[0x5100] = (0x01000031 if g.result_mode == 0x0050017a
                            else 0x00000031)
    for address, value in expected.items():
        assert words.get(address) == value, (
            f"{target} {g.name} 0x{address:04x}: "
            f"got {words.get(address)!r}, expected 0x{value:08x}")
    header = [int(value, 16) for value in task["header_words"]]
    assert len(header) == 9
    assert header[0] >> 16 == task["size_bytes"] // 4
    assert header[0] & 0xffff == 0
    assert header[2:] == [0x04000068, 0, 0x00fff868, 0, 0, 0, 0x00050009]
    return len(expected) + len(header)


def check_relu(target: str) -> int:
    plain = registers(oracle(
        target, "conv_desc_reg_k3_c3_n32_s224_st2")["task_descriptors"][0])
    relu = registers(oracle(
        target, "conv_desc_stem_k3_c3_n32_s224_st2_relu")
        ["task_descriptors"][0])
    assert set(plain) == set(relu)
    assert plain[0x4904] == 0x00103c00
    assert relu[0x4904] == 0x00113c00
    assert {address for address in plain if plain[address] != relu[address]} == {0x4904}
    return 3


def check_chain(target: str) -> int:
    tasks = oracle(target, "conv_desc_chain3_pw64_dw64s2_pw128_s112_relu")[
        "task_descriptors"]
    assert len(tasks) == 3
    words = [registers(task) for task in tasks]
    headers = [[int(value, 16) for value in task["header_words"]]
               for task in tasks]
    assert [header[0] & 0xffff for header in headers] == [0, 1, 2]
    assert [header[7] for header in headers] == [0x01ffffff, 0x7fffffff, 0]
    assert [header[8] for header in headers] == [0x00050001, 0x00050001,
                                                  0x00050009]
    assert 0x4d00 in words[0] and 0x4d00 not in words[1] and 0x4d00 not in words[2]
    assert 0x5100 not in words[0] and 0x5100 not in words[1] and 0x5100 in words[2]
    # Each result begins immediately after the task's resident input rows.
    assert words[0].get(0x414c, 0) == 0
    assert words[1][0x414c] == 112 * words[0][0x4154]
    assert words[2][0x414c] == words[2][0x4110] + 56 * words[2][0x4118]
    assert [word.get(0x5510, 0) for word in words] == [0, 0x1000, 0x1800]
    assert [word[0x5520] >> 16 for word in words] == [0, 1, 2]
    return 19


def main() -> int:
    checks = 0
    for target in ("h16", "h17"):
        for geometry in CASES:
            checks += check_geometry(target, geometry)
            print(f"PASS {target.upper()} {geometry.name}")
        checks += check_relu(target)
        checks += check_chain(target)
    print(f"H16G/H17 convolution field parity: PASS ({checks} assertions/fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
