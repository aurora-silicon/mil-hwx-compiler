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


@dataclass(frozen=True)
class WideGeometry:
    name: str
    cin: int
    cout: int
    width: int
    kernel: int
    stride: int
    groups: int
    relu: bool
    bias: bool
    task_couts: tuple[int, ...]
    common_h16: int
    common_h17: int
    ocg: int
    source_skew: int
    sparse_sizes: tuple[int, ...] = ()


WIDE_CASES = (
    WideGeometry("conv_desc_pw_c128_n256_s28", 128, 256, 28, 1, 1, 1,
                 False, False, (256,), 0x00220000, 0x00220000, 4, 0x50),
    WideGeometry("conv_desc_pw_c256_n256_s28", 256, 256, 28, 1, 1, 1,
                 False, False, (256,), 0x00220000, 0x00220000, 4, 0x50),
    WideGeometry("conv_desc_pw_c256_n512_s14", 256, 512, 14, 1, 1, 1,
                 False, False, (512,), 0x00230000, 0x00230000, 4, 0x30),
    WideGeometry("conv_desc_pw_c512_n512_s14", 512, 512, 14, 1, 1, 1,
                 False, False, (512,), 0x00233300, 0x00233300, 4, 0x30),
    WideGeometry("conv_desc_pw_c512_n1024_s7", 512, 1024, 7, 1, 1, 1,
                 False, False, (1024,), 0x00233304, 0x00233304, 5, 0x30),
    WideGeometry("conv_desc_pw_c1024_n1024_s7", 1024, 1024, 7, 1, 1, 1,
                 False, False, (1024,), 0x00244404, 0x00244404, 5, 0x30),
    WideGeometry("conv_desc_pw_c512_n1024_s14", 512, 1024, 14, 1, 1, 1,
                 True, False, (1024,), 0x00233300, 0x00233300, 4, 0x30),
    WideGeometry("conv_desc_pw_c1024_n1024_s14", 1024, 1024, 14, 1, 1, 1,
                 True, False, (512, 512), 0x00233300, 0x00233300, 4, 0x30),
    WideGeometry("conv_desc_pw_c512_n1024_s28", 512, 1024, 28, 1, 1, 1,
                 True, False, (1024,), 0x00222200, 0x00222200, 4, 0x50),
    WideGeometry("conv_desc_pw_c1024_n1024_s28", 1024, 1024, 28, 1, 1, 1,
                 True, False, (512, 512), 0x00222200, 0x00222200, 4, 0x50),
    WideGeometry("conv_desc_dw_k3_c256_s28", 256, 256, 28, 3, 1, 256,
                 False, False, (256,), 0x00220000, 0x00220000, 0, 0x50),
    WideGeometry("conv_desc_dw_k3_c256_s28_st2", 256, 256, 28, 3, 2, 256,
                 False, False, (256,), 0x00233300, 0x00230000, 0, 0x20,
                 (0x1c0, 0x180, 0x180, 0x1c0, 0x180, 0x180, 0x180,
                  0x1c0, 0x1c0, 0x1c0, 0x1c0, 0x1c0, 0x1c0, 0x1c0,
                  0x1c0, 0x1c0)),
    WideGeometry("conv_desc_dw_k3_c512_s14", 512, 512, 14, 3, 1, 512,
                 False, False, (512,), 0x00230000, 0x00230000, 0, 0x30),
    WideGeometry("conv_desc_dw_k3_c512_s14_st2", 512, 512, 14, 3, 2, 512,
                 False, False, (512,), 0x00244404, 0x00244404, 0, 0x20,
                 (0x3c0,) * 16),
    WideGeometry("conv_desc_dw_k3_c1024_s7", 1024, 1024, 7, 3, 1, 1024,
                 False, False, (1024,), 0x00244404, 0x00244404, 0, 0x30),
    WideGeometry("conv_desc_pw_c512_n1024_s7_bias", 512, 1024, 7, 1, 1, 1,
                 False, True, (1024,), 0x00244404, 0x00244404, 5, 0x30),
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


def check_expected(words: dict[int, int], expected: dict[int, int],
                   label: str) -> int:
    for address, value in expected.items():
        assert words.get(address) == value, (
            f"{label} 0x{address:04x}: got {words.get(address)!r}, "
            f"expected 0x{value:08x}")
    return len(expected)


def kernel_extent(words: dict[int, int]) -> int:
    return max(words.get(0x5560 + 4 * index, 0) +
               words[0x55a0 + 4 * index] for index in range(16))


def check_wide_geometry(target: str, g: WideGeometry) -> int:
    record = oracle(target, g.name)
    tasks = record["task_descriptors"]
    assert record["task_decode_errors"] == 0
    assert record["program_descriptor"]["task_count"] == len(g.task_couts)
    assert len(tasks) == len(g.task_couts)
    output_width = (g.width + g.stride - 1) // g.stride
    input_row = 2 * align(g.width, 8)
    source_stride = g.cin * input_row + g.source_skew
    rows_in = min(g.width, output_width * g.stride + g.kernel - 1)
    result_base = rows_in * source_stride
    surface_input_row = align(2 * g.width, 64)
    surface_output_row = align(2 * output_width, 64)
    common = g.common_h17 if target == "h17" else g.common_h16
    expected_section_end = 0
    checks = 3
    for index, (task, task_cout) in enumerate(zip(tasks, g.task_couts)):
        words = registers(task)
        expected = {
            0x0004: g.width, 0x0008: g.width, 0x000c: g.cin,
            0x0014: output_width, 0x0018: output_width,
            0x001c: task_cout, 0x0034: output_width, 0x003c: common,
            0x0040: g.ocg,
            0x004c: 0x00200000 if target == "h17" else 0x00100000,
            0x4114: input_row, 0x4118: source_stride,
            0x414c: result_base,
            0x4904: (0x00103c00 | (0x00010000 if g.relu else 0) |
                     (0x10 if g.bias else 0)),
            0x4910: 0x00003c00,
            0x5100: 0x01000000 | (index << 16) |
                    (0x21 if target == "h17" else 0x31),
            0x5108: (len(tasks) - 1 - index) * task_cout *
                    surface_output_row * output_width,
            0x5110: surface_output_row,
            0x5114: surface_output_row * output_width,
            0x5510: expected_section_end,
        }
        if index == 0:
            expected.update({
                0x4104: 0x00500171 if len(tasks) > 1 else 0x00500172,
                0x411c: g.cin * input_row, 0x4120: g.cin * input_row,
                0x4148: 0x0050017a,
                0x4d00: 0x21, 0x4d18: surface_input_row,
                0x4d1c: surface_input_row * g.width,
            })
        else:
            assert 0x4104 not in words and 0x4d00 not in words
            expected[0x4148] = 0x14a
            checks += 2
        if g.kernel > 1:
            expected[0x0028] = expected_conv_cfg(Geometry(
                g.name, g.cin, g.cout, g.width, output_width, g.kernel,
                g.stride, output_width, input_row // 2, g.source_skew,
                0, 0, common, common))
            expected[0x0030] = 0x00014000
        else:
            assert 0x0028 not in words and 0x0030 not in words
            checks += 2
        sparse = g.stride == 2
        if target == "h17":
            expected[0x4900] = 0x10000182 if sparse else 0x10000082
        elif sparse:
            expected[0x4900] = 0x00000182
        else:
            assert 0x4900 not in words
            checks += 1
        expected[0x5500] = ((index + 1) << 16) | (
            0x2e0 if sparse and len(tasks) == 1 else
            0x2c0 if index == len(tasks) - 1 else
            0x240)
        if g.sparse_sizes:
            sizes = g.sparse_sizes
        elif g.groups == g.cin == g.cout:
            sizes = (align(2 * (task_cout // 16) * g.kernel * g.kernel, 64),) * 16
        else:
            bias_per_chunk = 2 * (task_cout // 16) if g.bias else 0
            sizes = (align(2 * g.cin * g.kernel * g.kernel *
                           (task_cout // 16), 64) + bias_per_chunk,) * 16
        offset = 0
        for lane, size in enumerate(sizes):
            expected[0x5520 + 4 * lane] = (index << 16) | 0x21
            if offset:
                expected[0x5560 + 4 * lane] = offset
            expected[0x55a0 + 4 * lane] = size
            offset += size
        checks += check_expected(words, expected, f"{target} {g.name} task {index}")
        expected_section_end += offset

        header = [int(value, 16) for value in task["header_words"]]
        assert header[0] >> 16 == task["size_bytes"] // 4
        assert header[0] & 0xffff == index
        if len(tasks) == 1:
            assert header[2:] == [0x04000068, 0, 0x00fff868, 0, 0, 0,
                                  0x00050009]
        elif index == 0:
            assert header[2:] == [0x00000068, 0, 0x00fff868, 0, 0,
                                  0xff if output_width == 14 else 0x1fffffff,
                                  0x00050001]
        else:
            assert header[2:] == [0x04000000, 0, 0x00fff860, 0, 0, 0,
                                  0x00050009]
        checks += 9
    assert record["kernel_section"]["size"] == expected_section_end
    return checks + 1


def check_kernel_pair(name: str) -> int:
    h16 = oracle("h16", name)
    h17 = oracle("h17", name)
    assert h16["kernel_section"]["size"] == h17["kernel_section"]["size"]
    assert h16["kernel_section"]["sha256"] == h17["kernel_section"]["sha256"]
    return 2


def check_l2_chain_r2(target: str, name: str) -> int:
    record = oracle(target, name)
    tasks = record["task_descriptors"]
    assert len(tasks) == 6
    words = [registers(task) for task in tasks]
    headers = [[int(value, 16) for value in task["header_words"]]
               for task in tasks]
    assert [header[0] & 0xffff for header in headers] == list(range(6))
    assert 0x4d00 in words[0] and all(0x4d00 not in word for word in words[1:])
    assert all(0x5100 not in word for word in words[:-1]) and 0x5100 in words[-1]
    assert words[0][0x4148] == 0x00500170
    assert words[1][0x4104] == 0x120
    assert [word[0x4104] for word in words[2:]] == [0x104] * 4
    assert [word[0x4148] for word in words[1:-1]] == [0x140] * 4
    assert words[-1][0x4148] == 0x14a
    stage_tables = {
        "conv_desc_chain6_mob_c128_s28": (
            (128, 256, 28, 28, 1, 1, 0x00222200, 0x00222200, 4, False),
            (256, 256, 28, 28, 3, 1, 0x00222200, 0x00222200, 0, False),
            (256, 256, 28, 28, 1, 1, 0x00222200, 0x00222200, 4, False),
            (256, 256, 28, 14, 3, 2, 0x00233300, 0x00233300, 0, False),
            (256, 512, 14, 14, 1, 1, 0x00233300, 0x00233300, 4, False),
            (512, 512, 14, 14, 3, 1, 0x00233300, 0x00230000, 0, False),
        ),
        "conv_desc_chain_tail_c512_s14": (
            (512, 512, 14, 14, 3, 1, 0x00233300, 0x00230000, 0, False),
            (512, 512, 14, 14, 1, 1, 0x00233300, 0x00233300, 4, False),
            (512, 512, 14, 7, 3, 2, 0x00244404, 0x00244404, 0, False),
            (512, 1024, 7, 7, 1, 1, 0x00233304, 0x00233304, 5, False),
            (1024, 1024, 7, 7, 3, 1, 0x00244404, 0x00244404, 0, False),
            (1024, 1024, 7, 7, 1, 1, 0x00244404, 0x00244404, 5, True),
        ),
    }
    for index, (word, stage) in enumerate(zip(words, stage_tables[name])):
        cin, cout, width, output_width, kernel, stride = stage[:6]
        cfg_h16, cfg_h17, ocg, bias = stage[6:]
        assert word[0x0004] == word[0x0008] == width
        assert word[0x000c] == cin
        assert word[0x0014] == word[0x0018] == output_width
        assert word[0x001c] == cout
        assert word[0x0034] == output_width
        assert word[0x003c] == (cfg_h17 if target == "h17" else cfg_h16)
        assert word[0x0040] == ocg
        assert word[0x004c] == (0x00200000 if target == "h17" else 0x00100000)
        assert word[0x4904] == 0x00113c00 | (0x10 if bias else 0)
        if kernel == 1:
            assert 0x0028 not in word
        else:
            padding = 1 if stride == 1 else 0
            assert word[0x0028] == (kernel | kernel << 6 | stride << 13 |
                                    stride << 15 | padding << 17 |
                                    padding << 22 | 1 << 28 | 1 << 30)
    for producer, consumer in zip(words, words[1:]):
        assert producer.get(0x414c, 0) == consumer.get(0x4110, 0)
        assert producer[0x4150] == consumer[0x4114]
        assert producer[0x4154] == consumer[0x4118]
    running_kernel = 0
    for index, word in enumerate(words):
        assert word.get(0x5510, 0) == running_kernel
        assert word[0x5520] == (index << 16) | 0x21
        running_kernel += kernel_extent(word)
    assert record["kernel_section"]["size"] == running_kernel
    for index, header in enumerate(headers):
        assert header[0] >> 16 == tasks[index]["size_bytes"] // 4
        assert header[8] == (0x00050009 if index == 5 else 0x00050001)
        if index == 0:
            assert header[2] == 0x68 and header[4] == 0x00fff868
        elif index == 5:
            assert header[2] == 0x04000000 and header[4] == 0x00fff860
        else:
            assert header[2] == 0
            assert header[4] == (0x00fff800 if target == "h17" else 0x00fff820)
    return 116


def check_dram_chain(target: str) -> int:
    name = "conv_desc_chain6_dram_pw1024_s28"
    record = oracle(target, name)
    tasks = record["task_descriptors"]
    assert len(tasks) == 12
    words = [registers(task) for task in tasks]
    headers = [[int(value, 16) for value in task["header_words"]]
               for task in tasks]
    assert record["program_descriptor"]["task_count"] == 12
    assert record["kernel_section"]["size"] == 0x200000
    assert [header[0] & 0xffff for header in headers] == list(range(12))
    for index, (task, word, header) in enumerate(zip(tasks, words, headers)):
        assert word[0x0004] == word[0x0008] == 28
        assert word[0x000c] == 1024
        assert word[0x0014] == word[0x0018] == 28
        assert word[0x001c] == 512
        assert word[0x0034] == 28
        assert word[0x003c] == 0x00222200
        assert word[0x0040] == 4
        assert word[0x004c] == (0x00200000 if target == "h17" else 0x00100000)
        assert word[0x4114] == 0x40 and word[0x4118] == 0x10050
        assert word[0x414c] == 0x1c08c0
        assert word[0x4904] == 0x00113c00
        assert word[0x5510] == (0x100000 if index & 1 else 0)
        assert kernel_extent(word) == 0x100000
        assert all(word[0x55a0 + 4 * lane] == 0x10000
                   for lane in range(16))
        assert all(word.get(0x5560 + 4 * lane, 0) == lane * 0x10000
                   for lane in range(16))
        assert header[0] >> 16 == task["size_bytes"] // 4
        if index == 0:
            assert word[0x4104] == 0x00500171
            assert word[0x4d00] == 0x21
        elif index % 2 == 0:
            assert word[0x4104] == 0x841
            assert word[0x4d00] == 0x20000041 | (index << 16)
        else:
            assert 0x4104 not in word and 0x4d00 not in word
        final_layer = index >= 10
        expected_low = (0x21 if target == "h17" else 0x31)
        if not final_layer:
            expected_low |= 0xc0
        assert word[0x5100] == 0x01000000 | (index << 16) | expected_low
        assert word[0x5138] == (0x01302031 if final_layer else 0x08302031)
        assert word[0x5110] == (0x40 if final_layer else 0x1c0)
        assert word[0x5114] == (0x700 if final_layer else 0x3100)
        assert word[0x4148] == (
            0x0050087a if index == 0 else 0x14a if final_layer else 0x84a)
        if index < 10:
            assert header[8] == 0x00050001
        else:
            assert header[8] == (0x00050009 if index == 11 else 0x00050001)
    expected_5104 = {
        4: 0x09000000, 5: 0x11000000, 6: 0x09000000,
        7: 0x11000000, 8: 0x09000000, 9: 0x11000000,
    }
    for index, high in expected_5104.items():
        assert words[index][0x5104] == high | (0 if target == "h17" else 0x33)
    return 326


def check_reset_defaults(target: str) -> int:
    tasks = oracle(target, "conv_desc_chain2_dw64s2_pw128_s112_relu")[
        "task_descriptors"]
    first, second = map(registers, tasks)
    assert first[0x0040] == 0
    assert 0x0040 not in second       # Cout=128 reset/default is 3.
    assert first[0x4900] == (0x10000182 if target == "h17" else 0x182)
    if target == "h17":
        assert second[0x4900] == 0x10000082
    else:
        assert 0x4900 not in second  # Dense H16G reset/default is 0x82.
    return 4


def main() -> int:
    checks = 0
    for target in ("h16", "h17"):
        for geometry in CASES:
            checks += check_geometry(target, geometry)
            print(f"PASS {target.upper()} {geometry.name}")
        checks += check_relu(target)
        checks += check_chain(target)
        for geometry in WIDE_CASES:
            checks += check_wide_geometry(target, geometry)
            print(f"PASS {target.upper()} {geometry.name}")
        for name in ("conv_desc_chain6_mob_c128_s28",
                     "conv_desc_chain_tail_c512_s14"):
            checks += check_l2_chain_r2(target, name)
            print(f"PASS {target.upper()} {name} L2 alias/dependent edges")
        checks += check_dram_chain(target)
        print(f"PASS {target.upper()} conv_desc_chain6_dram_pw1024_s28 "
              "output split/DRAM dependency")
        checks += check_reset_defaults(target)
        print(f"PASS {target.upper()} per-task reset defaults")
    for geometry in WIDE_CASES:
        checks += check_kernel_pair(geometry.name)
    for name in ("conv_desc_chain6_mob_c128_s28",
                 "conv_desc_chain_tail_c512_s14",
                 "conv_desc_chain6_dram_pw1024_s28"):
        checks += check_kernel_pair(name)
    print(f"H16G/H17 convolution field parity: PASS ({checks} assertions/fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
