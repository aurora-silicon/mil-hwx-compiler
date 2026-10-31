#!/usr/bin/env python3
"""Re-derive round-2 H13 L2 rules and check decoded Apple field parity."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ORACLES = ROOT / "oracles" / "h13-l2-residency-r2"
L2_BYTES = 0x200000


class Checks:
    def __init__(self) -> None:
        self.count = 0
        self.mismatches: list[str] = []

    def equal(self, label: str, actual: int, expected: int) -> None:
        self.count += 1
        if actual != expected:
            self.mismatches.append(
                f"{label}: got {actual:#x}, expected {expected:#x}")


def align(value: int, amount: int) -> int:
    return (value + amount - 1) // amount * amount


def l2_skew(stride: int, width: int, channels: int) -> int:
    if stride == 2:
        return 0x30 if width >= 64 else 0x20
    if width >= 64:
        return 0
    if width > 16:
        return 0x50
    return 0x30 if channels >= 32 else 0x20


def integer(value: str) -> int:
    return int(value, 16)


def header(task: dict, index: int) -> int:
    return integer(task["header_words"][index])


def reg(task: dict, block: str, address: int) -> int:
    return integer(task["blocks"][block]["words"][f"0x{address:05x}"])


def check_links(checks: Checks, label: str, tasks: list[dict]) -> None:
    for index, task in enumerate(tasks[:-1]):
        next_size = (header(task, 1) >> 16) & 0x1ff
        checks.equal(f"{label} t{index} NextSize", next_size,
                     tasks[index + 1]["size_bytes"] // 4 - 1)
        checks.equal(f"{label} t{index} next pointer alignment",
                     header(task, 7) & 0xff, 0)
    checks.equal(f"{label} final next pointer", header(tasks[-1], 7), 0)


def check_tuple(checks: Checks, label: str, task: dict, source: bool,
                cfg: int, base: int, channel: int, row: int,
                depth: int, group: int) -> None:
    start = 0x4804 if source else 0x4830
    expected = (cfg, base, channel, row, depth, group)
    for offset, value in enumerate(expected):
        checks.equal(f"{label} {start + 4 * offset:#x}",
                     reg(task, "0x04800", start + 4 * offset), value)


def check_dma(checks: Checks, label: str, task: dict, source: bool,
              control: int, geometry: tuple[int, int, int, int, int],
              config: int) -> None:
    block = "0x13800" if source else "0x17800"
    start = 0x13800 if source else 0x17800
    checks.equal(f"{label} control", reg(task, block, start), control)
    base_offset = 8 if source else 4
    for index, value in enumerate(geometry):
        checks.equal(f"{label} geometry[{index}]",
                     reg(task, block, start + base_offset + 4 * index), value)
    config_offset = 0x38 if source else 0x18
    checks.equal(f"{label} config", reg(task, block, start + config_offset), config)


def verify_four_task_consumer(checks: Checks) -> dict:
    path = ORACLES / "mobile_dw_pw_dw_c1024_n2048_s16_st1_h13.json"
    record = json.loads(path.read_text())
    tasks = record["task_descriptors"]
    checks.equal("four-task stream task count", len(tasks), 6)
    check_links(checks, "four-task", tasks)

    spatial, channels_in, channels_out, slices = 16, 1024, 2048, 4
    row = align(max(16, spatial * 2), 16)
    in_depth = channels_in * row
    in_row = in_depth + l2_skew(1, spatial, channels_in)
    input_bytes = spatial * in_row
    out_depth = channels_out * row
    out_row = out_depth + l2_skew(1, spatial, channels_out)
    output_bytes = spatial * out_row
    slice_channels = channels_out // slices
    slice_depth = slice_channels * row
    output_base = 0
    input_base = output_bytes
    staged_input = (input_base + input_bytes) % L2_BYTES

    check_tuple(checks, "four-task producer source", tasks[0], True,
                0x500172, staged_input, row, in_row, in_depth, in_depth)
    check_tuple(checks, "four-task producer result", tasks[0], False,
                0x500170, input_base, row, in_row, in_depth, 0)
    checks.equal("four-task producer header[6]", header(tasks[0], 6), 0x1000d800)
    checks.equal("four-task producer header[8]", header(tasks[0], 8), 0x05823025)

    dram_row = align(spatial * 2, 64)
    dram_plane = dram_row * spatial
    dram_total = dram_plane * channels_in
    for part in range(slices):
        task = tasks[1 + part]
        first = part == 0
        check_tuple(checks, f"four-task consumer[{part}] source", task, True,
                    0x160 if first else 0x100, input_base,
                    row, in_row, in_depth, 0)
        result_base = output_base + (slices - 1 - part) * slice_depth
        check_tuple(checks, f"four-task consumer[{part}] result", task, False,
                    0x140, result_base, row, out_row, out_depth, 0)
        checks.equal(f"four-task consumer[{part}] Common 0x24",
                     reg(task, "0x00000", 0x24), 0x10001)
        checks.equal(f"four-task consumer[{part}] header[6]", header(task, 6),
                     0x2800 if first else 0)
        checks.equal(f"four-task consumer[{part}] header[8]", header(task, 8),
                     0x05800023 if first else 0x05800000)
        check_dma(checks, f"four-task consumer[{part}] source DMA", task, True,
                  0x48880 if first else 0x8880,
                  (0, dram_row, dram_plane, dram_total, dram_total)
                  if first else (0, 0, 0, 0, 0),
                  0x01002031 if first else 0x01000000)
        check_dma(checks, f"four-task consumer[{part}] result DMA", task, False,
                  0xc0, (0, 0, 0, 0, 0), 0x01000000)

    check_tuple(checks, "four-task aggregate consumer source", tasks[5], True,
                0x120, output_base, row, out_row, out_depth, 0)
    checks.equal("four-task aggregate source DMA",
                 reg(tasks[5], "0x13800", 0x13800), 0x8880)
    return {
        "case": record["case"],
        "input_edge_bytes": input_bytes,
        "output_edge_bytes": output_bytes,
        "input_base": input_base,
        "output_base": output_base,
        "slice_depth": slice_depth,
        "slice_result_bases": [
            output_base + (slices - 1 - part) * slice_depth
            for part in range(slices)
        ],
    }


def verify_six_layer_chain(checks: Checks, channels: int, spatial: int) -> dict:
    path = ORACLES / f"mobile_tail6_c{channels}_s{spatial}_h13.json"
    record = json.loads(path.read_text())
    tasks = record["task_descriptors"]
    label = f"six-layer c{channels}s{spatial}"
    checks.equal(f"{label} task count", len(tasks), 6)
    check_links(checks, label, tasks)

    row = align(max(16, spatial * 2), 16)
    depth = channels * row
    source_row = depth + l2_skew(1, spatial, channels)
    edge_bytes = spatial * source_row
    source_cfg = (0x500172, 0x160, 0x104, 0x144, 0x104, 0x144)
    result_cfg = (0x500170, 0x140, 0x140, 0x140, 0x140, 0x14a)
    common24 = (0x14001, 0x10001, 0x14001, 0x10001, 0x14001, 0x10001)
    header6 = (0x1000d800, 0x2800, 0xd000, 0x2800, 0xd000, 0x2000b800)
    header8 = (0x05823025, 0x05800023, 0x05823000,
               0x05800023, 0x05823000, 0x05024023)
    source_dma = (0x33881, 0x48880, 0x8880, 0x48880, 0x8880, 0x48880)
    dram_row = align(spatial * 2, 64)
    dram_plane = dram_row * spatial
    dram_total = dram_plane * channels
    populated = (0, dram_row, dram_plane, dram_total, dram_total)
    zero = (0, 0, 0, 0, 0)

    for index, task in enumerate(tasks):
        source_base = edge_bytes if index == 0 else ((index - 1) * edge_bytes) % L2_BYTES
        result_base = (index * edge_bytes) % L2_BYTES
        check_tuple(checks, f"{label} t{index} source", task, True,
                    source_cfg[index], source_base, row, source_row, depth,
                    depth if index == 0 else 0)
        check_tuple(checks, f"{label} t{index} result", task, False,
                    result_cfg[index], result_base,
                    0 if index == 5 else row,
                    0 if index == 5 else source_row,
                    0 if index == 5 else depth, 0)
        checks.equal(f"{label} t{index} Common 0x24",
                     reg(task, "0x00000", 0x24), common24[index])
        checks.equal(f"{label} t{index} header[6]", header(task, 6), header6[index])
        checks.equal(f"{label} t{index} header[8]", header(task, 8), header8[index])
        source_populated = index in (0, 1, 3, 5)
        check_dma(checks, f"{label} t{index} source DMA", task, True,
                  source_dma[index], populated if source_populated else zero,
                  0x01002031 if source_populated else 0x01000000)
        result_populated = index in (0, 2, 4, 5)
        check_dma(checks, f"{label} t{index} result DMA", task, False,
                  0x40000c1 if index == 5 else 0xc0,
                  populated if result_populated else zero,
                  0x01302031 if result_populated else 0x01000000)
    return {
        "case": record["case"],
        "row_l2": row,
        "source_row": source_row,
        "edge_bytes": edge_bytes,
        "edge_bases": [(index * edge_bytes) % L2_BYTES for index in range(5)],
        "final_staging_base": (5 * edge_bytes) % L2_BYTES,
    }


def main() -> None:
    checks = Checks()
    derived = {
        "four_task_consumer": verify_four_task_consumer(checks),
        "six_layer_chains": [
            verify_six_layer_chain(checks, 256, 28),
            verify_six_layer_chain(checks, 512, 14),
            verify_six_layer_chain(checks, 1024, 7),
        ],
    }
    print(json.dumps({
        "checked_fields": checks.count,
        "mismatches": checks.mismatches,
        "derived": derived,
    }, indent=2, sort_keys=True))
    raise SystemExit(bool(checks.mismatches))


if __name__ == "__main__":
    main()
