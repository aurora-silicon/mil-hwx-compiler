#!/usr/bin/env python3
"""Single-layer conv sweep: the general H16/H17 rules against decoded facts.

research/oracles/{h16,h17}/conv_sweep_*.json are decoded facts of our own
single-layer MIL (fp16, SAME, stride 1, bias + ReLU; deterministic random
weights) compiled by the macOS 26.6.2 ANE compiler for h16 and h17. The cases
were picked from the 2026-11-19 sweep to cover every branch of the rules in
docs/ane/h16-h17-conv.md ("Single-layer rules").

Records: every task's registers are repacked by the production packer and
must give Apple's records, in Apple's order, and Apple's task size.

Kernel layout: from Apple's F (Common 0x3c), NE count (header word 8) and
OCGSize (NECfg 0x40), the production layout must give Apple's chunk table
(0x5520/0x5560/0x55a0) and, packing the case's regenerated weights, Apple's
kernel section hash.

usage: test_h16_h17_conv_sweep_parity.py build/h16_conv_dump
"""
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import decode_task  # noqa: E402

DUMP = str(Path(sys.argv[1]).resolve())


def registers(task):
    return {int(a, 16): int(v, 16) for block in task['blocks'].values()
            for a, v in block['words'].items()}


def run(args, stdin):
    result = subprocess.run([DUMP, *args], input=stdin, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def weights(case):
    """The case's weights: random.Random(name), weight then bias, uniform(-0.5, 0.5) as fp16."""
    rnd = random.Random(case['name'])
    n = case['cout'] * case['cin'] // case['groups'] * case['kernel'] ** 2
    half = lambda m: b''.join(struct.pack('<e', rnd.uniform(-0.5, 0.5)) for _ in range(m))
    return half(n) + half(case['cout'])


def kernel_tiling(regs, header):
    """Apple's F, NE count and OCGSize (NECfg reset value 3)."""
    return [regs.get(0x3c, 0) >> 19 & 7, 1 << ((header[8] >> 16 & 7) - 1), regs.get(0x40, 3) & 7]


def check_kernel(gen, path, oracle):
    case, want = oracle['case'], oracle['task_descriptors'][0]
    regs = registers(want)
    header = [int(h, 16) for h in want['header_words']]
    target = {'h16': 'H16G', 'h17': 'H17'}[gen]
    # A one-channel depthwise layer is an ordinary dense conv.
    groups = case['groups'] if case['cin'] > 1 else 1
    shape = [target, case['cin'], case['cout'], case['kernel'], groups, 1, *kernel_tiling(regs, header)]
    sizes = [int(n) for n in run(['kernel-chunks', *map(str, shape)], b'').split()]
    table, offset = {}, 0
    for slot, size in enumerate(sizes):
        if not size:
            continue
        table[0x5520 + 4 * slot] = 0x21
        if offset:
            table[0x5560 + 4 * slot] = offset
        if size != 64:
            table[0x55a0 + 4 * slot] = size
        offset += size
    apple = {a: v for a, v in regs.items() if 0x5520 <= a < 0x55e0}
    assert table == apple, (gen, path.stem, table, apple)
    section = run(['kernel', *map(str, shape)], weights(case))
    ks = oracle['kernel_section']
    assert len(section) == ks['size'] and hashlib.sha256(section).hexdigest() == ks['sha256'], \
        (gen, path.stem, 'kernel section differs')


def check_records(gen, path, want):
    regs = registers(want)
    header = [int(h, 16) for h in want['header_words']]
    text = ' '.join(f'{h:x}' for h in header) + '\n' + ''.join(f'{a:x} {v:x}\n' for a, v in sorted(regs.items()))
    got = decode_task(run(['records'], text.encode()), gen)
    assert got['records'] == want['records'], (gen, path.stem, got['records'], want['records'])
    assert got['size_bytes'] == want['size_bytes'] and got['header_words'] == want['header_words'], (gen, path.stem)


def main():
    counts = {}
    for gen in ('h16', 'h17'):
        paths = sorted((ROOT / 'research/oracles' / gen).glob('conv_sweep_*.json'))
        assert len(paths) == 31, (gen, len(paths))
        for path in paths:
            oracle = json.loads(path.read_text())
            assert len(oracle['task_descriptors']) == 1, path.name
            check_records(gen, path, oracle['task_descriptors'][0])
            check_kernel(gen, path, oracle)
        counts[gen] = len(paths)
    print(f'conv sweep parity PASS: records, chunk tables and kernel sections of '
          f'{counts["h16"]} H16 and {counts["h17"]} H17 tasks')


if __name__ == '__main__':
    main()
