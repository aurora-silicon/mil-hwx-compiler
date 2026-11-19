#!/usr/bin/env python3
"""Single-layer conv sweep: the general H16/H17 rules against decoded facts.

research/oracles/{h16,h17}/conv_sweep_*.json are decoded facts of our own
single-layer MIL (fp16, SAME, stride 1, bias + ReLU; deterministic random
weights) compiled by the macOS 26.6.2 ANE compiler for h16 and h17. The cases
were picked from the 2026-11-19 sweep to cover every branch of the rules in
docs/ane/h16-h17-conv.md ("Single-layer rules").

Records: every task's registers are repacked by the production packer and
must give Apple's records, in Apple's order, and Apple's task size.

usage: test_h16_h17_conv_sweep_parity.py build/h16_conv_dump
"""
import json
from pathlib import Path
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
        counts[gen] = len(paths)
    print(f'conv sweep parity PASS: records of {counts["h16"]} H16 and {counts["h17"]} H17 tasks')


if __name__ == '__main__':
    main()
