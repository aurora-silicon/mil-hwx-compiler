#!/usr/bin/env python3
"""Rebuild native MobileNet prefixes and compare H13 schedule-role words.

The private t-5959 decode is optional and never copied into the repository.
Set MILHWXC_T5959_PRIVATE to its directory to enable the comparison; without
it this host parity gate reports SKIP.  Weight/kernel offsets and cost hints
are deliberately excluded: this checks the formula-built Common, L2,
Source/Result and TileDMA roles of prefixes p01..p28.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEADER = 0x1000
ROLE_REGS = ([0x28] + list(range(0x04804, 0x04848, 4)) +
             list(range(0x13800, 0x1386c + 4, 4)) +
             list(range(0x17800, 0x17818 + 4, 4)) + [0x0c800])


def decoded(anec):
    size, count = struct.unpack_from('<II', anec, 8)
    out, offset = [], 0
    for index in range(count):
        words = struct.unpack_from(f'<{size // 4}I', anec, HEADER + offset)
        regs, cursor = {}, 10 + (1 if words[9] & 3 == 3 else 0)
        while cursor < len(words):
            n = (words[cursor] >> 26) + 1
            base = words[cursor] & 0x03ffffff
            for j in range(n):
                regs[base + 4 * j] = words[cursor + 1 + j]
            cursor += n + 1
        out.append((words[:10], regs))
        if index + 1 < count:
            size = (((words[1] >> 16) & 0x1ff) + 1) * 4
            offset = words[7]
    return out


def apple_tasks(path):
    data = json.loads(path.read_text())
    result = []
    for task in data['task_descriptors']:
        regs = {int(a, 16): int(v, 16)
                for block in task['blocks'].values()
                for a, v in block['words'].items()}
        result.append((tuple(int(v, 16) for v in task['header_words']), regs))
    return result


def compare(prefix, ours, apple):
    bad = []
    if len(ours) != prefix or len(apple) != prefix:
        return [f'task count ours={len(ours)} apple={len(apple)} expected={prefix}']
    for i, ((oh, ore), (ah, are)) in enumerate(zip(ours, apple)):
        # IDs, link/event/control and surface selectors are schedule roles.
        for j in (0, 2, 4, 6, 8):
            a, b = oh[j], ah[j]
            if j == 0:
                a &= ~0x00ff0000
                b &= ~0x00ff0000
            if j == 8:
                a &= 0x00ffffff
                b &= 0x00ffffff
            if a != b:
                bad.append(f't{i} header[{j}] {a:#x} != {b:#x}')
        for address in ROLE_REGS:
            if ore.get(address) != are.get(address):
                bad.append(f't{i} {address:#07x} {ore.get(address)!r} != {are.get(address)!r}')
    return bad


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: h13_mobilenet_ring_parity.py MILHWXC')
    private = os.environ.get('MILHWXC_T5959_PRIVATE')
    if not private:
        print('H13 linked-ring private prefix parity: SKIP (MILHWXC_T5959_PRIVATE unset)')
        return
    private = Path(private)
    prefix_dir = private / 'decoded' / 'prefixes'
    if not prefix_dir.is_dir():
        print('H13 linked-ring private prefix parity: SKIP (decoded/prefixes absent)')
        return
    sys.path.insert(0, str(ROOT / 'tools' / 'h13-mobilenet'))
    from chain import conv_mil
    import mobilenet as M
    milc = str(Path(sys.argv[1]).resolve())
    layers = M.ane_layers('ane', 'native')
    params = M.to_ane_params(M.random_tf_weights(5964), 'ane', 'native')
    with tempfile.TemporaryDirectory(prefix='h13-linked-ring-') as tmp:
        tmp = Path(tmp)
        for prefix in range(1, 29):
            case = tmp / f'p{prefix:02d}'
            mil, _, _ = conv_mil(layers[:prefix], (3, 224, 224),
                                 str(case / 'src'), weights=params[:prefix])
            out = case / 'out'
            env = dict(os.environ, MILHWXC_H13_CONV_GENERIC='1')
            run = subprocess.run([milc, '--target', 'H13', '--schedule', 'linked-ring',
                                  '--mil', mil, '--model-root', str(case / 'src'),
                                  '--output', str(out)], text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 env=env)
            if run.returncode:
                raise AssertionError(f'p{prefix:02d} compile failed: {run.stderr[-500:]}')
            ours = decoded((out / 'program-0.anec').read_bytes())
            apple = apple_tasks(prefix_dir / f'mn_p{prefix:02d}.H13.hwx.json')
            bad = compare(prefix, ours, apple)
            if bad:
                raise AssertionError(f'p{prefix:02d}: {len(bad)} mismatches\n' +
                                     '\n'.join(bad[:40]))
            print(f'p{prefix:02d}: role parity')
    print('H13 linked-ring private prefix parity: PASS')


if __name__ == '__main__':
    main()
