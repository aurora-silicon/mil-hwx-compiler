#!/usr/bin/env python3
"""Offline proof for the three measured H16G MobileNetV1 L2 rungs."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import decode_task, register_words, split_h14_tasks  # noqa: E402

DUMP = str(Path(sys.argv[1]).resolve())

LAYERS = [
    (3,32,224,3,2,1), (32,32,112,3,1,32), (32,64,112,1,1,1),
    (64,64,112,3,2,64), (64,128,56,1,1,1), (128,128,56,3,1,128),
    (128,128,56,1,1,1), (128,128,56,3,2,128), (128,256,28,1,1,1),
    (256,256,28,3,1,256), (256,256,28,1,1,1), (256,256,28,3,2,256),
    (256,512,14,1,1,1), (512,512,14,3,1,512), (512,512,14,1,1,1),
    (512,512,14,3,1,512), (512,512,14,1,1,1), (512,512,14,3,1,512),
    (512,512,14,1,1,1), (512,512,14,3,1,512), (512,512,14,1,1,1),
    (512,512,14,3,1,512), (512,512,14,1,1,1), (512,512,14,3,2,512),
    (512,1024,7,1,1,1), (1024,1024,7,3,1,1024),
    (1024,1024,7,1,1,1), (1024,1024,7,1,1,1),
]
SCHEDULE = [
    0x230000,0x230000,0x231100,0x211100,0x243300,0x243300,0x243300,
    0x222200,0x222200,0x222200,0x222200,0x233300,0x233300,0x233300,
    0x233300,0x233300,0x233300,0x233300,0x233300,0x233300,0x233300,
    0x233300,0x233300,0x244404,0x244404,0x244404,0x244404,0x244404,
]
COST = [4,11,11,12,2,7,5,6,3,3,6,2,3,1,7,1,7,1,7,1,7,1,7,1,15,1,30,37]
MASK = [
    0x00001fff,0x00001fff,0x01ffffff,0x7fffffff,0x7f001fff,0x01ffffff,0xfffff01f,
    0xff00079f,0x000007ff,0x00007fff,0x003fff80,0x007fc000,0x0060000f,0x000000ff,
    0x000007f8,0x00007f80,0x0003fc00,0x003fc000,0x01fe0000,0x1fe00000,0xff000000,
    0xf000000f,0x8000007f,0x000000f8,0x000000c3,0x0000000f,0x0000003e,0,
]
BASE = [
    0,0,0,0x188e00,0,0x0c4a80,0x189500,0x0708c0,0,0x0708c0,0x0e1180,
    0x151880,0,0x0382a0,0x070540,0x0a87e0,0x0e0a80,0x118d20,0x150fc0,
    0x189260,0x1c1500,0x1f97a0,0x031a40,0x069c00,0,0x01c150,0x0382a0,
]
TASK_BYTES = [
    412,320,400,360,344,356,348,364,348,356,352,364,348,356,
    352,360,352,360,352,360,352,360,352,364,348,356,352,408,
]


def mobile_source_cfg(edge):
    if edge <= 5 or edge in (7, 8):
        return 0x120
    return 0x104


def run(mode, count=28):
    args = [DUMP, mode, 'H16G']
    for index, layer in enumerate(LAYERS[:count]):
        # Every layer is biased; tasks 0-26 fuse ReLU.
        args += [*map(str, layer), str(2 | (index != 27))]
    result = subprocess.run(args, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout, [decode_task(t, 'h16') for t in split_h14_tasks(result.stdout)]


def words(tasks):
    return [register_words(task) for task in tasks]


def main():
    dram_stream, dram = run('dram-dep')
    streams = {}
    rungs = {}
    for mode in ('mobilenet-l2', 'mobilenet-l2-schedule', 'mobilenet-l2-header'):
        streams[mode], rungs[mode] = run(mode)
        assert len(rungs[mode]) == 28
        actual_bytes = [t['size_bytes'] for t in rungs[mode]]
        if mode == 'mobilenet-l2-header':
            assert actual_bytes == TASK_BYTES, (mode, actual_bytes, TASK_BYTES)
            assert len(streams[mode]) == 10208
    assert len(dram) == 28 and len(dram_stream) > 10200

    # Placement and arithmetic are invariant across all three rungs.  The NE
    # blocks are also exactly those of the unchanged DRAM-linked compiler.
    base = rungs['mobilenet-l2']
    for tasks in rungs.values():
        assert [t['blocks'].get('0x04900') for t in tasks] == \
               [t['blocks'].get('0x04900') for t in dram]
        assert [(words(tasks)[i][4], words(tasks)[i][8], words(tasks)[i][12],
                 words(tasks)[i][0x14], words(tasks)[i][0x18], words(tasks)[i][0x1c])
                for i in range(28)] == \
               [(words(dram)[i][4], words(dram)[i][8], words(dram)[i][12],
                 words(dram)[i][0x14], words(dram)[i][0x18], words(dram)[i][0x1c])
                for i in range(28)]

    w = words(base)
    assert [i for i, t in enumerate(base) if '0x04d00' in t['blocks']] == [0, 2]
    assert [i for i, t in enumerate(base) if '0x05100' in t['blocks']] == [1, 27]
    assert w[1][0x5100] == 0x010100f1 and w[1][0x5138] == 0x08302031
    assert w[2][0x4d00] == 0x20020041 and w[2][0x4d68] == 0x08002031
    assert (w[1][0x5110], w[1][0x5114]) == (0x700, 0x31000)
    assert (w[2][0x4d18], w[2][0x4d1c]) == (0x700, 0x31000)
    # The spill's task-local L2 staging regions must not alias the live input
    # or result.  These independent decoded values were omitted from the
    # first implementation and are not derived from BASE/mobileResultR.
    assert w[1][0x414c] == 0x0c5500
    assert [w[2][a] for a in (0x4110,0x4114,0x4118,0x411c,0x4120)] == \
           [0x188e00,0x80,0x1050,0x1000,0x1000]
    assert w[27][0x414c] == 0x543f0 and w[27][0x5100] == 0x011b0031
    assert [w[i][0x5500] & 0x20 for i in range(28)] == \
           [0x20 if i in (0,3,7,11,23) else 0 for i in range(28)]
    assert [w[i][0x4148] for i in range(28)] == \
        [0x500170,0x84a] + [0x140] * 25 + [0x14a]
    assert [w[i][0x4104] for i in range(1, 28)] == \
        [0x120,0x842] + [0x120] * 4 + [0x104,0x120,0x120] + [0x104] * 18
    for edge in range(27):
        if edge == 1:
            continue
        assert w[edge].get(0x414c, 0) == BASE[edge]
        assert w[edge + 1].get(0x4110, 0) == BASE[edge]
        assert [w[edge][a] for a in (0x4150,0x4154,0x4158,0x415c)] == \
               [w[edge + 1][a] for a in (0x4114,0x4118,0x411c,0x4120)]
    # The decoded cursor wraps after task 21 inside the 0x200000 ring.
    assert (BASE[21] + 0x382a0) % 0x200000 == BASE[22]

    old = [words(base)[i][0x3c] for i in range(28)]
    assert [words(rungs['mobilenet-l2-schedule'])[i][0x3c] for i in range(28)] == SCHEDULE
    assert [words(rungs['mobilenet-l2-header'])[i][0x3c] for i in range(28)] == SCHEDULE
    assert sum(a != b for a, b in zip(old, SCHEDULE)) == 14

    # Prefix packages keep the first k eligible edges resident (edge 0, then
    # 2..k).  Every transition back to DRAM uses the explicit 0x84a/0x842
    # spill bridge; mixing a resident source with standalone 0x500172 leaves
    # TileDMA/BAR state live into the next execution.
    eligible = [0] + list(range(2, 27))
    for prefix in (1,2,4,8,26):
        _, tasks = run(f'mobilenet-prefix-{prefix}')
        ww = words(tasks)
        resident = set(eligible[:prefix])
        dram = set(range(27)) - resident
        bridges = {edge for edge in dram
                   if edge == 1 or (edge and edge - 1 in resident)}
        assert [i for i, t in enumerate(tasks) if '0x04d00' in t['blocks']] == \
               [0] + [edge + 1 for edge in sorted(dram)]
        assert [i for i, t in enumerate(tasks) if '0x05100' in t['blocks']] == \
               sorted(dram) + [27]
        for edge in resident:
            assert ww[edge][0x4148] in (0x500170,0x140)
            assert ww[edge + 1][0x4104] == mobile_source_cfg(edge)
        for edge in bridges:
            assert ww[edge][0x4148] == 0x84a
            assert ww[edge + 1][0x4104] == 0x842
            assert ww[edge][0x5100] == 0x010000f1 | edge << 16
            assert ww[edge + 1][0x4d00] == 0x20000041 | (edge + 1) << 16
            assert ww[edge][0x5138] == 0x08302031
            assert ww[edge + 1][0x4d68] == 0x08002031
        # The first bridge is the decoded Apple edge and must remain exact in
        # every prefix, including its task-local staging bases and geometry.
        assert ww[1][0x414c] == 0x0c5500
        assert ww[2][0x4d00] == 0x20020041
        assert (ww[1][0x5110],ww[1][0x5114]) == (0x700,0x31000)
        assert (ww[2][0x4d18],ww[2][0x4d1c]) == (0x700,0x31000)

    # Grow the prefix-1 placement from two tasks through the full graph.
    # Every nonterminal task retains the 28-task encoding; only each rung's
    # last task gains the external output endpoint.
    _, full_prefix = run('mobilenet-prefix-1')
    full_words = words(full_prefix)
    terminal_cfg = {2: 0x142, 3: 0x142, 4: 0x14a, 6: 0x14a,
                    9: 0x14a, 15: 0x14a, 28: 0x14a}
    for count in (2,3,4,6,9,15,28):
        _, tasks = run('mobilenet-prefix-1',count)
        ww = words(tasks)
        assert len(tasks) == count
        for index in range(count-1):
            assert ww[index] == full_words[index], (count,index)
        assert [i for i,t in enumerate(tasks) if '0x04d00' in t['blocks']] == \
               [0] + list(range(2,count))
        assert [i for i,t in enumerate(tasks) if '0x05100' in t['blocks']] == \
               list(range(1,count))
        # The first two truncations end on non-direct schedules.  They must
        # retain the ordinary staged terminal class rather than inheriting
        # the full graph's direct classifier bit.
        assert ww[-1][0x4148] == terminal_cfg[count]
        if count >= 3:
            assert ww[1][0x4148] == 0x84a and ww[2][0x4104] == 0x842
        if count >= 6:
            # Edge 3 is the first ping-pong edge with a nonzero BAR-3
            # addend.  It is one u64, not two copies of the low offset.
            assert (ww[3][0x5108],ww[3][0x510c]) == (0x1c0000,0)
            assert (ww[4][0x4d10],ww[4][0x4d14]) == (0x1c0000,0)

    # The v7 two-task packages isolate one field group at a time.  The
    # legacy control differs from the correction only in the terminal direct
    # bit; the other two hybrids independently import the linear program's
    # KernelDMA mode or the passing sentinel's first-edge staging geometry.
    _, corrected = run('mobilenet-prefix-1',2)
    cw = words(corrected)
    for mode, expected in (
        ('mobilenet-hybrid-legacy-terminal', {(1,0x4148)}),
        ('mobilenet-hybrid-linear-kernel', {(0,0x5500)}),
        ('mobilenet-hybrid-sentinel-geometry',
         {(0,0x4110),(0,0x414c),(0,0x4154),
          (1,0x4110),(1,0x4118),(1,0x414c)}),
    ):
        _, hybrid = run(mode,2)
        hw = words(hybrid)
        changed={(i,a) for i in range(2) for a in set(cw[i])|set(hw[i])
                 if cw[i].get(a)!=hw[i].get(a)}
        assert changed == expected, (mode,changed,expected)
    _, legacy = run('mobilenet-hybrid-legacy-terminal',2)
    _, kernel = run('mobilenet-hybrid-linear-kernel',2)
    _, geometry = run('mobilenet-hybrid-sentinel-geometry',2)
    assert words(legacy)[1][0x4148] == 0x14a
    assert words(kernel)[0][0x5500] & 0x20 == 0
    gw=words(geometry)
    assert gw[0].get(0x4110,0)==0 and gw[0][0x414c]==0x49800
    assert gw[0][0x4154]==0x1c80
    assert (gw[1][0x4110],gw[1][0x4118],gw[1][0x414c]) == \
           (0x49800,0x1c80,0x111000)

    _, baseline_spill = run('mobilenet-baseline-spill')
    bw = words(baseline_spill)
    assert [i for i, t in enumerate(baseline_spill) if '0x04d00' in t['blocks']] == [0,2]
    assert [i for i, t in enumerate(baseline_spill) if '0x05100' in t['blocks']] == [1,27]
    assert bw[1][0x5100] != 0x010100f1 and bw[2][0x4d00] != 0x20020041
    assert bw[1][0x414c] == 0x0c5500 and bw[2][0x4110] == 0x188e00

    headers = [[int(x, 16) for x in t['header_words']]
               for t in rungs['mobilenet-l2-header']]
    assert [h[1] for h in headers] == COST
    assert [h[4] for h in headers] == [0x68] + [0x28] * 27
    assert [h[7] for h in headers] == MASK
    assert [h[2] for h in headers] == [0x68] + [0] * 26 + [0x04000000]
    assert [h[8] for h in headers] == [0x00050001] * 27 + [0x00050009]
    print('H16G MobileNetV1 L2 rungs: PASS (28 tasks, 26 L2 edges, one 0xc4000 spill, four TileDMA endpoints)')


if __name__ == '__main__':
    main()
