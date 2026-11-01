#!/usr/bin/env python3
"""The decoded Apple H14 linked programs follow the input-reuse rule the H14
linker emits: a later output slice of a multi-task layer has no source
TileDMA, no SourceCfg/depth-stride words, NE repeat fields equal to the NE
class, and ResultCfg without the DMA-source bits; every task that is followed
by a reuse slice keeps L2 live (header word 6 nonzero)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / 'research/oracles/h14-mobile'


def words(task):
    w = {}
    for block in task['blocks'].values():
        for address, value in block['words'].items():
            w[int(address, 16)] = int(value, 16)
    return w


def main():
    records = [json.loads(p.read_text()) for p in sorted(ORACLES.glob('*.json'))
               if 'chain' in p.name or 's2link' in p.name]
    reuse_tasks = loaders = 0
    for r in records:
        assert r['error'] is None, r['case']
        tasks = r['task_descriptors']
        regs = [words(t) for t in tasks]
        for i, (t, w) in enumerate(zip(tasks, regs)):
            header = [int(x, 16) for x in t['header_words']]
            reads_dram = 0x1100 in w
            reads_l2 = 0x504 in w and not reads_dram
            if reads_dram or reads_l2 or i == 0:
                continue
            # A slice with neither a source DMA nor a SourceCfg reuses the
            # previous task's staged input: same input geometry.
            reuse_tasks += 1
            prev = regs[i - 1]
            assert w[0x0] == prev[0x0] and w[0xc] == prev[0xc], (r['case'], i)
            assert not any(0x1100 <= a < 0x1500 for a in w), (r['case'], i)
            assert 0x514 not in w and 0x518 not in w, (r['case'], i)
            cls = (w[0x38] >> 16) & 0xf
            assert (w[0x38] >> 8) & 0xff == (cls << 4) | cls, (r['case'], i, hex(w[0x38]))
            assert w[0x530] & 0x500030 == 0, (r['case'], i, hex(w[0x530]))
            prev_header = [int(x, 16) for x in tasks[i - 1]['header_words']]
            assert prev_header[6], (r['case'], i - 1)
            if prev.get(0x1100) is not None:
                loaders += 1
                assert prev[0x1100] >> 28 in (0, 1), (r['case'], i - 1)
            if i + 1 < len(tasks):
                assert header[6], (r['case'], i)
    assert (reuse_tasks, loaders) == (19, 5), (reuse_tasks, loaders)
    print(f'H14 linked reuse oracles: PASS ({len(records)} programs, {reuse_tasks} reuse slices, '
          f'{loaders} DMA loaders)')


if __name__ == '__main__':
    main()
