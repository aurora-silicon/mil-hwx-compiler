#!/usr/bin/env python3
"""Validate the generated 28-layer H14 package without opening a device."""
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import split_h14_tasks, decode_task, register_words  # noqa: E402


def contents(path):
    a = path.read_bytes()
    payload, first, count, text, constants = struct.unpack_from('<QIIQQ', a)
    assert payload + 0x1000 == len(a)
    offset = payload - constants
    assert offset % 64 == 0 and offset >= text
    tasks = split_h14_tasks(a[0x1000:0x1000 + text])
    assert len(tasks) == count and len(tasks[0]) == first
    return a, tasks, a[0x1000 + offset:]


def main():
    net = Path(sys.argv[1])
    m = json.loads((net / 'linked/manifest.json').read_text())
    assert m['target'] == 'H14' and m['schedule'] == 'linked'
    assert len(m['programs']) == 1 and m['dispatchPlan'] == [0]
    a, tasks, constants = contents(net / 'linked' / m['programs'][0]['file'])
    assert len(tasks) == 45 and len({t['layer'] for t in m['tasks']}) == 28
    assert len(m['scratch']['regions']) == 27
    assert m['scratch']['bytes'] == 17891328
    assert struct.unpack_from('<I', a, 0x28 + 3 * 4)[0] * 0x4000 == m['scratch']['bytes']
    regions = sorted(m['scratch']['regions'].values(), key=lambda r: r['offset'])
    for left, right in zip(regions, regions[1:]):
        assert left['offset'] + left['bytes'] <= right['offset']
    previous = -1
    for index, (raw, tr) in enumerate(zip(tasks, m['tasks'])):
        task = decode_task(raw, 'h14')
        words = register_words(task)
        layer = tr['layer']
        header = [int(w, 16) for w in task['header_words']]
        assert header[0] & 0xffff == index and header[6] == 0 and header[7] == 0x50001
        assert header[2] == (8 if index == 0 else 0) | (0x22 if index == 44 else 0)
        assert (words[0x1100] >> 28) == (1 if layer and layer != previous else 0)
        assert (words[0x1100] >> 16) & 255 == index
        assert (words[0x1500] >> 16) & 255 == index
        assert (words[0x1900] >> 16) & 255 == index + 1
        read_base, read_size = ((regions[layer - 1]['offset'], regions[layer - 1]['bytes'])
                               if layer else (0, m['inputs'][0]['allocationBytes']))
        write_base, write_size = ((regions[layer]['offset'], regions[layer]['bytes'])
                                 if layer != 27 else (0, m['outputs'][0]['allocationBytes']))
        assert read_base <= words[0x1110]
        assert words[0x1110] + words[0x111c] * words[0xc] <= read_base + read_size
        assert write_base <= words[0x1508]
        assert words[0x1508] + words[0x1514] * words[0x10] <= write_base + write_size
        for i in range(16):
            assert words.get(0x1908, 0) + words.get(0x1958 + 4*i, 0) + words[0x1998 + 4*i] <= len(constants)
        # Native BAR routing, independent of the header allocation table.
        bars = {int(r['address'], 16): (int(r['header'], 16) >> 23) & 63
                for r in task['records'] if int(r['header'], 16) & 0xa0000000 == 0x20000000}
        assert bars[0x1908] == 1 and bars[0x1110] == (3 if layer else 4)
        assert bars[0x1508] == (3 if layer != 27 else 5)
        previous = layer
    pm = json.loads((net / 'perop/manifest.json').read_text())
    assert len(pm['programs']) == 28
    parts = [contents(net / 'perop' / p['file']) for p in pm['programs']]
    assert sum(len(p[1]) for p in parts) == 45
    assert b''.join(p[2] for p in parts) == constants
    print(json.dumps(dict(status='PASS', layers=28, tasks=45, scratch=m['scratch']['bytes'],
                          program_bytes=len(a), sha256=hashlib.sha256(a).hexdigest())))

if __name__ == '__main__':
    main()
