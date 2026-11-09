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
from h14_libane_overrun import overruns  # noqa: E402


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
    assert m['target'] == 'H14' and m['schedule'] == 'linked-ring'
    assert len(m['programs']) == 1 and m['dispatchPlan'] == [0]
    a, tasks, constants = contents(net / 'linked' / m['programs'][0]['file'])
    assert len(tasks) == 28 and len({t['layer'] for t in m['tasks']}) == 28
    assert len(m['scratch']['regions']) == 1
    assert m['scratch']['bytes'] == 0x284000
    assert struct.unpack_from('<I', a, 0x28 + 3 * 4)[0] * 0x4000 == m['scratch']['bytes']
    assert not overruns(a)
    assert min(map(len, tasks)) >= 312 and max(map(len, tasks)) <= 384
    constant_bases = []
    for index, (raw, tr) in enumerate(zip(tasks, m['tasks'])):
        task = decode_task(raw, 'h14')
        words = register_words(task)
        header = [int(w, 16) for w in task['header_words']]
        assert header[0] & 0xffff == index and header[7] == 0x50001
        assert header[2] == (8 if index == 0 else 0) | (0x22 if index == 27 else 0)
        expected_source = (0x500172 if index == 0 else 0x842 if index == 2
                           else 0x104 if index == 7 or index >= 10 else 0x120)
        assert words[0x504] == expected_source
        assert words[0x530] == (0x500170 if index == 0 else
                               0x84a if index == 1 else
                               0x14a if index == 27 else 0x140)
        if index not in (0, 2):
            assert not any(0x1100 <= address < 0x1500 for address in words)
        if index not in (1, 27):
            assert 0x1500 not in words
        if index == 0:
            assert words[0x1100] == 0xe21 and words[0x1108] == 0xce
        elif index == 2:
            assert words[0x1100] == 0x10020e41 and words[0x1108] == 0x4e
        elif index == 1:
            assert words[0x1500] == 0x01010ef1
        elif index == 27:
            assert words[0x1500] == 0x011b0e31
        assert (words[0x1900] >> 16) & 255 == index + 1
        constant_bases.append(words[0x1908])
        assert (words[0x1918] >> 16) & 255 == index
        for i in range(16):
            if 0x1998 + 4*i in words:
                assert (words[0x1908] + words.get(0x1958 + 4*i, 0) +
                        words[0x1998 + 4*i] <= len(constants))
        bars = {int(r['address'], 16): (int(r['header'], 16) >> 23) & 63
                for r in task['records'] if int(r['header'], 16) & 0xa0000000 == 0x20000000}
        assert bars[0x1908] == 1
        if index in (0, 2):
            assert bars[0x1110] == (5 if index == 0 else 3)
        if index in (1, 27):
            assert bars[0x1508] == (3 if index == 1 else 4)
    assert constant_bases[0] == 0 and constant_bases == sorted(constant_bases)
    last_words = register_words(decode_task(tasks[-1], 'h14'))
    assert max(last_words[0x1908] + last_words.get(0x1958 + 4*i, 0) +
               last_words[0x1998 + 4*i] for i in range(16)) == len(constants)
    print(json.dumps(dict(status='PASS', layers=28, tasks=28, scratch=m['scratch']['bytes'],
                          program_bytes=len(a), sha256=hashlib.sha256(a).hexdigest())))

if __name__ == '__main__':
    main()
