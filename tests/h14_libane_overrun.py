#!/usr/bin/env python3
"""Device-free H14 surface-bound check matching libane's BAR walk."""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import decode_task, register_words, split_h14_tasks  # noqa: E402


def _bar_slots(task):
    return {int(record['address'], 16):
            (int(record['header'], 16) >> 23) & 63
            for record in task['records']
            if int(record['header'], 16) & 0x20000000}


def overruns(anec):
    """Return (channel, bytes-needed, bytes-allocated) H14 bind failures."""
    _, _, count, text_bytes, _ = struct.unpack_from('<QIIQQ', anec)
    tasks = split_h14_tasks(anec[0x1000:0x1000 + text_bytes])
    if len(tasks) != count:
        raise ValueError('H14 task count differs from the ANEC header')
    need = [0] * 32
    for raw in tasks:
        task = decode_task(raw, 'h14')
        words, bars = register_words(task), _bar_slots(task)
        if 0x1110 in bars:
            channel = bars[0x1110]
            channels = words[0x0c]
            if words.get(0x1150, 0) & 0x08000000:
                channels = (channels + 7) // 8
            end = words.get(0x1110, 0) + words.get(0x111c, 0) * channels
            need[channel] = max(need[channel], end)
        if 0x1508 in bars:
            channel = bars[0x1508]
            channels = words[0x10]
            if words.get(0x1520, 0) & 0x08000000:
                channels = (channels + 7) // 8
            end = words.get(0x1508, 0) + words.get(0x1514, 0) * channels
            need[channel] = max(need[channel], end)
    tiles = struct.unpack_from('<32I', anec, 0x28)
    return [(channel, required, tiles[channel] << 14)
            for channel, required in enumerate(need)
            if required > tiles[channel] << 14]


def main(paths):
    failed = False
    for path in map(Path, paths):
        found = overruns(path.read_bytes())
        for channel, need, have in found:
            print(f'OVERRUN {path}: channel {channel} needs {need}, has {have}')
            failed = True
        if not found:
            print(f'fits {path}')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
