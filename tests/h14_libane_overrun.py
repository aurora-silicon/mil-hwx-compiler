#!/usr/bin/env python3
"""Device-free H14 surface-bound check matching libane's BAR walk."""
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import decode_task, register_words, split_h14_tasks  # noqa: E402
from h14_fields import H14Fields  # noqa: E402


# The ABI-2 port table binds BAR 4 to the input buffer (header channel 5),
# BAR 5 to the output buffer (channel 4) and BAR 3 to scratch (channel 3).
BAR_CHANNEL = {4: 5, 5: 4}

# ANEC layout constants are format boundaries; H14 register and record fields
# are parsed from H14Fields.h so this checker cannot drift into a second map.
ANEC_TEXT_OFFSET = 4096
ANEC_TILE_TABLE_OFFSET = 40
TILE_BYTES_SHIFT = 14
FIELDS = H14Fields(ROOT)
INPUT_CHANNELS = FIELDS.register('inputChannels')
OUTPUT_CHANNELS = FIELDS.register('outputChannels')
SOURCE_DMA_BASE = FIELDS.register('sourceDmaBase')
SOURCE_DMA_PLANE_STRIDE = FIELDS.register('sourceDmaPlaneStride')
SOURCE_DMA_TILE = FIELDS.register('sourceDmaTile')
RESULT_DMA_BASE = FIELDS.register('resultDmaBase')
RESULT_DMA_PLANE_STRIDE = FIELDS.register('resultDmaPlaneStride')
RESULT_DMA_TILE = FIELDS.register('resultDmaTile')


def _bar_slots(task):
    return {int(record['address'], 16):
            BAR_CHANNEL.get(FIELDS.get('barSlot', int(record['header'], 16)),
                            FIELDS.get('barSlot', int(record['header'], 16)))
            for record in task['records']
            if int(record['header'], 16) & FIELDS.field('barPresent')}


def overruns(anec):
    """Return (channel, bytes-needed, bytes-allocated) H14 bind failures."""
    _, _, count, text_bytes, _ = struct.unpack_from('<QIIQQ', anec)
    tasks = split_h14_tasks(
        anec[ANEC_TEXT_OFFSET:ANEC_TEXT_OFFSET + text_bytes])
    if len(tasks) != count:
        raise ValueError('H14 task count differs from the ANEC header')
    need = [0] * 32
    for raw in tasks:
        task = decode_task(raw, 'h14')
        words, bars = register_words(task), _bar_slots(task)
        if SOURCE_DMA_BASE in bars:
            channel = bars[SOURCE_DMA_BASE]
            channels = words[INPUT_CHANNELS]
            if words.get(SOURCE_DMA_TILE, 0) & FIELDS.field('dmaTileScratch'):
                channels = (channels + 7) // 8
            end = (words.get(SOURCE_DMA_BASE, 0) +
                   words.get(SOURCE_DMA_PLANE_STRIDE, 0) * channels)
            need[channel] = max(need[channel], end)
        if RESULT_DMA_BASE in bars:
            channel = bars[RESULT_DMA_BASE]
            channels = words[OUTPUT_CHANNELS]
            if words.get(RESULT_DMA_TILE, 0) & FIELDS.field('dmaTileScratch'):
                channels = (channels + 7) // 8
            end = (words.get(RESULT_DMA_BASE, 0) +
                   words.get(RESULT_DMA_PLANE_STRIDE, 0) * channels)
            need[channel] = max(need[channel], end)
    tiles = struct.unpack_from('<32I', anec, ANEC_TILE_TABLE_OFFSET)
    return [(channel, required, tiles[channel] << TILE_BYTES_SHIFT)
            for channel, required in enumerate(need)
            if required > tiles[channel] << TILE_BYTES_SHIFT]


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
