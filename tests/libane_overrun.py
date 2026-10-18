#!/usr/bin/env python3
"""Device-free port of libane's surface overrun check (libane/ane_bind.h: ane_bind_walk and ane_bind_overrun).

An ANEC file is a 0x1000-byte header followed by the payload. The header holds the payload size, the task descriptor size and count, and the tile count of
every surface channel (the surface allocation is tiles << tile_shift, 14 by default). The task stream programs, per task, the byte size each live
surface transfer moves. libane refuses a program whose transfer on a surface channel is larger than that channel's allocation, because the engine would
read or write past the buffer. This module reports the same refusal without a device.

    python3 tests/libane_overrun.py PROGRAM.anec [...]
"""
import struct
import sys
from pathlib import Path

HEADER = 0x1000
TILE_COUNT = 0x20
FIRST_SURFACE = 4
DMA_DISABLED = 0x8880
# slot: (dma configuration register, selector shift in word 8, size register a, size register b)
SELECTORS = ((0x13800, 0, 0x13814, 0x13818), (0x13804, 6, 0, 0), (0x17800, 12, 0x17810, 0x17814))
MIN_TASK_BYTES = 40


def _word(data, index):
    return struct.unpack_from("<I", data, index * 4)[0]


def _task_dma(task):
    """The three tile-DMA configuration words and the byte extents one task programs, or None when the register walk leaves the task."""
    words = len(task) // 4
    if len(task) < MIN_TASK_BYTES or len(task) % 4:
        return None
    dma = [DMA_DISABLED] * 3
    extent = [0] * 3
    index = 10 + (1 if (_word(task, 9) & 3) == 3 else 0)
    while index < words:
        header = _word(task, index)
        count = (header >> 26) + 1
        base = header & 0x03FFFFFF
        if index + count >= words:
            return None
        for offset in range(count):
            register = base + offset * 4
            value = _word(task, index + 1 + offset)
            for slot, (config, _, size_a, size_b) in enumerate(SELECTORS):
                if register == config:
                    dma[slot] = value
                elif size_a and register in (size_a, size_b) and value > extent[slot]:
                    extent[slot] = value
        index += 1 + count
    return dma, extent


def channel_extents(anec):
    """Largest byte extent any task moves through each surface channel, or None when the stream cannot be walked."""
    td_size, td_count = struct.unpack_from("<II", anec, 8)
    stream = anec[HEADER:]
    if not td_count:
        return None
    extent = [0] * TILE_COUNT
    offset, size = 0, td_size
    for index in range(td_count):
        if size < MIN_TASK_BYTES or offset > len(stream) or size > len(stream) - offset:
            return None
        task = stream[offset:offset + size]
        walked = _task_dma(task)
        if walked is None:
            return None
        dma, moved = walked
        selectors = _word(task, 8)
        for slot, (_, shift, _, _) in enumerate(SELECTORS):
            channel = (selectors >> shift) & 0x1F
            if dma[slot] == DMA_DISABLED or channel < FIRST_SURFACE or channel >= TILE_COUNT:
                continue
            extent[channel] = max(extent[channel], moved[slot])
        if index + 1 == td_count:
            break
        following = _word(task, 7)
        size = (((_word(task, 1) >> 16) & 0x1FF) + 1) * 4
        if following % 4 or following > len(stream):
            return None
        offset = following
    return extent


def overruns(anec, tile_shift=14):
    """The (channel, bytes needed, bytes allocated) triples libane refuses, empty when every transfer fits."""
    extent = channel_extents(anec)
    if extent is None:
        return []
    tiles = struct.unpack_from("<32I", anec, 40)
    return [(channel, extent[channel], tiles[channel] << tile_shift)
            for channel in range(FIRST_SURFACE, TILE_COUNT)
            if extent[channel] > (tiles[channel] << tile_shift)]


def main(paths):
    status = 0
    for path in paths:
        found = overruns(Path(path).read_bytes())
        for channel, need, have in found:
            print(f"OVERRUN {path}: surface channel {channel} needs {need} bytes but is allocated {have}")
            status = 1
        if not found:
            print(f"fits {path}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
