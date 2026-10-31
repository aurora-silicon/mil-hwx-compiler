"""Source-built native layer constants; equality is checked by oracle hashes.

The bilinear kernel contains four phase filters over the 3x3 input footprint,
with separable half-pixel coefficients (1/4, 3/4). Sparse masks pack two taps
per header byte; H13 groups two taps, H14 groups four before coefficients.
No Apple constant-section bytes are read or retained here.
"""
import struct


def hard_swish_constants(size, channels, h14=False):
    out = bytearray(size)
    block = (channels + 31) // 32 * 32
    if channels > 1:
        for channel in range(channels):
            struct.pack_into("<H", out, (block + channel) * 2, 0x3155)
    if not h14:
        base = block * 4 if channels > 1 else 0
        # relu6's LUT; +3 and /6 are PE immediates in the task stream.
        for index, value in ((1, 6), (3, 6), (37, 1), (39, 1)):
            struct.pack_into("<e", out, base + index * 2, value)
        struct.pack_into("<H", out, base + 82, 64)
        struct.pack_into("<H", out, base + 84, 1)
    return bytes(out)


def resize_constants(operation, channels, h14=False):
    cores = min(16, channels)
    if operation == "Nearest2x":
        out = bytearray()
        for core in range(cores):
            block = bytearray(128)
            for index in range(1, 16):
                struct.pack_into("<e", block, 2 * index, 1)
            slots = (channels - core + 15) // 16
            if slots > 32:
                raise ValueError("nearest coefficient mask capacity exceeded")
            for slot in range(slots):
                struct.pack_into("<H", block, 64 + slot * 2, 0x1111)
            out += block
        return bytes(out)
    # Four output phases: top-left, top-right, bottom-left, bottom-right.
    rows = ((1, 0, 0, 0), (3, 3, 0, 0), (0, 1, 0, 0),
            (3, 0, 3, 0), (9, 9, 9, 9), (0, 3, 0, 3),
            (0, 0, 1, 0), (0, 0, 3, 3), (0, 0, 0, 1))
    step = 4 if h14 else 2
    body = bytearray([0])
    for start in range(0, 9, step):
        group = rows[start:start + step]
        masks = [sum(1 << i for i, value in enumerate(row) if value) for row in group]
        masks += [0] * (step - len(masks))
        body.extend(masks[i] | (masks[i + 1] << 4) for i in range(0, step, 2))
        for row in group:
            for value in row:
                if value:
                    body += struct.pack("<e", value / 16)
    channel = struct.pack("<H", len(body) - 1) + body
    out = bytearray()
    for core in range(cores):
        block = channel * ((channels - core + 15) // 16)
        out += block + bytes((-len(block)) % 64)
    return bytes(out)
