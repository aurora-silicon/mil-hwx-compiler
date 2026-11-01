#!/usr/bin/env python3
"""Mint the convolution probe grid and emit the H13 and H14 conv templates.

The envelope campaign (``mint_oracles.py``) accepted 22 convolutions per
target: kernel 1 and 3, stride 1 and 2, groups 1, 4 and 64, with and without a
BLOBFILE bias, always ``pad_type="same"`` and always one task. It swept the
channel counts coarsely and never varied the spatial extent independently of
the stride, and its uniform ``fp16(0x1p-1)`` weight makes the constant-section
permutation invisible except where the BLOBFILE header bytes land. These
probes close both gaps:

* a grid over kernel 1 and 3, ``Cin`` and ``Cout`` in
  {64, 128, 256, 512, 1024}, spatial in {1, 8, 16, 32, 64}, stride 1 and 2,
  groups 1 and depthwise, with and without a bias, and both ``pad_type``
  spellings MIL uses (``same`` and ``valid``);
* known-weight probes, whose weights.bin holds a distinct finite fp16 pattern
  per element, so the packing permutation is read off the constant section
  instead of inferred from a hash.

``--layout`` is the known-weight tool: it compiles a case on the Apple
compiler, compares the constant section with :func:`conv_constants`, and
prints the destination-to-source halfword mapping for a section small enough
to print. It keeps no HWX bytes; only the derived mapping and the hashes leave
the machine.

``--emit-templates`` reads the decoded JSON back and writes
``plugins/H13/H13ConvTemplates.inc`` and ``plugins/H14/H14ConvTemplates.inc``:
the exact task stream Apple emitted per geometry, checked first against the
encoder's own surface, scratch, binding-order and constant-section formulas.

    python3 research/mint_conv_probes.py --host studio-host --targets h13 h14
    python3 research/mint_conv_probes.py --host studio-host --layout \\
        --case 'conv_probe_k1_c64_n64_s8_st1_g1_bias0_same'
    python3 research/mint_conv_probes.py --emit-templates
    python3 research/mint_conv_probes.py --check
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402
from h13_td import decode_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {"h13": ROOT / "plugins/H13/H13ConvTemplates.inc",
           "h14": ROOT / "plugins/H14/H14ConvTemplates.inc"}
CONSTANT_ALIGNMENT = 0x40
SURFACE_ALIGNMENT = 0x4000
SURFACE_BASE = 0x30000000
ROW_FLOOR = 64
# H14 task streams open with a 16-byte zero-size task frame.
H14_STREAM_PREFIX_BYTES = 16
# The known-weight probes read the packing off bit planes: every element holds
# `LOW_PATTERN` in the baseline probe, and probe `k` flips every element whose
# index has bit `k` set to `HIGH_PATTERN`. Both patterns are finite normals
# that differ in both bytes, so a byte of the constant section names the
# element and the half it came from even when the packing works below halfword
# granularity. Neither pattern is zero, so the zero-skipping the strided
# sections apply stays constant across the whole probe set.
LOW_PATTERN = 0x3801
HIGH_PATTERN = 0x3C82


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def output_spatial(spatial: int, kernel: int, stride: int, pad_type: str) -> int:
    """The conv output extent, as coremltools defines `conv`.

    coremltools 9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4,
    `coremltools/converters/mil/mil/ops/defs/iOS15/conv.py`: `same` pads so the
    output covers `ceil(D / stride)`, `valid` applies no padding and yields
    `floor((D - dilation * (kernel - 1) - 1) / stride) + 1`. The probes leave
    `dilations = [1, 1]`, so the `valid` term reduces to
    `floor((D - kernel) / stride) + 1`.
    """
    if pad_type == "same":
        return (spatial + stride - 1) // stride
    if pad_type == "valid":
        return (spatial - kernel) // stride + 1
    raise ValueError(f"unsupported pad_type {pad_type!r}")


def conv_case(kernel: int, inputs: int, outputs: int, spatial: int,
              stride: int, groups: int, bias: bool, pad_type: str = "same",
              weights: bytes | None = None,
              name_prefix: str = "conv_probe") -> dict[str, Any]:
    """One convolution probe. `weights` overrides the uniform payload with the
    exact weights.bin body, which the known-weight probes use."""
    out_spatial = output_spatial(spatial, kernel, stride, pad_type)
    if out_spatial < 1:
        raise ValueError(f"kernel {kernel} does not fit in spatial {spatial}")
    x_shape = (1, inputs, spatial, spatial)
    y_shape = (1, outputs, out_spatial, out_spatial)
    w_shape = (outputs, inputs // groups, kernel, kernel)
    weight, elements = om.blobfile("w", w_shape)
    shapes = [w_shape]
    body = [
        f'string pt = const()[name = string("pt"), val = string("{pad_type}")];',
        'tensor<int32, [2]> st = const()[name = string("st"), '
        f'val = tensor<int32, [2]>([{stride}, {stride}])];',
        'tensor<int32, [4]> pd = const()[name = string("pd"), '
        'val = tensor<int32, [4]>([0, 0, 0, 0])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), '
        'val = tensor<int32, [2]>([1, 1])];',
        f'int32 gp = const()[name = string("gp"), val = int32({groups})];', weight,
    ]
    bias_argument = ""
    if bias:
        source, count = om.blobfile("b", (outputs,))
        body.append(source)
        elements = max(elements, count)
        shapes.append((outputs,))
        bias_argument = "bias = b, "
    body.append(
        f'{om.tensor_type(y_shape)} y = conv({bias_argument}dilations = dl, '
        'groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)'
        '[name = string("y")];')
    mil = om.program(f"{om.tensor_type(x_shape)} x", body, "y")
    name = (f"{name_prefix}_k{kernel}_c{inputs}_n{outputs}_s{spatial}"
            f"_st{stride}_g{groups}_bias{int(bias)}_{pad_type}")
    parameters = {
        "kernel": kernel, "input_channels": inputs, "output_channels": outputs,
        "spatial": spatial, "stride": stride, "groups": groups, "bias": bias,
        "pad_type": pad_type, "input_shape": x_shape, "output_shape": y_shape,
        "weight_shape": w_shape,
    }
    payload = weights if weights is not None else om.half_payload(elements)
    description: dict[str, Any] = {
        "storage": "BLOBFILE", "shapes": shapes,
        "payload_bytes": elements * 2,
        "value": "distinct" if weights is not None else "fp16(0x1p-1)",
    }
    return om.case(name, "conv_probe", parameters, mil, om.blob(payload),
                   description)


def bit_planes(elements: int) -> int:
    """How many bit-plane probes name every element of a constant."""
    return max(1, (elements - 1).bit_length())


def finite_index_payload(elements: int, seed: int = 3991) -> bytes:
    """Reproducible, nonzero finite fp16 values varying across every tap/core."""
    out = bytearray()
    state = seed
    for _ in range(elements):
        state = (1664525 * state + 1013904223) & 0xffffffff
        bits = ((state >> 16) & 0x8000) | ((8 + (state >> 26) % 7) << 10) | ((state >> 11) & 1023)
        out += struct.pack("<H", bits)
    return bytes(out)


def known_weights(elements: int, plane: int | None = None) -> bytes:
    """One bit-plane probe payload.

    `plane` is the index bit the probe flips, or None for the baseline where
    every element holds `LOW_PATTERN`. Reading a byte of the constant section
    across the baseline and every plane recovers which element and which half
    of it that byte holds.
    """
    if plane is None:
        return struct.pack(f"<{elements}H", *([LOW_PATTERN] * elements))
    if plane < 0:
        return struct.pack(f"<{elements}H", *([HIGH_PATTERN] * elements))
    return struct.pack(
        f"<{elements}H",
        *(HIGH_PATTERN if index >> plane & 1 else LOW_PATTERN
          for index in range(elements)))


def grid() -> list[dict[str, Any]]:
    """The probe grid. Cases whose weight payload exceeds 64 MiB are skipped:
    the campaign's compile budget, not a refusal by Apple."""
    cases: list[dict[str, Any]] = []
    channels = (64, 128, 256, 512, 1024)

    # 1. Kernel, stride and pad_type against the channel counts, at the
    # spatial extent the envelope already covered, so the grid separates a
    # channel-driven word from a spatial one.
    for kernel in (1, 3):
        for inputs in channels:
            for outputs in channels:
                for stride in (1, 2):
                    for pad_type in ("same", "valid"):
                        if output_spatial(16, kernel, stride, pad_type) < 1:
                            continue
                        if inputs * outputs * kernel * kernel * 2 > 64 << 20:
                            continue
                        cases.append(conv_case(kernel, inputs, outputs, 16,
                                               stride, 1, False, pad_type))

    # 2. The spatial sweep at fixed channels, with and without a bias.
    for kernel in (1, 3):
        for spatial in (1, 8, 16, 32, 64):
            for stride in (1, 2):
                for pad_type in ("same", "valid"):
                    if output_spatial(spatial, kernel, stride, pad_type) < 1:
                        continue
                    for bias in (False, True):
                        cases.append(conv_case(kernel, 64, 64, spatial, stride,
                                               1, bias, pad_type))
                        cases.append(conv_case(kernel, 256, 256, spatial,
                                               stride, 1, bias, pad_type))

    # 3. Depthwise, where the weight is [Cout, 1, kh, kw], and the groups=4
    # form the envelope already accepted, over the same channel counts.
    for kernel in (1, 3):
        for count in channels:
            for stride in (1, 2):
                for bias in (False, True):
                    cases.append(conv_case(kernel, count, count, 16, stride,
                                           count, bias))
                    cases.append(conv_case(kernel, count, count, 16, stride, 4,
                                           bias))

    # 4. Bias against the channel counts, which the envelope only probed at
    # 64, 256 and 768 output channels: the bias block's padding is a step
    # function of the output count, so every count in the grid is measured.
    for outputs in channels:
        for kernel in (1, 3):
            cases.append(conv_case(kernel, 64, outputs, 16, 1, 1, True))

    # 5. Known-weight probes. The permutation is read off these, so they stay
    # small enough that every element carries a distinct pattern.
    for kernel in (1, 3):
        for inputs, outputs in ((64, 64), (64, 128), (128, 64), (64, 256),
                                (256, 64)):
            for stride in (1, 2):
                for groups in (1, 4):
                    cases.append(known_weight_case(kernel, inputs, outputs, 16,
                                                   stride, groups, False))
                    cases.append(known_weight_case(kernel, inputs, outputs, 16,
                                                   stride, groups, True))
        # The row-group size is a function of the output count alone, so it is
        # probed at every count whose group the corpus needs: 768 outputs is
        # the envelope's widest convolution and the only count that is not a
        # power of two.
        for outputs in (512, 768, 1024):
            cases.append(known_weight_case(kernel, 64, outputs, 16, 1, 1, False))
            cases.append(known_weight_case(kernel, 64, outputs, 16, 1, 1, True))
        for count in (64, 128):
            cases.append(known_weight_case(kernel, count, count, 16, 1, count,
                                           False))
            cases.append(known_weight_case(kernel, count, count, 16, 2, count,
                                           True))

    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
        if not item["name"].startswith("conv_"):
            raise AssertionError(f"probe {item['name']} lacks the conv_ prefix")
    return list(unique.values())


def known_weight_case(kernel: int, inputs: int, outputs: int, spatial: int,
                      stride: int, groups: int, bias: bool,
                      pad_type: str = "same") -> dict[str, Any]:
    elements = outputs * (inputs // groups) * kernel * kernel
    if bias:
        elements = max(elements, outputs)
    return conv_case(kernel, inputs, outputs, spatial, stride, groups, bias,
                     pad_type, weights=known_weights(elements),
                     name_prefix="conv_known")


def campaign() -> list[dict[str, Any]]:
    return grid()


# --------------------------------------------------------------------------
# Constant sections
# --------------------------------------------------------------------------

def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def lane_cap(spatial: int) -> int:
    """The widest halfword interleave Apple uses for a dense convolution.

    Measured with known-weight probes at 512 outputs: an input surface of 8 or
    fewer pixels per side interleaves 32 output channels per plane, 9 or more
    interleaves 16. Both bounds are read off the row stride of the recovered
    permutation, and the boundary is pinned between spatial 8 and 9.
    """
    return 32 if spatial <= 8 else 16


def lane_chunks(outputs: int, cap: int) -> list[int]:
    """How Apple splits the output channels into interleave widths.

    Every plane group holds 16 planes, so the channels are consumed in chunks
    of `16 * lanes` where `lanes` is the largest power of two at or below
    `outputs / 16`, capped by :func:`lane_cap`. 768 outputs at cap 32 is the
    case that proves the split: 512 channels interleave 32 wide and the
    remaining 256 interleave 16 wide, in that order.
    """
    chunks = []
    remaining = outputs
    while remaining > 0:
        lanes = min(cap, 1 << ((remaining // 16).bit_length() - 1)) \
            if remaining >= 16 else 1
        chunks.append(lanes)
        remaining -= 16 * lanes
    return chunks


def pack_dense(reduction: int, outputs: int, chunks: list[int],
               weights: bytes, bias: bytes | None,
               plane_padding: bool) -> bytes:
    """Apple's dense convolution section.

    Each plane holds `lanes` output channels interleaved at halfword
    granularity over `reduction` rows, preceded by one bias row when the
    convolution has a bias. Sixteen plane groups follow, each holding one plane
    per chunk; a grouped convolution pads every plane to 64 bytes, a
    groups-1 convolution pads only the plane group.
    """
    rows = reduction + (bias is not None)
    plane_bytes = [align(rows * lanes * 2, 64) if plane_padding
                   else rows * lanes * 2 for lanes in chunks]
    group_bytes = align(sum(plane_bytes), 64)
    packed = bytearray(group_bytes * 16)
    start = 0
    for index, lanes in enumerate(chunks):
        offset = sum(plane_bytes[:index])
        for group in range(16):
            base = group * group_bytes + offset
            for lane in range(lanes):
                column = start + group * lanes + lane
                cursor = base + lane * 2
                if bias is not None:
                    packed[cursor:cursor + 2] = bias[column * 2:column * 2 + 2]
                    cursor += lanes * 2
                source = column * reduction * 2
                for _ in range(reduction):
                    packed[cursor:cursor + 2] = weights[source:source + 2]
                    cursor += lanes * 2
                    source += 2
        start += 16 * lanes
    return bytes(packed)


def pack_small_dense(reduction: int, outputs: int, weights: bytes, bias: bytes | None) -> bytes:
    """Balance adjacent output channels over up to 16 active cores.

    At c24, cores 0..7 interleave two consecutive channels each and cores
    8..15 carry one channel each. Each core is independently padded to 64.
    """
    cores = min(16, outputs)
    rows = reduction + (bias is not None)
    out = bytearray()
    first = 0
    for core in range(cores):
        lanes = outputs // cores + (core < outputs % cores)
        plane = bytearray(align(rows * lanes * 2, 64))
        for lane in range(lanes):
            cursor = lane * 2
            if bias is not None:
                plane[cursor:cursor+2] = bias[(first+lane)*2:(first+lane+1)*2]
                cursor += lanes * 2
            for row in range(reduction):
                source = ((first+lane)*reduction+row)*2
                plane[cursor:cursor+2] = weights[source:source+2]
                cursor += lanes * 2
        out += plane
        first += lanes
    return bytes(out)


def pack_dense_colgroups(reduction: int, outputs: int,
                         weights: bytes) -> bytes:
    """The captured F1 in-projection section (k1x1 g1 valid, 2048 outputs,
    1024 reduction, bias-free).

    16 regions x 8 slots; slot (g, j) is a 16-lane plane over `reduction`
    consecutive rows holding colgroup

        G(g, j) = 8 * (4 * (3 - g // 4) + ((g // 2) % 2) + 2 * (j % 2))
                  + 4 * (g % 2) + j // 2

    The order beyond colgroup 64 is invisible to any uint16 payload (e ->
    e + 64*1024 is value-invariant); it was decoded from the 32-bit-pair
    remint (receipts/2026-09-17-encoder-conv-lowering/oracles-u32) and is
    byte-exact against both that capture and the round-2 uint16 capture.
    """
    if (reduction, outputs) != (1024, 2048):
        raise ValueError("the colgroup-order dense capture is the F1 "
                         "in-projection geometry (1024 -> 2048)")
    regions, slots, lanes = 16, 8, 16
    packed = bytearray(regions * slots * reduction * lanes * 2)
    view = memoryview(packed).cast("H")
    for region in range(regions):
        for slot in range(slots):
            block = 4 * (3 - region // 4) + ((region // 2) % 2) \
                + 2 * (slot % 2)
            colgroup = 8 * block + 4 * (region % 2) + slot // 2
            base = region * slots * reduction * lanes + \
                slot * reduction * lanes
            for lane in range(lanes):
                source = (16 * colgroup + lane) * reduction
                cursor = base + lane
                view[cursor:cursor + reduction * lanes:lanes] = memoryview(
                    weights[source * 2:(source + reduction) * 2]).cast("H")
    return bytes(packed)


def writeout_inverse(slot: int) -> int:
    """The F3 engine write-out inverse: the ref run whose weights packer run
    ``slot`` must carry.

    The device emits the 1024 output planes in 64 runs of 16 under a fixed
    run permutation R: dev run k displays ref run

        R(k) = ((k >> 1) & 7) | (k & 0x10) | ((((~k) >> 5) & 1) << 3)
               | ((k & 1) << 5)

    (T6001 test host measurement, receipts/2026-09-18-encoder-conv-device-gate/
    f3-writeout-permutation.json). Exact numerics under the bijection mean
    the engine computes output run k from the weights packed at run slot
    R(k), so dev run k shows ref run k exactly when slot R(k) holds ref run
    k: slot p carries ref run

        g(p) = ((p >> 5) & 1) | ((p & 1) << 1) | (((p >> 1) & 1) << 2)
               | (((p >> 2) & 1) << 3) | (((p >> 4) & 1) << 4)
               | ((1 - ((p >> 3) & 1)) << 5)

    with within-run plane order preserved. The same engine class is what
    the m375 k1024 n1024 linear lowering already compensates (67dfcf1).
    """
    return (((slot >> 5) & 1)
            | ((slot & 1) << 1)
            | (((slot >> 1) & 1) << 2)
            | (((slot >> 2) & 1) << 3)
            | (((slot >> 4) & 1) << 4)
            | ((1 - ((slot >> 3) & 1)) << 5))


def pack_dense_writeout_inverse(reduction: int, outputs: int,
                                chunks: list[int], weights: bytes) -> bytes:
    """The F3 out-projection dense section (k1x1 g1 valid, 1024 outputs over
    1024 reduction, bias-free, [1, 1024, 1, 375] surfaces) with the engine
    write-out inverse decoded in: :func:`pack_dense` over weights whose
    16-channel runs sit at the slots the engine's write-out will read, so
    the device's planes land in manifest order.

    The record's captured section carries a uniform per-constant payload,
    so its bytes are invariant under the run placement and byte parity
    against the capture keeps holding; device execution is the oracle.
    """
    if (reduction, outputs) != (1024, 1024):
        raise ValueError("the write-out inverse is decoded only for the F3 "
                         "out-projection geometry (1024 -> 1024)")
    runs = outputs // 16
    run_halves = chunks[0] * reduction
    permuted = bytearray(len(weights))
    source = memoryview(weights).cast("H")
    view = memoryview(permuted).cast("H")
    for slot in range(runs):
        run = writeout_inverse(slot)
        view[slot * run_halves:(slot + 1) * run_halves] = \
            source[run * run_halves:(run + 1) * run_halves]
    return pack_dense(reduction, outputs, chunks, bytes(permuted), None,
                      False)


def pack_depthwise_slots(taps: int, outputs: int, weights: bytes,
                         bias: bytes | None) -> bytes:
    """The captured W-major depthwise section (k1x9 same, bias present).

    64 lanes x `outputs / 64` slots; slot s carries channel
    ``16 * (s % 64) + s // 64`` as 22 halfwords: the bias twice, then the
    nine taps in the decoded two-pass lane pairing with two structural
    zeros — ``b b t0 . t2 t1 t4 t3 t6 t5 t8 t7 t1 t0 t3 t2 t5 t4 t7 t6 . t8``.
    Verified against encoder_conv_idx_c1024_n1024_k1x9_s1_g1024_bias1_same_wmaj.
    """
    if taps != 9 or bias is None:
        raise ValueError("the slotted depthwise capture is k1x9 with a bias")
    lanes = 64
    slots = outputs // lanes
    slot_halves = len(_SLOT_ORDER)
    if slots * lanes != outputs:
        raise ValueError("the slotted depthwise capture packs 64-lane groups")
    packed = bytearray(slots * lanes * slot_halves * 2)
    halfs = memoryview(packed).cast("H")
    for s in range(lanes * slots):
        column = 16 * (s % lanes) + s // lanes
        bias_word = struct.unpack_from("<H", bias, column * 2)[0]
        base = s * slot_halves
        halfs[base] = bias_word
        halfs[base + 1] = bias_word
        for offset, k in enumerate(_SLOT_ORDER):
            if k is not None:
                halfs[base + offset] = struct.unpack_from(
                    "<H", weights, (column * taps + k) * 2)[0]
    return bytes(packed)


# The decoded slot template: None marks a structural zero, numbers name the
# channel's tap index within the 9-tap kernel.
_SLOT_ORDER = [None, None, 0, None, 2, 1, 4, 3, 6, 5, 8, 7,
               1, 0, 3, 2, 5, 4, 7, 6, None, 8]


def pack_padconv_columns(weights: bytes) -> bytes:
    """Apple's custom-pad depthwise section (k1x1 grouped, asymmetric W pad).

    Three 512-byte chunks of eight 64-byte lanes. Chunks one and two repeat
    the structural constants — even lanes 0x3c00 (fp16 1.0), odd lanes
    0x0001 then 0x0100 — and chunk three carries one raw weight halfword per
    lane in channel order. Verified against
    encoder_conv_pad_c8_n8_k1x1_s1_g8_bias0_p0010_f4
    and ..._tiny plus three same-format payload runs
    (receipts/2026-09-17-encoder-padconv-respell/): ascending payloads
    (1..8, 2..16) prove the identity lane order and that the structural
    chunks never carry weight bytes.
    """
    outputs = len(weights) // 2
    if outputs != 8:
        raise ValueError("the custom-pad depthwise capture is the 8-channel "
                         "grouped 1x1 padconv")
    lanes, lane_bytes = 8, 64
    chunk = lanes * lane_bytes
    packed = bytearray(chunk * 3)
    for lane in range(lanes):
        if lane % 2 == 0:
            struct.pack_into("<H", packed, lane * lane_bytes, 0x3C00)
            struct.pack_into("<H", packed, chunk + lane * lane_bytes, 0x3C00)
        else:
            struct.pack_into("<H", packed, lane * lane_bytes, 0x0001)
            struct.pack_into("<H", packed, lane * lane_bytes + 2, 0x0100)
            struct.pack_into("<H", packed, chunk + lane * lane_bytes, 0x0001)
            struct.pack_into("<H", packed, chunk + lane * lane_bytes + 2,
                             0x0100)
        packed[2 * chunk + lane * lane_bytes:2 * chunk + lane * lane_bytes + 2] \
            = weights[2 * lane:2 * lane + 2]
    return bytes(packed)


def pack_depthwise(taps: int, outputs: int, weights: bytes,
                   bias: bytes | None) -> bytes:
    """Apple's depthwise section: 16 lanes, each holding `outputs / 16`
    channels back to back with no padding between them, the lane padded to 64
    bytes. A bias precedes each channel's taps."""
    lanes = min(16, outputs)
    rows = taps + (bias is not None)
    lane_bytes = [align(((outputs - lane + lanes - 1) // lanes) * rows * 2, 64)
                  for lane in range(lanes)]
    lane_bases = [sum(lane_bytes[:lane]) for lane in range(lanes)]
    packed = bytearray(sum(lane_bytes))
    for column in range(outputs):
        cursor = lane_bases[column % lanes] + (column // lanes) * rows * 2
        if bias is not None:
            packed[cursor:cursor + 2] = bias[column * 2:column * 2 + 2]
            cursor += 2
        source = column * taps * 2
        packed[cursor:cursor + taps * 2] = weights[source:source + taps * 2]
    return bytes(packed)


def pack_strided(reduction: int, outputs: int, chunks: list[int],
                 weights: bytes, bias: bytes | None, target: str) -> bytes:
    """Apple's stride-2 section, which skips zero weights.

    A plane holds, in order: the plane's `lanes` bias halfwords when the
    convolution has a bias, a 16-bit count of the body bytes, then one row per
    reduction step. A row carries, per group of eight lanes, a mask byte whose
    bit `l` marks a lane whose weight is not zero, followed by those lanes'
    halfwords, and closes with `lanes / 2 - 2` zero bytes. H13 leads each row
    with a zero byte and H14 instead leads the whole body with one and follows
    every mask byte with a zero -- the same body count either way. Both signed
    zeros are skipped, so the section's size depends on the weight values.
    """
    packed = bytearray()
    start = 0
    for lanes in chunks:
        padding = max(0, lanes // 2 - 2)
        for group in range(16):
            first = start + group * lanes
            body = bytearray()
            count = 0
            if target == "h14":
                body.append(0)
            for row in range(reduction):
                if target == "h13":
                    body.append(0)
                count += 1
                for base in range(0, lanes, 8):
                    mask = 0
                    values = bytearray()
                    for lane in range(base, min(base + 8, lanes)):
                        index = ((first + lane) * reduction + row) * 2
                        value = weights[index:index + 2]
                        if int.from_bytes(value, "little") & 0x7FFF:
                            mask |= 1 << (lane - base)
                            values += value
                    body.append(mask)
                    if target == "h14":
                        body.append(0)
                    count += 1 + len(values)
                    body += values
                body += bytes(padding)
                count += padding
            plane = bytearray()
            if bias is not None:
                plane += bias[first * 2:(first + lanes) * 2]
            if count > 0xFFFF:
                raise ValueError(
                    "strided convolution plane exceeds a 16-bit body count")
            plane += count.to_bytes(2, "little") + body
            packed += plane + bytes(align(len(plane), 64) - len(plane))
        start += 16 * lanes
    return bytes(packed)


def weight_taps(parameters: dict[str, Any]) -> int:
    """Total kernel taps, from the recorded weight shape: rectangular
    kernels carry kh * kw taps, which the square-probe era spelled as
    ``kernel ** 2``."""
    weight_shape = parameters.get("weight_shape")
    if weight_shape and len(weight_shape) == 4:
        return weight_shape[2] * weight_shape[3]
    return parameters["kernel"] ** 2


def conv_constants(parameters: dict[str, Any], weights: bytes,
                   bias: bytes | None, target: str = "h13",
                   size: int | None = None) -> bytes:
    """The constant section for one convolution, from the resolved blobs."""
    inputs = parameters["input_channels"]
    outputs = parameters["output_channels"]
    groups = parameters["groups"]
    taps = weight_taps(parameters)
    reduction = (inputs // groups) * taps
    if (parameters["kernel"], inputs, outputs, parameters["spatial"],
        parameters["stride"], groups, parameters.get("pad_type")) in (
            (3,128,512,35,1,1,"same"),(3,512,128,35,1,1,"same")) and taps == 9 and \
            tuple(parameters.get("input_shape", ())) == (1,inputs,35,35) and \
            tuple(parameters.get("output_shape", ())) == (1,outputs,35,35):
        # Native 35x35 schedules write the upper output slice first (256+256
        # for gather, 32+96 for mix), each in a contiguous core section.
        parts = ((256,256),(0,256)) if outputs==512 else ((96,32),(0,96))
        return b"".join(pack_small_dense(reduction,count,
            weights[first*reduction*2:(first+count)*reduction*2],
            bias[first*2:(first+count)*2] if bias is not None else None)
            for first,count in parts)
    if groups == inputs and groups == outputs:
        weight_shape = parameters.get("weight_shape")
        if not bias and taps == 1 and weight_shape and \
                len(weight_shape) == 4 and weight_shape[2] == 1 and \
                weight_shape[3] == 1 and \
                parameters.get("pad_type") == "custom" and \
                any(parameters.get("pad") or ()):
            # The encoder's W-padded rel-pos padconv: a two-task program that
            # reads the unpadded input directly.
            section = pack_padconv_columns(weights)
        elif bias is not None and taps > 1 and weight_shape and \
                len(weight_shape) == 4 and weight_shape[2] == 1 and \
                weight_shape[3] == taps:
            # The W-major depthwise respell (unit height kernel) records a
            # 64-lane slotted section with a duplicated bias; the square and
            # H-major captures keep the classic per-channel lane form.
            section = pack_depthwise_slots(taps, outputs, weights, bias)
        else:
            section = pack_depthwise(taps, outputs, weights, bias)
    elif parameters["stride"] > 1:
        if outputs % 16:
            raise ValueError("small dense outputs have no qualified strided packing")
        chunks = lane_chunks(outputs, lane_cap(parameters["spatial"]))
        if groups != 1 or taps != 1 or max(chunks) > 8:
            raise ValueError(
                "no derived strided packing for grouped, multi-tap, or "
                "16-lane convolutions")
        section = pack_strided(reduction, outputs, chunks, weights, bias,
                               target)
    elif groups > 1:
        lanes = max(1, outputs // 64)
        section = pack_dense(reduction, outputs,
                             [lanes] * (outputs // lanes // 16),
                             weights, bias, True)
    else:
        if outputs % 16:
            return pack_small_dense(reduction, outputs, weights, bias)
        chunks = lane_chunks(outputs, lane_cap(max(
            parameters["spatial"],
            parameters.get("spatial_width", parameters["spatial"]))))
        if bias is None and (outputs, reduction) == (2048, 1024) \
                and chunks == [16] * 8:
            # The decoded F1 in-projection colgroup order; consecutive order
            # never reproduced past colgroup 64 (see pack_dense_colgroups).
            section = pack_dense_colgroups(reduction, outputs, weights)
        elif bias is None and (outputs, reduction) == (1024, 1024) \
                and chunks == [16] * 4 \
                and parameters.get("spatial_width") == 375:
            # The decoded F3 out-projection write-out inverse; the engine
            # readout permutation is measured only for this geometry (the
            # H14 16x16 probes with the same channel counts keep identity
            # order). See pack_dense_writeout_inverse.
            section = pack_dense_writeout_inverse(reduction, outputs, chunks,
                                                  weights)
        else:
            section = pack_dense(reduction, outputs, chunks, weights, bias,
                                 False)
    if size is not None and len(section) != size:
        raise ValueError(f"packed section is {len(section)} bytes, "
                         f"the oracle records {size}")
    return section


def blob_view(payload: bytes, elements: int) -> bytes:
    """The bytes a `weights.bin` holding `payload` resolves to for `elements`
    fp16 values.

    `mint_oracles.blob` writes its chunk header with unpadded fields, so both
    Apple's loader and `ANEBlobResolver` read the tensor from file offset 0:
    the first 64 elements of every constant are the 128-byte blob header, and
    the payload begins at element 64. Reproducing those header halfwords in
    the right places is what proves the packing rather than the payload value.
    """
    return om.blob(payload)[:elements * 2]


def _split_payload(parameters: dict[str, Any],
                   flat: bytes) -> tuple[bytes, bytes | None]:
    """Region 0 is the weight, region 1 the bias when the case carries one."""
    outputs = parameters["output_channels"]
    taps = weight_taps(parameters)
    weight_elements = outputs * \
        (parameters["input_channels"] // parameters["groups"]) * taps
    weights = flat[:weight_elements * 2]
    bias = flat[weight_elements * 2:
                (weight_elements + outputs) * 2] \
        if parameters["bias"] else None
    return weights, bias


def case_weights(record: dict[str, Any]) -> tuple[bytes, bytes | None]:
    """The weight and bias bytes the recorded case compiled against."""
    parameters = record["parameters"]
    description = record["weights"]
    elements = description["payload_bytes"] // 2
    outputs = parameters["output_channels"]
    if description.get("value") == \
            "fp16 bits 0x3400 + index, one value per constant":
        # The encoder-leftover campaign's weight blob: one uniform fp16
        # region per constant at VALUE_BASE + ordinal, laid out after a
        # 24-byte record table with no inline header. The description's
        # sha256 records the payload the mint generated; the sections these
        # records decode carry the uniform bytes the compiler actually
        # consumed, and the constant-section comparison in covered() is
        # what adjudicates that.
        flat = b"".join(
            struct.pack("<H", (0x3400 + index) & 0xFFFF) *
            math.prod(shape)
            for index, shape in enumerate(description["shapes"]))
        if len(flat) != elements * 2:
            raise ValueError(
                f"{record['case']} payload regions cover {len(flat)} bytes, "
                f"the description records {elements * 2}")
        return _split_payload(parameters, flat)
    if description.get("pattern") == "uint16_le_index_plus_one_wrapping":
        # The round-4 remint: one distinct uint16(index + 1) running payload
        # spanning every constant region, no inline header — the packing
        # permutation is fully visible in the recorded section.
        flat = bytearray()
        value = 0
        for shape in description["shapes"]:
            count = math.prod(shape)
            flat += struct.pack(
                f"<{count}H", *[(value + index + 1) & 0xFFFF
                                for index in range(count)])
            value += count
        if len(flat) != elements * 2:
            raise ValueError(
                f"{record['case']} payload regions cover {len(flat)} bytes, "
                f"the description records {elements * 2}")
        return _split_payload(parameters, bytes(flat))
    if description["value"] == "indexed_finite":
        payload = finite_index_payload(elements, description["seed"])
    else:
        payload = known_weights(elements) if description["value"] == "distinct" \
            else om.half_payload(elements)
    resolved = blob_view(payload, elements)
    taps = weight_taps(parameters)
    weight_elements = outputs * \
        (parameters["input_channels"] // parameters["groups"]) * taps
    weights = resolved[:weight_elements * 2]
    bias = resolved[:outputs * 2] if parameters["bias"] else None
    return weights, bias


# --------------------------------------------------------------------------
# Template emission
# --------------------------------------------------------------------------

def h13_task_words(task: dict[str, Any]) -> list[int]:
    """Rebuild one H13 task descriptor's words from its decoded form."""
    words = [int(word, 16) for word in task["header_words"]]
    registers = {int(address, 16): int(value, 16)
                 for block in task["blocks"].values()
                 for address, value in block["words"].items()}
    for record in task["records"]:
        base = int(record["address"], 16)
        words.append(int(record["header"], 16))
        words.extend(registers[base + index * 4]
                     for index in range(record["count"]))
    if len(words) * 4 != task["size_bytes"]:
        raise ValueError("rebuilt task size differs from the decoded size")
    return words


def h14_task_words(task: dict[str, Any]) -> list[int]:
    """Rebuild one H14 task's words. An H14 record either carries a bitmap of
    the registers it writes above its base or a dense run, so the addresses
    come from the record header rather than from the count alone."""
    words = [int(word, 16) for word in task["header_words"]]
    registers = {int(address, 16): int(value, 16)
                 for block in task["blocks"].values()
                 for address, value in block["words"].items()}
    for record in task["records"]:
        header = int(record["header"], 16)
        base = (header & 0x7FFF) * 4
        if header & 0x80000000:
            mask = (header >> 15) & 0xFFFF
            addresses = (base,) + tuple(
                base + (bit + 1) * 4 for bit in range(16) if mask & (1 << bit))
        else:
            addresses = tuple(
                base + index * 4
                for index in range(((header >> 15) & 0x3F) + 1))
        if len(addresses) != record["count"]:
            raise ValueError(f"record {record['header']} address count differs")
        words.append(header)
        words.extend(registers[address] for address in addresses)
    if len(words) * 4 != task["size_bytes"]:
        raise ValueError("rebuilt task size differs from the decoded size")
    return words


def stream_words(record: dict[str, Any]) -> list[int]:
    """The words a template carries: one H13 task descriptor, or the whole H14
    __TEXT/__text stream with its zero-size prefix frame. A linked H13 task
    array rebuilds with its alignment padding, exactly as the matvec emitter
    assembles one."""
    tasks = record["task_descriptors"]
    target = record["target"]
    rebuild = h13_task_words if target == "h13" else h14_task_words
    if target == "h13":
        words: list[int] = []
        for index, task in enumerate(tasks):
            rebuilt = rebuild(task)
            if decode_task(struct.pack(f"<{len(rebuilt)}I", *rebuilt),
                           target) != task:
                raise ValueError(
                    f"{record['case']} task {index} does not round-trip")
            words.extend(rebuilt)
            next_offset = int(task["header_words"][7], 16)
            if index + 1 == len(tasks):
                if next_offset:
                    raise ValueError(
                        f"{record['case']} final task links onward")
                continue
            if next_offset % 4 or next_offset < len(words) * 4:
                raise ValueError(
                    f"{record['case']} task {index} link is invalid")
            words.extend([0] * (next_offset // 4 - len(words)))
        return words
    stream = [0] * (H14_STREAM_PREFIX_BYTES // 4)
    for index, task in enumerate(tasks):
        rebuilt = rebuild(task)
        if decode_task(struct.pack(f"<{len(rebuilt)}I", *rebuilt), target) != task:
            raise ValueError(f"{record['case']} H14 task {index} does not round-trip")
        stream.extend(rebuilt)
        if index + 1 < len(tasks):
            stream.extend([0] * ((-len(stream)) % 4))
    declared = record["program_descriptor"]["text_words"]
    if len(stream) != declared:
        raise ValueError(f"{record['case']} assembled {len(stream)} text "
                         f"words, the descriptor declares {declared}")
    return stream


def surface(channels: int, height: int, width: int) -> dict[str, Any]:
    """A convolution surface, exactly as the encoders build it.

    The row is the width padded to 64 bytes, not merely floored at 64: a
    `pad_type="valid"` convolution produces odd widths, and 62 columns take a
    128-byte row rather than 124.
    """
    row = align(width * 2, ROW_FLOOR)
    plane = row * height
    element = plane * channels
    return {"shape": [1, channels, height, width],
            "strides": [element, plane, row, 2], "total": element,
            "allocation": align(element, SURFACE_ALIGNMENT)}


def check_surfaces(record: dict[str, Any],
                   surfaces: list[dict[str, Any]]) -> int:
    """Compares the encoder's surfaces and addresses with the oracle's and
    returns the scratch allocation Apple placed below them."""
    case = record["case"]
    decoded = record["tensor_descriptors"]
    if len(decoded) != len(surfaces):
        raise SystemExit(f"{case} has {len(decoded)} surfaces, the encoder "
                         f"builds {len(surfaces)}")
    for index, (tensor, expected) in enumerate(zip(decoded, surfaces)):
        actual = {key: value for key, value in tensor.items()
                  if key != "binding"}
        wanted = {"element_code": 5, "shape": expected["shape"],
                  "strides": expected["strides"],
                  "total_bytes": expected["total"]}
        if actual != wanted:
            raise SystemExit(f"{case} surface {index} is {actual}, the "
                             f"encoder formula gives {wanted}")
        role = 2 if index + 1 == len(surfaces) else 1
        if tensor["binding"] != role:
            raise SystemExit(f"{case} surface {index} binds as "
                             f"{tensor['binding']}, expected {role}")
    text = int(record["program_descriptor"]["text_address"], 16)
    scratch = text - SURFACE_BASE - sum(item["allocation"] for item in surfaces)
    if scratch < 0:
        raise SystemExit(f"{case} surfaces do not fit below its text address")
    cursor = SURFACE_BASE + scratch
    addresses = []
    for item in surfaces:
        addresses.append(cursor)
        cursor += item["allocation"]
    decoded_addresses = [
        int(value, 16) for value in
        record["program_descriptor"]["resource_addresses"][:len(surfaces)]]
    if decoded_addresses != addresses:
        raise SystemExit(
            f"{case} surface addresses are "
            f"{[hex(value) for value in decoded_addresses]}, the encoder's "
            f"order gives {[hex(value) for value in addresses]}")
    return scratch


def constant_offset(record: dict[str, Any], words: list[int]) -> int:
    descriptor = record["program_descriptor"]
    offset = int(descriptor["constant_address"], 16) - \
        int(descriptor["text_address"], 16)
    if offset != align(len(words) * 4, CONSTANT_ALIGNMENT):
        raise SystemExit(f"{record['case']} constant offset {offset} is not "
                         "the aligned end of its task stream")
    return offset


def template_key(record: dict[str, Any]) -> tuple:
    """What the compiler knows before it picks a program. The padding follows
    from the kernel, stride and the two surfaces, so `pad_type` never enters
    the key: a 1x1 convolution spells `same` and `valid` identically. Both
    kernel extents enter the key: a 9x1 and a 1x9 convolution are different
    programs even where a scalar kernel plus the surface pair would not
    collide."""
    parameters = record["parameters"]
    weight_shape = parameters["weight_shape"]
    return (weight_shape[2], weight_shape[3], parameters["stride"],
            parameters["groups"], parameters["bias"],
            tuple(parameters["input_shape"][1:]),
            tuple(parameters["output_shape"][1:]))


# Whole-program captures whose Apple task stream links several tasks. Every
# other decoded convolution stays one task: Apple partitions large
# convolutions into task arrays the single-task templates cannot carry, and
# those stay outside the tables until a capture qualifies them by name.
MULTI_TASK_CASES = frozenset({
    "encoder_conv_c1024_n1024_k1x1_s1_g1_bias0_valid",
    "encoder_conv_c1024_n2048_k1x1_s1_g1_bias0_valid",
    "encoder_conv_idx_c1024_n1024_k1x9_s1_g1024_bias1_same_wmaj",
    "encoder_conv_pad_c8_n8_k1x1_s1_g8_bias0_p0010_f4",
    "encoder_conv_pad_c8_n8_k1x1_s1_g8_bias0_p0010_tiny",
    "encoder_conv_idx_c1024_n2048_k1x1_s1_g1_bias0_valid_wmaj",
})


def covered(record: dict[str, Any], allow_multi_task: bool = False) -> bool:
    """Whether the encoders reproduce this oracle's constant section.

    The first campaign's convolutions predate the `groups` and `stride`
    parameters, which it never varied, so both default to one.
    """
    if record.get("error") is not None:
        return False
    tasks = record.get("task_descriptors") or []
    if not tasks or (len(tasks) != 1 and not allow_multi_task):
        return False
    parameters = record["parameters"]
    if "input_shape" not in parameters or "weight_shape" not in parameters:
        # The first campaign's convolutions predate the shape, stride and
        # group parameters an encoder template is keyed on.
        return False
    weights, bias = case_weights(record)
    try:
        section = conv_constants(record["parameters"], weights, bias,
                                 record["target"])
    except ValueError:
        return False
    return len(section) == record["constant_section"]["size"] and \
        hashlib.sha256(section).hexdigest() == \
        record["constant_section"]["sha256"]


def native_camera_multi(record):
    p = record.get("parameters", {})
    geometry = (p.get("kernel"),p.get("input_channels"),p.get("output_channels"),
                p.get("spatial"),p.get("stride"),p.get("groups"),p.get("pad_type"))
    return record["case"].startswith("layer_conv") and geometry in (
        (3,128,512,35,1,1,"same"),(3,512,128,35,1,1,"same"))


def conv_records(target: str) -> dict[tuple, dict[str, Any]]:
    selected: dict[tuple, dict[str, Any]] = {}
    for path in sorted((ROOT / "research/oracles" / target).glob("*.json")):
        record = json.loads(path.read_text())
        parameters = record.get("parameters") or {}
        if "kernel" not in parameters or "weight_shape" not in parameters:
            continue
        record.setdefault("target", target)
        if not covered(record, allow_multi_task=
                       record["case"] in MULTI_TASK_CASES or native_camera_multi(record)):
            continue
        key = template_key(record)
        previous = selected.setdefault(key, record)
        if previous is record:
            continue
        for field in ("task_descriptors", "program_descriptor",
                      "tensor_descriptors"):
            if previous[field] != record[field]:
                raise SystemExit(
                    f"{record['case']} and {previous['case']} share a template "
                    f"key but differ in {field}")
    return selected


def word_rows(words: list[int]) -> list[str]:
    return [f"    {', '.join(f'0x{word:08x}' for word in words[start:start + 8])},"
            for start in range(0, len(words), 8)]


def check_tensors(record: dict[str, Any],
                  surfaces: list[dict[str, Any]]) -> None:
    """Compares the encoder's surfaces with the oracle's tensor descriptors."""
    decoded = record["tensor_descriptors"]
    if len(decoded) != len(surfaces):
        raise SystemExit(f"{record['case']} has {len(decoded)} surfaces, the "
                         f"encoder builds {len(surfaces)}")
    for index, (tensor, expected) in enumerate(zip(decoded, surfaces)):
        actual = {key: value for key, value in tensor.items()
                  if key != "binding"}
        wanted = {"element_code": 5, "shape": expected["shape"],
                  "strides": expected["strides"],
                  "total_bytes": expected["total"]}
        if actual != wanted:
            raise SystemExit(f"{record['case']} surface {index} is {actual}, "
                             f"the encoder formula gives {wanted}")
        role = 2 if index + 1 == len(surfaces) else 1
        if tensor["binding"] != role:
            raise SystemExit(f"{record['case']} surface {index} binds as "
                             f"{tensor['binding']}, expected {role}")


def conv_row(record: dict[str, Any], symbol: str, words: list[int]) -> str:
    parameters = record["parameters"]
    descriptor = record["program_descriptor"]
    _, channels, height, width = parameters["input_shape"]
    _, out_channels, out_height, out_width = parameters["output_shape"]
    surfaces = [surface(channels, height, width),
                surface(out_channels, out_height, out_width)]
    weights, bias = case_weights(record)
    section = conv_constants(parameters, weights, bias, record["target"],
                             record["constant_section"]["size"])
    weight_shape = parameters["weight_shape"]
    geometry = (f"{weight_shape[2]}, {weight_shape[3]}, "
                f"{parameters['stride']}, "
                f"{parameters['groups']}, "
                f"{'true' if parameters['bias'] else 'false'}, "
                f"{{{channels}, {height}, {width}}}, "
                f"{{{out_channels}, {out_height}, {out_width}}}")
    if record["target"] == "h13":
        scratch = check_surfaces(record, surfaces)
        padding=parameters.get("pad") or []
        if parameters.get("pad_type")=="custom" and padding:
            top,left=(padding[0],padding[2]) if len(padding)==4 else (0,padding[0])
        elif parameters.get("pad_type")=="valid": top,left=0,0
        else:
            top=max(0,(out_height-1)*parameters["stride"]+weight_shape[2]-height)//2
            left=max(0,(out_width-1)*parameters["stride"]+weight_shape[3]-width)//2
        native=str(record["case"].startswith("layer_conv")).lower()
        return (f"    {{{geometry}, {symbol}, std::size({symbol}), "
                f"{(descriptor['task_words_minus_one'] + 1) * 4}, "
                f"{descriptor['task_count']}, "
                f"{constant_offset(record, words)}, "
                f"{len(section)}, {scratch}, {top}, {left}, {native}}},  // {record['case']}")
    check_tensors(record, surfaces)
    trailer = descriptor["trailing_words"]
    offset = int(descriptor["constant_address"], 16) - \
        int(descriptor["text_address"], 16)
    if offset != align(len(words) * 4, CONSTANT_ALIGNMENT):
        raise SystemExit(f"{record['case']} constant offset {offset} is not "
                         "the aligned end of its task stream")
    return (f"    {{{geometry}, {symbol}, std::size({symbol}), "
            f"{descriptor['task_count']}, {len(section)}, "
            f"0x{int(trailer[20], 16):08x}, 0x{int(trailer[28], 16):08x}, "
            f"0x{int(trailer[18], 16):08x}, "
            f"{str(record['case'].startswith('layer_conv')).lower()}}},  // {record['case']}")


def emit(target: str) -> str:
    lines = [
        f"// Generated from decoded Apple {target.upper()} convolution oracle "
        "task words by",
        "// research/mint_conv_probes.py --emit-templates. No HWX container",
        "// bytes; regenerate after re-minting the oracles.",
        "",
    ]
    symbols: dict[tuple[int, ...], str] = {}
    body: list[str] = []
    rows = []
    for key, record in sorted(conv_records(target).items()):
        words = stream_words(record)
        symbol = symbols.get(tuple(words))
        if symbol is None:
            symbol = f"k{target.upper()}ConvText{len(symbols)}"
            symbols[tuple(words)] = symbol
            body.append(f"static constexpr std::uint32_t {symbol}[] = {{")
            body.extend(word_rows(words))
            body.append("};")
        rows.append(conv_row(record, symbol, words))
    if not rows:
        raise SystemExit(f"no covered {target} convolution oracles found")
    lines.extend(body)
    lines.append("static constexpr OracleConvTemplate kConvTasks[] = {")
    lines.extend(rows)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def emit_templates(args: argparse.Namespace) -> int:
    for target in args.targets:
        generated = emit(target)
        path = Path(OUTPUTS[target])
        if args.check:
            if (path.read_text() if path.exists() else "") != generated:
                raise SystemExit(f"{path} is stale; regenerate it with "
                                 "research/mint_conv_probes.py --emit-templates")
            print(f"{target.upper()} conv templates: up to date ({path})")
            continue
        path.write_text(generated)
        print(f"wrote {path} ({len(generated)} bytes)")
    return 0


# --------------------------------------------------------------------------
# Layout probing
# --------------------------------------------------------------------------

def constant_bytes(data: bytes) -> bytes:
    """The __TEXT/__const body of an HWX image, in memory only."""
    _, _, _, _, count, command_bytes, _, _ = struct.unpack_from("<8I", data)
    cursor = 32
    for _ in range(count):
        command, size = struct.unpack_from("<2I", data, cursor)
        if command == 0x19:
            fields = struct.unpack_from("<2I16s4Q4I", data, cursor)
            segment = om.cstring(fields[2])
            section_cursor = cursor + 72
            for _ in range(fields[-2]):
                entry = struct.unpack_from("<16s16s2Q8I", data, section_cursor)
                if (segment, om.cstring(entry[0])) == ("__TEXT", "__const"):
                    return data[entry[4]:entry[4] + entry[3]]
                section_cursor += 80
        cursor += size
    raise ValueError("HWX has no __TEXT/__const section")


def compile_case(item: dict[str, Any], target: str, tool: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="mil-hwx-conv-layout-") as root:
        root_path = Path(root)
        capture = root_path / "capture"
        compiled = root_path / "compiled"
        capture.mkdir()
        compiled.mkdir()
        (capture / "model.mil").write_text(item["mil"])
        (capture / "weights.bin").write_bytes(item["weights"])
        result = subprocess.run([str(tool), str(capture), str(compiled), target],
                                capture_output=True, text=True,
                                timeout=om.COMPILE_TIMEOUT_SECONDS, check=False)
        image = compiled / "model.hwx"
        if result.returncode or not image.is_file():
            raise SystemExit(f"{item['name']} {target}: "
                             f"{(result.stderr or result.stdout).strip()}")
        return image.read_bytes()


def layout_report(item: dict[str, Any], baseline: bytes, saturated: bytes,
                  planes: list[bytes]) -> dict[str, Any]:
    """What one case's bit-plane probes say about the packing.

    `sources[p]` names what byte `p` of the constant section holds: `[element,
    half]` for the low or high byte of a weight element, or the baseline byte
    value for structural material the compiler wrote itself. Nothing but this
    derived map, the section sizes and the hashes leaves the machine.
    """
    sizes = {len(baseline), len(saturated)} | {len(plane) for plane in planes}
    if len(sizes) != 1:
        raise SystemExit(f"{item['name']} probe sections differ in size: {sizes}")
    low = (LOW_PATTERN & 0xFF, LOW_PATTERN >> 8)
    high = (HIGH_PATTERN & 0xFF, HIGH_PATTERN >> 8)
    sources: list[Any] = []
    for position in range(len(baseline)):
        if saturated[position] == baseline[position]:
            sources.append(baseline[position])
            continue
        element = sum(1 << plane for plane, image in enumerate(planes)
                      if image[position] != baseline[position])
        half = low.index(baseline[position]) \
            if baseline[position] in low else None
        if half is None or saturated[position] != high[half]:
            sources.append([None, f"0x{baseline[position]:02x}"])
        else:
            sources.append([element, half])
    return {
        "case": item["name"],
        "parameters": item["parameters"],
        "size": len(baseline),
        "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
        "sources": sources,
    }


def probe_geometry(specification: str) -> dict[str, Any]:
    """One known-weight case from a
    `kernel,inputs,outputs,spatial,stride,groups,bias,pad_type` spelling, so a
    layout probe can reach a geometry the grid does not carry."""
    fields = specification.split(",")
    if len(fields) != 8:
        raise SystemExit("--probe-case takes "
                         "kernel,inputs,outputs,spatial,stride,groups,bias,pad")
    kernel, inputs, outputs, spatial, stride, groups = map(int, fields[:6])
    return known_weight_case(kernel, inputs, outputs, spatial, stride, groups,
                             bool(int(fields[6])), fields[7])


def layout_run(args: argparse.Namespace) -> int:
    if platform.system() != "Darwin":
        raise SystemExit("--layout --local requires macOS")
    tool = Path(args.oracle_tool)
    if args.probe_case:
        selected = [probe_geometry(specification)
                    for specification in args.probe_case]
    else:
        selected = [item for item in campaign()
                    if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if not selected:
        raise SystemExit("no case matched --case")
    reports = []
    for target in args.targets:
        for item in selected:
            elements = item["weights_description"]["payload_bytes"] // 2

            def section(plane: int | None) -> bytes:
                probe = dict(item)
                probe["weights"] = om.blob(known_weights(elements, plane))
                return constant_bytes(compile_case(probe, target, tool))

            report = layout_report(
                item, section(None), section(-1),
                [section(plane) for plane in range(bit_planes(elements))])
            report["target"] = target
            reports.append(report)
            structural = sum(1 for entry in report["sources"]
                             if not isinstance(entry, list))
            print(f"{target} {item['name']} size={report['size']} "
                  f"structural={structural}", flush=True)
    Path(args.layout_output).write_text(json.dumps(reports) + "\n")
    print(f"wrote {args.layout_output} ({len(reports)} reports)")
    return 0


# --------------------------------------------------------------------------
# Minting
# --------------------------------------------------------------------------

def local_run(args: argparse.Namespace) -> int:
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    tool = Path(args.oracle_tool)
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)} targets={len(args.targets)}")
        return 0
    output = Path(args.output)
    decoded = rejected = 0
    for target in args.targets:
        for item in selected:
            destination = output / target / f"{item['name']}.json"
            if destination.exists() and not args.force:
                existing = json.loads(destination.read_text())
                decoded += existing.get("error") is None
                rejected += existing.get("error") is not None
                continue
            status = om.run_case(item, target, output, tool, args.source_commit)
            decoded += status == "decoded"
            rejected += status == "rejected"
            print(f"{target} {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected) * len(args.targets)} "
          f"decoded={decoded} rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-conv-probes.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-conv-probes."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    try:
        subprocess.run(["scp", "-q", str(script),
                        str(script.with_name("mint_oracles.py")),
                        str(script.with_name("h13_td.py")),
                        f"{args.host}:{root}/"], check=True)
        command = ["python3", f"{root}/{script.name}", "--local",
                   "--oracle-tool", args.oracle_tool,
                   "--output", f"{root}/oracles",
                   "--layout-output", f"{root}/layout.json",
                   "--source-commit", args.source_commit,
                   "--targets", *args.targets]
        if args.layout:
            command.append("--layout")
        if args.case:
            command.extend(["--case", args.case])
        for specification in args.probe_case or []:
            command.extend(["--probe-case", specification])
        if args.force:
            command.append("--force")
        if args.list:
            command.append("--list")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host,
             " ".join(shlex.quote(value) for value in command)], check=False)
        if args.list:
            return result.returncode
        if args.layout:
            subprocess.run(["scp", "-q", f"{args.host}:{root}/layout.json",
                            args.layout_output], check=True)
            return result.returncode
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mil-hwx-conv-json-") as staging:
            subprocess.run(["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                            staging], check=True)
            shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host")
    mode.add_argument("--emit-templates", action="store_true")
    mode.add_argument("--check", action="store_true",
                      help="fail when the checked-in templates are stale")
    parser.add_argument("--layout", action="store_true",
                        help="probe the constant-section layout instead of "
                             "minting oracle JSON")
    parser.add_argument("--targets", nargs="+", choices=sorted(om.SUBTYPES),
                        default=["h13", "h14"])
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--layout-output", default="conv-layout.json")
    parser.add_argument("--case")
    parser.add_argument("--probe-case", action="append",
                        help="layout-probe an ad-hoc "
                             "kernel,inputs,outputs,spatial,stride,groups,"
                             "bias,pad geometry instead of a grid case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.emit_templates or arguments.check:
        raise SystemExit(emit_templates(arguments))
    if arguments.local and arguments.layout:
        raise SystemExit(layout_run(arguments))
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
