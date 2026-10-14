#!/usr/bin/env python3
"""Why the H14 `exp` program missed its 2^-8 bound on an M2 Max (receipts/2026-10-13-h14-unary-m2/), pinned as arithmetic.

The compiler emits Apple's `exp` program unchanged: a 33-entry fp16 table of 2^(i/32) in the constant section and the fp16 value 0x3dc5
(1.4423828, log2(e) rounded to fp16) in the NE output-scale word 0x00d10. The model below is that program in plain numpy:
    t = raz16(x * scale)         one fp16 rounding of the exponent
    n = floor(t), s = 32 * (t - n), i = floor(s)
    y = raz16(T[i] + raz16((s - i) * (T[i+1] - T[i]))) * 2^n
with raz16 = round to nearest fp16, ties away from zero (the interpolator rule in docs/ane/numerics.md). It reproduces every saved device lane
bit for bit: 541 distinct (x, y) pairs, each seen in the calibration and the loop run of both widths (1,152 lane readings). The data pin the
final rounding (ties away), that the exponent is rounded, the interpolation, the scale word and every table entry. They do not pin the rounding
rule of the exponent or of the interpolation step; nearest-even gives the same lanes there. The mutants below show each pin.
The model then says where the 0.43 percent error comes from: with the scale as emitted, 60 of the 36,866 fp16 inputs in [-8, 8] miss a
relative error of 2^-8, the smallest at |x| = 5.598. With an exact scale and the same fp16 exponent, none do (largest 0.331 percent). This
is a statement about the model that reproduces the saved device output, not a device result for any changed program.
Usage: tests/test_h14_exp_arithmetic.py (no compiler needed; needs NumPy, so it is not part of `make test`: run `make test-h14-exp-model`)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "receipts/2026-10-13-h14-unary-m2"
ORACLE = ROOT / "research/oracles/h14/unary_exp_c512.json"
LANE_STRIDE = 32  # one fp16 value per 64-byte channel
SCALE_WORD = 0x00D10
BOUND = 2.0**-8
FLOOR = 2.0**-24  # the absolute floor of the unary gate's relative bound
LOG2E = 1.4426950408889634


def raz16(a):
    """Nearest fp16, ties away from zero."""
    a = np.asarray(a, dtype=np.float64)
    h = a.astype(np.float16)
    lo = np.where(h.astype(np.float64) <= a, h, np.nextafter(h, np.float16(-np.inf))).astype(np.float64)
    hi = np.where(h.astype(np.float64) >= a, h, np.nextafter(h, np.float16(np.inf))).astype(np.float64)
    below, above = a - lo, hi - a
    return np.where(below < above, lo, np.where(above < below, hi, np.where(np.abs(lo) > np.abs(hi), lo, hi)))


def nearest_even16(a):
    return np.asarray(a, dtype=np.float64).astype(np.float16).astype(np.float64)


def emitted_program():
    """The table and the scale of the oracle record the compiler replays."""
    record = json.loads(ORACLE.read_text())
    words = {w["index"]: w["value"] for w in record["constant_section"]["nonzero_fp16_words"]}
    table = np.array([words[4 + i] for i in range(33)], dtype=np.float64)
    ne = {int(k, 16): int(v, 16) for task in record["task_descriptors"] for b in task["blocks"].values() for k, v in b["words"].items()}
    scale = float(np.array([ne[SCALE_WORD] & 0xFFFF], dtype=np.uint16).view(np.float16)[0])
    return table, scale


def exp_model(x, table, scale, rnd=None, rnd_t=raz16, rnd_step=raz16, rnd_out=raz16, round_exponent=True, interpolate=True):
    if rnd is not None:  # one rule for all three stages
        rnd_t = rnd_step = rnd_out = rnd
    t = x * scale
    if round_exponent:
        t = rnd_t(t)
    n = np.floor(t)
    s = 32.0 * (t - n)
    i = np.minimum(np.floor(s).astype(int), 31)
    if not interpolate:
        return rnd_out(table[i]) * 2.0**n
    step = rnd_step((s - i) * (table[i + 1] - table[i]))
    return rnd_out(table[i] + step) * 2.0**n


def lanes(path, count):
    return np.fromfile(path, dtype=np.float16)[::LANE_STRIDE][:count].astype(np.float64)


def device_cases():
    for count in (64, 512):
        case = f"unary_exp_c{count}"
        x = lanes(RECEIPT / "inputs" / case / "x.f16", count)
        for run in ("cal", "long"):
            yield f"{case}.{run}", x, lanes(RECEIPT / "device-output" / f"{case}.{run}.y.f16", count)


def matches(table, scale, **kw):
    return sum(int(np.count_nonzero(exp_model(x, table, scale, **kw).astype(np.float16) == y.astype(np.float16))) for _, x, y in device_cases())



def test_raz16():
    """Every midpoint between positive fp16 values goes to the larger magnitude; nearest-even differs at half of them."""
    bits = np.arange(0, 0x7BFF, dtype=np.uint16)
    lo, hi = bits.view(np.float16).astype(np.float64), (bits + 1).view(np.float16).astype(np.float64)
    mid = (lo + hi) / 2
    assert np.array_equal(raz16(mid), hi) and np.array_equal(raz16(-mid), -hi), "raz16 must return the larger-magnitude neighbour"
    assert np.array_equal(raz16(lo), lo) and np.array_equal(raz16(-lo), -lo), "raz16 must keep exact fp16 values"
    assert int(np.count_nonzero(nearest_even16(mid) != hi)) == 15872, "nearest-even differs at half of the midpoints"


def sweep(table, scale):
    bits = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float64)
    xs = bits[np.isfinite(bits) & (np.abs(bits) <= 8)]
    rel = np.abs(exp_model(xs, table, scale) / np.exp(xs) - 1)
    return len(xs), rel, xs


def main():
    table, scale = emitted_program()
    assert scale == float(np.float16(LOG2E)) == 1.4423828125, scale
    assert table[0] == 1.0 and table[32] == 2.0 and np.array_equal(table, raz16(2.0 ** (np.arange(33) / 32))), "table is 2^(i/32) in fp16"
    test_raz16()

    total = sum(len(x) for _, x, _ in device_cases())
    assert total == 2 * (64 + 512)
    distinct = {(float(a), float(b)) for _, x, y in device_cases() for a, b in zip(x, y, strict=True)}
    assert len(distinct) == 541, len(distinct)
    for name, x, y in device_cases():
        got = exp_model(x, table, scale).astype(np.float16)
        assert np.array_equal(got, y.astype(np.float16)), f"{name}: model misses {int(np.count_nonzero(got != y.astype(np.float16)))} lanes"
    print(f"model reproduces {len(distinct)} distinct (x, y) pairs of the device, {total} lane readings, bit for bit")

    mutants = {
        "nearest-even rounding": {"rnd": nearest_even16},
        "exact log2(e)": {"scale": LOG2E},
        "unrounded exponent": {"round_exponent": False},
        "no interpolation": {"interpolate": False},
    }
    for name, kw in mutants.items():
        kw = dict(kw)
        hit = matches(table, kw.pop("scale", scale), **kw)
        assert hit < total, f"mutant '{name}' still matches every lane reading"
        print(f"mutant {name}: {hit} of {total} lane readings match")

    n, rel, xs = sweep(table, scale)
    outside = rel > BOUND  # plain relative error; the unary gate's rule (below) also has an absolute floor
    assert n == 36866 and int(outside.sum()) == 60 and float(np.abs(xs[outside]).min()) == 5.59765625, (n, int(outside.sum()))
    assert round(float(rel.max()), 5) == 0.00489
    _, rel2, _ = sweep(table, LOG2E)
    assert float(rel2.max()) < BOUND and round(float(rel2.max()), 5) == 0.00331

    # The gate rule of compare-unary.py: |y - exp(x)| <= 2^-8 * exp(x) + 2^-24 in float64. Lanes are counted per run, inputs once.
    def gate_outside(x, y):
        return np.abs(y - np.exp(x)) > BOUND * np.exp(x) + FLOOR

    per_width = {c: int(np.count_nonzero(gate_outside(x, y))) for name, x, y in device_cases() if name.endswith(".cal") for c in [len(x)]}
    assert per_width == {64: 2, 512: 9}, per_width
    device_x = {float(a) for _, x, _ in device_cases() for a in x}
    device_bad = {float(a) for _, x, y in device_cases() for a in x[gate_outside(x, y)]}
    model_y = exp_model(xs, table, scale)
    model_gate = {float(a) for a in xs[gate_outside(xs, model_y)]}
    assert len(model_gate) == 52 and device_bad == model_gate & device_x and len(device_bad) == 9, (len(model_gate), len(device_bad))
    print(f"as emitted, plain relative rule: {int(outside.sum())} of {n} fp16 inputs in [-8, 8] outside 2^-8, largest {rel.max():.5f}")
    print(f"as emitted, gate rule (floor 2^-24): {len(model_gate)} inputs; the device saw 9 of them and they are exactly its gate-failing inputs "
          "(2 lane readings at 64 channels, 9 at 512)")
    print(f"exact scale (model only): none outside, largest {rel2.max():.5f}")

if __name__ == "__main__":
    sys.exit(main())
