# H13 convolutions as one program per layer

This page explains why a 64-channel 64x64 1x1 conv plus relu used to compile to
4097 H13 programs, and what the compiler does now. It also lists the register
formulas the new encoder uses, the M1 Pro measurements, and what is still unknown.

## Where the 4097 programs came from

The convolution was never the problem. `conv_relu.mil` compiled to one
`apple-parity-conv` program, which runs in 65 us of engine time. The other 4096
programs were the **relu**. No decoded elementwise template covers a
`[1, 64, 64, 64]` relu, so the source-qualified fallback cut it into 64-element
slices, one program per slice. At about 29 us of fixed cost per program, that
took about 119 ms.

`--schedule chain` could not help. It refused the pair because the conv
encoder had no decoded post-operation field.

## What changed

1. **conv + relu fuse in the default per-op schedule.** When a conv's only
   consumer is a relu, and the conv result is not a function output, the pair
   becomes one program. The NE MAC word `0x0c804` gets the clamp bit
   `0x00010000`. This is the same bit Apple's decoded `chain_pair_mm_relu`
   matmul epilogue sets. On the M1 Pro the fused output equals relu of the
   unfused device output, bit for bit, on all 262,144 lanes.
2. **A formula-built convolution encoder** (`encodeConvGeneric`). It covers
   shapes no template has. It takes the invariant words from the closest
   decoded single-task template of the same family (same kernel, stride, group
   form and bias), then recomputes every shape-dependent register. The hardware
   tiles natively through Common `0x28` (TileHeight), so any spatial size is
   still one task.
3. **Output-channel tasks.** A kernel over 1 MiB splits into linked tasks in
   one program, each writing its own slice of the output surface.
4. **Stride-2 sections are value-dependent.** A stride-2 section skips zero
   weights, so its size depends on the values. A decoded template's recorded
   size fits only the oracle's own payload, and real weights used to be
   refused. The compiler now falls back to the generic form in that case. The
   task words are the same and the kernel-DMA table is sized from the actual
   section.

## Register formulas

With `Cin`, `Cout` (the per-task slice), input `H x W`, output `Ho x Wo`, kernel `k`, stride `s`:

| Register | Value |
| --- | --- |
| Common `0x00` / `0x14` | `H << 16 \| W` / `Ho << 16 \| Wo` |
| Common `0x0c` / `0x10` | `Cin` / `Cout` (depthwise: `C` / `C`) |
| Common `0x1c` ConvCfg | `Kw \| Kh<<5 \| OCG<<10 \| Sx<<13 \| Sy<<15 \| Px<<17 \| Py<<22`, top nibble from the template. `OCG = log2(lanes)`, 0 for depthwise |
| Common `0x28` | TileHeight: the largest tile with `rows_in * srcRow + tile * Cout * rowL2(Wo) <= 0x380000` |
| Common `0x34` | `0x20<<16 \| class<<16 \| low`. `class` is 4 up to 8 output columns, then 3, 2, 1 per doubling, and 0 from 128. `low` repeats the class in two nibbles (`0x4405` at class 4) for class 4, dense 3x3, and 1x1 with 512 or more channels on both sides; otherwise it is `0x0001` |
| L2 `0x480c` | `rowL2 = align16(max(16, 2W))` |
| L2 `0x4814`, `0x4818` | `Cin * rowL2` |
| L2 `0x4810` | `Cin * rowL2 + skew`. At stride 1 the skew is 0x20 (below 32 channels) or 0x30 up to W 16, 0x50 up to W 32, and 0 from W 64. At stride 2 it is 0x20, or 0x30 from W 64 |
| L2 `0x4834` | `rows_in * 0x4810`, where `rows_in = min(H, TileHeight * s + k - 1)` |
| TileDMA src `0x1380c..0x13818` | row `align64(2W)`, plane `row * H`, `plane * Cin` twice |
| TileDMA dst `0x17804` | `plane_out * first output channel` of the task |
| TileDMA dst `0x17808..0x17814` | row, plane, `plane * Cout_total`, `plane * Cout_task` |
| KernelDMA `0x1f848+4i` / `0x1f888+4i` | the 16 chunk offsets and sizes of the packed section: equal groups at stride 1, variable planes at stride 2 |
| header (linked tasks) | `header[0]` id plus `1<<24` after the first task and `1<<25` on the last. `header[1]` holds the next size. `header[7]` is the next offset (0x100-aligned). The end-of-stream bits (`header[2]` 0x22, `header[4]` 0x2, `header[6]` 0x9000) go on the last task only, and `header[8]` bit 23 on every task with a successor |

**Parity.** Every decoded single-task dense and depthwise template in range,
rebuilt from these formulas, equals Apple's words and constant section (112 of
121 shapes). The one exception is the low half of `header[1]`, which the
corpus does not explain. The 9 shapes that differ do so only by choice:

- 64x64x256 to 256 (Apple picks TileHeight 2);
- the 3x3 256 to 256 and 768 to 768 shapes at 1x1 and 8x8 pixels. Apple keeps
  1.18 MB in one task, and the 1 MiB task budget here splits it.

`tests/test_h13_generic_conv_cli.py` pins 16 of those parity shapes, the
stride-2 fallback, MobileNet-class coverage, the fusion and the refusals.

## M1 Pro measurements (31/10/2026, T6000, kernel 7.3.0-aurora-fleet12)

Every row is one program. `p50` is wall time per `ane_exec` and `busy` is
engine time from `ane_stats`. The CPU column is numpy fp32 on the same machine.
Accuracy is measured against a float64 reference rounded to fp16 (ties away).

| Layer | Before | After p50 / busy (us) | CPU fp32 (us) | exact / within 1 ulp | rel. L2 |
| --- | --- | ---: | ---: | --- | --- |
| 1x1 64->64 @64x64 + relu (`conv_relu.mil`) | 4097 programs, ~119 ms | 75.3 / 64.1 | 703 | 90.1 % / 98.0 % | 1.7e-4 |
| 3x3 64->64 @16x16, bias + relu | conv + 256 relu slices (~9.5 ms) | 51.7 / 38.7 | 276 | 80.1 % / 94.3 % | 3.0e-4 |
| 3x3 16->32 @112 (MobileNet stem as space-to-depth) | refused | 94.2 / 81.5 | 1926 | 87.0 % / 97.0 % | 2.1e-4 |
| dw 3x3 32 @112 | refused | 120.9 / 107.9 | 3826 | 94.9 % / 99.3 % | 1.0e-4 |
| 1x1 32->64 @112 | refused | 143.5 / 132.0 | 944 | 92.2 % / 98.6 % | 1.4e-4 |
| dw 3x3 64 @112 | refused | 199.4 / 188.0 | 8142 | 94.7 % / 99.3 % | 1.1e-4 |
| dw 3x3 128 @56 | refused | 109.7 / 96.6 | 5619 | 94.7 % / 99.2 % | 1.1e-4 |
| 1x1 128->128 @56 | refused | 106.4 / 95.5 | 936 | 87.7 % / 97.3 % | 2.0e-4 |
| 1x1 s2 64->128 @56->28 | refused (section size) | 64.0 / 53.3 | n/a | 80.2 % / 96.0 % | 1.7e-4 |
| dw 3x3 256 @28 | refused | 73.5 / 60.4 | 2807 | 94.8 % / 99.2 % | 1.1e-4 |
| 1x1 256->256 @28 | refused | 94.9 / 83.1 | 830 | 84.7 % / 96.0 % | 2.4e-4 |
| dw 3x3 512 @14 | refused | 73.4 / 60.2 | 1581 | 95.0 % / 99.3 % | 1.1e-4 |
| 1x1 512->512 @14 | refused | 121.9 / 110.5 | 679 | 80.9 % / 94.6 % | 2.9e-4 |
| dw 3x3 1024 @7 | refused | 70.8 / 59.6 | 947 | 94.6 % / 99.2 % | 1.1e-4 |
| 1x1 512->1024 @7 (2 tasks) | refused | 112.6 / 101.2 | 942 | 81.0 % / 94.6 % | 2.9e-4 |
| 1x1 1024->1024 @7 (4 tasks) | refused | 209.1 / 196.5 | 1230 | 76.8 % / 92.2 % | 3.5e-4 |

The output is not bit-exact against a float64 reference, and the template
programs show the same deviation. The engine accumulates at a precision below
fp32. The largest ulp distances sit next to zero, where the absolute error is
at most 0.002.

**MobileNetV1 224, estimate.** The 25 measured layer programs sum to about
2.5 ms of wall time (2.1 ms busy) on the M1 Pro. The three stride-2 pointwise
layers with 256 or more outputs are not supported yet, so this is not a
full-network measurement. With them included, expect about 3 ms. For
comparison, macOS does a whole MobileNetV1 in 0.30 ms on the M4 Pro (H16); for
H13 the scaled estimate is about 1 ms. The gap left is DRAM round trips
between layers and about 12 us of submit cost per program. Apple keeps
intermediates in L2 inside one program.

## Still refused or unknown

- **Stride 2 with 16 or more lanes** (1x1 s2 with more than 128 outputs) is
  refused. One such program (128 to 256 at 56 to 28) hung the M1 Pro engine.
  The driver's recovery could not clear it, so the engine needs a reboot. The
  decoded record already says this body count is two bytes off the model.
  Splitting into 128-output tasks would avoid it, but that has not run.
- **3x3 stride 2, grouped other than depthwise, and channel counts that are
  not multiples of 16** are refused.
- **The low half of `header[1]`** (a cost hint?) is copied from the template.
  The device does not seem to care.
- **The L2 capacity and the per-task kernel budget** are measured envelopes
  (0x380000 and 1 MiB), not known limits.
- **L2-resident chaining across layers.** Later tasks would read the previous
  task's L2 result, but the L2 `0x4804`/`0x4830` source and result configs
  that make this work have no formula.
- **H14** has none of this yet. The register blocks differ.
