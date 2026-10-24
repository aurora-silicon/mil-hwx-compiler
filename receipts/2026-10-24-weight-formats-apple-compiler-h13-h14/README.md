# Weight formats and a fused program through Apple's compiler for H13 and H14, 2026-10-24

Sixteen small matmul programs went through Apple's compiler for the H13 and the H14 target. They differ only in how the weights are stored. This receipt records what the compiler kept and what it refused. It records what two devices did with the programs. It also records one fused pair of decode programs.

## Programs

Each program multiplies one fp16 input [1, K] by constant weights [N, K] with `transpose_y`. The weights are pseudo-random with standard deviation 0.02. They are the same in every format. The shapes are 2048 x 2048, 6144 x 2048 and 2048 x 6144, and an MLP of three weights. Four formats make 16 programs.

- fp16 constants.
- int8 per output row, `constexpr_affine_dequantize` with an fp16 scale per row.
- 4-bit lookup table with one table of 16 values for the whole tensor, `constexpr_lut_to_dense`.
- int8 in blocks of 32 with an fp16 scale per block, `constexpr_blockwise_shift_scale`.

## What the compiler kept

Size in bytes of the compiled program file (the HWX). H13 and H14 gave the same size except where noted.

| Program | fp16 | int8 per row | 4-bit table | int8 in blocks |
|---|---|---|---|---|
| 2048 x 2048 | 8,421,376 | 4,227,072 | 2,129,920 | 8,421,376 |
| 6144 x 2048 | 25,198,592 | 12,615,680 | 6,324,224 | 25,198,592 |
| 2048 x 6144 | 25,198,592 | 12,615,680 | 6,324,224 | 25,198,592 |
| MLP | 75,546,624 | 37,830,656 (H14: 37,814,272) | 18,923,520 | 75,546,624 |

The int8 and the 4-bit formats stay packed. The blocked int8 format comes out at the fp16 size: the compiler folds it back to fp16 on both targets.

Two more compiles on H13, for the 6144 x 2048 linear:

- A 4-bit table per output row (one table of 16 values for each of the 6144 rows) compiles. The program file is 7,602,176 bytes.
- A 4-bit table per group of columns (groups of 512, 256, 128 and 64 along the input axis) is refused with `MIL conversion error: blockwise LUT is not supported at axis C`.

## Devices

The converted programs ran under `ane-run --repeat 200 --time` on a 13-inch M1 (H13) and on a 14-inch M2 Max (H14). The three 6144 x 2048 programs ran twice on each device and the three MLP programs once. The inputs were fixed random values. The outputs were not checked, because the weights are random. All runs returned 0 and logged no fault. The table gives the median exec time per call in ms of the first run, with the program file size in MB. The two runs of a program differ by at most 0.004 ms.

| Program (MB) | 13-inch M1 | M2 Max |
|---|---|---|
| 6144 x 2048 fp16 (25.2) | 0.582 | 1.480 |
| 6144 x 2048 int8 (12.6) | 0.342 | 1.120 |
| 6144 x 2048 4-bit (6.3) | 0.188 | 0.811 |
| MLP fp16 (75.5) | 1.708 | 3.494 |
| MLP int8 (37.8) | 0.982 | 2.493 |
| MLP 4-bit (18.9) | 0.507 | 1.483 |

On the M1 the time follows the file size. The fits use the mean of the two runs where there are two. The line 0.072 ms plus 0.022 ms per MB fits the six programs with R squared 0.994, about 45 GB/s. On the M2 Max the line is 0.66 ms plus 0.039 ms per MB with R squared 0.96, about 26 GB/s. The runs came from a package built here and were made by someone else.

## A fused pair

Two adjacent programs of a decode package were fused at the MIL level. The first has 34 MB of weights and produces five lane tensors. The second updates a state and consumes the lanes. It has no weights of its own. The lane tensors became internal tensors. The 1 MiB state stayed an input and an output.

- H14: compiles. The program file is 34,160,640 bytes. It converts to an ANEC of 34,099,648 bytes with 39 tasks. The port table has one open ambiguity: two outputs have the same shape.
- H13: refused, `ANECCompile=22` with no message. Each of the two programs compiles alone on H13. The cause is not known.

The fused program has not run on a device.

## Limits

- One boot of each machine. Timer steps are about 0.05 ms.
- Random weights. The receipt says nothing about the quality of 4-bit weights on real weights.
- The sizes come from one build of the bridge `ane-compile-hwx`, sha256 `2060776c98e385d091ca244a2083be25919ed0c340ba1c8c89840208473b05d5`.
- The per-row table was compiled for H13 only.

## Source records

The MIL, weights and compiled files are in the lab archive and are not in this directory. `SHA256SUMS-not-archived` lists the sha256 of the sealed packages the devices ran.
