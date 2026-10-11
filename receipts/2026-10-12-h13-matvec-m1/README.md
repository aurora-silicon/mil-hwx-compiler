# H13 matvec compile and device attempt, 2026-10-12

The compiler emitted H13 ANEC packages for fp16 matvecs at K = N = 256, 512 and 1024. The 256 and 512 packages were submitted to a 13-inch M1 and an M1 Max and invoked with omarchy-ane's ane-run --check matvec --repeat 20. Neither chip produced a numerical comparison.

## Setup
- Source: mil-hwx-compiler origin/main at c8506137c2001dd7f5f77abbbda4a5515ae7d954.
- Programs: matmul(x[1,K], w[N,K], transpose_y), fp16 input and constant weights. Seeded NumPy inputs are recorded in results.txt.
- Chip: 13-inch M1 (T8103) and M1 Max (T6001), Linux kernel 7.1.12-2-12.6-sep-ARCH.
- Runner binary SHA256 prefix: 480040b5f700f407.
- run-log.txt is a summary of the shell output, not the full terminal capture.

## Result
No lanes were compared. On both chips the 256 run stopped during ANE initialization with surface channel 5 needs 32768 bytes but is allocated 16384. On both chips the 512 run stopped with --check matvec: unexpected io shape. These are runtime/layout failures, not named H13 compile refusals. No lane-in-band, bit-exact or ULP values are available.

The 1024 geometry compiled as a five-program package: four matmul chunks and a final combine. It was not submitted because ane-run --check matvec checks one program with one input and one output, so passing a single chunk would not test the full result.

## Limits
- One device attempt per single-program package on each chip. Neither attempt reached numerical comparison.
- K = N = 1024 is compile-tested only.
- No speed claim. The pinned compiler test verifies output structure, not device behavior.
