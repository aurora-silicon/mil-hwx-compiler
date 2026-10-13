# H14 non-square matvec on an M2 Max: what is predicted before the run, 2026-10-13

This file is written and committed before any of the six programs runs on a device. The result file added later states
which predictions held.

## What runs
Six fp16 matvec programs built by `mil-hwxc --target H14` from this repository's `origin/main` (source 6f46c6a for the H14
encoder, binary sha256 `0a79d9c1a1236e99751499be1ca0a78ecd83d3073786686854cdedd810b6f62e`, the same binary as the
2026-10-11 square-shape run). Shapes (K inputs, N outputs): 256x512, 256x1024, 512x256, 512x1024, 1024x256, 1024x512.
The three square shapes already ran (`receipts/2026-10-11-h14-matvec-m2/`). Each program is one ANEC program. Each header
declares output N and input K as dense rows, which is the layout `ane-run --check matvec` reads.
Program sha256 prefixes: 256x512 `0e56fb17`, 256x1024 `68a60408`, 512x256 `71eda38e`, 512x1024 `be265566`,
1024x256 `e5ad21d3`, 1024x512 `4aba2a66`. Weights and input come from `make_matvec.py` in the square-shape directory,
seeded from K and N, so a rerun gives the same files. Build per shape: `python3 make_matvec.py K N DIR`, then
`mil-hwxc --mil DIR/matvec.mil --model-root DIR/model --target H14 --output DIR/out`.

## How it runs
`run-device.sh` in this directory, on an M2 Max (T6021) under Linux, through `ane-run --check matvec`:
1. Per shape, 20 calls with `--time` for a calibration, which also checks the result.
2. Per shape, a long loop sized to about 110 seconds from the calibration median, back to back, then the same check on the
   final output.
3. Duty for each loop and for the whole run is the change in the driver's `busy_ns` divided by wall time.
Fixed before the run: 6 shapes, a 110 second loop target, a floor of 200 calls and a cap of 3,000,000 calls, and a 90 percent
duty goal for the whole run. The script's logic is tested offline with a fake runner (`test-run-device.sh`, 10 checks).

## Predictions
1. Correctness. All six programs finish with exit status 0 in both the calibration and the long loop, every lane inside
   the 2 ulp or 4 condition-unit band, padding zero. Basis: the compiler output equals the decoded Apple output for these
   shapes, and the three square shapes passed the same check. A shape that fails is reported as a failure, not rerun until
   it passes.
2. Bit-exact lanes. About 90 to 97 percent of lanes bit-exact per shape (the square shapes gave 93.8, 94.1 and 94.8
   percent). This is a prediction, not a pass criterion.
3. Duty. At least 90 percent over the whole run, and at least 90 percent in each long loop. If the engine duty comes out
   lower, the number is reported as measured and the cause is not guessed.
4. Latency. Median call time per shape rises with the weight bytes (K times N). The small shapes (256x512 and 512x256) are
   the fastest and the large ones (512x1024 and 1024x512) the slowest, with the same weight bytes giving similar medians
   (256x1024 and 1024x256; 512x1024 and 1024x512). No comparison with macOS or with any other chip is made.
5. Wall time. About 12 to 14 minutes in total.

## What would change the plan
- A shape that is refused at load ("surface channel needs ... allocated", as H13 K=256 was) or reports an unexpected io
  shape is a finding about that shape's header or the checker. It is recorded with the exact message, and the other shapes
  still run.
- Any lane outside the band is recorded with the shape, the lane index and the values. It is not averaged away.

## Limits
One run on one boot, one weight set and one input per shape, M = 1, fp16. Not claimed: a speed result, a result for
M = 2, 8 or 64 rows, any shape outside 256, 512 and 1024, or any normalization, convolution or chained program.
