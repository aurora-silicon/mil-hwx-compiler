# H14 matvec with 2, 8 and 64 rows on an M2 Max: what is predicted before the run, 2026-10-13

This file is committed before any of the 27 programs runs on a device. The result file added later states which predictions held.

## What runs
Twenty-seven fp16 matvec programs built by `mil-hwxc --target H14` from this repository's `origin/main`: M rows of 2, 8 and 64, with K and
N each 256, 512 and 1024. Each is one ANEC program (`build-rows.py` writes them; a rerun gives the same bytes). The M = 1 shapes
already ran on an M2 Max (`receipts/2026-10-11-h14-matvec-m2/` and `receipts/2026-10-13-h14-matvec-nonsquare-m2/`) and are not repeated.
Before this run, a host check found no layout blocker: every header declares input `(1, 1, M, K)` and output `(1, 1, M, N)` with
dense rows, which is what `ane-run --check matvec` reads, and `tests/test_h14_matvec_row_programs.py` pins that for all 27 shapes.
The host check cannot show what the device writes for M above 1. This run is the first M above 1 on a device.

## How it runs
`run-rows.sh`, on an M2 Max (T6021) under Linux, through `ane-run --check matvec`:
1. Per shape, 20 calls with `--time`, which also checks the result against a float64 reference.
2. Per shape that passes, a back-to-back loop sized to about 40 seconds from the calibration median, then the same check on the final output.
3. A shape whose calibration check fails gets no long loop. It is recorded, not rerun.
4. Duty is the change in the driver's `busy_ns` divided by wall time.
Fixed before the run: 27 shapes, a 40 second loop target, a floor of 200 and a cap of 3,000,000 calls, and a limit of 30 minutes. The
script's logic is tested offline with a fake runner (`test-run-rows.sh`, 12 checks).

## Predictions
1. Correctness. All 27 programs pass the check (every lane inside 2 ulp or 4 condition units, padding zero) in the calibration and in
   the long loop. Confidence: high for 2 rows, moderate for 8 and 64 rows. The compiler output equals the decoded Apple output and the
   headers declare dense rows, but `ane-run`'s own source says the 8-row form runs N output channels with 8 row buffers, so a different
   device layout for 8 or 64 rows is possible. A shape that fails is a finding about that row count, reported with its exact output.
2. Call time. The median call time of every shape that passes lies between 0.30 and 0.50 ms, and for each K and N the 64-row median is
   no more than 0.10 ms above the 2-row median. Basis: all nine M = 1 shapes had a median of 0.356 or 0.357 ms, and a 64-row call does 32
   times the arithmetic of a 2-row call. If the call time is a fixed cost, it stays flat. If it rises with the rows, this prediction fails.
3. Duty. If all 27 pass, at least 90 percent over the run and at least 90 percent in each loop. If some fail, the duty is reported as
   measured, with the number of skipped loops.
4. Wall time. If all 27 pass, at most 22 minutes (27 loops of about 40 seconds plus calibrations).

## What would change the plan
- A refusal at load or an "unexpected io shape" message is recorded with the exact text. The other shapes still run.
- Lanes outside the band are recorded with the shape, the count in band and the largest error. They are not averaged away.

## Limits
One run on one boot, one weight set and one input per shape, fp16. Not claimed: any speed result, shapes outside 256, 512 and 1024 or other
row counts, normalization, convolution, or chained programs.
