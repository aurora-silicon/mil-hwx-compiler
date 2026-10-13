# H14 unary programs on an M2 Max: what is predicted before the run, 2026-10-13

This file is committed before any of the 23 programs runs on a device. The result file added later states which predictions held.

## What runs
Twenty-three fp16 unary programs built by `mil-hwxc --target H14` from this repository's `origin/main`, using the oracle MIL unchanged:
`abs` at 64, 128, 256, 512, 1024, 2048 and 4096 channels, and `exp`, `gelu`, `leaky_relu` (slope 0.125), `rsqrt`, `sigmoid`, `silu`, `sqrt` and
`tanh` at 64 and 512 channels. None has run on a device. `relu` and the two clip programs (`clip-low`, `clip-high`) already ran and are not
repeated. The oracle records for `clip` with both bounds and for `log` hold no Apple output (`callback_status=1`), so the compiler refuses them
and they are out of scope. Each program is one ANEC program with the layout of the programs that ran: one input and one output, each
declared `(1, C, 1, 1, 64, 64)`, one fp16 value per 64-byte channel. `tests/test_h14_unary_programs.py` pins all 23 (size and sha256).
`build-unary.py` writes the programs and the inputs; a rerun gives the same bytes.
Inputs are uniform over [-8, 8], except `sqrt` over [0, 64] and `rsqrt` over [0.0625, 64], with a few fixed values always included.

## How it runs
`run-unary.sh`, on an M2 Max (T6021) under Linux, through `ane-run` without a check:
1. Per program, 20 calls with `--time`. The output file is kept.
2. Per program, a back-to-back loop sized to about 30 seconds from the calibration median. The output file is kept.
3. A program whose calibration run fails gets no loop. It is recorded, not rerun.
The device only runs the programs. `compare-unary.py` judges the saved output afterwards, against a float64 reference of the fp16 input.
It fails a case for a missing output, a calibration output that differs from the long-loop output, a non-finite lane, nonzero padding, or any
lane outside the bound below. Its exit status is tested in this directory (10 checks, then the same checks against 6 deliberately broken copies of the gate). A plumbing check
with a simulated perfect device, whose output is the correctly rounded fp16 result, passes all 23 cases.
Fixed before the run: 23 programs, a 30 second loop target, a floor of 200 and a cap of 3,000,000 calls, and a limit of 20 minutes.

## Bounds and predictions
The bound is the prediction. A lane outside it is a finding, reported with the numbers.
1. `abs`: every lane bit-exact, at all seven sizes.
2. `leaky_relu`: every lane within 1 ulp of the correctly rounded result.
3. `sqrt` and `rsqrt`: every lane within a relative error of 2^-9 (0.2 percent).
4. `exp`: every lane within a relative error of 2^-8 (0.4 percent).
5. `sigmoid` and `tanh`: every lane within an absolute error of 2^-8 (0.0039). `silu` and `gelu`: within 2^-6 (0.0156).
6. For `gelu`, the output is closer to the exact (erf) form than to the tanh approximation. This is a measurement, not a bound. I give it even odds.
7. Every program runs to completion: both runs exit 0, and the outputs are finite with zero padding.
8. Duty at least 90 percent for the run and for each loop, and a total wall time of at most 15 minutes.
9. The median call time of every program is at most 0.60 ms. Basis: the matvec programs had medians of 0.36 to 0.51 ms. The 4096-channel `abs`
   moves 16 tiles per call and is the most likely to exceed this.
For reference, a perfectly rounded fp16 result uses about a quarter of each bound or less, so a pass here is not automatic.

## What would change the plan
- A refusal at load, or a nonzero exit, is recorded with the exact text. The other programs still run.
- A case that fails the gate is reported with the lane, the input and the output. It is not rerun or averaged away.

## Limits
One run on one boot, one input set per program, fp16 only. Not claimed: accuracy outside the input domains above, any speed result, other
channel counts, or chained programs. A pass says the output was inside the bounds on these inputs.
