# H14 exp with the scale correction: preregistration

Written before any device run. The exp cell of `receipts/2026-10-13-h14-unary-m2/` failed: 11 of 512 and 2 of 64 lanes were outside the
relative bound of 2^-8. That result stands, and this cell does not replace it. The whole unary cell is accepted only if this cell also meets
every criterion below.

## Why
`tests/test_h14_exp_arithmetic.py` reproduces the failed run bit for bit. Apple's program rounds x times fp16(log2 e) to fp16. That scale is 2.16e-4
low, so the error grows with |x|. A model of a second stage that adds back y * x * 2.165e-4 stays inside the bound. It is a model. It is not a device result.

## What runs
Four programs that already exist in the compiler's tests, chained on the host. The host copies bytes and does no arithmetic: each stage's output file is
the next stage's input file, unchanged. Each stage is one `ane-run` call with `--in IDX=FILE`.
1. s1: `exp(x)`, the program of the failed run (same bytes, sha256 `8599331b...` at 512 channels).
2. s2: `c = x * K`, an elementwise mul program with two inputs. K is fp16 0x0b18 (0.000216484) in every lane, from `in/k_c<C>.f16`.
3. s3: `d = e * c`, the same mul program, with the s1 and s2 outputs.
4. s4: `y = e + d`, an elementwise add program, with the s1 and s3 outputs.
The input x is the saved input of the failed run, byte for byte, at 64 and 512 channels. `build-expchain.py` writes the programs and inputs and lists their sha256.
Every surface is fp16 (1, C, 1, 1) with one value per 64-byte channel. That is 32768 bytes at 512 channels and 16384 at 64.
A host check of the program headers found no size or packing mismatch between any output and the input it feeds.

## How it runs
`run-expchain.sh`, on an M2 Max (T6021) under Linux, through `ane-run` without `--check`. Per stage: 20 calls with `--time`, then a back-to-back loop of about
30 seconds. Both outputs are kept. A stage whose calibration or long loop fails stops that size, and nothing is rerun.
The whole run has a 15 minute deadline in the script itself: no stage starts after it, and every call's timeout is capped to the time left, so the worst case is 15 minutes and not the sum of the per-call timeouts.
The run needs the ANE to itself. The script counts other `ane-run` processes before and after and writes both counts to the run file; the gate requires 0 and 0.

## Criteria (all fixed now; `compare-expchain.py` judges, `test-compare-expchain.py` tests it against 45 checks and 36 broken copies)
Numerical, for 64 and 512 channels:
1. x and K are the registered inputs: the sha256 of x is pinned in the gate, and K is fp16 0x0b18 in every lane with zero padding.
2. Every output has zero padding and no NaN or infinity, and its calibration and loop files are bit-identical.
3. s2, s3 and s4 equal the correctly rounded fp16 result of their own saved inputs on every lane (zero is compared exactly).
4. The final output s4 is within a relative error of 2^-8 of exp(x) on every lane, plus the absolute floor 2^-24 that the gate of the failed run used. The bound did not change.
   The gate asserts both values and pins the sha256 of the `compare-unary.py` it imports them from.
   The reference for criteria 3 and 4 is round to nearest, ties to even. If the device rounds ties away in mul or add, criterion 3 fails on those lanes
   and is reported as a criterion 3 failure. The bound is not relaxed for it.
Run identity, duty and time, reported apart and also required: `results.txt` names the T6021 chip, the program hash of each stage as in `SHA256SUMS-inputs`, the sha256 of every output file
(checked against the files), exactly one SUMMARY line, and 0 other `ane-run` processes before and after. Also: every stage's calibration median is at most 0.60 ms; every loop and the ticket have duty at least 90 percent;
total wall time is at most 15 minutes; all eight stage runs exit 0.

## Prediction
Take the saved device output of s1 and ordinary fp16 mul and add. The final output has a largest relative error of 0.280 percent at 64 channels and 0.309 percent
at 512 (bound 0.391), with no lane outside. 51 and 412 lanes change from s1. The eight elementwise programs that ran earlier were bit-exact. With ties-away rounding in mul and add the output would differ at 2 of 64 and 2 of 512 lanes, and the 64-channel maximum would be 0.283 percent. If s2, s3 or s4 are not,
criterion 3 fails and the cell is reported as failed. Over all 36,866 fp16 inputs in [-8, 8] the model's largest error for the chain is 0.338 percent. That sweep is not run on a device here.

## What would change the plan
A refusal at load or a nonzero exit is recorded with the exact text, and the later stages at that size do not run. A failed criterion is reported with the lane, input and
output. It is not rerun, averaged away or judged against a looser bound.
