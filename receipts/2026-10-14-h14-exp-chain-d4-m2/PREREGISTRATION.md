# H14 exp chain in scaled form (design D4): preregistration

Written before any device run. The first exp chain ran on 2026-10-14 (`receipts/2026-10-14-h14-exp-chain-run-m2/`). Its final output met the exp bound, and it failed criterion 3: 21 stage lanes
differed from the ties-to-even result. That failure stands, and so does the original unary cell at 21 of 23. This cell is separate. Five calls are not a speedup.

## Why a new design
The first chain differed from ties-to-even on two kinds of lane. 6 lanes were exact midpoints where the device gave the half-away result. 15 were mul results in the fp16 subnormal range,
which the device rounded one step larger in magnitude. This design changes the arithmetic so that, on the registered inputs, no stage has a lane where ties-to-even and half-away differ, and none has a subnormal result.
It does not change a criterion. It is a design change fitted to what one run showed. Whether it holds on other data or other shapes is not established.
On the registered inputs, checked offline: s2 has 2 exact-midpoint lanes at 512 channels (x = 4.25 and x = -6.5). Ties-to-even and half-away give the same value there, so they do not test the rule.
s3, s4 and s5 have no midpoint lane at either size, and no stage has a subnormal result. The smallest nonzero |x| is 0.0265, so |x * k16| is at least 9.2e-5, above 2^-14.
With 8 K in place of 16 K, two s2 lanes would be subnormal (x = 0.0311 at 64 channels and x = 0.0265 at 512). That is why the factor is 16.

## What runs
The same three programs as the first chain (exp, mul, add; same bytes), chained on the host. The host copies bytes and does no arithmetic. Five calls per size, at 64 and 512 channels.
1. s1: `e = exp(x)`, the program of the failed run.
2. s2: `c = x * k16`, mul. k16 = 16 K = fp16 0x1b18 in every lane. K = 0x0b18 is the relative error of the scale word 0x3dc5. Multiplying K by 16 is exact.
3. s3: `g = c + sixteen`, add. sixteen = 16.0 (0x4c00) in every lane. g is 16 (1 + K x), rounded at the same relative step as 1 + K x would be, since 16 is a power of two.
4. s4: `p = e * g`, mul. p is 16 e (1 + K x). It is at most 47712 on the registered lanes and about 47776 over all fp16 inputs in [-8, 8] with an exact exp, below the fp16 limit of 65504.
5. s5: `y = p * sixteenth`, mul. sixteenth = 0.0625 (0x2c00) in every lane. This scaling is exact because y is at least 3.3e-4, which is a normal fp16 value.
The input x is the saved input of the failed run, byte for byte. `build-expchain.py` writes the programs and inputs and lists their sha256 in `SHA256SUMS-inputs`.
Every surface is fp16 (1, C, 1, 1) with one value per 64-byte channel, as in the first chain.

## How it runs
`run-expchain.sh` on an M2 Max (T6021) under Linux, through `ane-run` without `--check`. Per stage: 20 calls with `--time`, then a back-to-back loop of about 30 seconds. Both outputs are kept.
A stage whose calibration or long loop fails stops that size, and nothing is rerun. The whole run has a 15 minute deadline in the script, and every call runs under `timeout -k 2`.
A TERM, INT or HUP sent to the script ends the running call and is recorded. The run needs the ANE to itself: the script counts other `ane-run` processes before and after, and the gate requires 0 and 0.
Ten stage runs take about 5.5 minutes.

## Criteria (all frozen; `compare-expchain.py` judges, `test-compare-expchain.py` tests it against 52 checks and 42 broken copies)
Numerical, for 64 and 512 channels. They are the criteria of the first chain, with the stage list extended:
1. x, k16, sixteen and sixteenth are the registered inputs. The sha256 of x is pinned. Each constant is its fp16 value in every lane with zero padding.
2. Every output has zero padding and no NaN or infinity, and its calibration and loop files are bit-identical.
3. s2, s3, s4 and s5 each equal the correctly rounded fp16 result of their own saved inputs on every lane. The rounding is ties to even, and the comparison is on fp16 bits, so the sign of a zero counts.
   There is no tolerance, no lane-specific exception and no acceptance of the half-away result. A device that gives it at a midpoint fails this criterion.
4. The final output s5 is within a relative error of 2^-8 of exp(x) on every lane, plus the absolute floor 2^-24 that the gate of the failed run used. The bound did not change.
   The gate asserts both values and pins the sha256 of the `compare-unary.py` it imports them from.
For a criterion 3 failure the gate lists up to 10 differing lanes per stage with the lane index, the two inputs, the ties-to-even result, the device result and an exact-midpoint flag.
These are diagnostics, and none turns a failure into a pass.
Run identity, duty and time, reported apart and also required: `results.txt` names `target_s=30`, the T6021 chip, the program hash of each stage as in `SHA256SUMS-inputs`, the sha256 of every output file
(checked against the files), exactly one SUMMARY line, and 0 other `ane-run` processes before and after. Also: every stage's calibration median is at most 0.60 ms; every loop and the ticket have
duty at least 90 percent; total wall time is at most 15 minutes; all ten stage runs exit 0.

## Prediction
Two predictions, both from the saved device output of s1 and neither a device result.
- With correctly rounded fp16 mul and add, the final output has a largest relative error of 0.283 percent at 64 channels and 0.348 percent at 512, against the bound of 0.391. No lane is outside.
  The margin at 512 channels is 0.043 percent. A different rounding at an unseen value could use it. Over all 36,866 fp16 inputs in [-8, 8] the largest error under this arithmetic is 0.360 percent.
- With the description of the device arithmetic fitted to the first run, no stage lane differs from ties-to-even. That description is a ranking tool. It was fitted after one run. The tie rule of its
  first rounding is not identified by the data, and it was fitted on a few hundred lanes of mul and add. If the device behaves otherwise at a value this design reaches, criterion 3 fails and the cell is reported as failed.

## What would change the plan
A refusal at load or a nonzero exit is recorded with the exact text, and the later stages at that size do not run. A failed criterion is reported with the lane, input and output.
It is not rerun, averaged away or judged against a looser bound, and the half-away result is not accepted.
