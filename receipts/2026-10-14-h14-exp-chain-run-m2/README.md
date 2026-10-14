# The exp chain on an M2 Max, 2026-10-14

The merged cell of `receipts/2026-10-13-h14-exp-chain-m2/` ran once on an M2 Max (T6021) under Linux. It ran alone on the ANE: no other `ane-run` process before or after, and the idle filler off.
Four programs ran in a chain per size, at 64 and 512 channels: exp, mul by K, mul, add. The host only copied bytes between them.
Nothing in the preregistration changed. Ties-to-even is the criterion 3 reference. The exp bound is relative 2^-8 plus the absolute floor 2^-24.
`compare-expchain.py` judged the saved output off the device (`verdict.txt`).

## Result
| Criterion | Result |
|---|---|
| 4: final output within the exp bound | Met. Largest relative error 0.283 percent at 64 channels and 0.309 percent at 512. No lane outside (bound 0.391). The failed run had 0.446 and 0.429 percent, with 2 and 9 lanes outside by the gate rule (11 by a plain relative 2^-8 at 512 channels) |
| 3: s2, s3 and s4 each equal the ties-to-even result of their saved inputs | Failed. 21 lanes differ: 3 at 64 channels (s3 1, s4 2) and 18 at 512 (s2 3, s3 13, s4 2) |
| 1 and 2: registered inputs, padding, finite values, identical calibration and loop outputs | Met |
| Run identity, duty and time | Met. T6021 chip, 0 other `ane-run`, registered program and output hashes, loop duty 99.5 to 99.6 percent, stage medians 0.307 to 0.360 ms, 235.5 s in total |

The numerical criteria fail, so the cell is not accepted. The unary cell stays at 21 of 23. No bound was relaxed and no rounding rule was added to the gate.

## What the 21 lanes are
`analysis.txt` comes from `analyze-chain-run.py`, which only reads the saved output.
- 6 lanes are exact midpoints where the device gave the half-away result. They are s4 at both sizes, plus one lane each in s2 and s3 at 512 channels. One is `2896 + 5`: ties-to-even gives 2900 and the device gave 2902. The stages hold 20 exact-midpoint lanes. In 14 of them ties-to-even and half-away give the same value. The 6 above are the lanes where the two rules differ, and all 6 gave the half-away result. For add alone it is 4 of 4 such lanes.
- 15 lanes are mul results in the fp16 subnormal range, below 2^-14. On all 15 the device result is one subnormal step larger in magnitude than the ties-to-even result. The fractional positions run from 0.265 to 0.481 of a step.
- 255 mul lanes have a subnormal result. Single rounding to the subnormal grid with ties to even matches 239 of them, and with ties away it matches 240. One model matches all 255. It rounds the product to 11 significant bits as if the exponent range were unbounded. Then it rounds to the subnormal grid with ties away.

That model was fitted after the run to these saved lanes. It was not registered, and it is not a prediction. Read its 255 matches with care:
- 240 of the 255 lanes also match single ties-away rounding. The model is tested by the 15 lanes that single rounding misses, and by the 240 it keeps right.
- The data fix the width. With 10 significant bits the model matches 226 lanes, with 12 it matches 246, and with 13 it matches 244. They do not fix the tie rule of the first rounding: ties to even there matches 255 too. A second rounding with ties to even matches 237.
- A mechanism fits all 15 lanes, also after the fact. Below 2^-14 the 11-bit step is finer than the subnormal grid. A product just under half a grid step rounds up to exactly half a step in the first rounding. The second rounding then carries it up. For a product in the binade from 2^b up to 2^(b+1), below 2^-14, the 11-bit step is 2^(b-10). The first rounding takes a fractional position of at least 0.5 - 0.5 * 2^(b-10) / 2^-24 up to exactly half. The threshold is 0.25 for b = -15, 0.375 for b = -16 and 0.4375 for b = -17. Every one of the 15 lanes is at or above its threshold. The lowest margin is 0.008, at 512 channels in s3, binade 2^-16, fraction 0.3828 against 0.375. The next is 0.015, at fraction 0.265 against 0.25.

The add stages here have no subnormal results, so the model says nothing about add. A separate timing run of an add 1x3x224x224 program on the same chip reported the half-away result at exact midpoints too. It was not re-checked here.

## Files
`results.txt` is the run file (sha256 prefix `a258fdd3`). `verdict.txt` is the gate output (`867c2398`). `device-output/` holds the kept outputs and the calibration and loop logs of all eight stage runs.
`inputs/` holds x and K. `SHA256SUMS` lists every file.
No speed is claimed. The chain is four calls where the failed exp program was one.
