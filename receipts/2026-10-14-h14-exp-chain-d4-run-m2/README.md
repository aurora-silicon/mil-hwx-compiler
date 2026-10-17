# The scaled exp chain (design D4) on an M2 Max, 2026-10-14

The merged cell of `receipts/2026-10-14-h14-exp-chain-d4-m2/` ran once on an M2 Max (T6021) under Linux. The run file counts other `ane-run` processes at the start and at the end, and both counts are 0. The scheduler's window had the idle filler off. That was logged outside this archive, and the run file does not record it.
Five programs ran in a chain per size, at 64 and 512 channels: exp, mul by 16 K, add 16, mul, mul by 0.0625. The host only copied bytes between them.
Nothing in the preregistration changed. Ties-to-even is the stage reference, with no tolerance and a comparison on fp16 bits. The exp bound is relative 2^-8 plus the absolute floor 2^-24.
`compare-expchain.py` judged the saved output off the device (`verdict.txt`).

## Result
| Criterion | Result |
|---|---|
| 3: s2, s3, s4 and s5 each equal the ties-to-even result of their saved inputs | Met. 64 of 64 and 512 of 512 lanes in every stage, at both sizes. The two s2 midpoint lanes at 512 channels (x = 4.25 and x = -6.5) are among them: both rules give the same value there |
| 4: final output within the exp bound | Met. Largest relative error 0.283 percent at 64 channels and 0.348 percent at 512, no lane outside (bound 0.391) |
| 1 and 2: registered inputs, padding, finite values, identical calibration and loop outputs | Met |
| Run identity, duty and time | Met. T6021 chip, 0 other `ane-run` at the start and the end, registered program and output hashes, loop duty 99.5 to 99.6 percent, calibration medians 0.306 to 0.360 ms, 287.6 s in total |

Gate exit 0: numerical criteria PASS, run identity, duty and time PASS.
Both preregistered predictions held: the largest errors (0.283 and 0.348 percent) and no stage lane off ties-to-even.
The first prediction came from correctly rounded fp16 arithmetic on the saved exp output. The second came from a description of the device arithmetic that was fitted after the first chain run.
This is one run on one boot.

## What this does and does not show
The stage s1 output equals the failed exp program's output byte for byte at both sizes, so the exp program is unchanged. The scaled chain after it meets the original exp bound at the registered inputs, with every stage equal to ties-to-even.
The failed exp run of the unary cell and the first four-call chain stay as recorded. The first chain failed criterion 3 on 21 lanes. The unary cell result of 21 of 23 is the record of its own run.
This receipt does not re-judge the unary cell.
No speed is claimed. The chain is five calls where the failed exp program was one, and the calibration median of each call was 0.306 to 0.360 ms.
The result covers these inputs and these shapes. Other data or other shapes are not covered, and the design was chosen from one earlier run. All arithmetic after the exp program ran on the ANE, and the host did none.

## Files
`results.txt` is the run file. `verdict.txt` is the gate output. `device-output/` holds the kept outputs and the calibration and loop logs of all ten stage runs. `inputs/` holds x and the three constants. `SHA256SUMS` lists every file.
