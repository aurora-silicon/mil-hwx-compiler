# The scaled exp as one ANEC on an M2 Max, 2026-10-14

The cell of `receipts/2026-10-14-h14-exp-scaled-one-anec-m2/` ran once on an M2 Max (T6021) under Linux, from 08:05:55Z to 08:06:48Z. The run file counts other `ane-run` processes at the start and at the end, and both counts are 0.
The scheduler's window had the idle filler off. That was logged outside this archive, and the run file does not record it.
One ANEC per size ran, at 64 and 512 channels. Each is `prog/expone_c<C>.anec`, 4736 bytes. `mil-hwxc` emits it for the MIL of exp, mul by 16 K, add 16, mul, mul by 0.0625. Apple's compiler made that program in three tasks.
The input is the saved input of the failed exp run, byte for byte. The criteria did not change. Two explanatory sections of the preregistration (what is decoded, and the bounds from models) were rewritten after review on 2026-10-14 at 01:52 CDT, before the run at 03:05 CDT.
The squash merge replaced that commit, and the history of the pull request branch shows it.
The exp bound is relative 2^-8 plus the absolute floor 2^-24. Each call must take at most 0.60 ms.
`compare-expchain.py` judged the saved output off the device (`verdict.txt`).

## Result
| Criterion | Result |
|---|---|
| Output within the exp bound | Met. Largest relative error 0.283 percent at 64 channels and 0.348 percent at 512. No lane outside (bound 0.391). The failed exp program had 0.446 and 0.429 percent |
| x pinned, zero padding, finite values, identical calibration and loop outputs | Met |
| One ANEC per size, with the registered program hash | Met. 4736-byte programs `2d8bc52d06db401e` (64) and `e78ed6f6847d09b2` (512), as in `SHA256SUMS-inputs` |
| Run identity, duty and time | Met. T6021 chip, 0 other `ane-run` at the start and the end, registered output hashes, loop duty 99.6 percent, calibration medians 0.361 ms (64) and 0.356 ms (512), 53.9 s in total |

Gate exit 0: numerical criteria PASS, run identity, duty and time PASS. The preregistration made no prediction. It gave model bounds, and the result is inside them.

## Comparison with the five-call chain and the failed exp program
`analysis.txt` holds the comparison, from `compare-with-chain.py`, which only reads the saved output.
- The one-ANEC output equals the five-call chain's final output (`receipts/2026-10-14-h14-exp-chain-d4-run-m2/`) byte for byte at both sizes: 0 differing lanes of 64 and of 512.
  The output equals the chain's bit for bit on these lanes. Round-toward-zero models of the fused tasks are excluded here, and the data are consistent with rounding to nearest. They do not tell one rounding from two.
- It differs from the failed exp program's output on 49 lanes at 64 channels and 407 at 512.
- The first task is the failed exp program, and the chain's correction follows it inside the same ANEC.
This is one run on these inputs and these two sizes. The decode of the fused tasks still does not give their order or internal precision, and this run does not either. It shows the result on these lanes.

## What this does and does not show
The compiler's own output for the scaled exp is one ANEC per size and meets the original exp bound on the original input, within the original call time. That meets the criteria the unary cell set for its exp programs on these inputs: one ANEC, the exp bound and at most 0.60 ms.
The failed exp run of the unary cell, the first four-call chain and the five-call chain stay as recorded. This receipt does not re-judge the unary cell.
No speed is claimed. The call took 0.356 to 0.361 ms, at the top of the 0.306 to 0.360 ms range of the single programs measured on this M2 in these runs, 0.001 ms above it. These are 20-call medians with a spread of about 0.05 ms. This one program does the work of five calls of the earlier chain.
The result covers these inputs and these shapes. Other data or other sizes are not covered.

## Files
`results.txt` is the run file. `verdict.txt` is the gate output. `device-output/` holds the kept outputs and the calibration and loop logs of both runs. `inputs/` holds x. `prog/` holds the two 4736-byte programs that ran, so their size and sha256 can be checked against `SHA256SUMS-inputs` of the cell. `build-and-test-071bc9a.txt` is the literal build and test output at the source tree the
compiler was built from (home paths replaced). It names two commits that the squash merge replaced. Its tree hash 7b9c0db7 equals the tree of the compile commit ab82de0. The only later change in this archive is under `receipts/` and in `CHANGELOG.md`. `SHA256SUMS` lists every file.
