# H14 scaled exp as one ANEC: preregistration

Written before any device run. The unary cell of `receipts/2026-10-13-h14-unary-m2/` required one ANEC per program, an exp bound of relative 2^-8 plus the absolute floor 2^-24, and a call time of
at most 0.60 ms. Its exp program (one ANEC) missed the bound. The five-call scaled chain (`receipts/2026-10-14-h14-exp-chain-d4-run-m2/`) met the bound, but it is five calls and not one program.
This cell runs the compiler's own output for the same arithmetic as ONE ANEC. The failed exp program, the first chain and the five-call chain stay as recorded. None of them is replaced.

## What runs
`mil-hwxc --target H14` compiles one MIL program per size (64 and 512 channels): `e = exp(x)`, `c = x * 16 K`, `g = c + 16`, `p = e * g`, `y = p * 0.0625`, with scalar constants and 16 K = fp16 0x1b18.
The compiler emits the three-task program Apple's compiler produced for that MIL. `research/oracles/h14/chain_exp_d4_c<C>.json` records it, minted compile-only on a macOS Mac with the compiler build the
record names. `tests/test_h14_parity.py` proves the emitted task words, constant section, program descriptor and tensor descriptors equal the record's, in both ANEC and HWX form.
`tests/test_h14_exp_chain_cli.py` proves that only the decoded form takes this path. The program file is one ANEC (`prog/expone_c<C>.anec`, 4736 bytes). `build-expchain.py` writes it and the input,
and `SHA256SUMS-inputs` lists their sha256. The input x is the saved input of the failed run, byte for byte. Every surface is fp16 (1, C, 1, 1) with one value per 64-byte channel.

## What is decoded, and what is not
Apple folded the scalar constants into task words. In the record, the first task is the failed exp program: its NE and tile blocks, scale word 0x3dc5 and 128-byte table equal the unary exp program's.
Only its routing differs, since its output goes to L2. The second task carries two PE words, 0x904 = {1.0, 16.0} and 0x908 = {16 K, 0}. The third carries the op word of the tensor-by-tensor mul and a post multiplier
0.0625. Reading a PE word as {multiplier, addend} is an inference from two scalar-constant records (add 0.5 and mul 0.5). It is not hardware documentation.
Not decoded: the order of the two slots in the second task, the internal precision of a slot or of the mul, and the routing between tasks. The arithmetic may not be that of five separate programs.
No stage output exists, so no stage criterion applies. The five-call chain's stage arithmetic is not assumed to carry over.

## Bounds from models, not predictions
This cell does not predict that the bound is met. Models of the fused tasks bound the outcome on the registered inputs, using the saved device output of the exp program. They are not evidence about the hardware.
- If the fused slots round to nearest, in one rounding or two, the result equals the five-call chain's: a largest error of 0.2825 percent at 64 channels and 0.3476 percent at 512, against 0.3906.
- If they round toward zero, 32 lanes at 64 channels and 250 at 512 change, and the largest error stays inside the bound (0.2838 and 0.3450 percent).
- A systematic one-ulp bias of the sum 16 + 16 K x would fail at 512 channels: 3 lanes outside for one ulp up (0.4187 percent) and 2 for one ulp down (0.3997 percent). The 512-channel slack is 0.043 percent.
A pass would show that the fused task rounds like one of the first two models on these lanes, and nothing more.

## How it runs
`run-expchain.sh` on an M2 Max (T6021) under Linux, through `ane-run` without `--check`, one call per size reading x and writing y. Per size: 20 calls with `--time`, then a back-to-back loop of about 30 seconds.
Both outputs are kept. A calibration or loop that fails stops that size, and nothing is rerun. The whole run has a 15 minute deadline in the script, and every call runs under `timeout -k 2`.
A TERM, INT or HUP sent to the script ends the running call and is recorded. The run needs the ANE to itself: the script counts other `ane-run` processes before and after, and the gate requires 0 and 0.
Two runs of about 31 seconds take about 70 seconds.

## Criteria (all frozen; `compare-expchain.py` judges, `test-compare-expchain.py` tests it against its checks and mutants)
Numerical, for 64 and 512 channels. They are the original unary criteria for exp:
1. x is the registered input: its sha256 is pinned and it equals the saved input of the failed run.
2. The output has zero padding and no NaN or infinity, and its calibration and loop files are bit-identical.
3. The output is within a relative error of 2^-8 of exp(x) on every lane, plus the absolute floor 2^-24. The bound did not change.
   The gate asserts both values and pins the sha256 of the `compare-unary.py` it imports them from.
4. The program is one ANEC: one program file per size, with the sha256 recorded in `SHA256SUMS-inputs` and in the run file.
Run identity, duty and time, reported apart and also required: `results.txt` names `target_s=30`, the T6021 chip, the program hash of each size as in `SHA256SUMS-inputs`, the sha256 of every output file
(checked against the files), exactly one SUMMARY line, and 0 other `ane-run` processes before and after. Also: every call's calibration median is at most 0.60 ms; every loop and the ticket have
duty at least 90 percent; total wall time is at most 15 minutes; both runs exit 0.

## What would change the plan
A refusal at load or a nonzero exit is recorded with the exact text. A failed criterion is reported with the lane, input and output. It is not rerun, averaged away or judged against a looser bound.
If the bound is missed, the cell is reported as failed and the five-call chain stays the only corrected implementation that has run. No speed is claimed either way: the call time is reported next to the failed
exp program's and the five-call chain's, and a pass shows an exp that meets the bound as one ANEC. It does not show the exp is faster.
