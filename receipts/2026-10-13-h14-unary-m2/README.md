# H14 unary programs on an M2 Max, run 2026-10-13

Twenty-three fp16 unary programs built by `mil-hwx-compiler` ran on an Apple M2 Max (T6021) under Linux through omarchy-ane's `ane-run`.
All 23 ran to completion. The numerical gate passed 21 of them and failed both `exp` programs. Their worst lanes exceed the relative-error
bound by at most 0.06 percentage points. So the cell is not accepted as a whole.
`PREREGISTRATION.md` was committed before the run. `results.txt` is the unmodified output of `run-unary.sh`.
The numerical verdict and the duty and timing results are kept apart below. No numerical claim rests on duty or timing.

## Setup
- Programs: the oracle MIL of each case, built by `build-unary.py`. Their sizes and sha256 are pinned by `tests/test_h14_unary_programs.py`.
- Chip: Apple M2 Max (T6021). Kernel 7.1.12-2-20261012.16-sep. Runner: the `ane-run` of omarchy-ane's slot tree (sha256 prefix `943ae8c67598175a`), used without a check.
- Run: 2026-10-13, one scheduler ticket submitted 13:53:24Z. Per program, 20 calls and then a loop of about 30 seconds. Both output files were kept.
- Inputs: uniform over [-8, 8], `sqrt` over [0, 64] and `rsqrt` over [0.0625, 64], with fixed values included. The 23 input files the device ran are in
  `inputs/`, with the MIL of each case. Their hashes equal the frozen staged files and a fresh rebuild.

## Numerical acceptance
The verdict comes only from `compare-unary.py`, run on the saved output (`verdict.txt`). Bounds were fixed before the run. For each program the gate
requires both output files, identical to each other, no NaN or infinity, zero padding, and every lane inside the bound. For `abs` the bound is the same
bits as the correctly rounded result.

| program | lanes | bit-exact | max abs error | max relative error | bound | verdict |
|---|---:|---:|---:|---:|---|---|
| unary_abs_c1024 | 1024 | 1024 | 0.00000 | 0.00000 | bits 0 | PASS |
| unary_abs_c128 | 128 | 128 | 0.00000 | 0.00000 | bits 0 | PASS |
| unary_abs_c2048 | 2048 | 2048 | 0.00000 | 0.00000 | bits 0 | PASS |
| unary_abs_c256 | 256 | 256 | 0.00000 | 0.00000 | bits 0 | PASS |
| unary_abs_c4096 | 4096 | 4096 | 0.00000 | 0.00000 | bits 0 | PASS |
| unary_abs_c512 | 512 | 512 | 0.00000 | 0.00000 | bits 0 | PASS |
| unary_abs_c64 | 64 | 64 | 0.00000 | 0.00000 | bits 0 | PASS |
| unary_exp_c512 | 512 | 141 | 7.43584 | 0.00429 | rel 0.0039062 | FAIL |
| unary_exp_c64 | 64 | 25 | 6.43012 | 0.00337 | rel 0.0039062 | FAIL |
| unary_gelu_c512 | 512 | 204 | 0.00451 | 0.06912 | abs 0.015625 | PASS |
| unary_gelu_c64 | 64 | 28 | 0.00294 | 0.06254 | abs 0.015625 | PASS |
| unary_leaky_relu_c512 | 512 | 512 | 0.00000 | 0.00000 | ulp 1 | PASS |
| unary_leaky_relu_c64 | 64 | 64 | 0.00000 | 0.00000 | ulp 1 | PASS |
| unary_rsqrt_c512 | 512 | 262 | 0.00086 | 0.00148 | rel 0.0019531 | PASS |
| unary_rsqrt_c64 | 64 | 37 | 0.00060 | 0.00114 | rel 0.0019531 | PASS |
| unary_sigmoid_c512 | 512 | 119 | 0.00311 | 0.02940 | abs 0.0039062 | PASS |
| unary_sigmoid_c64 | 64 | 18 | 0.00305 | 0.02633 | abs 0.0039062 | PASS |
| unary_silu_c512 | 512 | 179 | 0.01484 | 0.10632 | abs 0.015625 | PASS |
| unary_silu_c64 | 64 | 22 | 0.01484 | 0.07863 | abs 0.015625 | PASS |
| unary_sqrt_c512 | 512 | 389 | 0.00453 | 0.00092 | rel 0.0019531 | PASS |
| unary_sqrt_c64 | 64 | 53 | 0.00369 | 0.00067 | rel 0.0019531 | PASS |
| unary_tanh_c512 | 512 | 350 | 0.00143 | 0.00514 | abs 0.0039062 | PASS |
| unary_tanh_c64 | 64 | 42 | 0.00135 | 0.00401 | abs 0.0039062 | PASS |

Result: 23 expected cases, 23 judged, 2 failed. Both failures are `exp`.

### exp
Nine lanes of the 512-channel program and two lanes of the 64-channel program are outside the bound of 2^-8 (0.391 percent).
The largest relative error is 0.446 percent. The failing lanes are in the tails, at inputs from 5.9 to 7.5 and from -7.6 to -6.1.
Over [-4, 4] the largest relative error is 0.233 percent. Over the tails it is 0.43 percent (`analysis.txt`).
This run does not show why the error grows with the input magnitude, and no cause is claimed.

### silu and gelu
`silu` passed with little margin: its largest absolute error is 0.0148 against a bound of 0.0156, at an input of 0.5, at both sizes. Where its output
is small the relative error reaches 10.6 percent. `gelu` has a largest absolute error of 0.0045.
The two forms of gelu differ by less than the device's own error. So the data do not say which form the hardware follows.
Against the erf form the mean absolute error is 0.00031 (512 channels) and 0.00037 (64 channels). Against the tanh form it is 0.00035 and 0.00037.
By lane count, 346 of 512 lanes and 53 of 64 are closer to the tanh form.

## Duty and timing
This section does not bear on the numerical verdict.

| program | calls in loop | median call ms | loop seconds | loop duty % |
|---|---:|---:|---:|---:|
| unary_abs_c64 | 59,055 | 0.464 | 29.2 | 99.8 |
| unary_abs_c128 | 65,502 | 0.462 | 32.3 | 99.8 |
| unary_abs_c256 | 59,055 | 0.512 | 29.8 | 99.7 |
| unary_abs_c512 | 59,055 | 0.508 | 30.5 | 99.8 |
| unary_abs_c1024 | 55,762 | 0.508 | 28.8 | 99.8 |
| unary_abs_c2048 | 59,055 | 0.508 | 30.3 | 99.8 |
| unary_abs_c4096 | 58,593 | 0.508 | 30.3 | 99.8 |
| unary_exp_c64 | 63,559 | 0.508 | 33.2 | 99.8 |
| unary_exp_c512 | 59,055 | 0.508 | 31.0 | 99.8 |
| unary_gelu_c64 | 59,055 | 0.509 | 31.5 | 99.8 |
| unary_gelu_c512 | 59,055 | 0.512 | 31.8 | 99.7 |
| unary_leaky_relu_c64 | 59,055 | 0.508 | 31.8 | 99.8 |
| unary_leaky_relu_c512 | 56,390 | 0.508 | 30.3 | 99.8 |
| unary_rsqrt_c64 | 59,055 | 0.508 | 32.2 | 99.8 |
| unary_rsqrt_c512 | 58,939 | 0.509 | 32.4 | 99.8 |
| unary_sigmoid_c64 | 59,055 | 0.508 | 32.4 | 99.8 |
| unary_sigmoid_c512 | 59,055 | 0.508 | 32.7 | 99.8 |
| unary_silu_c64 | 49,261 | 0.517 | 27.6 | 99.7 |
| unary_silu_c512 | 53,667 | 0.514 | 30.2 | 99.8 |
| unary_sqrt_c64 | 53,667 | 0.559 | 30.7 | 99.8 |
| unary_sqrt_c512 | 53,763 | 0.559 | 31.2 | 99.8 |
| unary_tanh_c64 | 53,667 | 0.558 | 31.4 | 99.8 |
| unary_tanh_c512 | 59,055 | 0.512 | 34.6 | 99.8 |

`SUMMARY pass=23 fail=0 skipped_loops=0 total_wall_s=716.4 ticket_duty_pct=99.7`

Median call times span 0.462 to 0.559 ms. Loop duty is 99.7 to 99.8 percent in every loop and 99.7 percent over the ticket. The run took 716 seconds.

## Predictions against the result
Numerical predictions:
1. `abs` bit-exact at all seven sizes. Held. Every lane matched the same bits.
2. `leaky_relu` within 1 ulp. Held. Every lane is bit-exact.
3. `sqrt` and `rsqrt` within a relative error of 2^-9. Held. The largest are 0.00092 and 0.00148. The bound is 0.00195.
4. `exp` within a relative error of 2^-8. Did not hold. 11 lanes are outside. The largest is 0.446 percent.
5. `sigmoid` and `tanh` within 2^-8 absolute. `silu` and `gelu` within 2^-6. Held. The largest are 0.0031, 0.0014, 0.0148 and 0.0045.
6. `gelu` closer to the erf form than to the tanh form. Did not hold by lane count. The tanh form is closer on most lanes. The mean absolute error is nearly equal.
7. Every program runs to completion with finite output and zero padding. Held.
Run predictions:
8. Duty at least 90 percent and a wall time of at most 15 minutes. Held. Duty was 99.7 percent and the wall time 11.9 minutes.
9. Median call time of at most 0.60 ms. Held. The largest is 0.559 ms.

## How the gate changed after review
The gate in the first commit (`ed6b75aa`) gave the verdict above. After the run, two reviews found that it could let a wrong case set or a wrong
reference through. It did not know that 23 cases were expected. It accepted `-0.0` where `abs` must give `+0.0`. It ignored the `rsqrt` epsilon and the
`leaky_relu` slope of the MIL.
The corrected gate requires exactly the case list the builder uses (`unary_cases.py`). It compares `abs` bit for bit. It reads each operation's
parameters from the case's own MIL, as fp32. It fails on any parameter or string argument it does not know. It prints the input hash.
`test-compare-unary.py` has 26 checks. It also runs them against 13 deliberately broken copies of the gate and requires each to be caught.
On the first gate, 10 of the first 23 checks failed.
The corrected gate gives the same result as the first one on every program, number for number (`verdict-preregistered-gate-ed6b75aa.txt`).
The change did not alter any verdict.

## Reproduce
From this directory. The first line checks the archived files and the second judges them.
```
sha256sum -c SHA256SUMS-inputs && sha256sum -c SHA256SUMS-outputs
python3 compare-unary.py inputs device-output
```
Expected: the 21 PASS lines, the two `exp` FAIL lines, "23 expected cases, 23 judged, 2 failed", and exit status 1. `build-unary.py` rebuilds
the programs and inputs. A different numpy version may draw different inputs, so judge against the archived `inputs/`.

## Limits
- One run on one boot, one input set per program, fp16 only.
- Accuracy is measured inside the input domains above and says nothing outside them.
- The cause of the `exp` error and the form that `gelu` follows are not established.
- A gate that passes a program says the output was inside the bound on these inputs. It is not a proof for other inputs.

