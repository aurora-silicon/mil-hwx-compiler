# H16G staged programs on an M4 Pro under Linux, run 2026-11-08

The staged H16G backend had only run on M4 hardware under macOS. This run
loads its programs on an M4 Pro (J616s, T6041) under Linux, through the ABI 3
accel node, and compares every output with a float64 CPU reference.

Every single-task program, and the two-task square matmul, loaded, ran and
matched. Every program with more tasks, and the single-task ALU and LUT
programs at 1024 and 2048, were refused by the kernel at load
(`H16_LOAD: Exec format error`) before reaching the engine. Those are
kernel-side acceptance limits, described below, not numerical failures.

## Setup

- Chip: Apple M4 Pro (J616s). Kernel `7.3.0-aurora-fleet30`. Runner: the
  ABI 3 test runner `ane-run.py`, which loads one HWX, poisons the output,
  runs each case 30 times and saves the last output.
- Programs: built by `build-kit1.py` and `build-kit3.py` from the compiler at
  this commit (seeded inputs; `SHA256SUMS-programs`). Inputs are standard
  normal (matmul right operand scaled by `1/sqrt(N)`, ALU and GELU inputs
  scaled by 2).
- `results.txt` is the unmodified runner output. `analysis.txt` is
  `analyze.py` run on the saved outputs.

## Results

Relative L2 error against the float64 reference rounded to fp16, the share of
bit-exact lanes, and the host p50 of one EXECUTE (submit to completion,
30 runs).

| Program | Form | Tasks | Result | rel L2 | exact | p50 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `matmul` 128 | runtime x runtime | 2 | match (x @ y) | 1.5e-4 | 83 % | 168 us |
| `matmul` 256 | runtime x runtime | 2 | match | 1.8e-4 | 79 % | 175 us |
| `matmul` 512 | runtime x runtime | 2 | match | 2.2e-4 | 73 % | 223 us |
| `matmul` 128 then `mul` 0.125 | fused output scale | 2 | match | 4.4e-5 | 97 % | 169 us |
| `add` 128 / 512 | binary ALU | 1 | match | 2.7e-4 | 89 % | 175 / 170 us |
| `mul` 128 / 512 | binary ALU | 1 | match | 3e-5 | 99.8 % | 173 / 198 us |
| `gelu` 128 / 512 | LUT | 1 | match | 8.4e-4 | 28 % | 162 / 147 us |
| `exp` 128 / 512 | LUT | 1 | match | 7.1e-4 | 66 % | 167 / 147 us |
| `add` `[1,128,768]` | ALU, 512 x 512 contiguous view | 1 | match | 2.7e-4 | 89 % | 176 us |
| `softmax` `[1,1,128,128]` | row softmax | 7 | refused at load | | | |
| `matmul` 768 / 896 / 1536 / 3072 | tiled runtime matmul | 4 / 5 / 7 / 25 | refused at load | | | |
| `add`, `mul`, `gelu`, `exp` 2048; `gelu`, `add` 1024 views | ALU / LUT | 1 | refused at load | | | |
| four-head attention `[1,64,4,192]` | decoded mixed form | 10 | refused at load | | | |

The GELU and exponential tables are approximations: their errors match the
same tables on H14. The engine does not round a plain fp16 add exactly
(89 % bit-exact); every differing lane is one unit in the last place.

## Why the kernel refuses the larger programs

The kernel's H16 HWX loader accepts only the program-descriptor values seen
in its first corpus: the word at `0x860` must be one of `0x10`, `0x12`, `0x1f`,
`0x20`, and the word at `0x890` (`0x8b0` in the mixed descriptor) must be one
of an enumerated list. In the staged programs both scale with the program:

| Program | Tasks | `0x860` | `0x890` / `0x8b0` |
| --- | ---: | --- | ---: |
| `add` 512 | 1 | `0x10` | 23 |
| `add` 1024 view | 1 | `0x10` | 92 |
| `add` 2048 | 1 | `0x10` | 368 |
| `gelu` 1024 view / 2048 | 1 | `0x10` | 61 / 245 |
| `matmul` 512 | 2 | `0x1f` | 0 |
| `matmul` 768 / 896 / 1536 / 3072 | 4 / 5 / 7 / 25 | `0x2e` / `0x3c` / `0x58` / `0x154` | 0 |
| `softmax` 128 | 7 | `0x58` | 1 |

The first grows with the task count and the second with the element count,
so the fix belongs in the loader's checks rather than in the compiler. The
compiler's own `tools/h16g_validate.py` mirrors the same lists and reports
these programs as `h16g.program-record-code` / `h16g.program-format-code`.

## Reproduce

```
python3 build-kit1.py build/mil-hwxc kit1      # numpy; writes programs, inputs, references
python3 build-kit3.py build/mil-hwxc kit3
# on the J616s, per program:
ane-run.py --hwx kit1/mm512/program-0.hwx --case in0.bin,in1.bin --runs 30 --out-dir out
python3 analyze.py . out
```
