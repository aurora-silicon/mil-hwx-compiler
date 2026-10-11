# H14 matvec programs built by this compiler, run on an M2 Max, 2026-10-11

Three fp16 matvec programs were compiled with `mil-hwxc --target H14`. The shapes were K = N = 256, 512 and 1024.
They ran on an Apple M2 Max (T6021) under Linux through omarchy-ane's `ane-run --check matvec`. All 3 passed.

## Setup
- Compiler: mil-hwx-compiler `origin/main` 6f46c6a. It was built on an x86 Linux host with the documented GNUstep toolchain.
  The binary sha256 is `0a79d9c1a1236e99751499be1ca0a78ecd83d3073786686854cdedd810b6f62e`.
- Programs: `matmul(x[1,K], w[N,K], transpose_y)` with fp16 weights from a BLOBFILE.
- `make_matvec.py` writes the MIL, the weights and one random input. The weights come from numpy, seeded from K and N.
  Program and MIL hashes are in `results.txt`.
- Chip: Apple M2 Max (T6021). Kernel 7.1.12-2-12.6-sep, driver module `ane_t6021` build `bd34a12`.
- Runner: `ane-run --anec program-0.anec --in 0=x --out 0=y --weights w --check matvec --repeat 20`. It ran once per size
  under the `ane-run` lock (`run.sh`).
- `--check matvec` accepts a band of 2 ulp or 4 condition units, because the device's accumulation order is not known.
  It also checks that padding lanes are zero.
- Nothing else ran on the ANE during the run.

## Result
| K = N | program bytes | lanes in band | bit-exact lanes | max ulp | max condition units | padding |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 135,744 | 256 of 256 | 240 | 1 | 0.114 | zero |
| 512 | 528,960 | 512 of 512 | 482 | 15 | 0.122 | zero |
| 1024 | 2,101,824 | 1024 of 1024 | 971 | 16 | 0.111 | zero |

Every lane is inside the band. The larger ulp differences are lanes where the sum nearly cancels. In condition units, the
error relative to the sum of the absolute terms, the largest is 0.122. The 4-unit band has a large margin.

The first 4,672 bytes of the K = N = 256 program are the header and task streams. They equal the first 4,672 bytes of the
omarchy-ane fixture `fixtures/h14-anec/matvec/program-0.anec`. The weight section differs, because the weights differ.
`tests/test_h14_matvec_programs.py` pins the weight-independent head and the program size for all three shapes.

## Limits
- One run on one boot, 20 calls per program, one random input and one weight set per size.
- M = 1 only, and square shapes only. Other shapes have not run from this backend on a device. Neither have normalization,
  convolution or chained programs.
- The result is not bit-exact on every lane. The check allows 2 ulp or 4 condition units.
- No speed claim: this run checks results only. `run-log.txt` lists the dry run and the device run.
