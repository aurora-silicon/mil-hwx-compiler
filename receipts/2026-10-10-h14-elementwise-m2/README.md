# H14 elementwise programs built by this compiler, run on an M2 Max, 2026-10-10

Eight programs were compiled with `mil-hwxc --target H14`. They ran on an Apple M2 Max (T6021) under Linux through omarchy-ane's `ane-run --check`. This is one run on one boot.

## Setup
- Compiler: mil-hwx-compiler `origin/main` 6f46c6a. It was built on an x86 Linux host with the documented GNUstep toolchain (`scripts/verify-linux-compiler.sh`). The binary sha256 is 0a79d9c1a1236e99751499be1ca0a78ecd83d3073786686854cdedd810b6f62e. Each program compiles in under half a second.
- Programs: `tests/fixtures/h14/*.mil`, one program per file, tensor shape [1, 512, 1, 1] in fp16. The program and MIL hashes are in `PROVENANCE.txt`.
- Chip: Apple M2 Max (T6021). Kernel 7.1.12-2-12.6-sep-ARCH. Driver module `ane_t6021` build `bd34a12`.
- Runner: `ane-run --anec program-0.anec --in ... --check OP --repeat 20` from omarchy-ane, run under the `ane-run` lock. `--check OP` compares the output with the reference for the op. The arithmetic ops round half away from zero. Every padding lane must be zero.
- Nothing else ran on the ANE during the run.

## Result
All 8 passed. Each of add, mul, relu, add-scalar, mul-scalar, real-div-scalar, clip-low and clip-high had 512 of 512 valid lanes bit-exact against the reference, zero padding lanes and exit code 0. The raw output is `results.txt`. `run.sh` is the script that ran it.

Each program is also byte-identical to the H14 fixture program that omarchy-ane checks in (`fixtures/h14-anec/<op>/program-0.anec`). The `add` program was proven on this chip before. The other seven had not run from a compiler-built package.

## Limits
- One run on one boot. 20 calls per program. One random input per operand, in the sparse planes the runner generates.
- These are single elementwise programs. Matvec, normalization, convolution and chained programs have not run from this backend on a device.
- The run covers programs this compiler produces. It shows they execute correctly. It does not show they are faster than Apple's compiler output.
