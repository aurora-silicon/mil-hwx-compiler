# H13 elementwise programs built by this compiler, run on a 13-inch M1 and an M1 Max, 2026-10-11

Eight programs were compiled with `mil-hwxc --target H13`. They ran under Linux through omarchy-ane's `ane-run`. The chips
were a 13-inch M1 (T8103) and a 16-inch M1 Max (T6001). All 8 passed on both.

## Setup
- Compiler: mil-hwx-compiler `origin/main` 6f46c6a. It was built on an x86 Linux host with the documented GNUstep
  toolchain. Binary sha256 `0a79d9c1a1236e99751499be1ca0a78ecd83d3073786686854cdedd810b6f62e`.
- Programs: `tests/fixtures/h13/*.mil`, one program per file, tensor shape [1, 64, 1, 1] in fp16. Program and MIL hashes
  are in `PROVENANCE.txt`.
- Two H13 encoders produced them. `h13-source-qualified` made add, mul and mul-scalar (4,736 bytes each). Mul and
  mul-scalar are the same program, because the scalar travels as a second operand surface. `h13-oracle-parity` made relu,
  add-scalar, real-div-scalar, clip-low and clip-high.
- Runner: `ane-run` from omarchy-ane main plus the `--tile-shift` option, under the `ane-run` lock.
- `gate_h13.py` writes the inputs and runs the program once. It compares all 64 lanes with an exact reference (round half
  away from zero, exact comparison) and checks that padding is zero. `run.sh` is the ticket that calls it.
- Kernel 7.1.12-2-12.6-sep, in-tree `ane` module. Nothing else ran on the ANE during a run.

## Result
All 8 programs passed on both chips: 64 of 64 lanes bit-exact, zero nonzero padding lanes. `results-m1.txt` and
`results-m1-max.txt` hold the latest run on each chip. The last three runs on each chip passed 8 of 8.

## Limits
- One input per operand per run, 64 lanes. These are single elementwise programs. Matvec, normalization, convolution and
  chained programs have not run from this backend on an M1.
- The new encoder's add is 4,736 bytes. It is not byte-identical to the older 20,992-byte add fixture in omarchy-ane,
  which also ran on an M1. So `tests/test_h13_fixture_programs.py` pins this compiler's own output. It shows the output
  is stable. It does not show the output equals Apple's.
- An earlier gate run on the 13-inch M1 passed 7 of 8 and failed mul-scalar. The gate script set the scalar surface
  wrongly. The script was fixed before the runs in this folder, and the program was unchanged.
- No speed claim: these runs check results only.
