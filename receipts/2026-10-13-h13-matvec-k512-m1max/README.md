# H13 matvec, K = N = 512, on an M1 Max, run 2026-10-12

One fp16 matvec program built by `mil-hwxc --target H13` ran on an Apple M1 Max (T6001) under Linux. All 512 output lanes
were inside the error band and every padding value was zero. This is the first H13 matvec from this compiler that has
reached a numerical comparison on a device; the earlier attempt (`receipts/2026-10-12-h13-matvec-m1/`) stopped before it.

## Why a new gate
`ane-run --check matvec` reads the dense layout of the H14 programs. Output N and input K are each one row of halves.
The H13 program declares both tensors differently. Each fp16 value sits in its own 64-byte channel. The input on channel 5
is nchw `[1, K, 1, 1, 64, 64]` and the output on channel 4 is `[1, N, 1, 1, 64, 64]`. That checker answers "unexpected io
shape" for it.
`tests/hardware/gate_h13_matvec.py` packs and reads that channel layout and does its own comparison. It still goes
through `ane-run` and libane. The load-time checks of the program stay in force.

## Setup
- Compiler: mil-hwx-compiler `origin/main` c8506137 (v0.2.0). One program, 529,024 bytes, sha256
  `3128f95559d3b758c416e34728aa534fc93d2b3d9e6c9875d64d75718daabd9a`.
- Program source: `matmul(x[1,512], w[512,512], transpose_y)` with fp16 weights from a BLOBFILE. The weights come from
  `receipts/2026-10-11-h14-matvec-m2/make_matvec.py 512 512 DIR` (numpy seed `512 * 100003 + 512`), then
  `mil-hwxc --mil DIR/matvec.mil --model-root DIR/model --target H13 --output DIR/out`. The gate rebuilds the same weights
  from the seed and refuses to run if they differ from the package's `w.raw.f16`.
- Input: 512 fp16 values from the same generator, packed one per 64-byte channel into a 32,768-byte surface.
- Chip: Apple M1 Max (T6001). Kernel 7.1.12-2-12.6-sep, in-tree `ane.ko`. Runner: the `ane-run` of omarchy-ane's slot
  tree, sha256 prefix `480040b5f700f407`.
- Run: one call (`--repeat 1`) under the `ane-run` lock, from a scheduler ticket, on 2026-10-12 about 12:43Z.
- Reference: float64 accumulation of the fp16 products, rounded to fp16 with ties away from zero.
- Band: a lane passes inside 2 ulp or 4 condition units. One condition unit is 2^-11 times the sum of the absolute
  products. `ane-run --check matvec` uses the same band for H14. Padding values in the 32,768-byte output must all be zero.

## Result
`results.json`: `pass` true, 512 of 512 lanes in band, padding nonzero 0, largest absolute difference from the float64 sum
0.00755. The gate does not count bit-exact lanes, so no bit-exact figure is given.

## Limits
- One run, one call, one weight set and one input, one chip. The 13-inch M1 was not run with this gate.
- K = N = 256 is not run: libane refuses that program at load ("surface channel 5 needs 32768 bytes but is allocated
  16384"). K = N = 1024 compiles to five programs and the gate runs only complete one-program packages.
- No speed claim. H13 matvec stands at 1 of 3 shapes device-verified.
- The self-test (`gate_h13_matvec.py --self-test`) checks the packing and the comparison with made-up outputs. It does not touch a device.
