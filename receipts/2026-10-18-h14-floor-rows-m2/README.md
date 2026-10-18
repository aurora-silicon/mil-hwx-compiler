# H14 programs at 200 calls with timing, on an M2 Max, 2026-10-18

Fifteen programs that already ran on an M2 Max ran again under `ane-run` with `--repeat 200 --time`. All 15 passed the device-side check. The per-call time of every program lies between 0.306 and 0.457 ms.

## Programs

The programs were built by `mil-hwxc --target H14`. Their sha256 equals the program sha256 of the earlier receipts `2026-10-10-h14-elementwise-m2`, `2026-10-11-h14-matvec-m2` and `2026-10-13-h14-matvec-rows-m2`, so the weights and inputs of the matvecs are the same too.

Eight elementwise programs on [1, 512, 1, 1] fp16: add, mul, relu, add-scalar, mul-scalar, real-div-scalar, clip-low and clip-high. Sizes 4,480 to 20,800 bytes.

Three square matvecs with one row: K = N = 256, 512 and 1024 (135,744, 528,960 and 2,101,824 bytes).

Four matvecs with more rows. They have 2 rows at K = N = 1024, 8 rows at K = N = 512, 8 rows at K = N = 1024 and 64 rows at K = N = 1024.

The elementwise inputs are 32,768 bytes each: one fp16 at the start of every 64-byte plane, zero elsewhere, from a seeded generator. The matvec inputs are the files of `receipts/2026-10-11-h14-matvec-m2/make_matvec.py` and `receipts/2026-10-13-h14-matvec-rows-m2/build-rows.py`.

## Source

Compiler: main at 00c0e2af66608a0e380093d19bb5df9678deca24 for the H14 code, which the last compiler change did not touch. The binary has sha256 `5a809a2b6b1d19341a862d65da3e8486600fda9517bb484c4658e2da5afb37a9`.

## Machine

14-inch M2 Max (T6021), Linux kernel 7.1.12-2-20261013.18-sep-ARCH, in-tree `ane_t6021` module of that kernel, CPU governor performance.

OS firmware 13.5, system firmware 27.0, iBoot-8422.141.2, read from the device tree after the runs.

Runner: `ane-run` with sha256 `943ae8c67598175a542f9a2c370852d60ba9a810cb1e8db11fed92efc2daf2e1`. The GPU queue was paused and nothing else ran on the ANE.

## Command

One `ane-run` process per program, under the `ane-run` lock, in the program's directory. The deadline was 60 seconds for elementwise programs and 90 seconds for matvecs. The wrapper stops after the first non-zero exit code.

Elementwise: `ane-run --anec program.anec --in 0=in0.f16 [--in 1=in1.f16] --out 0=out0.f16 --check OP --repeat 200 --time`

Matvec: `ane-run --anec program.anec --in 0=x.f16 --out 0=y.f16 --weights w.raw.f16 --check matvec --repeat 200 --time`

`--check` runs on the device side after the calls and makes `ane-run` exit non-zero on a mismatch. The elementwise programs ran at 07:46Z and the four row programs at 08:22Z, on one boot.

## Result

All 15 runs returned 0, and the engine job counter rose by 200 per program. Exec time per call in ms, as median with the 10th to 90th percentile range, then the busy time per call from the driver counter.

Add: 0.310 (0.258 to 0.361), busy 0.3125. 512 of 512 lanes bit-exact with zero padding.

Mul: 0.310 (0.258 to 0.360), busy 0.3109. 512 of 512 lanes bit-exact with zero padding.

Relu: 0.306 (0.256 to 0.357), busy 0.2957. 512 of 512 lanes bit-exact with zero padding.

Add-scalar: 0.307 (0.256 to 0.357), busy 0.3087. 512 of 512 lanes bit-exact with zero padding.

Mul-scalar: 0.307 (0.256 to 0.357), busy 0.3041. 512 of 512 lanes bit-exact with zero padding.

Real-div-scalar: 0.307 (0.255 to 0.376), busy 0.3203. 512 of 512 lanes bit-exact with zero padding.

Clip-low: 0.306 (0.255 to 0.407), busy 0.3103. 512 of 512 lanes bit-exact with zero padding.

Clip-high: 0.307 (0.256 to 0.407), busy 0.3176. 512 of 512 lanes bit-exact with zero padding.

1 row 256 x 256: 0.357 (0.256 to 0.407), busy 0.3369. All 256 lanes within the band, 240 bit-exact, largest difference 1 ulp, 0.114 condition units.

1 row 512 x 512: 0.361 (0.306 to 0.412), busy 0.3703. 512 lanes in band, 482 bit-exact, 15 ulp, 0.122 condition units.

1 row 1024 x 1024: 0.360 (0.357 to 0.411), busy 0.3834. 1024 lanes in band, 971 bit-exact, 16 ulp, 0.111 condition units.

2 rows 1024 x 1024: 0.361 (0.357 to 0.440), busy 0.3862. 2048 lanes in band, 1932 bit-exact, 29 ulp, 0.109 condition units.

8 rows 512 x 512: 0.358 (0.357 to 0.411), busy 0.3764. 4096 lanes in band, 3865 bit-exact, 1554 ulp, 0.197 condition units.

8 rows 1024 x 1024: 0.407 (0.357 to 0.458), busy 0.3975. 8192 lanes in band, 7690 bit-exact, 125 ulp, 0.117 condition units.

64 rows 1024 x 1024: 0.457 (0.357 to 0.508), busy 0.4356. 65536 lanes in band, 61604 bit-exact, 1039 ulp, 0.129 condition units.

The matvec check accepts a lane within 2 ulp or 4 condition units, because the accumulation order of the device is not known. The large ulp values are single lanes where the sum nearly cancels, and the largest condition value is 0.197. Padding lanes are zero in every output.

The exec values lie on a grid of about 0.05 ms, the step of the timer. The busy time is a mean over 200 jobs and is finer. From the 1-row to the 64-row program with the same weights, the compute grows 64 times and the busy time grows by 0.0522 ms (14 percent). The 2-row 1024 x 1024 and 8-row 512 x 512 programs have the same work and a fourfold weight size difference, and their medians are 0.361 and 0.358.

## Procedure

Timing: 200 calls per program, one process per program, the per-call exec time from the driver. No temperature, fan or throttle reading was taken. The CPU governor was performance.

Reopen and recovery: each process opened the ANE device, ran its calls and exited with 0. The job counters rose by 2,200 over the elementwise and 1-row programs and by 800 over the four row programs. No worker restart, reboot or self-test was needed, and no wedge occurred.

Dispatch trace: not applicable to a single-program ANE run.

## Limits

One run per program on one boot. The medians of the elementwise programs differ by 0.004 ms and the timer step is 0.05 ms, so they are one value.

The exec distributions of the row programs overlap: the 10th percentile of the 64-row program equals that of the 2-row program. The busy time shows the same order.

The medians of 2026-10-13 (0.357 to 0.512 ms, loops of 73,394 to 110,803 calls, another boot) agree in range with these.

## Source records

`SHA256SUMS-not-archived` lists the sha256 of the source records in the lab archive: the device result files of both runs, the per-program timing files and the sealed program directories. They are not in this directory.
