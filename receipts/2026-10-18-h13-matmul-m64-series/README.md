# H13 constant-weight matmul series with 64 rows, on a 13-inch M1 and a 16-inch M1 Max, 2026-10-18

Ten single-program matmuls ran exact on both chips under `ane-run`, from 4.2 to 537 million multiply-adds per call. This is the first device run of the H13 `apple-parity-matvec` encoder through `ane-run`. The per-call time of the 16-inch is 1.9 to 2.8 times that of the 13-inch.

## Programs

Each is `matmul(x[64, K], w[N, K], transpose_y)` with constant fp16 weights, one program, built by `mil-hwxc --target H13` from `tests/h13_first_run/first_run.py` rungs 11 to 20. The weights and inputs are the deterministic patterns of `tests/h13_first_run/rungs.json`.

Rung 11: K = 256, N = 256. 4.19 million multiply-adds, 136,320 bytes, anec sha256 prefix `9bc8e76ac2bef8f0`.

Rung 12: K = 512, N = 256. 8.39 million, 267,392 bytes, `df8ff827584508d7`.

Rung 13: K = 256, N = 512. 8.39 million, 267,392 bytes, `aeb94e8aedc657d0`.

Rung 14: K = 512, N = 512. 16.8 million, 529,536 bytes, `ab8260b18afa97e9`.

Rung 15: K = 1024, N = 512. 33.6 million, 1,053,824 bytes, `ed4cacc193cc0104`.

Rung 16: K = 1024, N = 256. 16.8 million, 529,536 bytes, `5fe9187e0ed6bb40`.

Rung 17: K = 256, N = 1024. 16.8 million, 530,176 bytes, `42d3dbd88ee34b20`.

Rung 18: K = 1024, N = 1024. 67.1 million, 2,102,400 bytes, `4ec950a40139eb1c`.

Rung 19: K = 2048, N = 2048. 268 million, 8,394,496 bytes, `6c5334063b7166af`.

Rung 20: K = 4096, N = 2048. 537 million, 16,782,464 bytes, `32db4b24f39ca234`.

Other sizes in the probed range compile to one program per row or row group and cannot run through `ane-run`. The compiler emits one program for these ten shapes only. The probe is `receipts/2026-10-18-h13-matmul-m64-series/probe.txt`.

## Source

Compiler: main at 00c0e2af66608a0e380093d19bb5df9678deca24 for H13 code. The binary has sha256 `5a809a2b6b1d19341a862d65da3e8486600fda9517bb484c4658e2da5afb37a9` and was built from the same tree. Every program fits its surfaces under the device-free libane overrun check of `tests/libane_overrun.py`.

## Machines

13-inch M1 (T8103), Linux kernel 7.1.12-2-20261013.18-sep-ARCH, in-tree `ane` module, CPU governor performance. OS firmware 13.5, system firmware 27.0, iBoot-8422.141.2.

16-inch M1 Max (T6001), Linux kernel 7.1.12-2-12.6-sep-ARCH, in-tree `ane` module, governor performance. OS firmware 13.5, system firmware 26.6.2, iBoot-8422.141.2.

The firmware values are read from each device tree after the runs. Runner: `ane-run` with sha256 `480040b5f700f407f11563f5349e7614df9fefe0efe57327c68ebc35b9779805`. The GPU queue was paused and nothing else ran on the ANE.

## Command

Each rung ran in its own directory under the `ane-run` lock, in one process, with a deadline of 90 seconds (180 seconds for rungs 19 and 20):

`ane-run --anec program.anec --in 0=in0.bin --out 0=out0.bin --repeat 200 --time`

`in0.bin` is the packed input surface that `tools/h13_run_linux.py` sends through libane: 32,768 to 524,288 bytes. The wrapper stops after the first rung with a non-zero exit code or a deadline hit. Run 1 ran rungs 11 to 15 (13-inch 07:21Z, 16-inch 07:24Z) and run 2 ran rungs 16, 17, 14, 18, 19 and 20 (13-inch 08:12Z, 16-inch 08:15Z).

## Result

All 22 process runs returned 0, and the engine job counter rose by 200 per rung. Every output equals the exact fp16 reference of `tools/h13_reference.py`, unpacked the way `tools/h13_run_linux.py` unpacks it. The output bytes of each rung are identical on the two chips. `out0-rung-NN.bin` is that output.

Exec time per call in ms, as median with the 10th to 90th percentile range over 200 calls, then the busy time per call from the driver counter. The ratio is the 16-inch median over the 13-inch median.

Rung 11: 13-inch 0.024 (0.024 to 0.024), busy 0.0163. 16-inch 0.067 (0.066 to 0.067), busy 0.0554. Ratio 2.79.

Rung 12: 13-inch 0.033 (0.033 to 0.033), busy 0.0244. 16-inch 0.075 (0.074 to 0.077), busy 0.0641. Ratio 2.27.

Rung 13: 13-inch 0.033 (0.033 to 0.033), busy 0.0245. 16-inch 0.063 (0.061 to 0.065), busy 0.0502. Ratio 1.91.

Rung 14, run 1: 13-inch 0.037 (0.036 to 0.037), busy 0.0291. 16-inch 0.091 (0.089 to 0.093), busy 0.0790. Ratio 2.46.

Rung 14, run 2: 13-inch 0.036 (0.035 to 0.038), busy 0.0288. 16-inch 0.079 (0.076 to 0.093), busy 0.0711. Ratio 2.19.

Rung 15: 13-inch 0.058 (0.055 to 0.061), busy 0.0500. 16-inch 0.124 (0.110 to 0.127), busy 0.1112. Ratio 2.14.

Rung 16: 13-inch 0.041 (0.040 to 0.041), busy 0.0330. 16-inch 0.099 (0.098 to 0.102), busy 0.0860. Ratio 2.41.

Rung 17: 13-inch 0.041 (0.041 to 0.041), busy 0.0332. 16-inch 0.100 (0.096 to 0.104), busy 0.0873. Ratio 2.44.

Rung 18: 13-inch 0.074 (0.070 to 0.075), busy 0.0659. 16-inch 0.178 (0.168 to 0.186), busy 0.1662. Ratio 2.41.

Rung 19: 13-inch 0.231 (0.225 to 0.242), busy 0.2236. 16-inch 0.602 (0.569 to 0.632), busy 0.5886. Ratio 2.61.

Rung 20: 13-inch 0.406 (0.401 to 0.422), busy 0.4005. 16-inch 1.101 (0.994 to 1.181), busy 1.0849. Ratio 2.71.

The busy ratio is 2.47 to 2.71 over run 2 and 2.63 to 2.71 from rung 19 up. Between rungs 18 and 19 the busy time grows by 0.1577 ms on the 13-inch and 0.4224 ms on the 16-inch, a ratio of 2.68. Between rungs 19 and 20 it grows by 0.1769 and 0.4963 ms, a ratio of 2.81. From rung 19 to rung 20 the added busy time gives a marginal rate of 1.52 trillion multiply-adds per second on the 13-inch and 0.54 on the 16-inch.

Rungs 16 and 17 have the same work as rung 14 and swapped surface sizes. They take the same time as each other (ratio 0.99 to 1.00 on both chips). Both are above rung 14 in run 2, by 14 percent on the 13-inch and 8 to 30 percent on the 16-inch, depending on which cluster of rung 14 the comparison uses (see below). Rungs 12 and 13 have the same work too. The 13-inch times are equal and the 16-inch times differ, 0.075 against 0.063.

On the 16-inch, rung 14 in run 2 has two clusters of call times. The quantiles of the 200 calls are 0.075 (minimum), 0.076, 0.076, 0.079 (median), 0.092, 0.093 and 0.094 (99th percentile). The median lies at the boundary between a cluster near 0.076 and one near 0.092. Run 1 of rung 14 lies wholly in the upper cluster. The cause was not looked up.

## Procedure

Timing: 200 calls per rung, one process per rung, the per-call exec time from the driver. No temperature, fan or throttle reading was taken. The CPU governor was performance.

Reopen and recovery: each process opened the ANE device, ran its calls and exited with 0. The engine job counters rose by 200 per rung. No worker restart, reboot or self-test was needed, and no wedge occurred. The 8.4 and 16.8 MB programs of rungs 19 and 20 loaded and ran.

Dispatch trace: not applicable to a single-program ANE run.

## Limits

One run per rung and chip, two runs of rung 14. Two chips, one boot each.

The criterion is exact fp16 equality with the reference. The patterns of `rungs.json` keep the sums small.

Work and surface sizes grow together along the series.

## Source records

`SHA256SUMS` lists the files in this directory, which can be checked from here. `SHA256SUMS-not-archived` lists the source records in the lab archive: the device result files of both runs on both machines, the check files and the sealed program directories. They are not in this directory.
