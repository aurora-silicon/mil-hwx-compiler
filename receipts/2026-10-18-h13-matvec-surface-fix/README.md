# H13 matvec surface fix: device run on a 13-inch M1 and a 16-inch M1 Max, 2026-10-18

libane refused the K = 256, N = 512 matvec of the first-run kit on a 13-inch M1 before the fix. With the fix it runs on that machine and on a 16-inch M1 Max. Every output equals the exact fp16 reference. Two further first-execution programs ran exact on both chips.

## Before the fix

The program of rung 04 was built from main at 002a84dc. It has 529,024 bytes, the anec sha256 prefix `f9cf6cdf21521a3c` and a 16,384-byte input surface. It was submitted through `ane-run` on the 13-inch M1 at 05:36Z. libane refused it before any device job with this message:

`surface channel 5 needs 32768 bytes but is allocated 16384; refusing the program`

The job counter did not move, and the engine stayed healthy. The device-free port in `tests/libane_overrun.py` gives the same message for this program.

## After the fix

Source: compiler commit c8403715380d64ed6b2ac57e640d1222b9c12f68 (on main 002a84dc), binary sha256 `5a809a2b6b1d19341a862d65da3e8486600fda9517bb484c4658e2da5afb37a9`. The programs come from `tests/h13_first_run/first_run.py` (rungs 4, 9 and 10) and were packed into surfaces by the runner code of `tools/h13_run_linux.py`.

- Rung 04 is a matvec with K = 256 and N = 512 and constant weights. The program has 529,024 bytes and the anec sha256 prefix `9141ef2fd0b06060`. The input surface has 32,768 bytes.
- Rung 09 is a matvec with K = 512 and N = 512 and constant weights. The program has 529,024 bytes and the prefix `908ef299c1f7d760`. The input surface has 32,768 bytes.
- Rung 10 is a runtime matmul of 128 by 128 by 128. The program has 21,632 bytes and the prefix `65922d2d4adf697e`. It has two input surfaces of 32,768 bytes.

The output surface of each program is 32,768 bytes. Rung 04 is the program that was refused, now with a 32,768-byte input surface.

## Machines

- 13-inch M1 (T8103), Linux kernel 7.1.12-2-20261013.18-sep-ARCH, in-tree `ane` module of that kernel, CPU governor performance. OS firmware 13.5, system firmware 27.0, iBoot-8422.141.2.
- 16-inch M1 Max (T6001), Linux kernel 7.1.12-2-12.6-sep-ARCH, in-tree `ane` module of that kernel, governor performance. OS firmware 13.5, system firmware 26.6.2, iBoot-8422.141.2.
- The firmware values are read from each device tree after the runs. Runner: `ane-run` with sha256 `480040b5f700f407f11563f5349e7614df9fefe0efe57327c68ebc35b9779805`. Nothing else ran on the ANE, and the GPU queue was paused.

## Command

Each rung ran in its own directory under the `ane-run` lock with a 90 second deadline. The command was:

`ane-run --anec program.anec --in 0=in0.bin [--in 1=in1.bin] --out 0=out0.bin --repeat 200 --time`

The order was rung 10, rung 04, rung 09 on each machine. The wrapper stops after the first rung with a non-zero exit code or a deadline hit. The 13-inch ran at 06:58Z and the 16-inch at 07:01Z.

## Result

All six runs returned 0. The engine job counter rose by 200 per rung. The output of each rung was compared with the exact fp16 reference, unpacked the way `tools/h13_run_linux.py` unpacks it, and every output was equal. The output bytes of each rung are identical on the two chips. `out0-rung-NN.bin` in this directory is that output.

Exec time per call in ms, as median, minimum and maximum over 200 calls.

- Rung 10, matmul 128: 0.027, 0.027 and 0.046 on the 13-inch. 0.048, 0.046 and 0.063 on the 16-inch. The output sha256 prefix is `32e2f62d7cb9bcfb`.
- Rung 04, matvec K = 256, N = 512: 0.034, 0.032 and 0.040 on the 13-inch. 0.073, 0.068 and 0.080 on the 16-inch. The output sha256 prefix is `4136e46ef2e39f9c`.
- Rung 09, matvec K = 512, N = 512: 0.035, 0.035 and 0.072 on the 13-inch. 0.073, 0.068 and 0.083 on the 16-inch. The output sha256 prefix is `4489ac8eb4519488`.

The times are the per-call execution times that `ane-run --time` prints for 200 calls. The programs of this size sit near a per-call floor of about 0.024 to 0.027 ms on the 13-inch, so the table is not a measurement of compute cost.

## Procedure

- Timing: 200 calls per rung, one process per rung, the per-call exec time from the driver. No temperature, fan or throttle reading was taken. The CPU governor was performance.
- Reopen and recovery: each process opened the ANE device, ran its 200 calls and exited with 0. The job counters rose by 600 per machine, 1920 to 2520 on the 13-inch and 880 to 1480 on the 16-inch. No worker restart, reboot or self-test was needed, and no wedge occurred.
- Dispatch trace: not applicable to a single-program ANE run.

## Limits

- Three programs, one input each, two machines, one run per rung and machine.
- The result is about the compiler head named above, which is the head of the pull request that carries this receipt.
- The criterion is exact fp16 equality with the reference of `tools/h13_reference.py`. The weight and input patterns are the deterministic patterns of `tests/h13_first_run/rungs.json`.

## Source records

`SHA256SUMS` lists the sha256 of the source records in the lab archive. They are the device result of each machine, the check output, the sealed program directory and the earlier refusal record. They are not in this directory.
