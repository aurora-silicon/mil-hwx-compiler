# H14 matvec with 2, 8 and 64 rows on an M2 Max, run 2026-10-13

Twenty-seven fp16 matvec programs ran on an Apple M2 Max (T6021) under Linux through omarchy-ane's `ane-run`. They were built by
`mil-hwxc --target H14`. All 27 gave output inside the error band with zero padding. Twenty-one passed `ane-run --check matvec`.
Six 64-row programs ran their 20 calibration calls, but `ane-run` refused to compare them. Their output was compared off the device.
`PREREGISTRATION.md` was committed before the run. `results.txt` is the unmodified output of `run-rows.sh`.

## Setup
- Compiler and programs: as in `PREREGISTRATION.md`. `build-rows.py` rebuilds the same files.
- Chip: Apple M2 Max (T6021). Kernel 7.1.12-2-20261012.16-sep.
- Runner: the `ane-run` of omarchy-ane's slot tree. Its sha256 prefix is `943ae8c67598175a`.
- Run: 2026-10-13, 12:26:57Z to 12:40:49Z. One scheduler ticket. 832 seconds of wall time. Run once.
- Per shape: 20 calls with `--time` and `--check matvec`.
- Per passing shape: a loop of 73,394 to 110,803 calls, sized to about 40 seconds.
- Duty: the change in the driver's `busy_ns` divided by wall time.
- Duty was 99.5 percent over the ticket and 98.5 to 99.7 percent in each loop.

## Result
| M | K x N | lanes in band | bit-exact | max ulp | max cond. units | compared by | loop median call ms | loop duty % |
|---:|---|---:|---:|---:|---:|---|---:|---:|
| 2 | 256 x 256 | 512 of 512 | 479 (93.6%) | 8 | 0.124 | ane-run | 0.357 | 99.7 |
| 2 | 256 x 512 | 1024 of 1024 | 953 (93.1%) | 12 | 0.193 | ane-run | 0.361 | 99.7 |
| 2 | 256 x 1024 | 2048 of 2048 | 1919 (93.7%) | 19 | 0.214 | ane-run | 0.362 | 99.7 |
| 2 | 512 x 256 | 512 of 512 | 480 (93.8%) | 83 | 0.117 | ane-run | 0.362 | 99.7 |
| 2 | 512 x 512 | 1024 of 1024 | 965 (94.2%) | 16 | 0.126 | ane-run | 0.407 | 99.7 |
| 2 | 512 x 1024 | 2048 of 2048 | 1904 (93.0%) | 8 | 0.121 | ane-run | 0.407 | 99.7 |
| 2 | 1024 x 256 | 512 of 512 | 478 (93.4%) | 27 | 0.065 | ane-run | 0.407 | 99.7 |
| 2 | 1024 x 512 | 1024 of 1024 | 948 (92.6%) | 23 | 0.111 | ane-run | 0.407 | 99.7 |
| 2 | 1024 x 1024 | 2048 of 2048 | 1932 (94.3%) | 29 | 0.109 | ane-run | 0.457 | 99.7 |
| 8 | 256 x 256 | 2048 of 2048 | 1932 (94.3%) | 51 | 0.234 | ane-run | 0.408 | 99.7 |
| 8 | 256 x 512 | 4096 of 4096 | 3837 (93.7%) | 112 | 0.213 | ane-run | 0.408 | 99.7 |
| 8 | 256 x 1024 | 8192 of 8192 | 7687 (93.8%) | 120 | 0.214 | ane-run | 0.411 | 99.7 |
| 8 | 512 x 256 | 2048 of 2048 | 1918 (93.7%) | 90 | 0.133 | ane-run | 0.411 | 99.7 |
| 8 | 512 x 512 | 4096 of 4096 | 3865 (94.4%) | 1554 | 0.197 | ane-run | 0.458 | 99.7 |
| 8 | 512 x 1024 | 8192 of 8192 | 7670 (93.6%) | 153 | 0.134 | ane-run | 0.458 | 99.6 |
| 8 | 1024 x 256 | 2048 of 2048 | 1919 (93.7%) | 11 | 0.064 | ane-run | 0.458 | 99.6 |
| 8 | 1024 x 512 | 4096 of 4096 | 3849 (94.0%) | 77 | 0.117 | ane-run | 0.457 | 99.6 |
| 8 | 1024 x 1024 | 8192 of 8192 | 7690 (93.9%) | 125 | 0.117 | ane-run | 0.458 | 99.6 |
| 64 | 256 x 256 | 16384 of 16384 | 15367 (93.8%) | 2097 | 0.232 | ane-run | 0.458 | 99.6 |
| 64 | 256 x 512 | 32768 of 32768 | 30759 (93.9%) | 185 | 0.255 | offline | not run | not run |
| 64 | 256 x 1024 | 65536 of 65536 | 61678 (94.1%) | 240 | 0.243 | offline | not run | not run |
| 64 | 512 x 256 | 16384 of 16384 | 15418 (94.1%) | 1049 | 0.132 | offline | not run | not run |
| 64 | 512 x 512 | 32768 of 32768 | 30686 (93.6%) | 2210 | 0.135 | ane-run | 0.458 | 99.5 |
| 64 | 512 x 1024 | 65536 of 65536 | 61523 (93.9%) | 2035 | 0.185 | offline | not run | not run |
| 64 | 1024 x 256 | 16384 of 16384 | 15399 (94.0%) | 471 | 0.118 | offline | not run | not run |
| 64 | 1024 x 512 | 32768 of 32768 | 30712 (93.7%) | 543 | 0.125 | offline | not run | not run |
| 64 | 1024 x 1024 | 65536 of 65536 | 61604 (94.0%) | 1039 | 0.129 | ane-run | 0.512 | 98.5 |

The `compared by` column says which check produced each row. `ane-run` printed the first 21.
The six marked `offline` were compared by `compare-offline.py`. It read the output file that `ane-run` wrote after its 20 calls.
It applies the rules of `ane-run --check matvec` for every lane with a scale. Lanes whose products are all zero are treated differently (below).
The reference is a float64 sum of the fp16 products, rounded to fp16.
A lane is in band within 2 ulp or 4 condition units. Padding must be zero.
On the 21 shapes that `ane-run` checked, the script reproduced `ane-run`'s printed counts exactly.
That covers lanes, bit-exact lanes, largest ulp and largest condition units. `offline-compare.txt` shows 0 mismatches. Only the six outputs that `ane-run` refused are archived, in `device-output/`.
`SHA256SUMS` lists those six, with paths that check from this directory. `SHA256SUMS-not-archived` lists the other 21 hashes.
Those 21 output files are not in the repository, so that second list cannot be checked from here.

## Reproduce the six offline lines
Run both commands from the repository root. The first needs a built `mil-hwxc` and numpy.
```
python3 receipts/2026-10-13-h14-matvec-rows-m2/build-rows.py /tmp/rows-six build/mil-hwxc --only m64k256n512,m64k256n1024,m64k512n256,m64k512n1024,m64k1024n256,m64k1024n512
python3 receipts/2026-10-13-h14-matvec-rows-m2/compare-offline.py /tmp/rows-six receipts/2026-10-13-h14-matvec-rows-m2/device-output
```
Expected: six PASS lines, equal to the six `offline` rows above, then "6 shapes compared", 0 failed, 0 missing, exit status 0.
To check the archived files: `cd receipts/2026-10-13-h14-matvec-rows-m2 && sha256sum -c SHA256SUMS`.
`test-compare-offline.py` runs 16 checks of the comparator's exit status. It then runs the same checks against 7 deliberately broken copies
of the comparator and requires every copy to be caught.

## Why six shapes were refused
`ane-run` stopped with "io channel sizes disagree". It requires the input and output surfaces to have the same size. Those sizes come in
whole 16 KiB tiles. The rule "input tiles differ from output tiles" fits all 27 shapes with no mismatch. It selects exactly the six
64-row shapes with K different from N. Their input needs M x K x 2 bytes. Their output needs M x N x 2 bytes. The tile counts differ.
At 2 and 8 rows every surface fits one tile, so K different from N is accepted. The six programs ran and their output is in band.
So the limit is in the checker, not in the compiler or the device.
The M = 1 non-square shapes passed only because they fit one tile.

## Predictions against the result
1. All 27 pass, every lane in band, padding zero. Did not hold as written. 21 passed `ane-run`'s check and 6 were refused by it.
   The device output of all 27 is in band with zero padding. The layout concern for 8 and 64 rows did not materialize.
   No row count gave a wrong result.
2. Median call time between 0.30 and 0.50 ms, and the 64-row median at most 0.10 ms above the 2-row median. Did not hold at the edges.
   One median is 0.512 ms (64 rows, 1024 x 1024). At 256 x 256 the 64-row median is 0.101 ms above the 2-row median.
   The other two square pairs differ by 0.051 and 0.055 ms. Call time rises with the row count.
   Loop medians at 2 rows span 0.357 to 0.457 ms. At 8 rows they span 0.408 to 0.458. At 64 rows they span 0.458 to 0.512.
3. Duty at least 90 percent for the run and each loop. Held. It was 99.5 percent for the run and 98.5 to 99.7 in the loops.
   Six shapes had no loop.
4. Wall time at most 22 minutes if all 27 pass. The condition did not apply, since six had no loop. The run took 13.9 minutes.

## Limits
- One run on one boot, one weight set and one input per shape, fp16. No speed claim.
- Each offline comparison rests on one 20-call output. The comparator was validated against `ane-run` on the other 21 shapes.
- Not run: row counts other than 2, 8 and 64 (1 row ran earlier), shapes outside 256, 512 and 1024.
- Not run: normalization, convolution, chained programs.
- The refusal rule is read from `ane-run`'s check and fits this run. Another checker version could differ.
- The first version of `compare-offline.py` in this receipt (0f9d99e9) exited 0 for a failed shape, a wrong output and a missing output.
  Only a mismatch with `ane-run`'s printed counts gave a nonzero status. Its printed per-shape lines were right, and they are unchanged in the fixed script.
- The six offline results have not been checked by a second, independent comparator. They rest on this one.
- A second fault was found afterwards. When the sum of the absolute products was zero, the comparator counted the condition-unit error as 0.
  Any output then passed the "or" test, including all ones and all NaN for a zero input. It now accepts only an output within 2 denormal ulps of zero there (the 2 ulp test of fp16 around 0), and never a NaN or infinity.
  `ane-run` itself still accepts any output on such a lane until omarchy-ane PR 161 changes that to an exact zero.
  A NaN or infinite lane is never in band. None of the 27 real shapes has such a lane, so no reported row changed.
- `ane-run --check matvec` computes the condition-unit error the same way (it uses 0 when the sum is 0). It shares this flaw.
  That does not affect the 21 shapes it checked here, since none has a zero-sum lane.
