# H14 non-square matvec on an M2 Max, run 2026-10-13

Six fp16 matvec programs built by `mil-hwxc --target H14` ran on an Apple M2 Max (T6021) under Linux through omarchy-ane's
`ane-run --check matvec`. All six passed. `PREREGISTRATION.md` in this directory was committed before the run and lists what
was predicted. `results.txt` is the unmodified output of `run-device.sh`.

## Setup
- Compiler: the binary of the 2026-10-11 square-shape run. Its sha256 is `0a79d9c1a1236e99751499be1ca0a78ecd83d3073786686854cdedd810b6f62e`.
  The H14 encoder source is 6f46c6a.
- Chip: Apple M2 Max (T6021). Kernel 7.1.12-2-20261012.16-sep. Boot id `b52b7bd7`. Driver module `ane_t6021`.
- Runner: the `ane-run` of omarchy-ane's slot tree. Its sha256 prefix is `943ae8c67598175a`.
- Run: 2026-10-13, 11:28:46Z to 11:39:31Z. One scheduler ticket, launched after the scheduler's idle settle. 645 seconds of wall time.
- Per shape: 20 calls with `--time` and `--check matvec`. Then a long back-to-back loop with the same check on the final output.
- Loop length: set from the calibration median, aiming at 110 seconds.
- Duty: the change in the driver's `busy_ns` divided by wall time.

## Result
| K x N | program bytes | lanes in band | bit-exact | max ulp | max cond. units | calls in loop | loop seconds | loop duty | median call ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 x 512 | 266,816 | 512 of 512 | 490 (95.7%) | 6 | 0.120 | 308,123 | 102.1 | 99.6% | 0.356 |
| 256 x 1024 | 528,960 | 1024 of 1024 | 945 (92.3%) | 8 | 0.239 | 308,123 | 109.5 | 99.7% | 0.357 |
| 512 x 256 | 266,816 | 256 of 256 | 245 (95.7%) | 27 | 0.108 | 307,262 | 113.8 | 99.7% | 0.357 |
| 512 x 1024 | 1,053,248 | 1024 of 1024 | 962 (93.9%) | 21 | 0.118 | 308,123 | 111.7 | 99.7% | 0.357 |
| 1024 x 256 | 528,960 | 256 of 256 | 242 (94.5%) | 7 | 0.063 | 305,555 | 109.6 | 99.7% | 0.357 |
| 1024 x 512 | 1,053,248 | 512 of 512 | 486 (94.9%) | 5 | 0.063 | 268,292 | 98.3 | 99.7% | 0.357 |

Every lane is inside the 2 ulp or 4 condition-unit band. Padding is zero. The calibration and the long loop gave the same counts for every shape.
The largest ulp differences are lanes where the sum nearly cancels. The largest error in condition units is 0.239, against a band of 4.
Duty over the whole run was 99.7 percent.

## Predictions against the result
1. All six pass, every lane in band, padding zero. Held.
2. About 90 to 97 percent of lanes bit-exact. Held. The range is 92.3 to 95.7 percent.
3. Duty at least 90 percent for the run and for each loop. Held. It was 99.7 percent for the run and 99.6 to 99.7 for the loops.
4. Median call time rises with the weight bytes. Did not hold. The medians are 0.356 to 0.357 ms for all six shapes.
   The weights span 262,144 to 1,048,576 bytes. One calibration median differs (0.410 ms for 1024 x 512, from 20 calls). Its long loop gave 0.357.
5. Total wall time of 12 to 14 minutes. Did not hold. It took 10.8 minutes. Each loop ran 98 to 114 seconds, near the 110 second target.

Call time does not depend on weight size over this range. This run does not show why, and no cause is claimed.

## Limits
- One run on one boot, one weight set and one input per shape, M = 1, fp16.
- Not run: 2, 8 or 64 row variants, shapes outside 256, 512 and 1024, normalization, convolution, chained programs.
- The median call time is the host's view of one `ane_exec`. It is not a speed comparison with macOS or any other chip.
