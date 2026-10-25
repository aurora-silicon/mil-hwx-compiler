# Int8 weights from a mixed calibration: prose, multilingual and code, 2026-10-25

The first GPTQ int8 package for the Qwen decode was calibrated on prose and multilingual text only. It passed the technical prompt set. On code prompts it failed the top-1 rule. On long prompts it cost twice the shipped KL. This receipt records a second calibration that adds code. It also records what that did on three prompt sets. The recipe scripts are the ones in `2026-10-24-int8-gptq-qwen38-h13-h14`. Only the calibration capture and the damping list changed.

## Calibration

The activation capture holds 64 prompts of 128 forced positions. With the prompt tokens that is 9,136 rows per tensor, for the 15 int8 programs: 20 prose prompts, 20 multilingual prompts and 24 code prompts. The code prompts come from the odd-numbered source documents of the code test set, 12 of 25. The 26 prompts of the even-numbered documents stay unseen as the code test. The long-position set is not in the calibration at all. It is made of windows of 10 base streams.

GPTQ ran on all 125 large tensors with damping 0.3, 1, 3 and 10. The last 20 percent of the rows chose per tensor between plain rounding and each damping. The chosen damping was 0.3 on 101 tensors, 1 on 19 and 3 on 5. The earlier set chose 1 on 115 of 125, the lowest value it was offered.

## The packages

Compiled with Apple's compiler for H13 and H14, 15 programs each with return code 0, then sealed. The digest is the sha256 of the package's `SHA256SUMS` file.

| Target | Package | Digest |
|---|---|---|
| H13 | `20261024-h13-qwen38-control38-scaled-convffnexpdiv-int8w-S1gptqmix-L192-pkg-v1` | `829b43ab2f42e962d85617a8ce47ac08d17286c6c035c842f5a29f1fce4d1bf5` |
| H14 | `20261024-h14-qwen38-convffnexpdiv-L192-k08-int8w-S1gptqmix-pkg-v1` | `539b3c73d09e187474d6777f95b9068c9234f8cef92abc3e7e4542daac31a116` |

The package files are sealed lab directories, not part of this repository.

## Results on the 13-inch M1

KL divergence of the decode logits against the CPU fp32 reference, with top-1 agreement. The rule fixed before the runs is: KL at most twice the shipped package's KL on the same set, and top-1 within 0.001 of it.

| Set | Shipped | First GPTQ set | Mixed |
|---|---|---|---|
| Technical, 50 prompts x 128 | 0.000267 (the gate, 0.000534, is twice this) | 0.000397 / 0.9958 | 0.000390 / 0.9956 |
| Code, 26 unseen prompts | 0.000254 / 0.9958 | 0.000407 (1.60x) / 0.9934 | 0.000351 (1.38x) / 0.9952 |
| Long position, 50 prompts x 170 | 0.000293 / 0.9935 | 0.000605 (2.06x) / 0.9896 | 0.000663 (2.26x) / 0.9898 |

The mixed set passes the technical set and the code set. It fails the long-position set on both KL and top-1. There it is 10 percent worse than the first set. Adding code cut the code cost by 14 percent. It did not help the family it never saw.

## Not verified

The mixed package has no run on the 14-inch M2 Max or the 16-inch M1 Max. Its speed was not measured, because another job ran on the GPU during the scoring run. The long-position failure is open. A per-program scan on a CPU simulation is running. It looks for the programs that carry the cost, to keep those in fp16.
