# Int8 weights chosen by GPTQ for the Qwen decode package, H13 and H14, 2026-10-24

The decode package keeps its large matmul weights as int8 with one fp16 scale per output row. Plain rounding of those weights misses the quality gate on a 14-inch M2 Max. Choosing the integers with GPTQ error feedback, on captured matmul inputs, passes it on all three machines that ran it, at the same speed. This receipt holds the recipe scripts and the seals of the two packages.

## The packages

Fifteen of the 17 large programs are int8 (programs 4, 6, 8, 10, 12, 14, 16, 18, 23, 25, 27, 29, 31, 33 and 35). Every two dimensional constant named `*_w` with at least 2^20 elements becomes `constexpr_affine_dequantize` with int8 data [N, K] and an fp16 scale [N]. Everything else stays fp16. Programs 2 and 37 stay fp16: a simulation with GPTQ on those two as well gave 1.72 times the KL, a projected 0.00068 on the gate set against a gate of 0.000534.

The packages were compiled with Apple's compiler, 15 programs for each target with return code 0. Each package directory has a `SHA256SUMS` file. The digest of that file is the seal.

| Target | Package | SHA256SUMS digest |
|---|---|---|
| H13 | `20261024-h13-qwen38-control38-scaled-convffnexpdiv-int8w-S1gptq-L192-pkg-v1` | `91ff7b47b50ac6014f35278e9c23a60a2f148d743ec0735835f8e0ce181e3a3c` |
| H14 | `20261024-h14-qwen38-convffnexpdiv-L192-k08-int8w-S1gptq-pkg-v1` | `de3d4dab6cea2fe8997f83207266c77902d64ed9b4ec1535833fbb60be4cbcf8` |

The package files are not in this repository. They are sealed lab directories.

## How the integers are chosen

For one weight W [N, K] and captured inputs X [T, K] at that matmul:

1. The scale of each row is fixed first. It is the max divided by 127, or a clipped value picked by the lowest weight error (the MSE clip). The scale is stored as fp16 and used in that exact form.
2. GPTQ picks the integers column by column. With H = X^T X / T + d * mean(diag) * I, each rounding error is pushed onto the columns not yet rounded, so the output error of X (W - Wq)^T stays small, not the weight error.
3. The first 80 percent of the rows build H. The last 20 percent score the choices. The candidates are plain rounding and GPTQ at d = 1, 3 and 10. The lowest held-out output error wins, so a tensor is never worse than plain rounding on the held-out rows. In the shipped set the chosen d was 1 on 115 tensors, 3 on 8 and 10 on 1. One tensor kept plain rounding.
4. The final integers use all rows with the chosen d.

## Quality and speed

Quality is the KL divergence of the decode logits against the fp16 package, on 50 prompts of 128 tokens, with the top-1 agreement.

| Machine | Variant | KL | Top-1 | Gate |
|---|---|---|---|---|
| 13-inch M1 | GPTQ int8 | 0.000397 | 0.9958 | 0.000534 |
| 14-inch M2 Max | plain int8 | 0.000547 | 0.9945 | 0.000534 |
| 14-inch M2 Max | GPTQ int8 | 0.000400 | 0.9958 | 0.000534 |

Paired by prompt on the M2 Max, GPTQ lowers the KL by 1.47e-4 (bootstrap 95 percent interval 1.13e-4 to 1.85e-4), on 45 of 50 prompts. The 16-inch M1 Max uses the 13-inch package and has not run it yet.

Speed per decode step, measured in separate lab sessions, not reproducible from this receipt: 13-inch M1 0.0868 s for the earlier shipped package and 0.0620 s with GPTQ int8. M2 Max 0.1234 s plain int8 and 0.1232 s GPTQ int8, the same within 0.2 percent. The integers change, the program does not.

## Recipe

All paths come from the environment. `ANEFORGE_DIR` is a checkout of ANEForge (its blob writer). `GPTQ_MIL_ROOT` holds `mil192` and `mil50`, the MIL of the fp16 programs. `GPTQ_EXPORT` holds `prog<NN>/weights.bin`. `GPTQ_ACT` holds the captured inputs `p<prog>-<const>.npy`, [T, K] fp16.

1. `python3 gptq_s1.py <prog> <out>` for each of the 15 programs. It needs torch and writes `<out>/p<prog>-<const>.npz` (int8 `q`, fp32 `scale` holding the fp16 value) and a report. `GPTQ_DAMPS` overrides the damping list.
2. `GPTQ_NPZ=<out> GPTQ_PKG_MANIFEST=<manifest.json> python3 make_int8_weights_npz.py <dir>` rewrites the MIL and the weights blob of each program. It asserts the source hashes from the manifest.
3. Compile each program with Apple's compiler for the target. This step ran on a Mac with the compiler installed and is not scripted here.

`gptq_int8.py` is the algorithm module and also runs alone with a held-out report.

## Not verified here

The packages were not rebuilt from these scripts after the paths were turned into environment variables. The scripts are the ones that built them, with path lines changed. The 16-inch M1 Max has no run.
