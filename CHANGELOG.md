# Changelog

One entry per day the compiler changes. Newest first. Releases are tagged on GitHub.

## Unreleased

- H13/H14: add the explicit spatial schedule, using existing copy, convolution
  and matrix kernels for halo dispatch, nearest/bilinear resize and concat.
  Support the MobileNetV1 convolution/head geometries at batch 1/8, including
  H14 biased linear. Add a versioned reusable-kernel dispatch contract,
  structure checks and a software executor with CPU reference comparisons.
  Existing per-op/chain packages retain their behavior. See
  `docs/spatial-dispatch.md` for arithmetic and execution limits.

- Aurora: reuse the existing H16G backend and descriptor writer. Add an
  independent SPEC-r3 fp16 container validator and source-generated owner
  identity regression. Preserve the add-128 hash and expose the consumer's
  zero-prologue disagreement. Tighten H16 binding names, integer widths and
  relocation identities; separate batch spans from allocation padding.
- Aurora: CPU halo geometry and fp16 reference checks for spatial convolution,
  resize and concat. These do not expand a hardware encoding envelope. See
  `docs/aurora-compiler-status.md` for the branch audit and remaining coverage.

## 0.3.1 (2026-10-25)

New since 0.3.0:

- H14: two more matmul shapes compile, both decoded from Apple's compiler output: M = 64 with K = 2048, N = 2048 and with K = 4096, N = 2048. Each has two tasks and a 520 byte task stream. The encoder still needs an exact decoded shape and does not interpolate. Before, both shapes failed with `h14.outside-parity-envelope`. The parity gate now covers 722 cases (PR #28).

- Receipt `2026-10-25-int8-gptq-mixed-calibration`: a second GPTQ int8 calibration for the Qwen decode that adds code prompts to prose and multilingual text, with the seals of the H13 and H14 packages. On a 13-inch M1 it passes the technical set (KL 0.000390) and the code set (1.38 times the shipped KL, against 1.60 for the first set). It fails the long-position set at 2.26 times, 10 percent worse than the first set. No run on the other chips yet.

- Receipt `2026-10-24-int8-gptq-qwen38-h13-h14`: the recipe scripts and the seals of the Qwen decode packages whose int8 weights are chosen by GPTQ on captured matmul inputs. KL against the CPU fp32 reference is 0.000397 on a 13-inch M1 and 0.000400 on a 14-inch M2 Max. The gate is 0.000534. Plain int8 gave 0.000547 on the M2 Max. The speed is the same. The packages were not rebuilt from the scripts after the path lines were changed to environment variables.

- Receipt `2026-10-24-weight-formats-apple-compiler-h13-h14`: what Apple's compiler keeps of int8, 4-bit and blocked int8 weights for H13 and H14. A 4-bit table per output row compiles on H13. Tables per group of columns are refused. It gives the device times of six programs on a 13-inch M1 and on an M2 Max. It also records a fused pair of decode programs that compiles for H14 and not for H13. No compiler code changed.

## 0.3.0 (2026-10-18)

New since 0.2.0:
- Fix: the H13 matvec at reduction 256 sizes its channel 5 input surface to the 32,768 bytes its task moves (PR #25).
  A 13-inch M1 refused the old K = 256 program. The fixed program runs exact on a 13-inch M1 and a 16-inch M1 Max.
  Its package manifest changes. The physical surface is 512 channels and the logical tensor stays 256.
- Compiler: the scaled exp chain compiles to one H14 program. It ran on an M2 Max within its preregistered criteria.
- Inspector: `research/inspect_hwx.py` names CPU subtype 11 as H18G (PR #24).
- Tests: a device-free port of libane's overrun check. Pinned H14 matvec programs with 2, 8 and 64 rows. Non-square matvecs.
  Twenty-three unary programs. A gate for the H13 matvec channel layout.
- First-run kit: rungs 9 to 20 join the catalog (PRs #25 and #26). Among them are constant-weight matmuls with 64 rows.
- Bootstrap: the toolchain script installs libobjc2 into `lib` on lib64 distributions (PR #10).
- Device runs: the H13 and H14 runs of this period are listed by date below. Each is one run on one boot.

### 2026-10-18
- Kit: the first-run catalog gains rungs 11 to 20, ten constant-weight matmuls with 64 rows that the compiler emits as one
  program each (4.2 to 537 million multiply-adds). Other sizes compile to one program per row or row group.
- Device status: the ten programs ran exact on a 13-inch M1 and a 16-inch M1 Max under `ane-run`, 200 calls each. The outputs
  are identical on the two chips. The calls took 0.024 to 0.406 ms on the 13-inch and 0.067 to 1.101 ms on the 16-inch.
  The 8.4 and 16.8 MB programs loaded and ran (receipt `receipts/2026-10-18-h13-matmul-m64-series/`).
- Device status: fifteen H14 programs that ran on an M2 Max before ran again with `--repeat 200 --time`: the eight elementwise
  programs, matvecs of 1 row at K = N of 256, 512 and 1024, and of 2, 8 and 64 rows. All passed the device-side check. The calls took
  0.306 to 0.457 ms (receipt `receipts/2026-10-18-h14-floor-rows-m2/`).
- Receipt: the sum lists of `receipts/2026-10-18-h13-matvec-surface-fix/` are split into the files in the directory and the source
  records in the lab archive, and both copies of the refusal record are named.
- Fix: the H13 source-qualified matvec at reduction 256 declared a 16,384-byte input surface on channel 5. Its task moves
  32,768 bytes from that channel at both reductions. libane refuses such a program. A 13-inch M1 refused the K = 256,
  N = 512 matvec of the first-run kit with "surface channel 5 needs 32768 bytes but is allocated 16384". The input
  surface is now 512 channels at both reductions. The logical tensor stays 256 channels and the padding lanes are zero.
  The task bytes are unchanged. The K = 256 program now has the header of the K = 512 program. So
  `tests/test_h13_matvec_programs.py` pins one prefix for both.
- Test: `tests/libane_overrun.py` ports libane's task walk and overrun check, with no device. `tests/test_libane_overrun.py`
  checks it on synthetic programs and on H13 matvec programs at K and N of 256 and 512. It failed on K = 256 before the
  fix. `first_run.py` runs it on every program of every rung. Two programs of the 77 in the MLP rung overran before the fix.
- Kit: the first-run catalog gains `09-matvec-k512-n512` and `10-runtime-matmul-128`. A latency table then has points
  above the per-call floor.
- Device status: the K = 256, N = 512 matvec with the fixed surface ran exact on a 13-inch M1 and on a 16-inch M1 Max under
  `ane-run`, 200 calls each. So did the K = N = 512 matvec and a 128 x 128 x 128 runtime matmul. Every output equals the
  exact fp16 reference, and the outputs are identical on the two chips. The calls took 0.027 to 0.035 ms on the 13-inch and
  0.048 to 0.073 ms on the 16-inch (receipt `receipts/2026-10-18-h13-matvec-surface-fix/`). One run per rung and chip.

### 2026-10-17
- Inspector: `research/inspect_hwx.py` names CPU subtype 11 as H18G. An HWX that the ANE compiler produced for target `h18g`
  from a one-add fp16 MIL program (ios18 dialect, shape [1,64,1,64]) has Mach-O magic 0xbeefface, CPU type 0x80, subtype 11,
  seven load commands, a 0x4138-byte text section and a 0x6000-byte const section. The existing H14-format splitter reads
  that text as two tasks of 73 and 74 words. The ISA version of H18G prints as unknown because no field in the file
  carries one. `tests/test_hwx_inspection.py` checks the name on a synthetic envelope and fails if the table entry is
  removed.

### 2026-10-14
- Device status: the scaled exp as ONE ANEC ran on an M2 Max (`receipts/2026-10-14-h14-exp-scaled-one-anec-run-m2/`). The compiler's own output, a 4736-byte program of three Apple tasks per size, met every
  preregistered criterion on the original input. The largest relative error against the exp bound of 2^-8 plus 2^-24 was 0.283 percent at 64 channels and 0.348 percent at 512, with no lane outside, against 0.446
  and 0.429 percent for the failed exp program. Each call took 0.361 ms and 0.356 ms against the 0.60 ms limit, and loop duty was 99.6 percent. The output equals the final output of the five-call chain byte for byte at both
  sizes. Round-toward-zero models of the fused tasks are excluded here, and the data are consistent with rounding to nearest. They do not tell one rounding from two. The order and internal precision of the fused tasks are still not decoded. One run on one boot, these inputs and these two sizes. No speed
  claim: one program does the work of five calls, and its call time is within the range of the single programs measured on this M2. The unary cell result of 21 of 23, the first chain's failure and the five-call chain are kept as recorded, and the device-verified program count is not changed here.
- Compiler: the scaled exp chain now compiles to ONE program. A MIL function of exactly exp, mul by fp16 0x1b18 (16 K), add 16, mul, mul by 0.0625 over a [1, C, 1, 1] fp16 input, with C of 64 or 512,
  emits the three-task program Apple's compiler produced for it, in ANEC and HWX form, under the per-op and the chain schedule. `research/oracles/h14/chain_exp_d4_c64.json` and `_c512.json` are the
  decoded records, minted compile-only on a macOS Mac with a compiler build that differs from the one the older H14 records name (the older build is gone; the identity of the one used is in each record).
  `tests/test_h14_parity.py` proves the task words, constant section, descriptors and tensors equal the records (720 cases now, 2 of them this program). `tests/test_h14_exp_chain_cli.py` proves that only the
  decoded form takes the path: 8 changed forms (constant, order, size, operation) leave it, and mutants of the matcher are caught except two. The product's second operand check changes nothing, because the dataflow rules already force it. The channel-support check changes
  only the refusal text for an unmatched size, and under the chain schedule the code path that refuses.
  The mint cases are in `research/mint_oracles.py`, and `tests/test_exp_scaled_chain_case.py` pins their constants.
  Not a device result: Apple folded the scalar constants into task words, so the arithmetic of the one program is not the arithmetic of the five-call chain, and only a run shows whether it meets the exp bound.
  The cell for that run is prepared in `receipts/2026-10-14-h14-exp-scaled-one-anec-m2/` (preregistration, gate, run script). No device run and no speed claim.
- Device status: the scaled exp chain ran on an M2 Max, alone on the ANE (`receipts/2026-10-14-h14-exp-chain-d4-run-m2/`). It met every preregistered criterion. Each of the four
  stages after exp equalled the ties-to-even result on all 64 and all 512 lanes, against 21 differing lanes in the first chain. The final output stayed inside the unchanged exp bound,
  with a largest relative error of 0.283 percent at 64 channels and 0.348 percent at 512 and no lane outside. The failed exp program had 0.446 and 0.429 percent. The exp stage
  output equals the failed run's byte for byte. Loop duty was 99.5 to 99.6 percent and the calibration medians were 0.306 to 0.360 ms. The chain is five calls where the failed program was one,
  so there is no speed claim. One run on one boot. The unary cell result of 21 of 23 and the first chain's failure are kept as recorded, and the device-verified program count is not changed here.
- Prepared, not run: `receipts/2026-10-14-h14-exp-chain-d4-m2/` holds a scaled form of the exp chain for an M2 Max. The first chain failed criterion 3 on 21 stage lanes. This
  design uses the same three programs in five calls: c = x * 16K, g = c + 16, p = e * g, y = p * 0.0625. Scaling by a power of two is exact, so no stage value falls where the
  first run's device rounding differed from ties to even. That is a design chosen from one run, not an established fact for other data. The frozen criteria are unchanged:
  ties-to-even stage results with no tolerance, the original 2^-8 bound with its 2^-24 floor, and the original input. The prediction from the saved device output is a largest error
  of 0.283 percent at 64 channels and 0.348 percent at 512, against a bound of 0.391. The gate has 52 checks and 42 broken copies, and the run script has 10. Nothing ran on a device.
  Five calls are not a speedup.
- Device status: the exp chain ran on an M2 Max, alone on the ANE (`receipts/2026-10-14-h14-exp-chain-run-m2/`). The final output met the unchanged exp bound. The largest
  relative error was 0.283 percent at 64 channels and 0.309 percent at 512, with no lane outside 2^-8. The failed run had 0.446 and 0.429 percent, with 2 and 9 lanes outside by the gate rule.
  The cell failed criterion 3 and is not accepted. 21 lanes in the mul and add stages differ from the ties-to-even result of their inputs. 6 are exact midpoints where the
  device gave the half-away result, and these are all the lanes where the two rules differ. 15 are mul results in the subnormal range that the device rounded one step larger in magnitude.
  One model fits all 255 mul lanes with a subnormal result: round to 11 significant bits, then to the subnormal grid with ties away. It was fitted after the run and is not a prediction.
  Loop duty was 99.5 to 99.6 percent and stage medians were 0.307 to 0.360 ms. The unary cell stays at 21 of 23. No bound or reference changed. The device-verified program count
  is unchanged. One run, no speed claim: the chain is four calls where the failed exp program was one.

### 2026-10-13
- Test and device status: `tests/hardware/gate_h13_matvec.py` runs a K = N = 512 H13 matvec through `ane-run` with the program's own
  channel layout (one fp16 value per 64-byte channel), which `ane-run --check matvec` does not read. It ran on an M1 Max: 512 of 512
  lanes inside 2 ulp or 4 condition units, padding zero, largest absolute difference 0.00755 (receipt
  `receipts/2026-10-13-h13-matvec-k512-m1max/`). One run, one call, no speed claim. K = N = 256 and 1024 have not run: the 256 program is
  refused at load, and the 1024 package has five programs. Device-verified compiler output is now 28 program runs; H13 matvec is 1 of 3
  shapes.
- Test: `tests/test_h14_matvec_programs.py` now covers nine H14 matvec shapes, the three square ones and the six non-square ones of 256, 512
  and 1024. Each shape pins its program size and the sha256 of its weight-independent 4,672-byte head. The six new shapes failed first
  with empty pins and pass with the compiler's output.
- Device status: the six non-square programs (256x512, 256x1024, 512x256, 512x1024, 1024x256, 1024x512) ran on an M2 Max through
  omarchy-ane's `ane-run --check matvec`. All six passed: every lane inside 2 ulp or 4 condition units (largest error 0.239 condition
  units), 92.3 to 95.7 percent of lanes bit-exact, padding zero. Each ran a long back-to-back loop of 268,292 to 308,123 calls after a
  20-call calibration. Engine duty over the 645 second ticket was 99.7 percent. The median call time was 0.356 to 0.357 ms for every shape,
  which contradicted the preregistered prediction that it would rise with the weight size. The fastest calls rose slightly, from
  0.217 ms to 0.255 ms across the 4x weight range. The mean call time (loop wall time over calls) rose 13.6 microseconds between the
  smallest and largest weight classes, less than the 39 microsecond gap between the two smallest shapes. Receipt:
  `receipts/2026-10-13-h14-matvec-nonsquare-m2/`. One run, M = 1, no speed claim.
- Device-verified compiler output is now 34 program runs (28 before the six M2 programs), 26 programs by name and 25 distinct by bytes.
- Test: `tests/test_h14_matvec_row_programs.py` pins 27 H14 matvec programs with 2, 8 and 64 rows (K and N of 256, 512, 1024): size, the sha256 of the
  weight-independent 4,672-byte head, and the dense-row layout rules. It failed first with empty pins.
- Device status: the 27 row-variant matvec programs ran on an M2 Max. 21 passed `ane-run --check matvec` (every lane inside 2 ulp or 4 condition units).
  `ane-run` refused the six 64-row shapes with K different from N ("io channel sizes disagree": it needs equal input and output tile counts). Their
  output was compared off the device and is in band; they are not counted below. Engine duty was 99.5 percent over the 832 second ticket. Two
  preregistered predictions did not hold. Receipt: `receipts/2026-10-13-h14-matvec-rows-m2/`. One run, M = 2, 8 and 64, no speed claim.
- Test: `tests/test_h14_unary_programs.py` pins the 23 H14 unary programs that have Apple output in the oracle records (abs at seven sizes, and
  exp, gelu, leaky_relu, rsqrt, sigmoid, silu, sqrt and tanh at 64 and 512 channels): whole-program size and sha256, and the header layout. It failed
  first with empty pins. clip and log have oracle records without Apple output, so the compiler refuses them.
- Device status: the 23 unary programs ran on an M2 Max. Preregistered bounds were met by 21: abs bit-exact, leaky_relu, sqrt, rsqrt, sigmoid, silu,
  tanh and gelu. exp missed its relative-error bound at both sizes: 11 lanes outside the gate's bound (13 without its 2^-24 absolute
  allowance), the largest 0.446 percent against 0.391 percent. silu passed with
  little margin (0.0148 absolute against 0.0156). Engine duty was 99.7 percent over the 716 second ticket, reported apart from the numerical verdict.
  The offline gate was corrected after review and gives the same result on every program. Receipt: `receipts/2026-10-13-h14-unary-m2/`. One run, fp16.
- Device-verified compiler output is now 76 program runs (55 before the unary programs), 68 programs by name and 67 distinct by bytes. The count
  includes the 21 row-variant programs that passed `ane-run`'s check and the 21 unary programs inside their bound. It leaves out the six offline-compared
  programs and the two exp programs.
- Analysis: why the H14 `exp` program missed its bound on the M2 Max. The compiler replays Apple's program: a 33-entry fp16 table of 2^(i/32) and
  the fp16 value 0x3dc5 (1.4423828, log2(e) rounded) in the NE output-scale word 0x00d10. `tests/test_h14_exp_arithmetic.py` models it. The model rounds x * scale to fp16, interpolates the table
  linearly, and rounds the result to fp16 (ties away from zero). It reproduces
  all 541 distinct (x, y) pairs the device produced (1,152 lane readings, 64 and 512 channels, calibration and loop outputs) bit for bit. Mutants of the final rounding, the scale, the exponent rounding or the interpolation miss lanes. The data do not pin the rounding rule of the exponent or of the interpolation step. The
  scale is 2.16e-4 low, so the error grows with |x|. In the model 60 of the 36,866 fp16 inputs in [-8, 8] miss a plain relative 2^-8, the smallest at |x| = 5.598,
  and the largest error is 0.489 percent. The gate of the failed run adds an absolute floor of 2^-24. Under that rule 52 inputs miss in the model. The device saw 9 of them, and they are exactly the 9 inputs behind its 11 failing lane readings (2 at 64 channels, 9 at 512). With an exact scale and the same fp16 exponent none miss (largest 0.331 percent). A second stage that
  adds back y * x * 2.165e-4 has a model maximum of 0.338 percent. That is a prediction. It is not a device result. The compiler's output did not change, the
  `exp` cell stays failed, and no bound was relaxed. A fix needs one program with two inputs, or a chained program. `ane-run` has run neither.
- Prepared, not run: `receipts/2026-10-13-h14-exp-chain-m2/` holds a corrected `exp` cell for an M2 Max. It chains four existing programs on the host (exp, mul by K,
  mul, add) through `ane-run`, with the host copying bytes only. The bound is the original 2^-8, and the failed run stays in `receipts/2026-10-13-h14-unary-m2/`.
  The preregistered prediction is a largest error of 0.280 percent at 64 channels and 0.309 percent at 512, from the saved device output of the exp stage and
  ordinary fp16 mul and add. It is a prediction. The gate `compare-expchain.py` was tested first: 50 checks, and 40 broken copies of it are each caught. A criterion 3 failure lists up to 10 differing lanes per stage with their inputs and results. It also
  checks the chip, the program hashes, the output hashes and that no other `ane-run` ran. The run script has a fake-runner test with 8 broken copies,
  a failed long loop stops the chain at that size, a 15 minute deadline in the script caps every call's timeout, calls run under `timeout -k`, and a TERM to the script ends the running call. Nothing ran on a device.
- Test: `tests/test_h14_exp_arithmetic.py` needs NumPy, which no other test here uses. It runs as `make test-h14-exp-model` and is not part of `make test`, so a host without NumPy still passes the suite.

### 2026-10-12
- Test: `tests/test_h13_matvec_programs.py` compiles H13 fp16 matvecs at K=N=256, 512, and 1024 and pins program count, sizes, and the 640-byte header/task prefix. The geometries emit 1, 1, and 5 programs. Two weight sets keep each prefix stable while changing matmul constants. Wired into `make test-h13-parity`.
- Device attempt: the 256 and 512 programs were submitted on a 13-inch M1 and an M1 Max. Neither reached numerical comparison. K=256 was refused during ANE initialization because channel 5 needed 32,768 bytes but the program allocated 16,384; K=512 was rejected by `ane-run --check matvec` as an unexpected I/O shape. K=1024 compiled to five programs and was not passed to the single-program checker. No lane or ULP results are claimed ([run record](receipts/2026-10-12-h13-matvec-m1/README.md)).
## 0.2.0 (2026-10-11)

New since 0.1.0:
- `--target H17` and `--target H18` emit Apple's exact HWX for the 165 decoded elementwise, unary and scalar-constant points of each generation and
  refuse every other shape, constant and family by name (PR #4). Byte parity with decoded Apple output, not device execution.
- Test subprocess timeouts scale with `MIL_HWX_TEST_TIMEOUT_SCALE` (PR #2). The README is rewritten for first-time users (PR #3).
  The pre-commit hook runs the fleet identity check.
- Device-verified compiler output: 27 program runs in the committed receipts. H13: 8 elementwise programs on a 13-inch M1 and the same 8 on an
  M1 Max, each bit-exact on 64 of 64 lanes. H14: 8 elementwise programs bit-exact on 512 of 512 lanes and 3 matvec programs (K = N of 256,
  512 and 1024) inside a 2 ulp or 4 condition-unit band, on an M2 Max. That is 19 programs by name and 18 distinct by bytes (the H13
  mul and mul-scalar programs are the same bytes). One run per receipt; no speed claim.

### 2026-10-11
- Docs: the README targets table and usage line list H17 and H18.
- Test: `tests/test_h14_matvec_programs.py` compiles an H14 fp16 matvec at K = N = 256, 512 and 1024 and pins each program's
  size and the sha256 of its first 4,672 bytes (header and task streams), which do not depend on the weights. It checks
  that two different weight sets give the same head and different weight sections. For 256 the head equals the omarchy-ane
  fixture's. Weights come from the standard library, so no numpy. Wired into `make test-h14-parity`.
- Device status: the three matvec programs ran on an M2 Max through omarchy-ane's `ane-run --check matvec`. Every lane
  was inside the 2 ulp or 4 condition-unit band (256 of 256, 512 of 512, 1024 of 1024), with 240, 482 and 971 lanes
  bit-exact and padding zero (receipt `receipts/2026-10-11-h14-matvec-m2/`). One run, M = 1, square shapes only, no
  speed claim.

### 2026-10-10
- Test: `tests/test_h13_fixture_programs.py` compiles eight H13 elementwise programs (add, mul, relu, add-scalar,
  mul-scalar, real-div-scalar, clip-low, clip-high; MIL in `tests/fixtures/h13/`) and checks each result against a pinned
  sha256 and size. The hashes pin this compiler's own output (the new H13 encoder's add is not byte-identical to the older
  add fixture in omarchy-ane). Wired into `make test-h13-parity`.
- Device status: all 8 programs ran on a 13-inch M1 and on an M1 Max through omarchy-ane's `ane-run`, each bit-exact on
  64 of 64 lanes with zero padding (receipt `receipts/2026-10-11-h13-elementwise-m1/`). One input per operand per run;
  no speed claim.
- Test: `tests/test_h14_fixture_programs.py` compiles eight H14 elementwise programs (add, mul, relu, add-scalar,
  mul-scalar, real-div-scalar, clip-low, clip-high; MIL in `tests/fixtures/h14/`) and checks each result against a
  pinned sha256. The pinned hashes are those of the H14 fixture programs that omarchy-ane checks in for the M2, so
  the compiler output equals them byte for byte. Wired into `make test-h14-parity`.
- Device status: all 8 programs compile identical to the fixtures and all 8 ran on an M2 Max through omarchy-ane's
  `ane-run --check`, each bit-exact on 512 of 512 lanes (receipt `receipts/2026-10-10-h14-elementwise-m2/`). The same eight
  fixtures had run on an M2 Max before, in the omarchy-ane receipts `2026-10-03-t6021-v042-gate` (all eight) and
  `2026-10-01-t6021-disk-boot` (six). This change shows that this compiler's output is byte-identical to them. The test
  proves byte reproduction and no device behavior.
