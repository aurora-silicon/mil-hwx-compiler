# Changelog

One entry per day the compiler changes. Newest first. Releases are tagged on GitHub.

## Unreleased

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
