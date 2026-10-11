# Changelog

One entry per day the compiler changes. Newest first. Releases are tagged on GitHub.

## Unreleased

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
