# Changelog

One entry per day the compiler changes. Newest first. Releases are tagged on GitHub.

## Unreleased

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
