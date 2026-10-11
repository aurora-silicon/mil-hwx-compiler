# Changelog

One entry per day the compiler changes. Newest first. Releases are tagged on GitHub.

## Unreleased

### 2026-10-10
- Test: `tests/test_h14_fixture_programs.py` compiles eight H14 elementwise programs (add, mul, relu, add-scalar,
  mul-scalar, real-div-scalar, clip-low, clip-high; MIL in `tests/fixtures/h14/`) and checks each result against a
  pinned sha256. The pinned hashes are those of the H14 fixture programs that omarchy-ane checks in for the M2, so
  the compiler output equals them byte for byte. Wired into `make test-h14-parity`.
- Device status: all 8 programs compile identical to the fixtures and all 8 ran on an M2 Max through omarchy-ane's
  `ane-run --check`, each bit-exact on 512 of 512 lanes (receipt `receipts/2026-10-10-h14-elementwise-m2/`). The same eight
  fixtures had run on an M2 Max before, in the omarchy-ane receipts `2026-10-03-t6021-v042-gate` (all eight) and
  `2026-10-01-t6021-disk-boot` (six). This change shows that this compiler's output is byte-identical to them. The test
  proves byte reproduction and no device behavior.
