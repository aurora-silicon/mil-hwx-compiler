# Changelog

One entry per day the compiler changes. Newest first. Releases are tagged on GitHub.

## Unreleased

### 2026-10-10
- Test: `tests/test_h14_fixture_programs.py` compiles eight H14 elementwise programs (add, mul, relu, add-scalar,
  mul-scalar, real-div-scalar, clip-low, clip-high; MIL in `tests/fixtures/h14/`) and checks each result against a
  pinned sha256. The pinned hashes are those of the H14 fixture programs that omarchy-ane checks in for the M2, so
  the compiler output equals them byte for byte. Wired into `make test-h14-parity`.
- Status of the count: 8 of 8 H14 elementwise programs compile identical to the fixtures. On a device, only `add`
  is proven so far; the other seven run on the M2 once its ANE queue frees up.
