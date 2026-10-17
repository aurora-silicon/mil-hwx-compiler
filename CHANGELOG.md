# Changelog

One entry per day the compiler changes. Newest first. Releases are tagged on GitHub.

## Unreleased

### 2026-10-17
- Inspector: `research/inspect_hwx.py` names CPU subtype 11 as H18G. An HWX that the ANE compiler produced for target `h18g`
  from a one-add fp16 MIL program (ios18 dialect, shape [1,64,1,64]) has Mach-O magic 0xbeefface, CPU type 0x80, subtype 11,
  seven load commands, a 0x4138-byte text section and a 0x6000-byte const section. The existing H14-format splitter reads
  that text as two tasks of 73 and 74 words. The ISA version of H18G prints as unknown because no field in the file
  carries one. `tests/test_hwx_inspection.py` checks the name on a synthetic envelope and fails if the table entry is
  removed.

### 2026-10-11
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
