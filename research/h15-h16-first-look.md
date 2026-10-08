# H15/H16 first look: Apple compiler oracles, compile only

## Status

Five of six MIL cases compiled for CPU subtype 6 (H15) and subtype 7 (H16) on
macstudio. The records are `research/oracles/h15/*.json` and
`research/oracles/h16/*.json`, minted by `research/mint_oracles.py` and stored
raw: raw task-section words plus Mach-O and section facts, `decoded: false`.
No ANE job ran. No Apple HWX bytes left macstudio. Nothing here is a support
claim for any device.

## Provenance (measured)

- Host: `macstudio`, Mac13,2, Apple M1 Ultra, macOS 26.6.2 build 25G83.
- Tool: `/tmp/w73-t7/tool/ane-compile-hwx`, rebuilt 2026-10-08 from
  `~/src/ane-linux-experiments/tools/ane-compile-hwx.mm` (same source the
  2026-09-05 receipt used) with
  `xcrun clang++ -std=c++17 -fblocks -framework Foundation
  -F/System/Library/PrivateFrameworks -framework ANECompiler`.
  SHA-256 `2060776c98e385d091ca244a2083be25919ed0c340ba1c8c89840208473b05d5`,
  52,336 bytes. The 2026-09-05 binary (`b1bab437…`, receipt) no longer exists;
  `/tmp/h13-oracle/bin/ane-compile-hwx` is a dangling symlink to a removed
  `/private/tmp/ane-compile-hwx`. The 2026-09-17 lineage tool was 52,336 bytes
  with a different hash (`3d13fc85c2a6baa0…`); same source, different build.
- Driver: `research/mint_oracles.py` (this branch), per-record SHA-256 in each
  record. Decoder: `research/h13_td.py`
  `125da0249ca3c3ba0ca1f9f0cabfd1db54a965abb75f9f4cdcc2598e255c8fbc`
  (unused for h15/h16 records; recorded for provenance parity).
- Repository commit: `5271ab0` (branch `agent/t7-h15-h16-oracles`).
- MIL: the same six cases the H13/H14 first campaign used — `add`, `mul`,
  `sub` on `[1,64,1,1]` fp16, `relu` and `clip` at 64 channels, and the matvec
  `matmul_m1_k256_n512_ty1` (`[1,256] x const[512,256]`, `transpose_y=true`).
- Command: `python3 mint_oracles.py --local --oracle-tool …/ane-compile-hwx
  --output …/oracles --source-commit 5271ab0d --targets h15 h16 --case
  binary_add_1x64x1x1 binary_mul_1x64x1x1 binary_sub_1x64x1x1 unary_relu_c64
  unary_clip_c64 matmul_m1_k256_n512_ty1` — printed
  `SUMMARY cases=12 decoded=0 raw=10 rejected=2`.

## Accepted and refused (measured)

| Case | h15 | h16 |
|---|---|---|
| `binary_add_1x64x1x1` | raw record | raw record |
| `binary_mul_1x64x1x1` | raw record | raw record |
| `binary_sub_1x64x1x1` | raw record | raw record |
| `unary_relu_c64` | raw record | raw record |
| `unary_clip_c64` | refused, exit 1, error `callback_status=1` | refused, exit 1, error `callback_status=1` |
| `matmul_m1_k256_n512_ty1` | raw record | raw record |

Control: with the same tool and host state today, `unary_clip_c64` is refused
on h13 and h14 too (records kept in the lab notebook artifacts, not in this
repository), while the 2026-09-05 h13/h14 records have the same case decoded.
Inference: the clip refusal tracks the macstudio compiler-service state on
2026-10-08, not the h15/h16 targets.

## File sizes (measured)

| Case | h15 bytes | h16 bytes |
|---|---:|---:|
| `binary_add_1x64x1x1` | 65,536 | 65,536 |
| `binary_mul_1x64x1x1` | 65,536 | 65,536 |
| `binary_sub_1x64x1x1` | 65,536 | 65,536 |
| `unary_relu_c64` | 49,152 | 49,152 |
| `matmul_m1_k256_n512_ty1` | 294,912 | 311,296 |

The file size is not stable: the same h16 add program compiled twice produced
49,152 bytes (probe run) and 65,536 bytes (minted record). File size and file
hash are run identifiers only; the section facts below are the stable
comparison. This matches the known behavior recorded in
`docs/ane/parity-method.md` for repeated compiles.

## Section layout deltas vs H14 (measured)

Both h15 and h16 objects carry `__FVMLIB` segments, a `__TEXT` segment with
`__text` and `__const`, and the same Mach-O header form as H13/H14
(magic `0xbeefface`, cputype `0x80`, filetype `0x2`). Measured cpusubtype is 6
for h15 and 7 for h16, matching the 2026-09-05 cross-target receipt.

| Case | target | `__TEXT/__text` | `__TEXT/__const` | weight home |
|---|---|---:|---:|---|
| add/mul/sub | H14 (record) | one 244-byte task | 16,384, zero | — |
| add/mul/sub | H15 | 284 | 16,384, zero | — |
| add/mul/sub | H16 | 304 | 16,384, zero | — |
| relu | H14 (record) | one 188-byte task | 16,384, zero | — |
| relu | H15 | 240 | 16,384, zero | — |
| relu | H16 | 260 | 16,384, zero | — |
| matvec | H14 (record) | 152 + 340 (2 tasks) | 262,144 | `__TEXT/__const` |
| matvec | H15 | 548 | 262,144 | `__TEXT/__const` |
| matvec | H16 | 584 | 16,384, zero | `__KERN_0/__kern_0`, 262,144 |

Three measured deltas:

1. The H15 matvec constant section is byte-identical to the H14 one: SHA-256
   equal for all five shared cases, including the 262,144-byte matvec weight
   section and its 131,016 nonzero bytes. Whatever packing H14 uses, H15
   emits the same bytes in the same section.
2. The H16 matvec moves the weights to a new segment: `__KERN_0/__kern_0`,
   262,144 bytes at file offset 49,152, while `__TEXT/__const` stays at the
   16,384-byte all-zero form. The h16 object has 12 load commands where h15
   has 11. This is a container-form change, not a weight change.
3. Elementwise task sections are longer than one H14 task by a fixed
   overhead (H15 add: 284 bytes against H14's 244-byte task; relu: 240
   against 188). The extra bytes are unexplained; the words are recorded.

## Task stream header (measured)

The H15/H16 task section does not carry the H14 `header[0]` size field.
`words[0]` is `0x00000001` in every h15/h16 record; H14 tasks encode the task
word count in `words[0][31:16]` (H14 add: `0x003d0001` for 61 words). The
first 16 bytes read as a zero-size prologue. After that prologue the H14
splitter consumes the whole section without error, which bounds the framing:
the H14 size-prefixed, 16-byte-aligned task array form reads through, but no
field behind it is claimed. Measured task sizes the splitter yields: H15
add 268, relu 224, matvec 160+372; H16 add 288, relu 244, matvec 172+392.
Inference only: the streams may otherwise differ; the raw words in each
record are the evidence.

## What this does not claim

- No device executed anything. Compile only.
- No decoded field meaning for h15/h16 task words, tensor descriptors, or
  program descriptors. The records keep raw words and structural facts only.
- No M3/M4 hardware support claim of any kind. The job path for H15/H16 stays
  closed; this mint only preserves what Apple's compiler emits for them.
