# H16G rule-built LUT/reduction round 5

Round 4 established a bit-exact no-op gate and reduced the remaining unary and
reduction failures to six field groups.  This round records every differing
field, repairs the fields whose effect is unambiguous, and emits single-field
probes for the rest.

## Unary exp fields

The L2-only result differed at exactly 512 elements: rows 0, 32, 64, and 96
contained input rows 16, 48, 80, and 112 instead of unary results.  All other
rows were bit-identical to the baseline.  This is a tile-stride error, not an
offset, channel-order, or NaN-fill error.

| Group | Field | Baseline | Round 4 |
|---|---|---:|---:|
| L2 | input plane stride `0x4118` | `0x100` | `0x8000` |
| L2 | input batch stride `0x411c` | `0x100` | `0x8000` |
| L2 | input surface span `0x4120` | `0x100` | `0x8000` |
| L2 | output tile base `0x414c` | `0x1000` | absent |
| L2 | output row stride `0x4150` | absent | `0x100` |
| L2 | output plane stride `0x4154` | absent | `0x8000` |
| L2 | output batch stride `0x4158` | absent | `0x8000` |
| L2 | output surface span `0x415c` | absent | `0x8000` |
| Common | resident-layout field `0x0030` | `0x14000` | absent |
| Common | tile/partial-base field `0x0034` | `0x10` | `0x1` |
| Common | kernel/output-base field `0x003c` | `0x10000` | `0x5` |
| Common | tile-count field `0x0040` | `0x38` | `0x4` |

The encoder now emits a power-of-two tile row capped at 256 fp16 lanes in the
four L2 input-stride fields, emits the named local output-tile base, and does
not describe the output as a second whole-surface L2 walk.  Common remains a
single-field device bisect because its four words cannot yet be assigned a
safe general rule.

## Direct reduce-max fields

The task-header-only and Common-only switches timed out at EXECUTE.  The
Other-only switch wrote 64 values at the correct padded row positions, but the
values were out-of-range garbage rather than a shifted or permuted input.
SourceDMA produced only the first correct maximum; the rest were zero-like,
with one non-finite value.  ResultDMA produced the first correct maximum and
left the remaining 63 logical rows as NaN.  These distinguish selector,
source traversal, and result traversal failures respectively.

| Group | Field | Baseline | Round 4 |
|---|---|---:|---:|
| task header | task control word 8 | `0x00000009` | `0x00010009` |
| Other | maximum-reduction selector `0x4500` | `0x2` | absent |
| SourceDMA | row stride `0x4d18` | absent | `0x80` |
| SourceDMA | plane stride `0x4d1c` | `0x80` | `0x2000` |
| SourceDMA | batch span `0x4d24` | absent | `0x2000` |
| ResultDMA | row stride `0x5110` | absent | `0x40` |
| ResultDMA | plane stride `0x5114` | `0x40` | `0x1000` |
| ResultDMA | batch span `0x511c` | absent | `0x1000` |
| Common | height `0x0008` | `0x1` | `0x40` |
| Common | channels `0x000c` | `0x40` | `0x1` |
| Common | output width `0x0014` | `0x40` | `0x1` |
| Common | output height `0x0018` | `0x1` | `0x40` |
| Common | output channels `0x001c` | `0x40` | `0x1` |
| Common | reduction axis `0x0024` | absent | `0x3` |
| Common | tile/partial base `0x0034` | `0x1` | `0x100` |
| Common | kernel/output base `0x003c` | `0x50` | `0x200` |
| Common | tile count `0x0040` | absent | `0x1` |
| Common | schedule class `0x0044` | `0x6` | absent |
| Common | surface limit `0x004c` | `0` | `0x100000` |

Task control now selects engines by pipeline phase: direct and finalize use no
additional engine set, reorder uses five, an internal axis-2 partial uses
none, axis-2 store uses four, and an external one-task maximum partial uses
one.  Direct maximum emits its named selector.  SourceDMA traversal is
axis-specific; ResultDMA traversal is phase-specific.  Common remains split
into its eleven individual fields in the r5 morph kit.

## Axis-2 LOAD header fields

The task-size/identifier word 0 differs because the task bodies have different
record counts; it is always recomputed from the body and is not switchable.
Every other differing header field is a LOAD-only single-field probe:

| Task | Field | Baseline | Round 4 |
|---:|---|---:|---:|
| 0 | execution cycles (word 1) | `1` | `0` |
| 0 | dispatch/class/format (word 4) | `0x68` | `0x00fff820` |
| 0 | dependencies (word 7) | `3` | `0` |
| 0 | task control (word 8, max only) | `0x00050001` | `0x00010001` |
| 1 | execution cycles (word 1) | `2` | `0` |
| 1 | output/record format (word 2) | `0` | `0x68` |
| 1 | dependencies (word 7) | `3` | `1` |
| 1 | task control (word 8) | `1` | max `0x00010001`, mean/sum `0x00050001` |
| 2 | output/record format (word 2) | `0x04000000` | `0x04000068` |
| 2 | dispatch/class/format (word 4) | `0x60` | `0x00fff868` |
| 2 | dependencies (word 7) | `0` | `3` |
| 2 | task control (word 8) | `0x00040009` | max `0x00010009`, mean/sum `0x00050009` |

The probes retain the old-main task bodies and switch one header word in one
task.  Their manifests set `execute: false`; they are LOAD/unload tests only.
The fixed r5 objects remain separate `99-rule-built-exact` endpoints.
