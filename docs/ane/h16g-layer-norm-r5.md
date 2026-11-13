# H16G layer normalization r5 prefix bisect

Round 4 established that the standalone objects complete but do not produce
the expected values. Its research boundary objects were not valid evidence for
the individual mathematical stages:

- the mean, variance, and reciprocal-standard-deviation variants all forced a
  16-byte source row into a full-width output writer;
- the centered variant used a source/staging base, not the producing task's
  result base; and
- the measured prefix does not expose independent centered or reciprocal-
  standard-deviation result views.

This explains the device pattern without assuming that the subtract task was
skipped. The BERT centered object read an unwritten staging view and returned
zeros. The Whisper centered objects read an internal staged-input view whose
layout is not the external row-major tensor, so their output was uncorrelated
with the CPU-centered tensor. In neither case did the object read a centered
result.

## Hanging mean versus completing centered object

The two objects retained their original prefix tasks. Their relevant
differences were:

| Field | Mean object | Centered object |
|---|---|---|
| Retained producer tasks | Load/reduce-mean prefix | Load plus variance/subtract prefix |
| Writer source | Mean producer result | Input/staging view |
| Writer source row | Forced 16 bytes | Full logical row (`2*C`) |
| Writer task index | Immediately after the load prefix | Immediately after the variance prefix |
| Last producer dispatch/dependency | Load task's measured values | Variance task's measured values |
| Writer dispatch/dependency | Same final-writer values | Same final-writer values |
| Terminal DMA/control | Index follows retained count | Index follows retained count |

The writer geometry is the clear hang candidate: only the forced 16-byte/full-
width combinations hung across all three geometries. The producer scheduling
fields still need a device bisect, so production prefix tasks are unchanged.

## R5 instrumentation

R5 emits one object after every measured prefix task. Each appended writer's
source base, row stride, row count, and token start come from the producing
task's named result fields. Compact scalar/tile results are copied compactly;
they are not presented as a broadcast logical stage.

Main does not admit one wide last-axis reduction for these shapes. Each
geometry therefore carries the admitted 128-by-128 main-branch `reduce_mean`
and tiles it over the complete `[S,C]` input, zero-padding only the final token
tile. These known-good input/reduction/runtime controls run first, followed by
the widest prefix result views before narrower views. The judge retains task
order and reports the first missing, non-finite, or numerically bad producer.

The production layer-normalization gate remains closed. The H=6 and H=8
attention objects remain byte-identical to the qualified r3/r4 objects.
