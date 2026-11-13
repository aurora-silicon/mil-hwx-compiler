# H16G rule-built LUT/reduction round 6

Round 5's one-field device probes separated fields that were already correct
from the fields that either changed results or prevented completion.  This
round changes only the latter rules, plus the independently isolated axis-2
task-2 dependency word.

## Unary rules

The geometry inputs and the three device-named fields are:

| Shape | Surface bytes | Tile row bytes | Field | Round 4 | Round 5 | Derived value |
|---|---:|---:|---|---:|---:|---:|
| exp 128x128 | 32768 | 256 | L2 output tile base `0x414c` | absent | 4096 | 4096 |
| exp 128x128 | 32768 | 256 | Common kernel/output base `0x003c` | 5 | 5 | 65536 |
| exp 128x128 | 32768 | 256 | Common tile count `0x0040` | 4 | 4 | 56 |
| exp 192x192 | 73728 | 512 | L2 output tile base `0x414c` | absent | 4096 | 4096 |
| exp 192x192 | 73728 | 512 | Common kernel/output base `0x003c` | 10 | 10 | absent |
| exp 192x192 | 73728 | 512 | Common tile count `0x0040` | 9 | 9 | 56 |

The output tile base is a local byte offset, fixed at one 4096-byte tile for
table/direct operations.  The narrow Common kernel field is a byte base: for
128 it is the end of the two 32768-byte surfaces, not an allocation-page
number.  A 512-byte tile row selects its table through KernelDMA and omits the
Common base.  The Common tile count is seven groups of eight completion tiles;
it is not `ceil(surfaceBytes / 16384)`.  Reciprocal-family output bases retain
their separate `nextPowerOfTwo(size) * 256` local-layout rule.

## Reduction Common geometry

For compactness, each Common tuple below is
`channels/output-width/output-height/output-channels/kernel-or-output-base`.
An em dash means the field is omitted.  Multi-task axis-2 rows list
`reorder | partial | store`.

| Input and axis | Round 4 | Round 5 | Derived tuple |
|---|---|---|---|
| C64 H8 W8, axis 1 | `64/8/8/1/768` | same | `64/8/8/1/262144` |
| C96 H12 W20, axis 1 | `96/20/12/1/1536` | same | `96/20/12/1/262144` |
| C32 H64 W16, axis 2 | `32/16/1/32/2304` in every task | same | `64/16/32/64/807608832` \| `32/64/1/32/80` \| `-/1/32/-/270270464` |
| C7 H33 W19, axis 2 | `7/19/1/7/512` in every task | same | `33/19/7/33/807604480` \| `7/33/1/7/80` \| `-/1/7/-/270270464` |
| C1 H64 W64, axis 3 | `1/1/64/1/512` | same | `64/64/1/64/80` |
| C5 H11 W37, axis 3 | `5/1/11/5/512` | same | `11/37/1/11/80` |

The rules behind the table are:

- Direct width reduction exposes `H` independent contiguous rows:
  `C'=H, W'=W, H'=1, Cout'=H`.
- Axis-2 reorder swaps C and H while preserving W; its partial is
  `C'=C, W'=H, H'=1, Cout'=C`; store is `W'=1, H'=C` and omits both channel
  fields.
- Axis-1 maximum preserves C/H/W and writes one output channel.  Sum/mean
  partials preserve C output channels; their finalize packs C into one row up
  to 64 channels, then into 32-wide rows, with W as the lane set.
- Compute/finalize phases place the accumulator after one padded output row
  plus four control words, which is 80 for every matrix shape.
- Partial/reorder/store pipeline words are assembled from the phase, the
  ceiling-log2 dimension class, and the lane-group count.  They are not output
  allocation offsets.

The remaining device-named fields follow directly from geometry.  Maximum's
selector is emitted in direct and axis-2 partial compute phases.  Axis 3 source
and result plane strides are the padded input and output row bytes (128 and 64
for both width-reduction shapes).  Axis-2 reorder uses the padded input row as
its source plane stride.  The already-correct phase task-control sequence is
`0x00050001, 0x00000001, 0x00040009` for reorder/partial/store and `0x9` for a
one-task direct width reduction.

## Dependency rule

Header word 7 is a producer mask, not a mask of earlier task indices.  Every
non-final producer advertises one bit for each inter-task consumer slot; the
final consumer advertises zero:

`producerMask(i, n) = i == n - 1 ? 0 : (1 << (n - 1)) - 1`.

| Pipeline | Round 5 words | Derived words |
|---|---|---|
| two tasks | `0, 1` | `1, 0` |
| three tasks | `0, 1, 3` | `3, 3, 0` |

This makes task 2 in every three-task axis-2 reduction a consumer with no
successor, matching the isolated LOAD result.

## Verification artifacts

The structured encoder test checks every device-named exp-128 and direct
reduce-max field against the established field map, checks the wide unary
presence rule, and pins the axis-2 dependency words to `3, 3, 0`.  The short
morph-kit mode emits r4 `00`/`c01` gates, fixed r6 `99` endpoints, reverse
single-field switches from r6 into r4, and six LOAD-only task-2 dependency
probes.
