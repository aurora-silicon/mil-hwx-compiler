# H13 ResNet-50 round 15: external-source bank phase

Round 15 starts from the one-task `block01-conv1-only` result.  Its kernel
payload is correct: unpacking the emitted dense 1x1 section and evaluating it
against the kit input reproduces the CPU reference at relative L2
`1.45e-7`.  The device result is also deterministic, and every device pixel's
64-channel vector is a CPU-reference pixel vector.  The spatial order is
wrong: device row `y` begins at flat reference pixel `32*y` instead of
`56*y`.  Thus the earlier three-row least-squares description was only an
approximation to a deterministic row permutation, not evidence of a 3x1
kernel.

`tools/h13-resnet50/round15.py` decodes the task, unpacks its own weights and
bias, and models that zero-phase row walk.  Against the retained round-14
device output the model has relative L2 `3.10e-4`, below the required `1e-2`.
No target or compiler capture is an input to the model.

## Decoded geometry

All byte values below come from this compiler's generated ANEC tasks.  Row,
plane, and depth entries are bytes.

| field | passing stem | passing max-pool | failing conv1-only | passing resident projection |
|---|---:|---:|---:|---:|
| kernel HxW | 7x7 | 3x3 | 1x1 | 1x1 |
| stride HxW | 2x2 | 2x2 | 1x1 | 1x1 |
| leading padding HxW | 2x2 | 0x0 | 0x0 | 0x0 |
| input HxW | 224x224 | 112x112 | 56x56 | 56x56 |
| output HxW | 112x112 | 56x56 | 56x56 | 56x56 |
| tile height | 112 | 56 | 56 | 56 |
| L2 source channel | 448 | 224 | 112 | 112 |
| L2 source row | 1,344 | 14,384 | **7,168** | **7,248** |
| L2 source depth | 1,344 | 14,336 | 7,168 | 7,168 |
| source surface row | 448 | disabled resident read | 112 | disabled resident read |
| source surface plane | 100,352 | disabled resident read | 6,272 | disabled resident read |
| source surface depth | 301,056 | disabled resident read | 401,408 | disabled resident read |
| destination surface row | 256 | 128 | 128 | 128 |
| destination surface plane | 28,672 | 7,168 | 7,168 | 7,168 |
| destination surface depth | 1,835,008 | 458,752 | 458,752 | 1,835,008 |

The conv1 kernel height, stride, padding, tile height, external NCHW row and
plane strides, and output strides rule out a 3-row kernel window, overlapping
vertical output tiles, and a 128-byte external input-row read.  The remaining
geometry difference is the named `l2_source_row_stride` bank phase.  The
failing external task made row and depth equal; the same 56-wide projection
passes when its resident source row is depth plus the width-derived 80-byte
bank rotation.

The encoder now uses two explicit source-layout rules:

- the independently proven native stem repeats its unskewed three-channel
  pixel-row span;
- every later external convolution source retains the consumer layout's
  `l2Skew(stride, width, channels)` phase.

The fix changes conv1-only and projection-only from row 7,168 to 7,248 while
leaving their TileDMA surface row/plane at 112/6,272.  The passing `stem`,
`stem-pool-via-dram`, `maxpool`, and `next-stall-projection` packages remain
byte-identical to round 14.

## Add-only repeatability

Round 14 made both add operands external after their L2 routes had been
planned.  Both enabled TileDMA sources then carried a zero bank phase: their
L2 row and depth fields were both 28,672.  A single convolution's zero-phase
walk is deterministic; two independently scheduled source DMAs feeding one PE
task can contend for the same unrotated bank half-row.  The PE reads whichever
source write wins, which explains why all three add-only runs completed but
did not agree bit-for-bit.  This is a staging read-before-completion hazard,
not an uninitialized constant or accumulator.

The add route now derives both branch layouts for a processing-element
consumer.  Its named PE bank rotation is 32 bytes, so both L2 source rows are
28,704 while their depth remains 28,672.  Their L2 bases and declared staging
ranges remain disjoint, and the source and destination surface ABI does not
change.

## Offline and host gates

The round-15 field/model tool reports the old task's 32-pixel effective walk,
the corrected task's 56-pixel walk, and optionally a device-dump relative L2.
The C++ residual test asserts the convolution phase separately from the PE-add
phase, checks both add operands, and continues to cover 224/192 bottleneck and
basic-block networks.
