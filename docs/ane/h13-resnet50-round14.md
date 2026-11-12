# H13 ResNet-50 round-14 dependency audit

Round 14 audits the seven-task `block01` rung after the pool writer was
verified independently. The compiler now publishes each residual task's
logical surface range, L2 range, and dependency in the package manifest;
`tools/h13-resnet50/task_graph.py` rejects a read without an earlier covering
producer, an input/result L2 overlap, or a residual add without two inputs.

## Fault

The first block has two scratch edges: `main1 -> main2` and `main3 -> add`.
Their first consumers used a guessed TileDMA DependencyMode 5. The decoded
and device-proven H13 scratch handoff is producer TSE+TSR followed by
DependencyMode 7 (`0x13800 = 0x0007bb81`); only later slices rereading the
same completed surface use mode 3 (`0x0003bb81`). Mode 5 did not wait for the
producer's output DMA and explains why cold and warm outputs differed.

The offline allocator audit found a second read-before-write hazard in the
add. Staging allocations were each checked against the live tensor set, but
not against staging ranges allocated earlier for the same task. The add's
scratch source and result could therefore receive the same L2 tile. Staging
ranges are now allocated cumulatively: source first, then a disjoint result,
then a disjoint second source.

Neither fault is an uninitialized arithmetic accumulator. Convolution tasks
write their complete output slice. The fused add reads two producer-backed
inputs and writes a distinct result.

## Fixed one-block graph

Ranges are half-open `base + bytes`. External and scratch ranges are surface
ranges; every row also records the L2 resident or staging range used by the
task.

| # | layer | role | reads | writes | dependency | producer(s) |
|---:|---|---|---|---|---|---|
| 0 | stem | stem | external `0x0+0x49800`; L2 `0x189500+0x4c200` | L2 `0x0+0x189500` | program-input mode 3 | external |
| 1 | pool | pool | L2 `0x0+0x189500` | L2 `0x189500+0x63180` | initial alias | 0 |
| 2 | block01 | projection | L2 `0x189500+0x63180` | L2 `0x0+0x189180` | L2 dependent | 1 |
| 3 | block01 | main1 | L2 `0x189500+0x63180` | scratch `0x0+0x62000`; L2 `0x1ec680+0xc630` | L2 dependent | 1 |
| 4 | block01 | main2 | scratch `0x0+0x62000`; L2 `0x1ec300+0xc630` | L2 `0x189180+0x63180` | TileDMA mode 7 | 3 |
| 5 | block01 | main3 | L2 `0x189180+0x63180` | scratch `0x64000+0x188000`; L2 `0x1ec300+0x10000` | L2 dependent | 4 |
| 6 | block01 | add+relu | scratch `0x64000+0x188000`; L2 `0x189180+0x40000`; shortcut L2 `0x0+0x189180` | external `0x0+0x1c0000`; L2 `0x1c9180+0x20000` | TileDMA mode 7 | 5, 2 |

Every non-external read has an earlier covering producer. Live logical
surfaces do not overlap, and no task reuses one L2 staging range for an input
and its result.

## Device sub-rungs

Round 14 adds compiler-built standalone programs for `conv1`, `conv1+conv2`,
`conv1+conv2+conv3`, projection, and fused residual add. The first four read
the float64 pool boundary. The add binds the float64 main3 and projection
boundaries as two inputs. Each kit entry contains its own float64 output and
requires three cold outputs to be bit-identical before its accuracy result is
accepted.
