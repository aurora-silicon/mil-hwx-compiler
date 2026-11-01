# H13 linked programs: a whole conv chain in one submit

This page describes `--schedule linked`, which compiles a straight-line chain
of convolutions (each optionally followed by its relu) into one H13 program.
It also records what a whole MobileNetV1 224 costs on the M1 Pro with it, how
the three wide stride-2 layers now run, and what is still slow.

It builds on the per-layer encoder in [fused-conv.md](fused-conv.md) and on
the task-descriptor notes in [task-descriptors.md](task-descriptors.md).

## What the linker does

`ane::h13::linkConvChain` takes the per-layer programs that
`encodeConvGeneric` (plus the relu fusion) already builds and relinks their
tasks into one stream:

- **Order and headers.** Tasks keep their layer order. Each task gets its
  index in `header[0]`, the continuation bit after the first task and the end
  bit on the last, `header[7]` pointing at the next task (0x100 aligned) and
  the next task's size in `header[1]` bits 24:16. The header words are the
  form the multi-task generic convolution already runs on the device: every
  task keeps the DRAM source and destination bits (`header[6]` 0x30000800),
  the end-of-stream event bits go on the last task only, and `header[8]` bit
  23 (TBE) is set on every task with a successor.
- **Intermediates in DRAM.** The first layer reads the program input
  (channel 5) and the last writes the program output (channel 4). Every
  intermediate goes through the scratch channel (3): the producer's write
  selector and the consumer's read selector point at channel 3, and the
  producer's tile-DMA destination base (`0x17804`) and the consumer's source
  base (`0x13808`) both point at that intermediate's own region. Apple's own
  decoded programs stage through channel 3 the same way (the boolean select
  captures).
- **Inter-layer dependency.** The last task of every layer carries TSE and
  TSR (`header[6]` bits 15 and 12), and the first task of the next layer sets
  its input DMA's DependencyMode (`0x13800` bits 19:16, named in the public
  H13 register map) to 7 instead of the program-input value 3. Apple sets the
  same pair on its DRAM-chained streams (chain_ffn_d1024_s375: task 16 TSE+TSR,
  task 17 mode 7; encoder_conv_c1024_n2048: task 4 TSE+TSR). The first version
  of this branch had neither. Warm runs looked bit-exact, but a cold single run
  read stale scratch (rel. L2 0.22 to 0.43, first wrong at layer 16, dw 3x3 512
  @14x14), and it converged only after about 20 passes on the same input.
- **Scratch DMA hints.** Apple's
  scratch traffic never uses the program-I/O DMA configs. Tasks that write
  scratch set bits 9:8 of the output DMA config (`0x17800` = 0x40003c1, all
  decoded scratch writers). Tasks that read scratch use CacheHintReuse =
  CacheHintNoReuse = 0xb (`0x13800` = 0x3bb81 on 1076 decoded tasks, 0x7bb81
  on the first read after a TSE/TSR producer). The linker now emits the same
  values. They went in after the M1 Pro still showed a recoverable task
  error at pass 68 of a 5000-pass boost-on soak with the dependency bits
  alone (and another agent loading the CPU). With them, on a quiet machine,
  10,000 boost-on passes ran with no task error (01/11/2026, ane692 kernel).
- **A simple allocator.** Each intermediate gets its own tile-aligned region
  and no region is reused, so no task ever writes a region that an earlier
  task still reads. The only ordering the stream relies on is producer before
  consumer, which the hardware gives us (see the bit-exact check below).
  Every task's write span (`0x17804 + 0x17814`) and read span (`0x13814`) is
  range-checked against its region. MobileNetV1 needs 17.9 MB of scratch.
- **Constants.** Each layer's kernel section is appended to one constant
  section and every task's 16 kernel-DMA offsets (`0x1f848..`) move by its
  layer's base. The linker checks that the driver's kernel base (task length
  rounded to 16 bytes) equals the ANEC constant offset (rounded to 64).

The compiler front end accepts the schedule only for a graph that is one
input, a straight chain of `conv` and `relu`, at least two convolutions and
one result. Everything else is refused with `h13.linked-outside-envelope`.

```sh
mil-hwxc --target H13 --schedule linked --mil net.mil --model-root . --output build/net
```

## Stride 2 above 128 outputs

The zero-skipping stride-2 kernel section is byte-proven only up to 8
interleaved lanes, and one 16-lane program (128 to 256 at 56 to 28) hung the
M1 Pro engine. A wider stride-2 1x1 layer now runs as output-channel tasks of
128 channels each. Every task is the proven one-chunk 8-lane form (`ConvCfg`
OCG 3) and writes its own slice of the output surface. No program ever
carries a 16-lane stride-2 section.

| Layer (bias + relu) | Tasks | p50 / busy (us) | within 1 ulp |
| --- | ---: | ---: | ---: |
| 1x1 s2 128->128 @56->28 | 1 | 95.0 / 83.9 | 97.2 % |
| 1x1 s2 128->256 @56->28 | 2 | 147.9 / 142.6 | 97.2 % |
| 1x1 s2 256->512 @28->14 | 4 | 188.2 / 183.3 | 96.0 % |
| 1x1 s2 512->1024 @14->7 | 8 | 255.5 / 258.1 | 94.5 % |

M1 Pro, 31/10/2026, `wedged` 0 after every run.

## MobileNetV1 224 as one program

`tools/h13-mobilenet/mobilenet.py` builds the network as a 28-layer chain.
The ANE runs it in a mirrored frame: the host flips the image on both
spatial axes and every 3x3 kernel likewise. TensorFlow's SAME padding at
stride 2 samples odd input positions; in the mirrored frame those are the
even positions, which is exactly what a symmetric-pad stride-1 layer followed
by a stride-2 1x1 layer samples. So:

- the stem (3x3 s2, 3 to 32) is a space-to-depth of the mirrored image (12
  channels, zero-padded to 16) followed by a 3x3 s1 conv whose kernel holds
  the stem taps in a 2x2 window;
- each depthwise s2 layer runs at stride 1 and the stride moves onto the next
  1x1 layer;
- the classifier (1x1, 1024 to 1001, padded to 1024) runs on the ANE at 7x7,
  before the average pool. Pooling commutes with it, so the host only averages
  49 values per logit.

On the host, in float64, this chain equals the TensorFlow-frame network to
7e-16 (`mobilenet.tf_reference` against `chain_ref`).

### Measured on the M1 Pro

31/10/2026, T6000, kernel 7.3.0-aurora-fleet12d, default ANE clock (perf
level 5, linux #684, is not in this kernel). Random He-scaled fp16 weights.
Wall time includes the input and output copies unless marked exec only.

| Form | Submits | p50 | p99 | inf/s | engine busy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 28 per-layer programs, host copy between layers | 28 | 4.82 ms | 4.93 ms | 208 | 2.99 ms |
| one linked program (45 tasks), first version, no dependency bits | 1 | 2.35 ms | 2.38 ms | 426 | 2.31 ms |
| one linked program with the dependency fix, 1000 passes | 1 | 2.30 ms | 2.32 ms | 434 | 2.24 ms |
| same, exec only, 1000 passes | 1 | 2.25 ms | 2.26 ms | 445 | 2.24 ms |

The linked output equals the per-layer output bit for bit on **cold** single
runs with a new input each time (`tools/h13-mobilenet/coldcheck.py`: a fresh
process, no warm-up, 6 inputs each for random and real weights; prefix 16, the
first layer that failed before the dependency fix, is also exact). Rel. L2 vs
float64 is 0.002 on the whole net.

Real weights (the Apache-2.0 quantized MobileNetV1 from Mesa's teflon tests,
dequantized, ReLU in place of ReLU6), 16 test images, single cold runs: top-1
agrees with the float64 CPU reference of the same network on 16 of 16, top-5
overlap 5 of 5, logits rel. L2 at most 1.3e-3. The images are procedural
patterns and smooth noise, not photos, and they land in 10 different classes.

1000 passes (fleet12e, CPU boost on, real weights, after the fix): p50
2.30 ms and p99 2.32 ms with I/O (434 inf/s); exec only p50 2.25 ms, p99
2.26 ms (445 inf/s); engine busy 2.24 ms. The barrier costs about nothing.

With the scratch DMA configs as well (01/11/2026, quiet machine): 5000
passes with I/O p50 2.27 ms, p99 2.30 ms (441 inf/s); 5000 exec only p50
2.25 ms, p99 2.26 ms (445 inf/s); busy 2.23 ms; no task errors; cold runs
still equal to per-layer; top-1 16/16.

**Open hazard.** With the driver's CPU boost off (`boost_idle_ms=0`, slower
fabric) the linked program raised a task error (-5) at pass 455 and the
engine wedged. It did not happen in 1000 passes with boost on. The 28 per-layer programs ran 5000 passes clean at boost 0, and so did
the linked 8-layer prefix, so the fault is in the longer linked stream. With the scratch DMA configs in place (01/11/2026, #692
recovery kernel, so each error cost one -EIO and the engine recovered), the
boost-0 error remains, as a -5 task error at a random pass:

- clean over 5000 passes: linked prefixes 1-8, 1-16, 1-24, layer slices 5-25,
  9-25, 13-25, 17-25, 21-25, 23-28, 25-28, and the 28 per-layer programs;
- failing within 5000 passes: the full net, prefixes 1-25/26/27, slices 3-25,
  4-25, 3-8 and 1-4.

The dependency variants did not move it: every consumer task waiting, the
program-input cache hints on scratch reads, and TSE+TSR on every task. Nor
did the scratch address (up to 48 MB), the constant size (8.4 MB), the task
count (40) or capping the L2 tile at 2 MiB. Every failing stream contains the
112x112 layers 3 and 4 together with a long enough tail, but not every such
stream fails in 5000 passes, so this is a rate, not a deterministic hazard.
With the CPU boost on (the shipping configuration), 10,000 passes are clean. Linking halves the wall time: it removes
27 submits (about 25 us each), the host copies, and about 0.7 ms of engine
time per inference.

Per-task cost inside a linked stream, from chains of identical layers: about
7 us per extra 64->64 1x1 task at 8x8 and 12 us at 32x32, against 30 to 50 us
per extra program when each layer is its own submit.

### Where the 2.2 ms goes

`tools/h13-mobilenet/prefixes.py` compiles every prefix of the chain, so the
difference of busy time between prefixes is the cost of one layer inside the
stream:

| Layers | Busy (us) |
| --- | ---: |
| stem + first dw/pw pair at 112x112 | 273 |
| dw 64 @112 (stride moved to the next layer) | 212 |
| the 56x56 block (5 layers incl. the 2-task s2 layer) | 377 |
| the 28x28 block (4 layers incl. the 4-task s2 layer) | 270 |
| five dw/pw pairs at 14x14, then the dw before the last s2 | 536 |
| s2 512->1024 (8 tasks), dw 1024, pw 1024 (4 tasks) | 368 |
| classifier 1024->1024 at 7x7 (4 tasks) | 178 |

Every layer is bound by DMA, not by the MAC array. A single-task layer moves
its input and output at about 19 to 27 GB/s (for example the 1x1 128->128
layer at 56x56 moves 1.8 MB in 70 us), and changing the tile height from the
whole image down to 2 rows changes nothing (measured on the 1x1 128 @56 and
32->64 @112 layers). The DRAM surfaces also pad rows to 64 bytes, which costs
2.3x at 14x14 and 4.6x at 7x7. The weights add 8.4 MB per inference.

## Dependency and DMA fields per task (the spec for other targets)

Every field below is set by `linkConvChain`; the H14/H16G/H17 ports should
carry the same roles. `0x13800` and `0x17800` are the TileDMA source and
destination DMAConfig words. In the public H13 register map, source bits 19:16
are DependencyMode, 15:12 CacheHintNoReuse, 11:8 CacheHintReuse, 7:4 CacheHint
and 0 the enable.

| Task role | `header[6]` | `0x13800` | `0x17800` | selectors (`header[8]`) |
| --- | --- | --- | --- | --- |
| layer 0, reads the program input | SrcLoc+DstLoc+SPL (0x30000800) | 0x33881 (mode 3) | as below | RBase0 = input |
| first task of a layer reading DRAM scratch | same | 0x7bb81 (mode 7: waits on the producer) | | RBase0 = 3 |
| later tasks of that layer | same | 0x3bb81 (mode 3) | | RBase0 = 3 |
| writes DRAM scratch | | | 0x40003c1 | WBase = 3 |
| writes the program output | | | 0x40000c1 | WBase = output |
| last task of every layer | adds TSE+TSR (0x9000) | | | |
| every task but the last | | | | TBE (bit 23) |

DependencyMode values in Apple's decoded H13 programs: 3 on program inputs and
on later reads of a scratch region; 7 on the first scratch read after a
TSE+TSR producer (chain_ffn_d1024_s375 task 17); 5 on encoder_conv_c1024_n2048
task 5; 4 on L2 alias consumers (0x48880); 1 on some batched-matmul reads.
Read as bits, mode 7, 5 and 4 share bit 18, which is what a "bits 19:18 = 1"
decode calls the wait.

## L2-resident intermediates (`--schedule linked-l2`, host parity only)

The first attempt hung the M1 Pro: it mixed two of Apple's edge forms and
dropped every handoff signal. Daybreak t-3891 decoded both forms from fresh
Apple chains, and `linked-l2` now emits them:

- **Alias edge** (dense producer, dense or depthwise consumer). Producer:
  ResultCfg 0x500170, result strides = the consumer's source strides,
  output DMA disabled (0xc0) with its geometry kept, `header[6]` adds TSE+DPC+TSR,
  WBase=3 kept. Consumer: SourceCfg 0x160, input DMA 0x48880 with geometry kept,
  `header[6]` SPC+SPL, RBase0=3 kept.
- **Dependent edge** (depthwise producer, 1x1 consumer, spatial 8 to 64).
  Producer: ResultCfg 0x140, Common 0x24 bit 14, output DMA 0xc0 with zero
  geometry, no write selector. Consumer: SourceCfg 0x104 (Dependent=1), input
  DMA 0x8880 with zero geometry, no read selector, no SPL.
- **Placement** is Apple's: the first producer leaves its result at 0 and
  stages its DRAM input above it; every later task puts its result or output
  staging right above its resident input. Bases are 16-byte aligned and
  everything stays below 2 MiB.

Only the task forms Apple is observed to emit are used: dense, then alias,
then depthwise, then dependent, then a 1x1 that writes DRAM; or dense, then
alias, then dense. Every layer is a single task with its whole output in one
tile, at spatial 8 to 64. `research/h13_l2_edge_parity.py` rebuilds Apple's
decoded pw-pw and conv-dw-pw chains at 8x8 and matches every header,
Common, L2 and TileDMA word (`make test-h13`). At 16x16 the only difference is
Common 0x34 on the depthwise task, which is not an edge field. At 32x32 and
64x64 Apple uses different L2 row skews for resident edges; ours stay
self-consistent.

In MobileNetV1 these rules take two triplets at 14x14 (layers 15 to 17 and 19
to 21). The 56x56 triplet does not fit in 2 MiB, and the others meet a
multi-task or stride-2 layer. On the M1 Pro (01/11/2026) every step ran exact and cold against the
per-layer programs, with no task error: pw-pw 8x8 (the pair that hung the
first attempt), conv-dw-pw at 8x8 and 16x16, with relu and bias, and 512
channels at 14x14. The 512-channel triplet takes 0.188 ms busy against
0.228 ms with DRAM edges (-18 %). The whole real-weight MobileNetV1 with its
two L2 triplets is exact cold, top-1 16/16, and 2.190 ms p50 exec against
2.243 ms (-2.4 %) over 2000 passes.

## L2 residency, round 2 (t-4020 rules)

`linked-l2` now plans long resident runs, as Apple's compiler does:

- **Edge phases.** A run starts with an initial alias edge (ResultCfg 0x500170
  to SourceCfg 0x160) out of a producer that reads DRAM. It then alternates a
  dependent edge (0x140 to 0x104) and a chained alias edge (0x140 to 0x144,
  with DPC/SPC, channel-3 selectors and the disabled DMA geometry kept).
- **Split consumers.** Only the first output-channel task takes the incoming
  handoff. Later tasks reread the same resident input with SourceCfg 0x100,
  input DMA 0x8880 with zero geometry and no handoff bits.
- **Split producers.** In the dependent phase, a multi-task producer writes
  one aggregate result (open encoder slice j at base + j * slice channels *
  rowL2, full strides). The depthwise consumer reads it with SourceCfg 0x120.
- **Ring placement.** Each edge spans the consumer's full input and starts
  right after the live input, modulo 2 MiB. A task's live input never
  overlaps its result or staging.
- Common 0x24 bit 14 is set on depthwise tasks next to a resident edge. It
  is not an edge mode.
- **Spills to DRAM:** a dependent edge above 0x80800 bytes out of a single
  task (except the exact terminal 56x56 forms described below);
  a tiled task next to an edge; a chained alias out of a split producer; and
  anything that does not fit.

Field parity (`make test-h13`) against Apple's decoded chains: the six-layer
tails at 28x28/c256 (including the ring wrap) and 14x14/c512, the two-task
aggregate producer, the 7x7 triplet and the round-1 8x8 pair and triplet.
Cases where the open compiler splits layers differently from Apple (128-output
stride-2 slices, the 1 MiB kernel budget) are not compared field by field.

MobileNetV1 under these rules keeps 23 of 27 edges in L2, a continuous run
from layer 8 to layer 27, and scratch drops from 17.9 MB to 6.9 MB. Some of
the forms in that run are inferred rather than seen at the same shape: a
chained alias into a split consumer, stride-2 split consumers reading L2, and
a depthwise task after a 0x120 edge emitting a chained alias. They still
need the device ladder.

**On the M1 Pro (01/11/2026, kernel tt3 = aurora-wip with the #717 recovery and #718 boost
fallback, real weights).** The ladder ran cold against the per-layer programs, each step exact:

- the six-layer tails at 14x14 and 28x28;
- the split-producer aggregate with 2 and 4 slices, after one fix (below);
- MobileNet layers 8 to 13 and 8 to 27;
- the whole network.

Top-1 is 16/16 and top-5 5/5 against the CPU reference.

| MobileNetV1 224 | p50 exec | p99 | inf/s | busy |
| --- | ---: | ---: | ---: | ---: |
| `linked` (DRAM edges), 1000 passes | 2.249 ms | 2.316 ms | 443 | 2.244 ms |
| `linked-l2`, 1000 passes | 1.641 ms | 1.659 ms | 609 | 1.627 ms |
| `linked-l2`, 5000 passes, boost on | 1.645 ms | 1.678 ms | 605 | 1.639 ms |
| `linked-l2`, 5000 passes, boost_idle_ms=0 | 1.629 ms | 2.051 ms | 605 | 1.633 ms |

That is 27 % faster than DRAM edges, with no task errors in either 5000-pass
soak. The boost-0 error that the DRAM-chained net shows did not appear here.

The one fix: Apple's decoded split-producer slices descend (slice 0 at the
highest base). This encoder runs its slices in ascending channel order, and
with descending bases the device output was wrong (cold rel. L2 1.15). Slice j
now sits at base + j * slice channels * rowL2, and the parity check compares
slice bases as a set.

## Round 3: bounded 56x56 terminal forms and explicit spills

The decoded round-2 facts do **not** describe spatially sliced inter-layer
edges. A row-band compute task still produces a complete height edge. The
56x56 c160 triplet has two 0xf5700-byte resident edges; c176 needs
0x21b700 bytes for two edges, exceeds the 2 MiB ring, and spills. Reducing
TileHeight does not change this decision. There is no decoded 112x112
spatial handoff in that campaign.

The planner now accepts exactly the three-layer, same-channel 56x56
`pw -> dw -> pw` chain at c128. The second handoff uses SourceCfg
0x120 together with the complete terminal ResultCfg 0x142 form: output
C/R/D/G = `0x80 / (channels*0x80+0x40) / channels*0x80 / channels*0x80`,
TileDMADst 0xc1, and TileHeight 56. Source row strides
also follow that exact decoded case. The c160 case remains refused by the generic kernel encoder because its
lane chunks have different widths. This form is restricted to a program
output. Extending it to a scratch writer or a resident successor would
combine fields that the supplied facts do not prove.

Other DRAM edges now report `candidateL2Bytes` (the full consumer input span)
and `l2SpillReason` in the manifest. These distinguish unsupported row-band
handoffs, unsupported wide dependent consumers, and overlapping full spans.
Unknown forms retain the existing DRAM schedule.

`tests/test_h13_l2_residency.py` (part of `make test-h13-parity`) compares the
new terminal forms with every decoded header/Common/L2/TileDMA field, apart
from the existing per-layer Common 0x34 exclusion, ExeCycles, and binding
number normalization. It separately checks four/eight-task Apple handoffs,
the c176/c192/c256 spill boundary, and host scatter/gather transport through
the emitted L2 tuples against the per-layer float64 reference. The transport
tests include ring wrap, ascending output-channel assembly, bias and ReLU;
kernel sections and numerical task registers must match the per-layer build.
These host checks do not establish device arithmetic or execution safety.

Rebuilding `main` at `86dcb19` and this branch gives **21/27** resident edges,
45 tasks and 6,881,280 scratch bytes, rather than the 23/27 count recorded
above. The DRAM producers are layers 1, 2, 3, 4, 6 and 26 (zero-based).
The whole-network descriptor traffic estimate is unchanged: 24,924,160 bytes
per inference, versus 53,825,536 for the per-layer DRAM programs. It sums
enabled TileDMA extents and per-task kernel-DMA lengths; cache effects and
DRAM transaction overhead are excluded. The terminal c128 probe has three
tasks, no scratch and a 1,905,664-byte estimate.

This bounded extension does not eliminate MobileNetV1's remaining DRAM
edges. Use `tools/h13-mobilenet/l2_stats.py` for descriptor byte counts and
`prepare_l2.py` for the real-weight device package. No new MobileNet latency
is claimed before the lead runs the package.

## Remaining gap to macOS and next steps

macOS runs MobileNetV1 224 on the M4 Pro (H16) in 0.30 ms (measured). For
H13 the macOS figure is about 1 ms, scaled from that, not measured. We are at
2.2 to 2.35 ms at the default ANE clock. Joshua's perf-level results (+31 %
on M1, 1.45x on M1 Max) would put us near 1.6 to 1.8 ms once linux #684 is in
the kernel; that is an estimate.

1. L2-resident intermediates: run `linked-l2` on the device, then widen
   the edge rules (multi-task consumers, 7x7, chains longer than three) one
   measured step at a time. The 28x28 and smaller layers take 1.46 ms of the
   2.21 ms and are DMA bound.
2. Fewer kernel tasks: the 1 MiB per-task budget splits the 1024-wide layers
   into 4 tasks; Apple keeps up to 1.18 MB in one task. With L2 input reuse
   (the decoded two-task c1024 form) the extra tasks would also stop
   re-reading their input.
3. Run with perf level 5 and measure the real uplift.
