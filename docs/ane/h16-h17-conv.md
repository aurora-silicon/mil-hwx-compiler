# Formula-built H16G and H17 convolutions

**MobileNetV1 224 runs on the H17 ANE as one linked program.** On the J700
(Neo, A18 Pro, T8140) the 28-task program compiled by
`tools/h16-mobilenet/mobilenet.py --target H17` with real weights gave top-1
16/16 against the float64 reference, at a host p50 of 1.744 ms per inference
(200 runs, one submit, research harness). See "Device evidence" below. H16G
emits the same network; it has not executed on a J616s yet.

What the backend covers:

- every MobileNetV1 224 convolution: 1x1 at 32..1024 channels, 3x3 depthwise
  at stride 1 and 2, the 3->32 stride-2 stem, all with bias and fused ReLU;
- stride 2 in Apple's zero-skipping sparse kernel form
  ([stride-2 facts](h16-h17-stride2-sparse-kernel-facts.md)); dense stride-2
  kernels hang the H17 engine and are never emitted;
- `--schedule linked`: all layers in one program, intermediates in DRAM
  scratch (slot 3, two ping-pong regions) with the DependencyMode edge.

Descriptor parity holds for all 38 single-task and two-chain oracles per
target and every kernel section, dense and sparse, matches Apple's SHA-256.

The source facts are the sanitized `research/oracles/{h13,h16,h17}/conv_desc_*.json`
from `research/h16g-h17-conv-oracles`. They contain decoded fields, source MIL,
metadata and hashes, never Apple-produced binary payloads. Implementation uses
our existing MIT-licensed stride-1 packer. Public register names and packed
record grammar are documented by
[coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h16_register_map.md)
(BSD-3-Clause). No code from unlicensed sources was copied.

## Reproduce

```
make test-h16g test-h17
make test-h13 test-h13-parity test-h17-parity
python3 tools/h16-mobilenet/emit_measured.py build/measured-h16-h17
python3 tools/h16-mobilenet/mobilenet.py build/mn-h17 --target H17 --weights mobilenet_v1_1_224_quant.tflite
build/mil-hwxc --target H16G --format hwx --mil model.mil --model-root model --output out
build/mil-hwxc --target H17 --format hwx --mil model.mil --model-root model --output out
```

H16G's existing default staged path remains available; `--format hwx`
selects the new measured conv backend. H17 routes conv graphs to this backend
and preserves its existing elementwise parity path. The new manifest states
`deviceValidated: false`. Output contains a `program-0.hwx` and its surface
layout in `manifest.json`; weights reside in `__KERN_0`, and the relocation
points there as a 64-bit operand. `__TEXT/__const` contains zeros. The H17
convolution path uses H16-style resource metadata with the H17 architecture
and ISA tags, rather than routing its weight section to elementwise constants.
Legacy corpus-based size caps have been replaced with checks against the
32-bit HWX file-offset representation; no macOS software size cap is imposed.

## Shape formulas and measured schedule

External fp16 NCHW rows are `align64(2*W)`, planes are `row*H`, and allocations
are rounded to 16 KiB. Internal L2 rows are distinct: pointwise uses
`2*min(W,64)`; regular/depthwise 3x3 uses `2*W`. Source stride is
`Cin*internalRow + schedule.sourceSkew`. Result base is input rows times this
stride for standalone forms; resident chains use the measured ping-pong
allocation and cumulative kernel bases. Result stride uses the measured
result-row mode plus skew. In particular a resident depthwise result has a
measured `0x80` skew, not the standalone width-56 skew of `0x40`.

| Geometry (Cin→Cout, square input) | TileHeight | Source skew | Common 0x3c | Executable |
| --- | ---: | ---: | ---: | --- |
| 1x1 64→64 @64 | 64 | 0 | 0x210000 | yes |
| 1x1 32→64 @112 | 112 | 0x50 | 0x231100 | yes |
| 1x1 64→128 @112 | 80 | 0x50 | 0x231100 | yes |
| 1x1 64→128 @56 | 56 | 0x50 | 0x240000 | yes |
| dense 3x3 64→64 @112 | 112 | 0 | 0x200000 | yes |
| depthwise 3x3 C32 @112 | 112 | 0x50 | 0x230000 | yes |
| dense 3x3/s2 3→32 @224 | 112 | 0 | 0x230000 | yes (sparse) |
| depthwise 3x3/s2 C64 @112 | 56 | 0 | 0x210000 | yes (sparse) |
| depthwise 3x3/s2 C128 @56 | 28 | 0x30 | 0x220000 | yes (sparse) |

The round-2 and round-3 rows (MobileNetV1 widths 128..1024 at 56/28/14/7,
stride-2 depthwise C256/C512, bias) are in the table in
`plugins/H16G/Encoding/H16ConvProgram.cpp`. A biased layer without its own
measured row takes the plain row's schedule; the measured bias rows show the
same Common value except 512->1024 @7.

Only square batch-one SAME, unit dilation, dense or depthwise groups are
covered. Bias sets NE `0x4904` bit `0x10` and adds one bias row per lane. ReLU sets NE MAC bit 16. Dense stride-1 chunks are
`align64(2*Cin*k*k*Cout/16)`; depthwise substitutes one for Cin. Sixteen local
chunk offsets follow the cumulative sizes. Reset-default size 64 is omitted.

H17 applies the explicit generation delta: Common `0x4c=0x200000`, mandatory
NE `0x4900=0x10000082` plus sparse bit `0x100`, destination low byte `0x21`,
bit 27 of Common `0x3c` on measured dense 3x3, and the measured middle-task
header flag. H16G uses `0x100000`, omits the dense NE baseline, and writes
destination `0x31`. Packed-record selection recomputes task sizes after the
H17 NE write.

Two resident schedules are descriptor-tested: depthwise C64/s2→pointwise
C128, and pointwise C64→depthwise C64/s2→pointwise C128. External DMA appears
at the endpoints only; task IDs, local chunk offsets, cumulative kernel bases,
L2 bases and chain flags are rebuilt. These L2 forms stay a descriptor API;
executable chains use the DRAM edges described below.

## Parity and report corrections

Every register (including omitted/default fields), packed record, task size,
ID and known header word is checked against all 38 single-task and two-chain
oracles for each target. **Only header[1] ExeCycles is excluded**; the encoder
writes zero and makes no runtime claim about that unknown hint. All 36
single-task kernel sections, dense and sparse, plain and biased, must match
Apple's SHA-256, and nonuniform input weights must change the kernel section.

The campaign uses the legacy `mint_oracles.blob` header layout. As already
explained by `research/mint_conv_probes.py:blob_view`, its resolved constant
starts at file offset zero and includes the synthetic file header before the
0.5 payload. Hash tests reconstruct that source-owned input view, rather than
assuming a flat array of 0.5 values or changing production packing to fit an
incorrect test input. Ordinary correctly formed BLOBFILE inputs are supported;
the nonuniform-weight package test checks that their real payload is used.

The field report counts seventeen tasks; the twelve decoded programs actually
contain fifteen (ten singles, one pair and one triple).

The field report says no CacheDMA words are written. The decoded JSON instead
contains `0x5900`, `0x5904`, `0x5918`, `0x5928`, `0x592c` on final tasks and
stage-tagged `0x5908` on final resident-chain tasks. The encoder preserves
these decoded facts and tests them. The resident depthwise result skew above
is another schedule-specific fact which must not be extrapolated from the
standalone result-skew formula.

## Unknowns and refusals

- Sparse H16/H17 byte grammar. The stride-2 section hashes differ from H13;
  neither H13 sparse bytes nor a zero placeholder can substitute for weights.
  Descriptor-only chunk sizes describe the campaign BLOBFILE view.
- Bias packing/epilogue, later MobileNet channel counts/spatial widths,
  output-channel splitting, general tiling/skew/result-mode selection, VALID,
  custom padding, dilation, rectangular kernels and non-depthwise groups.
- DRAM dependency synchronization. The H13 linker retains its tested producer
  TSE+TSR and consumer TileDMASrc DependencyMode behavior. H16/H17 have a
  different header: header[6] is not H13's control word. The public map places
  source DependencyMode at bits 28–29 and exposes CacheDMA TaskSync, but the
  focused conv corpus has no DRAM producer/consumer edge. Copying H13 control
  bits or enabling a guessed H16 handshake would not be a validated port.
- General L2 capacity and allocator policy, and the meanings of generation
  bits, destination modes, and cost hints beyond the recorded forms.
- Harness/kernel ABI execution. A readable HWX package and field parity do not
  establish ABI compatibility or numerical correctness on a device.

The existing lane-a graph has 28 bias-enabled convolutions, starts with the
space-to-depth 16→32 stem at 112, and uses later 256/512/1024 widths plus
stride-2 pointwise layers. Those are outside the supplied H16/H17 facts.
Reproduce the exact attempt and a layer-by-layer descriptor coverage report:

```
python3 tools/h16-mobilenet/coverage.py build/mobilenet-h16-h17
```

The tool saves the same random-weight graph and parameters as
`tools/h13-mobilenet`, attempts `--schedule linked` on both targets, writes
`coverage.json`, and exits 2 on refusal. It never labels a refusal or an
H13-relabelled object as an H16/H17 package. Wider decoded schedule/bias/sparse
and DRAM-edge evidence is needed before full-model loadable packages can be
emitted.

## Linked programs (DRAM edges)

`--schedule linked` keeps every layer's measured standalone task and links the
tasks through DRAM scratch. Measured fields (round-2 DRAM chain):

- producer (every non-final task): TileDMADst `0x5100` low byte `0xe1` (H17)
  or `0xf1` (H16G), task index in bits 16..23, destination relocated on BAR
  slot 3; no CacheDMA block;
- consumer (every non-first task): TileDMASrc `0x4d00 = 0x20000021 | index<<16`
  (DependencyMode bit 29), source relocated on BAR slot 3;
- headers: first `0x68`/`0x00fff868`, middle `0x00fff800` (H17) or
  `0x00fff820` (H16G), last `0x04000000`/`0x00fff860`; word 7 `0x1fffffff`
  on every non-final task;
- kernel bases are the running sum of the per-layer sections, which are
  packed back to back; scratch holds two regions, each the largest
  intermediate allocation; the program descriptor is the scratch-backed
  `0x8c0` form.

On the J700 the round-2 consumer form (low byte `0x41`, `0x4d08 = 0x4e`) and
this DependencyMode-only form gave identical, deterministic results; without
the dependency fields the chain races. L2-resident chains remain descriptor
parity only.

## Device evidence (J700, 2026-11-01)

All runs used a research harness that loads the HWX sections directly; the
clean kernel driver path is separate work. Mismatch counts are against the
float64 per-layer fp16 chain; the engine's accumulator is narrower than fp32,
so 1-ulp differences of a few percent per layer are expected.

| Run | Result |
| --- | --- |
| 23 stride-1 layers, real weights, standalone | all outputs written; 2.5-11 % of values 1 ulp off (estimated rel-L2 ~1e-4) |
| 5 stride-2 layers, sparse kernels, standalone | stem 1.8 %, depthwise 1.7-5.3 % 1-ulp mismatches, deterministic |
| 5 linked stride-1 segments (2-12 tasks) | dependency forms identical and deterministic; 11 layers linked take 54 % of their standalone engine time |
| full network, one 28-task program, 16 images | top-1 16/16 vs float64; host p50 1.744 ms, p90 1.747 ms, p99 2.204 ms; engine p50 38251 ticks at 24 MHz |

The scratch regions in the device run were 2 MiB each; the compiler sizes them
to the largest intermediate (1.75 MiB). The emitted object is otherwise
identical: same descriptors, same kernel bytes.
