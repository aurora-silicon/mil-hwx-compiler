# H14 formula-built convolution and linked DRAM schedules

`--target H14 --schedule linked` accepts a straight chain of dense and
depthwise convolutions, with optional ReLU after each convolution. The
MobileNetV1 graph and preprocessing are the same as
[the H13 package](../../tools/h13-mobilenet/README.md). Device execution and
accuracy on the base M2 remain a lead-side qualification step.

## Evidence and field map

The existing H14 corpus includes 546 decoded `env_conv`/`conv_probe` records.
The original strict parity suite covers the 284 convolution records whose
constant packing is derived, alongside the other 438 parity cases. The new
[compile-only probe driver](../../research/mint_h14_mobile.py) adds 64
MobileNet layer records (bias/no-bias and ReLU/no-ReLU pairs at 16/32/64/128/
256/512/1024 channels and spatial sizes 112/56/28/14/7) and six two/three-layer
DRAM chains. These records retain decoded fields, sizes, status and hashes;
the temporary compiler output is removed. The compiler is never asked to
load or execute a program.

Field names and positions were checked against the BSD-3-Clause
[freedomtan H14 map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/main/hwx_dump/h14_register_map.md).
The formulas and corrections below come from this repository's decoded
corpus. No implementation code or binary payload is copied from that map.

| Meaning | H13 | H14 | Encoding or formula |
| --- | --- | --- | --- |
| Input/output W,H | `0x0`, `0x14` | same | `(H << 16) \| W` |
| Input/output channels | `0xc`, `0x10` | same | full input, this task's output slice |
| OCG | part of `0x1c` | `0x1c` | `log2(lanes)`; depthwise 0 |
| Convolution geometry | `0x1c` | `0x20` | KW 0–5, KH 6–11, SX 13–14, SY 15–16, PX 17–21, PY 22–26 |
| Depthwise mode | `0x24` | `0x28` | decoded `0x14001` family |
| Output tile height | `0x28` | `0x2c` | rows fitting the L2 budget |
| NE scheduling | `0x34` | `0x38` | width/lane scheduling classes; a performance choice |
| L2 source channel stride | `0x480c` | `0x50c` | `align16(max(16, 2*W))` |
| L2 source row stride | `0x4810` | `0x510` | channel stride * Cin + bank skew |
| L2 depth/group stride | `0x4814/18` | `0x514/18` | channel stride * Cin |
| L2 result staging base | `0x4834` | `0x534` | staged input rows * source row stride |
| Source DMA config | `0x13800` | `0x1100` | H14 DependencyMode at 28–31 |
| Source cache extensions | embedded | `0x1108` | program input `0xce`, scratch `0x4e` |
| Source base, row, plane | `0x13808/c/10` | `0x1110/18/1c` | byte offsets/strides, 64-byte aligned |
| Destination config | `0x17800` | `0x1500` | program output cache hint 3, scratch hint 15 |
| Destination base,row,plane | `0x17804/08/0c` | `0x1508/10/14` | output plane * first channel; physical row/plane strides |
| Kernel section addend | per-chunk bases | BAR pair at `0x1908/c` | this task's section base in concatenated constants |
| 16 coefficient bases | `0x1f848..` | `0x1958..` | offsets within the task's packed section |
| 16 coefficient sizes | `0x1f888..` | `0x1998..` | actual packed plane lengths, each 64-byte aligned |
| ReLU | `0xc804` bit 16 | `0xd04` bit 16 | fused on every output-channel task |

The public map's L2 labels around `0x530` are shifted relative to the corpus:
`0x530` carries ResultCfg (`0x50017a` in ordinary DRAM output tasks), and
`0x534` carries the staging base. The encoder follows the decoded values.
The coefficient BAR pair writes the same addresses as an earlier zero
Prefetch record: **last write wins**. Constant relocation must update the BAR
payload, not the earlier zero.

## Task framing, ordering and routing

H14 uses an eight-word header, a 16-byte zero prefix and 16-byte task
alignment. Word 0 contains its own word count in bits 26–16 and TID in
15–0. There is no NextPointer, NextSize, H13 continuation/end flag, or ninth
DTID word. Word 6 is LiveOuts; word 7 holds TSR bit 0 and engine selection.
In particular, copying H13 TSE/TSR (`0x9000`) into word 6 would corrupt
LiveOuts. The independent generic tasks close their live-outs and keep the
observed H14 TSR form `0x50001`; the consumer's DMA dependency supplies the
DRAM hand-off.

A dense BAR record has bit 29 set, and BAR slot in bits 28–23. The native
convolution records bind slot 1 to constants, slot 4 to input and slot 5 to
output. Linked scratch reads and writes both use slot 3. They do not use an
H13 header selector. Every intermediate gets a separate 16-KiB-aligned
region. No region is reused.

On each layer's first task after the first layer, the source config sets
DependencyMode 1 (`0x10000000`). Further output slices read the same region
with DependencyMode 0. Scratch readers use the decoded low config `0xe41`
and extension `0x4e`; scratch writers use cache hint `0xf`, producing
`0x01000ef1` in the ordinary output mode. The fresh `chain2_c128_s112` and
`chain3_c512_s56` probes exhibit these exact settings. Per-DMA task tags in
bits 23–16 advance with task index; the kernel master tag is index + 1.
The linker refuses more than 254 tasks, keeping these tags in range.

The ANEC header records scratch allocation on channel 3. The M2 runner
binds an explicit port table: BAR 4 → input buffer 5, BAR 5 → output buffer
4, and BAR 3 → scratch buffer `0x40`. This avoids the runtime's heuristic
scratch-size inference, which uses the final output size and is insufficient
for a MobileNet arena containing much larger early intermediates.

## Envelope and verification limits

Formula-built convolutions emit ANEC. HWX remains available for the existing
exact-parity forms; generic HWX is refused until its descriptor cost fields
are derived.

The generic path covers fp16 NCHW batch 1, square 1x1/3x3 dense or depthwise
stride 1 and dense 1x1 stride 2, symmetric same/valid/custom padding and
channel counts divisible by 16. It borrows invariant family registers from
a small decoded task, like H13; all dimensions, tiling, strides, coefficient
sizes/offsets and output slices are calculated. The default per-op path
retains the existing exact-parity encodings; `MILHWXC_H14_CONV_GENERIC=1`
forces the generic path for comparison.

The generic planner uses conservative budgets (1 MiB kernel bytes and
`0x1e0000` L2 bytes per task). Apple's H14 scheduler sometimes uses larger
single tasks, input reuse or a different width class. Therefore generic task
bytes and task counts are not claimed to equal every Apple schedule. The
strict parity suite still checks all 722 original cases byte for byte; the
new corpus audit visits all 610 old/new convolution records (403 generic
geometry cases and 207 explicit refusals), checks supported packed constants,
and requires explicit refusal for unsupported families. No host test proves
execution accuracy for a new schedule.

Wide stride-2 layers split into 128-channel tasks; **no task carries a sparse
form above eight lanes**. Grouped non-depthwise, multi-tap stride 2 and mixed
lane chunks remain refused. Spatial dimensions and scratch/constant/task
limits are checked before publishing a package.

```sh
make test-h13 test-h13-parity test-h14-parity
# Portable encoding checks on Linux, without Foundation/GNUstep:
make CXX=clang++ test-h14-conv-portable
PYTHON=python3 tools/h14-mobilenet/package.sh build/mobilenet-h14
```

See [the lead-side runner instructions](../../tools/h14-mobilenet/README.md).
