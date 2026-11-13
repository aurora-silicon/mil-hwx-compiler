# H14 formula-built convolution, DRAM chains, and native ring

`--target H14 --schedule linked` accepts a straight DRAM chain of dense and
depthwise convolutions. `--schedule linked-ring` accepts the native-frame
MobileNetV1 chain and applies the 2 MiB circular L2 policy. Both accept an
optional ReLU or finite fp16 clip after each convolution. The MobileNetV1
graph and preprocessing are the same as
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
| Clip | `0xc804` bit 17 | `0xd04` bit 17 | 128-byte duplicated-bound LUT descriptor prepended to every kernel lane |

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
LiveOuts. Single-task layers close their live-outs and keep the observed H14
TSR form `0x50001`; the consumer's DMA dependency supplies the DRAM hand-off.
Every task of a multi-task layer keeps LiveOuts `0x1fffffff`, except the
program's last task; the consumer dependency closes the hand-off.

A dense BAR record has bit 29 set, and BAR slot in bits 28–23. The native
convolution records bind slot 1 to constants, slot 4 to input and slot 5 to
output. Linked scratch reads and writes both use slot 3. They do not use an
H13 header selector. Every intermediate gets a separate 16-KiB-aligned
region. No region is reused.

On each layer's first task after the first layer, the source config sets
DependencyMode 1 (`0x10000000`). Further output slices of a multi-task layer
do not read the input again. They reuse what the first slice staged in L2,
which is the reuse form in every decoded multi-task corpus layer
(`chain3_c1024_s28` tasks 1-3 and 6-8, and the stride-2 consumer of
`h14_s2link_pw1024_pws2_2048`, tasks 5-9, both in research/oracles/h14-mobile).
Such a task has no source TileDMA records and no `0x504`/`0x514`/`0x518` words, its NE repeat fields (`0x38` bits 15:8)
take the NE class, and ResultCfg (`0x530`) drops the DMA-source bits
`0x500030`. Before this, a second slice reloaded the input from scratch DRAM
with DependencyMode 0. On the M2 that slice computed on zeros at MobileNet
layer 9 (the first two-task consumer): its 128 channels equalled relu(bias)
while the first slice was correct, although the same two tasks are correct
when they read the program input. With the reuse form both slices match the
CPU (rel. L2 5.5e-4) and the whole linked MobileNetV1 agrees on top-1 for
16/16 images at 1.13 to 1.21 ms p50 (01/11/2026, 7.3.0-aurora-tt3). Scratch readers use the decoded low config `0xe41`
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

## Rule-built native ring

The native ring path is separate from the template-parity and generic paths.
A layer carries geometry, its 16 coefficient-lane extents, and its
fused activation until the whole chain is known. The allocator then derives
source/result roles, L2 bases, 64 KiB live-bank masks, TileDMA presence, BAR
routing, task tags, and per-task coefficient addends. It emits packed records
directly from those rules and does not read `H14ConvTemplates.inc`.

At 224 pixels, the first depthwise result is the sole interleave-8 scratch
spill. The following pointwise layer narrows its staged source to the widest
16-byte-aligned channel stride that fits beside the whole resident result:
467,712 + 1,609,216 = 2,076,928 bytes. Later placements advance around the
2 MiB ring and preserve the live input. The same fit loop compiles a synthetic
192-pixel network without another geometry case; its bridge input fits at full
width.

Resident tasks omit source and destination TileDMA records. Program input,
scratch read, scratch write and final output are emitted from their roles;
coefficient BAR addends are the accumulated packed-section sizes. The public
tests compare these semantic fields, run a BAR-based port of libane's
bind-overrun walk, and execute NumPy references at 224 and 192 pixels.

Resident L2 rows use the smallest positive convolution execution-lane skew.
One lane is 16 fp16 values (32 bytes). The allocator starts there and advances
in whole-lane quanta until the next row start has a different offset within a
64-KiB L2 bank. This bank-conflict rule uses the actual channel count and
channel stride; it contains no width, stride, or channel-class table. The skew
only changes physical L2 placement, never the convolution result.

Coefficient-lane configuration begins six words after the coefficient DMA
header. A scatter mask has 16 bits, with bit zero naming the following word,
so lanes 0 through 10 fit in that record. The remaining five of the 16 lanes
are consecutive and use one dense record beginning at lane 11.

`compactTaskMinimumBytes` is 312 bytes: the smallest padded task size accepted
by our existing H14 device tests. This is an empirical limit, not a structure
formula. Tasks whose semantic records are shorter are padded with writes to
the ignored destination-extension word. `tools/h14-mobilenet/envelope_kit.py`
builds ReLU ring prefixes p04 and p08 plus the full ReLU ring at padding
minima of 0 (no padding), 128, 256 and 312 bytes. Its index lists every full
ring task's semantic and padded size and the actual size of every task in each
probe program. The device result table is to be filled from device runs:

| padding minimum | load | run | numerical result |
|---:|:---:|:---:|:---:|
| 0 | to be filled | to be filled | to be filled |
| 128 | to be filled | to be filled | to be filled |
| 256 | to be filled | to be filled | to be filled |
| 312 | to be filled | to be filled | to be filled |

A terminal task normally uses `L2Role::DirectResult`. A wide resident source
whose external result exceeds 128×56×56 fp16 bytes plus the maximum compact
row allowance instead uses `L2Role::StagedDirectResult`: channel stride
`align16(2*W)`, row stride `C*channel + 32 fp16 lanes`, and matching
depth/group strides. Its destination DMA drops the compact form's external
enable field. In MobileNetV1 this geometry rule selects exactly the three
prefixes ending at 128x56x56; the adjacent 64x56x56 and 128x28x28 terminals
remain compact. This avoids a prefix-number table and applies to other frame
sizes at the same allocation boundary.

Finite-clip rings use the same split contract as the accurate linked
convolution emitter. `fields::macFiniteClip` selects the prepended bound
descriptor, while `fields::neLutEnable` stays clear. Sparse convolution
independently retains `fields::neSparsePhase`; dense forms retain reset state.
Header word 1 becomes an activation mode: three-tap kernels select
`fields::taskModeKernelClipDescriptor`, and one-tap kernels select mode zero.
ReLU changes only `MacCfg` and retains its scheduler cost. On the native ring,
device comparison found that changing only this header mode and
`neLutEnable` was output-neutral. They remain matched to the linked emitter,
but are not the explanation for the ring's ReLU6 numerical fault.

### ReLU6 safety

The coefficient-lane sizes are stateful KernelDMA registers. A task may omit
them only when it deliberately reuses the preceding task's exact sizes. The
ring's scratch-result compact form used to omit all 16 sizes unconditionally.
At p04 that made L01 inherit the ReLU6 stem's 256-byte lanes even though L01
owns 192 bytes per lane, crossing into the following descriptor. ReLU happened
to tolerate the analogous over-extent; finite clip did not. Finite-clip tasks
now always state `registers::kernelLaneSize`, including scratch writers, while
the established ReLU byte stream stays unchanged.

`KernelCfg.neLutEnable`, `MacCfg.macFiniteClip`, header word 1, and every
coefficient-lane offset/size can still affect command completion. A mismatched
mode can select a contract the coefficient stream does not provide, and a
wrong lane extent can parse another lane's descriptor or kernel bytes. Either
can prevent task retirement and surface as an ANE EXEC timeout. The emitter
keeps the linked modes, builds one `clipDescriptorBytes` prefix through
`clipBoundsDescriptor`, and bounds every explicit lane extent to its constant
section.

On the base M2 the fused ReLU6 MobileNetV1 ring matches the float64
reference (relative L2 1.5e-3, same top-5 as the reference), as does every
ReLU6 prefix from one to 28 tasks, at a p50 of about 587 us per inference.
A standalone pointwise 32-to-64 layer at 112x112 was inaccurate and not
repeatable, with or without the clip: its complete input and result spans
exceed the 2 MiB ring arena. Inside the MobileNet chain the same layer works
because its input is already resident. The ring schedule is for the
MobileNetV1 chain. For a standalone boundary task the compiler now derives a
working-span budget with four named contributors: the source surface, result
surface, external scratch traffic, and largest simultaneous coefficient-lane
prefetch. It refuses the task with `h14.linked-ring-outside-arena` when their
sum exceeds `mobileL2Bytes`, and directs the caller to `--schedule linked`.
The ordinary linked schedule accepts this one-layer generic fallback.
The rejected 32-to-64 form needs 802,816 source bytes plus 1,609,216 result
bytes before the small coefficient prefetch, already exceeding the arena.
Resident chain tasks retain their device-proven streaming allocation and byte
stream; they do not stage two complete external boundary surfaces at once.

For ReLU and no-post-operation ring tasks, header word 1 is a scheduler hint
derived from geometry. The compiler counts multiply-accumulates as
`ceil(input-width/stride) * ceil(input-height/stride) * output-channels *
(input-channels/groups) * kernel-width * kernel-height`, rounds that work up
to one-MiMAC quanta, and, for a terminal task, adds the fp16 result span
rounded up to 64-KiB L2-bank drain quanta. Finite clips replace that low-half
hint with the linked emitter's activation mode described above.

The Common schedule word treats issue class and repeat as separate fields. At
the widest spatial stage the issue class is the mask of the first two issue
pipes. At every narrower stage, including stage one, issue class and both
repeat fields equal the geometry stage capped at four. The inner and outer
fields specify the number of schedule replays; no issue-pipe mask is reused as
a repeat count.

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
`0x1e0000` L2 bytes per task). Other valid H14 schedules can use larger single
tasks, input reuse or a different width class. Therefore generic task bytes
and task counts are not a universal schedule contract. The
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
PYTHON=python3 tools/h14-mobilenet/ring_kit.py build/mobilenet-h14-kit
```

See [the lead-side runner instructions](../../tools/h14-mobilenet/README.md).
