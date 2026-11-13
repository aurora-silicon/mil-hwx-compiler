# H16G transformer coverage: BERT-base and Whisper encoders

This page records which operations of a BERT-base encoder layer and of the
Whisper-tiny and Whisper-base encoders the H16G (M4) backend can emit today,
which ones it can emit only through zero padding, and which ones need new
field facts. "Device" means the program ran on an M4 Pro (J616s) under Linux
through the ABI 3 accel node and matched a float64 CPU reference.

## Model shapes

| Model | Tokens S | d_model | Heads x d_head | FFN | Layers |
| --- | ---: | ---: | --- | ---: | ---: |
| BERT-base layer | 128 | 768 | 12 x 64 | 3072 | 1 |
| Whisper-tiny encoder | 1500 | 384 | 6 x 64 | 1536 | 4 (+2 conv stem) |
| Whisper-base encoder | 1500 | 512 | 8 x 64 | 2048 | 6 (+2 conv stem) |

A BERT layer is 23 MIL operations after Core ML conversion; the Whisper-tiny
encoder is 99. Before this work every one of them was refused by name on
both the H16G and the H17 routes.

## Native transformer-linear formula update

The H16G route emits constant-weight rank-3 `linear` as a 1x1 convolution
over tokens, including optional bias, fused GELU, and fused output scale.  The
24 measured BERT/Whisper descriptor and packed-kernel rows remain exact after
normalizing standalone-object BAR relocation.

Standalone objects bind input, output, and `__KERN_0` at BAR slots 6, 7, and
8.  The packed kernel region also contains bias and post-op tables.  Earlier
linear objects corrected only the kernel operand; their source/result records
still named slots 5/6 and could load with unbound or aliased DMA.  All external
linear records now use 6/7, scratch-backed contractions retain BAR 3, and all
coefficient records use BAR 8.  `h16g_validate.py` recognizes the linear task
signature, checks these aperture roles, and bounds the input and output DMA
walks.  The corrected v3 linear kit ran without DART faults.  Plain, biased,
scaled, and scratch-backed cases measured relative L2 `4.4e-4` to `7.8e-4`.

The fused BERT up-projection initially measured `1.77e-2` against a reference
that evaluated mathematical erf.  Its packed table and placement match the
measured task.  The decoded H16G table instead contains 33 fp16 samples over
`[-4,4]`, linearly interpolates between them, clamps the negative tail to zero,
and passes through the positive tail.  Using that hardware function with the
unchanged inputs and program gives relative L2 `6.46e-4` against the preserved
J616s output.

## Whisper stem object packaging update

Whisper stem1, stem2, and encoder transpose retained captured source/result
BAR slots 5/6 while standalone objects bind their input/output surfaces at
6/7.  Stem v2 therefore loaded with its kernel correctly at BAR 8, then the
first case produced only 66 exact values out of 1,148,944 compared lanes and
logged ten DART translation faults.  All stem source/result records now use
the same `taskBar(6)`/`taskBar(7)` law as transformer linear; kernel records
remain at `taskBar(8)`.  Row-padded stem surfaces also declare their physical
strides and allocation spans in the object and runtime metadata.

The offline validator recognizes the BAR-independent stem and transpose task
signatures, so every v2 stem/transpose program is refused for an unbound slot.
It also bounds the already-correct positional-add 6/8/7 DMA walks.  The v3
device run showed that transpose had mapped only the dense 3000-byte rows even
though its source task walks 3008-byte physical rows.  That truncated the tail
of every channel and accounted for the 1,528/2,040 non-exact tiny/base values.
The object and runtime descriptors now expose the complete padded source; the
validator bounds DMA against that tensor mapping rather than the larger,
aligned Mach-O segment.  A v4 runner saves every output and reports relative
L2 against the reference.

## Native attention formula update

The H16G route now emits native rank-3/rank-4 head movement and batched
runtime matmuls for the Whisper S=1500 geometries:

| Form | Tiny | Base | Tasks |
| --- | --- | --- | ---: |
| split / split-transpose / merge | 6 x 64 heads | 8 x 64 heads | 1 |
| QK | `[1,6,1500,64]` | `[1,8,1500,64]` | 3 |
| transpose-y QK | same | same | 4 |
| PV | same | same | heads + 1 |
| last-axis softmax | eight-plane program, two zero planes | measured and enabled | 5 |
| scaled QK + softmax + PV | three proven programs, gate closed | three proven programs, gate closed | 3 dispatches |

The compiler keeps public held-out movement and matmul axes as unrevealed
predictions. They pass the structural validator but do not claim descriptor
or numerical parity. Resident single-program attention remains refused. The
candidate path instead emits scaled QK, measured eight-plane softmax, and PV
as three ordinary programs with identifier-shared DRAM score surfaces.

The first corrected Whisper-tiny device run made the numerical and scheduling
boundaries more precise.  Split/split-transpose/merge were bit exact, QK and
transpose-y QK had relative L2 error `4.6e-5`, and all programs produced zero
DART faults.  Standalone PV had relative L2 error `2.6e-2`.  Its error was
uniform across all six heads with near-zero bias; omitting the 92-element
tail, any periodic 128-element tile lane, or shifting rows made the result
worse.  The standalone encoder therefore stages normalized probabilities at
the exact fp16 scale 128 and folds the exact 1/128 reciprocal into the head
result.  This changes only the two decoded scale words; structural parity is
retained after normalizing that intentional numeric pair.  The first fused
path kept its resident probability plane at unit scale and showed the same
relative-L2 signature.  Its corrected research form preserves the captured
QK/softmax prefix, adds one resident-to-resident P*128 staging task into the
second score plane, and folds 1/128 into each PV head.  This heads+12 form is
structurally validated but remains refused pending a numerical device run.

The next device run completed that resident candidate without translation
faults, but H=6 still measured relative L2 `1.12` despite the right output
magnitude. It is therefore not a qualified fusion. The replacement chain
uses only independently qualified boundaries: the QK result-scale word folds
the MIL scalar, softmax is the measured eight-plane program, and standalone
PV stages P*128 and folds 1/128 into each head. H=6 score tensors occupy the
first six physical planes of an eight-plane intermediate; the runtime clears
shared intermediates before every dispatch, so the two padding planes are
deterministic. The compiler manifest records three programs, dispatch order,
two shared surfaces, and per-program bindings. Qualification gates remain
closed until both H=6 and H=8 pass the reviewer kit.

The first run found that the unrevealed H=6 softmax loaded but never completed.
Its exponential task retained the captured H=8 dependency mask `0x3ff`; the
head-derived `0xff` correction alone did not resolve the wedge.  The remaining
difference was the five task-header schedule words.  The H=6 prediction had
copied the short `0x68`/`0x28` form from the contextual 219-task Whisper
program, although standalone softmax is a serialized five-task program.  It
now retains the captured standalone sequence `0x00fff868`, three
`0x00fff820` words and terminal `0x00fff860`, independent of head count.  The
dependency mask remains `(1 << (heads + 2)) - 1`, bounded to the 32-bit field.
The H=8 measured row remains byte-identical. A second device run with that
serialized H=6 schedule still wedged. Prefix bisection then showed tasks
0 and 0--1 refused at LOAD, 0--2 and 0--3 timed out at LOAD, and the full
program timed out at EXECUTE. In contrast the measured H=8 program completed
with relative L2 `9.4e-4`.

Last-axis softmax normalizes every row on its own, so the heads of a
`[1,H,1500,1500]` tensor are independent planes. The compiler now emits any
S=1500 softmax with one to eight heads as the measured eight-plane program:
the tensor occupies the first H planes, the remaining planes are zero padding,
and their results are discarded. The H=6 object is byte-identical to the H=8
one; only the binding's logical byte count differs. The predicted H=6 schedule
is no longer emitted. Other token counts and more than eight heads stay
refused with `h16g.encode.transformer-softmax-hardware-unproved`.
On a J616s with Linux the H=6 object, run on a padded `[1,6,1500,1500]`
input of N(0,3) scores, matches a float64 reference with relative L2
`5.5e-4` (max abs error `9.9e-4`).

Core ML placed isolated H=6 softmax on the CPU, so there is no captured H=6
standalone program. The compiler closes that gap without predicting a new
schedule: last-axis rows are independent, so it executes the byte-identical
measured H=8 object over an eight-plane allocation and zero pads the final two
planes. The failed H=6 schedule is no longer emitted.

Packaged H16G objects expose external surfaces beginning at BAR slot 6, with
the kernel table after the final surface.  The captured attention records used
the preceding slot numbers, which left the first source unbound and aliased
later inputs/results.  The encoder rebases only those DMA records when it
packages a standalone object.  Fused attention exposes one padded score
surface to the task but requires two complete score planes for the engine's
implicit ping-pong walk.  The first Q staging write begins at the first score
surface's end and fits within the second plane.  Allocating only the visible
score plus Q staging left the hidden remainder unmapped; the device produced
ten translation faults.  `h16g_validate.py` recognizes the decoded
attention and transformer-linear task families, checks source and result
direction, and bounds every DMA extent for which the task supplies complete
geometry, including this hidden second score extent.

## Last-axis layer-normalization evidence and refusal

The decoded construction covers the identity-affine normalization used by
the captured BERT and Whisper encoder programs:

| Shape | Tasks | Measured part | Predicted part |
| --- | ---: | --- | --- |
| `[1,128,768]` | 5 | three-task external-input prefix + normalization | zero-beta output writer |
| `[1,1500,384]` | 11 | nine-task tiled prefix + normalization | zero-beta output writer |
| `[1,1500,512]` | 11 | nine-task tiled prefix + normalization | zero-beta output writer |

The long form splits 1500 rows into three 384-row tiles and one 348-row tail.
It encodes the fp16 value of `1e-5`, promoted to fp32 in the task, and fp32
`1/d` for the reduction.  Reciprocal square root uses the task's direct
normalization unit; there is no external reciprocal or rsqrt lookup table.
Identity gamma is a no-op, so the standalone form does not materialize or
multiply it. The final writer adds the packaged zero beta while writing the
external result. The round-2 device dump showed that removing the preceding
normalization task was not an identity specialization: it produced BERT
zeros, hung Whisper-tiny, and exposed an unnormalized Whisper-base value. The
constructor now retains that measured task and then uses the zero-beta writer
with the serialized first/middle/final dispatch-limit schedule.

Research objects use BAR 6 for the external input, BAR 8 for zero beta
constant table, and BAR 7 for the external output.  The fail-closed validator
recognizes the family and checks direction, aperture binding, and computable
DMA extents. The first BERT build returned all ones; the five/eleven-task
round-1 candidate returned all-zero BERT output; and the four/ten-task
round-2 candidate again returned zeros after it removed the task that actually
performs normalization. The restored five/eleven-task candidates pass the
offline validator, but host structural success is not treated as device
qualification. The compiler refuses the family with
`h16g.encode.transformer-layer-norm-hardware-unproved`.

The full encoder capture stores each explicit affine pair as contiguous fp16
`gamma` followed by fp16 `beta`. Every pair in that capture is identity/zero,
and the isolated random-affine probes were CPU-only. The decoded builder
retains that evidence for offline parity, but no affine or non-affine
standalone layer-normalization object is emitted publicly until the
normalization path passes hardware qualification.

## What the staged H16G backend already encodes

The default H16G route (no `--format hwx`) already holds measured forms that
cover pieces of a transformer, each at a fixed geometry:

| Form | Geometry | MIL accepted |
| --- | --- | --- |
| Runtime x runtime square matmul | `[1,N,N] x [1,N,N]`, N in {128, 256, 512} and multiples of 128 in [768, 4096] | `matmul`, both operands runtime |
| Matmul + scalar output scale | N = 128, 256 | `matmul` -> `mul` by a scalar |
| Matmul + GELU post-operation | N = 128, 256 | `matmul` -> `gelu` |
| Row softmax | `[1,1,128,128]` | `softmax(axis=-1)` |
| Attention head chain | `[1,1,128,128]` q, k, v: three programs (scores and scale, softmax, PV) | `matmul`, `mul`, `softmax`, `matmul` |
| Four-head attention | one program, input `[1,64,4,192]`, S=64, d_head=64 | slices, transposes, `matmul`, `mul`, `softmax`, `reshape` |
| Binary ALU | `[1,1,N,N]`, N in {128, 256, 512, 1024, 2048} | `add`, `mul`, `max`, `min` |
| LUT unary | `[1,1,N,N]`, N in {128, 256, 512, 1024, 2048} | `gelu`, `exp`, `sigmoid`, `tanh`, `silu`, `sqrt` |

None of these had run on Linux before 2026-11-08.

### Contiguous square views

A standalone binary ALU or table-unary operation on external fp16 tensors of
rank 1 to 4 with matching shapes runs as the smallest measured `[1,1,N,N]`
program whose `N*N` lanes cover the tensor. The program reads and writes the
tensor's bytes in order; lanes past the tensor are padding and their results
are discarded. The emitted object is byte-identical to the square program.
The manifest gives the tensor's logical bytes, the square allocation, and the
square's row stride, so a runtime copies the tensor contiguously from offset
zero and zero-fills the rest of the allocation. Examples:

| Tensor | Elements | View | Lane use |
| --- | ---: | --- | ---: |
| BERT residual `[1,128,768]` | 98,304 | 512 x 512 | 38 % |
| BERT FFN `gelu` `[1,128,3072]` | 393,216 | 1024 x 1024 | 38 % |
| Whisper-tiny residual `[1,1500,384]` | 576,000 | 1024 x 1024 | 55 % |
| Whisper-tiny FFN `gelu` `[1,1500,1536]` | 2,304,000 | 2048 x 2048 | 55 % |
| Whisper-base FFN `gelu` `[1,1500,2048]` | 3,072,000 | 2048 x 2048 | 73 % |

Operations fed by another program keep the old square-only rule, because a
producer's surface layout need not be contiguous.

## Operation gap table

Status keys: **device** = runs on the J616s under Linux and matches the CPU
reference; **padded** = runs through an existing form after zero padding,
with wasted work; **compiler** = no new hardware facts needed, only compiler
work; **facts** = needs new field facts for this geometry.

### BERT-base layer (S=128)

| Op (count) | Shape | Today | Route now | Status | Proper route |
| --- | --- | --- | --- | --- | --- |
| `layer_norm` (2) | `[1,128,768]` last axis | refused with hardware diagnostic | five-task normalize + zero-beta writer candidate | round-2 omitted normalize and returned all zero; reviewer run pending | fixed boundary |
| `linear` d->d (4) | `[1,128,768] x 768x768` | native formula | constant-weight 1x1 convolution over tokens | compiler; corrected device run pending | done |
| `linear` d->4d (1) | `768 -> 3072` | native formula | same | compiler; corrected device run pending | done |
| `linear` 4d->d (1) | `3072 -> 768` | native formula | same | compiler; corrected device run pending | done |
| `reshape` (4) | head split/merge | refused | layout only | compiler | fold into producer/consumer strides |
| `transpose` (4) | perm 0,2,1,3 / 0,2,3,1 | refused | host | facts | transposes as DMA strides or layout tasks |
| `mul` 1/8 (1) | scores | refused alone | output-scale word of the score matmul | device (fused) | done |
| `matmul` QK^T (1) | `[1,12,128,64] x [1,12,64,128]` | refused | per head: d_head zero-padded 64->128, attention chain program 0 | device, padded (2x) | batched runtime matmul over heads |
| `softmax` (1) | `[1,12,128,128]` | refused | per head: attention chain program 1 (7 tasks) | compiles; kernel loader refuses | batched heads |
| `matmul` PV (1) | `[1,12,128,128] x [1,12,128,64]` | refused | per head: attention chain program 2 (same form as the score matmul) | padded (2x); waits for softmax | batched runtime matmul |
| `gelu` (1) | `[1,128,3072]` | refused | LUT on a contiguous view: 1024 view, or two 512 views of the halves | done in this change (512 form device-checked) | rank-3 LUT form |
| `add` residual (2) | `[1,128,768]` | refused | ALU on a `[1,1,512,512]` view | device (this change) | rank-3 ALU form |

### Whisper-tiny / Whisper-base encoder (S=1500)

| Op (count per layer) | Shape (tiny / base) | Today | Route now | Status | Proper route |
| --- | --- | --- | --- | --- | --- |
| conv1d k3 s1 stem (1 per model) | `[1,80,3000] -> [1,384/512,3000]` | native formula | one task | v4 relL2 `4.4e-4` | done |
| conv1d k3 s2 stem (1 per model) | `[1,d,3000] -> [1,d,1500]` | native formula | 1-3 tasks | v4 relL2 `1.4e-3` to `1.7e-3` | done for d=384/512 |
| `gelu` stem (2 per model) | `[1,d,3000]`, `[1,d,1500]` | fused into each stem convolution | same program; decoded LUT reference | device-qualified with stem rows | done |
| `transpose` + positional `add` (1 per model) | `[1,d,1500] -> [1,1500,d]` | native transpose then constant-backed 1024-square add | two programs | transpose exact; add relL2 `2.1e-4` | done |
| `layer_norm` (2, +1 final) | `[1,1500,d]` | refused with hardware diagnostic | eleven-task normalize + zero-beta writer candidate | round-2 tiny hung/base wrong; reviewer run pending | fixed boundary |
| `linear` (6) | `384/512 -> 384/512, 1536/2048` | native formula | constant-weight 1x1 convolution over tokens | compiler; corrected device run pending | done |
| `matmul` QK^T, PV (2) | `[1,h,1500,64]`, `[1,h,1500,1500]` | native formula | batched QK/PV programs | device-qualified separately | done for h=6/8, d=64 |
| `softmax` (1) | `[1,h,1500,1500]` | eight-plane program | pad unused heads to eight | H=6 relL2 `5.5e-4` | done for h<=8 |
| attention chain (1) | Q/K/V, h=6/8 | native three-program chain | scaled QK -> softmax -> PV through DRAM | J616s relL2 7.50e-4 / 7.49e-4, 0 DART | done |
| `mul`, `reshape`, `transpose` | as BERT | refused | as BERT | as BERT | as BERT |

## Device results

[The J616s receipt](../../receipts/2026-11-08-h16g-staged-linux-j616s/README.md)
ran the staged forms under Linux on 2026-11-08. Single-task programs and the
two-task square matmul (128 to 512, with or without the fused output scale)
load, run and match their float64 references, including the contiguous-view
residual add `[1,128,768]`. Programs with more tasks (softmax, square matmul
from 768 up, the decoded four-head attention) and the single-task ALU and LUT
programs at 1024 and 2048 are refused by the kernel loader, whose accepted
values for two program-descriptor words are an enumerated list. So on Linux
today:

| Piece | Runs on the Linux ANE now |
| --- | --- |
| Attention score matmul per head, with the 1/8 scale fused | yes (S=128, d_head padded to 128) |
| Attention softmax, context matmul | no (softmax is a 7-task program) |
| Projections and FFN matmuls | only through 512-square tiles plus adds |
| Residual add, BERT shapes | yes (512 view) |
| FFN GELU `[1,128,3072]` | yes as two 512 views; the 1024 view is refused |
| layer_norm | no; standalone BERT/Whisper forms are refused after failed device qualification |

Once the loader accepts these words by their meaning, the padded BERT
self-attention and FFN sub-blocks built from the forms above need no compiler
change.

## What is needed next

The normalization dataflow and non-identity affine consumption remain open.
S=1500 attention data movement and batched runtime matmul over heads have
native encoders; the qualified three-program chain is enabled while the
resident single-program fusion remains a named refusal. The two
Whisper stem convolutions and encoder-axis transpose have measured descriptor
parity; the positional constant uses the established contiguous ALU packet
with its constant in the fourth aperture.
