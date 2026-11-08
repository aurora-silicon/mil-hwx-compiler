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
It also bounds the already-correct positional-add 6/8/7 DMA walks.  Corrected
v3 device qualification remains pending.

## Native attention formula update

The H16G route now emits native rank-3/rank-4 head movement and batched
runtime matmuls for the Whisper S=1500 geometries:

| Form | Tiny | Base | Tasks |
| --- | --- | --- | ---: |
| split / split-transpose / merge | 6 x 64 heads | 8 x 64 heads | 1 |
| QK | `[1,6,1500,64]` | `[1,8,1500,64]` | 3 |
| transpose-y QK | same | same | 4 |
| PV | same | same | heads + 1 |
| last-axis softmax | unrevealed prediction | measured | 5 |
| fused scaled QK + softmax + PV | measured | measured | heads + 11 |

The compiler records S=128 softmax and the public held-out movement and
matmul axes as unrevealed predictions.  They pass the structural validator
but do not claim descriptor or numerical parity.  Fused attention is limited
to the measured S=1500, D=64, H=6/8 forms: its smaller predictions reused
S=1500 scratch offsets and are now refused.  Device qualification for the
corrected rows is pending.  Affine last-axis layer normalization is still
outside the formula envelope.

The first corrected Whisper-tiny device run made the numerical and scheduling
boundaries more precise.  Split/split-transpose/merge were bit exact, QK and
transpose-y QK had relative L2 error `4.6e-5`, and all programs produced zero
DART faults.  Standalone PV had relative L2 error `2.6e-2`.  Its error was
uniform across all six heads with near-zero bias; omitting the 92-element
tail, any periodic 128-element tile lane, or shifting rows made the result
worse.  The standalone encoder therefore stages normalized probabilities at
the exact fp16 scale 128 and folds the exact 1/128 reciprocal into the head
result.  This changes only the two decoded scale words; structural parity is
retained after normalizing that intentional numeric pair.  The fused path
does not cross the external fp16 probability boundary and keeps its captured
unit-scale tasks.

The first run found that the unrevealed H=6 softmax loaded but never completed.
Its exponential task retained the captured H=8 dependency mask `0x3ff`; the
head-derived `0xff` correction alone did not resolve the wedge.  The remaining
difference was the five task-header schedule words.  The H=6 prediction had
copied the short `0x68`/`0x28` form from the contextual 219-task Whisper
program, although standalone softmax is a serialized five-task program.  It
now retains the captured standalone sequence `0x00fff868`, three
`0x00fff820` words and terminal `0x00fff860`, independent of head count.  The
dependency mask remains `(1 << (heads + 2)) - 1`, bounded to the 32-bit field.
The H=8 measured row remains byte-identical.  Device qualification of the
corrected H=6 softmax and fused attention remains pending.

Packaged H16G objects expose external surfaces beginning at BAR slot 6, with
the kernel table after the final surface.  The captured attention records used
the preceding slot numbers, which left the first source unbound and aliased
later inputs/results.  The encoder rebases only those DMA records when it
packages a standalone object.  Fused attention also allocates both the padded
score surface and the padded Q/K staging planes; its first staging write begins
at the end of the score surface.  `h16g_validate.py` recognizes the decoded
attention and transformer-linear task families, checks source and result
direction, and bounds every DMA extent for which the task supplies complete
geometry.

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
| `layer_norm` (2) | `[1,128,768]` last axis | refused | host | facts | layer_norm at transformer widths |
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
| conv1d k3 s1 stem (1 per model) | `[1,80,3000] -> [1,384/512,3000]` | native formula | one task | corrected v3 device run pending | done |
| conv1d k3 s2 stem (1 per model) | `[1,d,3000] -> [1,d,1500]` | native formula | 1-3 tasks | corrected v3 device run pending | done for d=384/512 |
| `gelu` stem (2 per model) | `[1,d,3000]`, `[1,d,1500]` | fused into each stem convolution | same program; decoded LUT reference | corrected v3 device run pending | done |
| `transpose` + positional `add` (1 per model) | `[1,d,1500] -> [1,1500,d]` | native transpose then constant-backed 1024-square add | two programs | compiler; device pending | device qualification |
| `layer_norm` (2, +1 final) | `[1,1500,d]` | refused (planner SRAM overflow) | host | facts | tiled layer_norm |
| `linear` (6) | `384/512 -> 384/512, 1536/2048` | native formula | constant-weight 1x1 convolution over tokens | compiler; corrected device run pending | done |
| `matmul` QK^T, PV (2) | `[1,h,1500,64]`, `[1,h,1500,1500]` | refused | square 1536 matmul per head, d_head padded to 1536 | padded (24x) | batched non-square runtime matmul |
| `softmax` (1) | `[1,h,1500,1500]` | refused (SRAM overflow) | none | facts | tiled row softmax |
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
| layer_norm | no form |

Once the loader accepts these words by their meaning, the padded BERT
self-attention and FFN sub-blocks built from the forms above need no compiler
change.

## What is needed next

The remaining formula gap is affine native layer_norm.  S=1500 softmax,
attention data movement and batched runtime matmul over heads now have native
encoders; the corrected objects still need device qualification.  The two
Whisper stem convolutions and encoder-axis transpose have measured descriptor
parity; the positional constant uses the established contiguous ALU packet
with its constant in the fourth aperture.
