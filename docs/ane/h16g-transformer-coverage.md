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
| `linear` d->d (4) | `[1,128,768] x 768x768` | refused | square 896 matmul, weights as a runtime surface, bias via a ones column | padded (7x MACs) | constant-weight linear (1x1 conv over tokens) |
| `linear` d->4d (1) | `768 -> 3072` | refused | 4 x square 896 | padded | same |
| `linear` 4d->d (1) | `3072 -> 768` | refused | 4 x square 896 + adds | padded | same |
| `reshape` (4) | head split/merge | refused | layout only | compiler | fold into producer/consumer strides |
| `transpose` (4) | perm 0,2,1,3 / 0,2,3,1 | refused | host | facts | transposes as DMA strides or layout tasks |
| `mul` 1/8 (1) | scores | refused alone | output-scale word of the score matmul | device (fused) | done |
| `matmul` QK^T (1) | `[1,12,128,64] x [1,12,64,128]` | refused | per head: d_head zero-padded 64->128, attention chain program 0 | padded (2x) | batched runtime matmul over heads |
| `softmax` (1) | `[1,12,128,128]` | refused | per head: attention chain program 1 | device per head | batched heads |
| `matmul` PV (1) | `[1,12,128,128] x [1,12,128,64]` | refused | per head: attention chain program 2 | padded (2x) | batched runtime matmul |
| `gelu` (1) | `[1,128,3072]` | refused | host, or LUT on a `[1,1,1024,1024]` view of the contiguous tensor | compiler | rank-3 LUT form |
| `add` residual (2) | `[1,128,768]` | refused | host, or ALU on a `[1,1,512,512]` view | compiler | rank-3 ALU form |

### Whisper-tiny / Whisper-base encoder (S=1500)

| Op (count per layer) | Shape (tiny / base) | Today | Route now | Status | Proper route |
| --- | --- | --- | --- | --- | --- |
| conv1d k3 s1 stem (1 per model) | `[1,80,3000] -> [1,384/512,3000]` | refused (rank 3) | none | facts | 1x3 kernels over T=3000 |
| conv1d k3 s2 stem (1 per model) | `[1,d,3000] -> [1,d,1500]` | refused | none | facts | stride-2 1x3 kernels |
| `gelu` stem (2 per model) | `[1,d,3000]`, `[1,d,1500]` | refused | LUT on a 2048x2048 view | compiler | rank-3 LUT form |
| `transpose` + positional `add` (1 per model) | `[1,d,1500] -> [1,1500,d]` | refused | host | facts / compiler | |
| `layer_norm` (2, +1 final) | `[1,1500,d]` | refused (planner SRAM overflow) | host | facts | tiled layer_norm |
| `linear` (6) | `384/512 -> 384/512, 1536/2048` | refused | square 1536/2048 matmul with padding | padded | constant-weight linear |
| `matmul` QK^T, PV (2) | `[1,h,1500,64]`, `[1,h,1500,1500]` | refused | square 1536 matmul per head, d_head padded to 1536 | padded (24x) | batched non-square runtime matmul |
| `softmax` (1) | `[1,h,1500,1500]` | refused (SRAM overflow) | none | facts | tiled row softmax |
| `mul`, `reshape`, `transpose` | as BERT | refused | as BERT | as BERT | as BERT |

## Device results

See the receipt for the J616s run of the staged forms and of a BERT
self-attention sub-block built from them.

## What is needed next

The missing facts are tracked as four research items: constant-weight linear
at the transformer widths (with bias, GELU and output scale); layer_norm,
softmax and rank-3 elementwise at transformer shapes; attention data movement
and batched runtime matmul over heads; and the Whisper conv1d stem. Until
they land, the padded routes above are the only H16G paths for these models,
and they waste most of their multiply-accumulates.
