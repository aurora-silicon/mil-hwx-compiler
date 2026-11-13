# H16G layer normalization r6 mean-prefix repair

The r5 prefix bisect found the first bad boundary at task 1 for all three
geometries. The exact non-finite count identifies the appended compact writer,
not the reduction accumulator, as the first defect visible at the external
surface.

For either Whisper width, prefix 1 exposes the edge tile at 348 rows by 16
bytes. That is 5,568 bytes, or 2,784 fp16 lanes. The device wrote exactly one
64-byte cache line (32 lanes) and left 2,752 lanes in the output poison:

```
348 * 16 / 2 - 64 / 2 = 2784 - 32 = 2752
```

The BERT view is still narrower (128 rows by one fp16). With no complete
result-DMA plane named, none of its 128 scalar lanes became externally
visible. This accounts for all of the non-finite output without requiring an
accumulator-width failure.

## Mean producer field table

The table lists every mean-task word in the axis, accumulator/divisor, and
result-layout groups. The comparison is the passing H=8 attention softmax sum
reduction. Values marked *derived default* are intentionally absent from the
measured task stream; the task shape supplies them.

| Named field | Address | BERT mean | Whisper edge-tile mean | Passing H=8 softmax sum |
|---|---:|---:|---:|---:|
| Reduction width | `0x0004` | 768 | `C` (384 or 512) | 1500 |
| Reduction groups | `0x0008` | 1 | 1 | 1 |
| Outer rows/heads | `0x000c` | 128 | 348 | 8 |
| Outer tile count | `0x0024` | derived default | derived default | 1500 |
| Tile height | `0x0034` | 1 | 1 | 1 |
| Reduction compute class | `0x003c` | `0x50` | `0x50` | `0x50` |
| Reduction/lane selector | `0x0044` | 9 | 9 | 9 |
| Auxiliary axis selector | `0x0048` | `0x400` | derived default | derived default |
| Accumulator/finalize selector | `0x4500` | derived default | derived default | `0x1000` |
| Mean divisor, fp32 | `0x4514` | `1/768` | `1/C` | not present (sum) |
| Result mode | `0x4148` | external-to-resident mean | same on the first tile | derived resident default |
| Result base | `0x414c` | named BERT mean base | edge-tile position × 384 × 16 | implicit continuation |
| Result row bytes | `0x4150` | derived 2 | 16 | 16 |
| Result row/plane stride | `0x4154` | derived default | derived default | derived default |
| Result plane alias A | `0x4158` | derived default | derived default | derived default |
| Result plane alias B | `0x415c` | derived default | derived default | `8 * 16` |

The r6 change deliberately leaves these measured producer words byte-identical
to r5. A same-value repeat of `0x4514` is packaged as the dead-write control:
it changes task bytes and record count but leaves the decoded divisor and all
other register state unchanged.

## External writer field table

Prefix 1 adds a final writer after the mean producer. This is where r5 used
two full-program defaults that do not survive compacting the view.

| Named field | Address | r5 compact prefix | r6 compact prefix | Passing attention final writer |
|---|---:|---:|---:|---:|
| Zero-beta source BAR offset | `0x4d28` | `2 * compact_elements` | `2 * model_channels` | not affine |
| Result BAR offset | `0x5108` | 0 | 0 | 0 |
| Result DMA row bytes | `0x5110` | producer row bytes | producer row bytes | padded score row |
| Result DMA plane bytes | `0x5114` | omitted | `producer_rows * producer_row_bytes` | complete score plane |
| Result DMA mode | `0x5138` | external fp16 result | external fp16 result | same external fp16 result mode |

The constant table stores the complete identity-gamma vector before zero beta.
Using the compact width therefore selected gamma and added approximately one
to the few values that were written. The judge computes correlation only over
finite pairs, so four visible compact rows can look direction-right by chance:
the base geometry reached about 0.85, tiny about 0.39, and BERT had no finite
pairs and therefore NaN. The different correlations are not evidence for
different reduction axes.

R6 uses the original model width for the beta offset and explicitly supplies
the contiguous result-plane span. The same writer is used after every retained
prefix, so once task 1 is finite the kit immediately measures the subsequent
variance and normalized producer views. Each layer-normalization object is
loaded afresh three times; the judge requires bit-identical cold outputs and a
final relative L2 error no greater than 0.02. The production gate remains
closed pending that device review. The H=6 and H=8 attention objects are
checked byte-for-byte against r5 and gate #44 while the kit is built.
