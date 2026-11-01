# H16G/H17 stride-2 sparse convolution kernel grammar

Task: `t-4235-daybreak-re-h16g-h17-stride-2-sparse-kernel-gram`
Compiler: macOS 26.6.2 (25G83), compile-only
Targets: H16G and H17

## Result

H16G and H17 use the same sparse kernel bytes for every measured form.  The
reference implementation is `pack_h16g_h17_sparse_conv.py`; it reconstructs
the public BLOBFILE view and matches the committed kernel-section SHA-256 for
all five MobileNetV1 stride-2 shapes, their bias variants, and two nonuniform
signed-zero probes on both targets.

Sparse mode is selected independently in NE `0x4900` bit `0x100` and
Kernel-DMA `0x5500` bit `0x20`.  The measured values are H16G `0x00000182`
and H17 `0x10000182` at NE, and `0x000102e0` at Kernel-DMA for a standalone
task.

Only decoded formulas, generated test patterns, hashes and register facts are
retained.  Compiler-produced objects existed only in temporary directories.

## Chunk and channel order

There are always 16 chunks and each complete chunk is zero-padded to a
64-byte boundary.

- Depthwise C64/C128/C256/C512 is striped.  Chunk `i` contains channels
  `i, i+16, i+32, ...`, in that order.  Each channel is one coefficient
  plane.
- The regular 3->32 stem is contiguous.  Chunk `i` contains output channels
  `2*i, 2*i+1`, in that order.  The chunk has one coefficient plane whose
  phase rows repeat for input channels 0, 1 and 2.

The 16 chunk byte strings are concatenated without another section-level
alignment.

## Plane header, masks and zero rule

A plane is:

```text
[bias halfwords, when present]
u16le n
u8  zero                         # structural, not included in n
n bytes of phase masks and nonzero fp16 values
```

For each phase row, all mask bytes come first.  The selected fp16 halfwords
then follow in mask-byte order and ascending bit order.  A cleared bit has no
halfword in the stream.  `n` is exactly the mask-byte count plus twice the
number of set bits; it excludes the initial structural zero and all bias
bytes.

The zero predicate is `(half_bits & 0x7fff) == 0`.  Thus both `+0`
(`0x0000`) and `-0` (`0x8000`) clear the mask bit and are omitted.  Every
other fp16 bit pattern is retained.

Depthwise bias is one channel's halfword immediately before that channel's
`n`.  Stem bias is the two output-channel halfwords, in lane order,
immediately before the chunk's `n`.  Bias does not contribute to `n`.

## Depthwise phase rows

Tap numbers are the source tensor's flat 3x3 order 0..8.

For C64/C128/C256 (`channels_per_chunk` 4, 8 or 16), each channel has two
phase rows:

| row | mask byte / bit -> tap | emitted tap order when all are nonzero |
|---|---|---|
| A | m0: b0->0, b2->2, b4->6, b6->8; m1: b0->1, b4->7 | 0, 2, 6, 8, 1, 7 |
| B | m0: b0->3, b2->5; m1: b0->4 | 3, 5, 4 |

The all-nonzero masks are `55 11` and `05 01`; `n=0x16`.

For C512 (`channels_per_chunk` 32), the same tap order is split across four
phase rows:

| row | mask byte / bit -> tap | emitted taps |
|---|---|---|
| A | m0: b0->0, b4->2; m1: b0->6, b4->8 | 0, 2, 6, 8 |
| B | m0: b0->1; m1: b0->7 | 1, 7 |
| C | m0: b0->3, b4->5; m1: none | 3, 5 |
| D | m0: b0->4; m1: none | 4 |

The all-nonzero masks are `11 11`, `01 01`, `11 00`, `01 00`; `n=0x1a`.

These are stride-2 phase rows, not nine linear tap rows.

## Regular 3->32 stem phase rows

Each phase entry expands to two adjacent bits and two fp16 values, lane 0
then lane 1.  For each input channel:

| row | mask byte / bits | tap and lane mapping |
|---|---|---|
| A m0 | b0/1, b2/3, b4/5, b6/7 | taps 0, 2, 6, 8; lane 0 then 1 |
| A m1 | b0/1, b4/5 | taps 1, 7; lane 0 then 1 |
| B m0 | b0/1, b2/3 | taps 3, 5; lane 0 then 1 |
| B m1 | b0/1 | tap 4; lane 0 then 1 |

The all-nonzero masks per input channel are `ff 33`, `0f 03`.  The two rows
repeat for input channels 0, 1 and 2 inside one plane.  With all weights
nonzero, `n=0x78`.

## Hash verification

The plain uniform BLOBFILE campaign produced:

| form | section bytes | SHA-256 |
|---|---:|---|
| stem 3->32 | 1,984 | `59e42bb358420d1bf7e9d7bc12f146904ba22f6f43664d523fad9deaea375ce4` |
| DW C64 | 2,048 | `7446cb44a8cdf02086289d5c54cb6e23bd103105f221d3f3e8558ebb9c5b6736` |
| DW C128 | 3,648 | `c89e420cce49a5bdaf876327869081a50321c2e444bc834d8776e1c4cf9b45de` |
| DW C256 | 6,848 | `13a0aa7091d655c8f8cbfe66f49c21b2d3804e234803f21b62d753c035be2b13` |
| DW C512 | 15,360 | `4cb5a781789b792494354334a0eac4b83e4acfce940a05874479d5e28cf569cf` |

The bias variants produced:

| form | section bytes | SHA-256 |
|---|---:|---|
| stem 3->32 + bias | 1,984 | `27f2777778ef0df5a7862f356a8e5862ef2848ee25968bae4d11b9e59c121aaf` |
| DW C64 + bias | 2,048 | `180d7f2aa84c082675b730db088773b96533470b99578c67bea77135c3ec4de8` |
| DW C128 + bias | 4,096 | `65f97e1212a515eb3b1ce18c0891ea4fe374f556052e16ce50572f218916fd40` |
| DW C256 + bias | 7,168 | `052795ead812e7402c869fd8435c551e391a81714d4d24a6e1361d9c97578cc5` |
| DW C512 + bias | 16,384 | `b8a810b262603356587f4888fa5cae01940815d64b2c185904072b01c581525a` |

The generated nonuniform probes, containing both exact signed zeros, match:

| form | section bytes | SHA-256 |
|---|---:|---|
| DW C64 signed-zero probe | 2,048 | `6e117f7d524e4afaee1b510913c7f22a579cf3fbd5064c551f54866b9e4c1c79` |
| stem signed-zero probe | 1,984 | `d608e7e7819c54bb220c7c6f8a3198f3f56b2369b1a95a4bddcf19dfa4585e87` |

Every hash in these tables is identical between H16G and H17.

## Sparse selection

No dense stride-2 form was observed.  In addition to all MobileNet forms,
selection controls measured 1x1 regular, 1x1 depthwise and 3x3 groups=4 at
stride 2.  All set NE bit `0x100` and Kernel-DMA bit `0x20`; bias and the
nonuniform signed-zero patterns did not change the selection.  The matched
stride-1 corpus uses dense mode.  Therefore the fact supported by 25G83 is:
stride 2 selects sparse across the measured k1/k3 and g1/g4/depthwise grid;
there is no measured dense-stride-2 condition.

## Newly measured scheduler rows

| form | bias | TileHeight `0x34` | Common `0x3c` | source skew | ResultCfg `0x4148` |
|---|---:|---:|---:|---:|---:|
| DW 3x3 C128 @56 | no/yes | `0x38` | `0x00210000` | `0x00` | `0x0050017a` |
| PW 1x1 128->128 @56 | no/yes | `0x38` | `0x00240000` | `0x50` | `0x00500172` |
| DW 3x3 C512 @14 | yes | `0x0e` | `0x00230000` | `0x30` | `0x0050017a` |
| PW 1x1 512->512 @14 | yes | `0x0e` | `0x00233300` | `0x30` | `0x0050017a` |
| PW 1x1 1024->1024 @7 | yes | `0x07` | `0x00244404` | `0x30` | `0x0050017a` |

For all seven rows H16G and H17 agree.  Bias sets NE `0x4904` bit `0x10`.
The three newly measured full-width bias rows keep the same Common value as
their own plain form; the earlier 512->1024 @7 bias exception remains a
different geometry and is not generalized.
