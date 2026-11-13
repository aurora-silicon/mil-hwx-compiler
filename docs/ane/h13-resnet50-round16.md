# H13 ResNet-50 round 16: external source-DMA rows

Round 15 changed `block01-conv1-only` L2 Source row from 7,168 to
7,248. The M1 Pro output remained byte-identical to round 14. That is a
direct negative result: an external NCHW source does not take its spatial row
advance from this L2 phase word. The word is retained only as the round-16
dead-field control.

The causal field is the TileDMA source row at `0x1380c`. Round 14 and 15
encoded 112 bytes for a 56-pixel fp16 row. H13's external source path
consumes whole 64-byte row beats, so the effective advance was
`112 & ~63 = 64` bytes, or 32 fp16 pixels. This is exactly the spatial
permutation measured on device: output row `y` begins at logical pixel
`32*y`. The earlier offline model reproduced that permutation but assigned
it to the downstream L2 word instead of the DMA that fills L2.

The existing formula-built convolution path already uses the legal external
row class `align64(2*W)`. Round 16 applies the same named rule to residual
program inputs. At W=56 the coherent external source geometry becomes row
128, plane 7,168, depth/group 458,752. The package binding uses the same
128-byte physical row. The native W=224 stem remains byte-identical because
its 448-byte row was already a multiple of 64.

## Input-side field table

All values are bytes except dimensions, channels, and tile height. The full
machine-readable table, including configs, bases, kernel/stride/padding,
Common traversal/group words, and every L2/TileDMA input stride, is emitted as
`round16-fields.json` by `tools/h13-resnet50/round16.py`.

| input-side field | passing stem | passing max-pool resident read | r14 conv1 baseline | r16 conv1 fixed |
|---|---:|---:|---:|---:|
| input HxW / channels | 224x224 / 3 | 112x112 / 64 | 56x56 / 64 | 56x56 / 64 |
| kernel / stride / leading pad | 7 / 2 / 2 | 3 / 2 / 0 | 1 / 1 / 0 | 1 / 1 / 0 |
| tile height | 112 | 56 | 56 | 56 |
| L2 source channel | 448 | 224 | 112 | 112 |
| L2 source row | 1,344 | 14,384 | 7,168 | 7,168 |
| L2 source depth / group | 1,344 | 14,336 | 7,168 | 7,168 |
| TileDMA source row | 448 | disabled (224 retained) | **112** | **128** |
| TileDMA source plane | 100,352 | disabled (25,088 retained) | **6,272** | **7,168** |
| TileDMA source depth / group | 301,056 | disabled (1,605,632 retained) | **401,408** | **458,752** |
| package input row / plane | 448 / 100,352 | n/a (resident) | 112 / 6,272 | 128 / 7,168 |

`stem-pool-via-dram` is also included in the JSON table. Its pool input uses
the scratch interleave-8 source path, not the external NCHW source path, so it
does not exercise the failing alignment class.

## Morph and row probes

The kit contains matching morph sets for conv1-only and projection-only.
Each object has one task and requests three cold runs. The order is chosen to
put size-only changes before addressing changes:

1. r14 baseline;
2. dead L2-row control (only 7,168 -> 7,248);
3. TileDMA source group only;
4. TileDMA source depth only;
5. TileDMA source plane only;
6. TileDMA source row only.

The unit test and `round16.py` require each morph to differ from the r14 task
in exactly its named 32-bit register. The ordinary round-16 sub-rungs carry
the coherent four-word fix, rather than a patched diagnostic object.

`block01-conv1-row1` changes the standalone conv input and output to H=1. It
removes every inter-row advance while retaining the 64-channel plane walk.
If this probe is correct while an H=56 row morph is wrong, the row word is the
remaining discriminator; if H=1 is wrong, the plane/channel walk is still
incorrect.

## Host gates

The ResNet geometry test asserts the r14 endpoint, the dead control, all ten
single-word candidate objects, the coherent fix, and the H=1 dimensions.
The kit audit models the 64-byte TileDMA beat directly: the r14 task advances
32 logical pixels, while the fixed task advances one padded 56-pixel row.
Passing stem, pool, and next-stall programs are compared byte-for-byte with
round 15 during kit generation.
