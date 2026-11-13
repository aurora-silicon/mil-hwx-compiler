# H13 ResNet-50 round 17: residual-add operand gates

Round 16 fixed the external TileDMA row used by every standalone first-block
convolution. Those tasks are correct and remain byte-identical in round 17.
The remaining non-repeatable block01-add-only result was a separate two-source
problem: both physical operands already used coherent 128-byte rows, but the
add descriptor did not enable and phase its second L2 operand like the
repository's existing H13 binary-add implementation.

The round-16 add combined two incompatible forms. Its first and second
TileDMA configurations were both the source-1 value 0x33881, Common Cfg was
the one-source PE value 0x31, and L2 SourceCfg omitted the binary
second-source bit. Both L2 rows also equalled their 28,672-byte depth, despite
the existing elementwise add rule requiring a 32-byte PE bank phase. The
physical main, skip, and result surfaces were already correct:
row 128, plane 7,168, and depth 1,835,008 bytes.

This explains the cold-run variation. The descriptor selected a second
surface channel and supplied its address, but neither Common nor L2 admitted
that operand through the binary path. The secondary DMA word also duplicated
source 1's transfer-enable bit. The PE could therefore consume prior contents
of the nominal second staging region instead of a defined projected-skip
operand. This is an uninitialised/stale L2 read, not fp16 add error and not a
112-byte physical-row regression.

## Add field table

All sizes are bytes. Main is the conv3 output and skip is the projected
shortcut. Values are constructed from the tensor geometry and storage class.

| named field | passing external conv | r16 add-only | r17 add-only |
|---|---:|---:|---:|
| Common Cfg / binary stage | convolution schedule | 0x31 | 0x33 |
| L2 SourceCfg | 0x00500172 | 0x00500172 | 0x01500172 |
| main L2 channel | 112 | 112 | 112 |
| main L2 row / depth | 7,168 / 7,168 | 28,672 / 28,672 | 28,704 / 28,672 |
| skip L2 channel | n/a | 112 | 112 |
| skip L2 row / depth | n/a | 28,672 / 28,672 | 28,704 / 28,672 |
| source-1 TileDMA Cfg | 0x33881 | 0x33881 | 0x33881 |
| source-2 TileDMA Cfg | n/a | 0x33881 | 0x33880 |
| main external base / row / plane / depth | 0 / 128 / 7,168 / 458,752 at C64 | 0 / 128 / 7,168 / 1,835,008 | unchanged |
| skip external base / row / plane / depth | n/a | 0 / 128 / 7,168 / 1,835,008 | unchanged |
| L2 ResultCfg | 0x0000014a | 0x0000014a | 0x0050017a |
| result external base / row / plane / depth | 0 / 128 / 7,168 / shape depth | 0 / 128 / 7,168 / 1,835,008 | unchanged |

For a resident block add, the same rule sets Common Cfg 0x33, adds bit 24
to the storage-specific L2 SourceCfg, and keeps both branch rows at
depth + 32. Source 2 remains resident and its TileDMA stays disabled. A
terminal add uses the PE external result configuration; an internal ring
result retains the resident result constructor.

## Device morphs

The ordinary round-17 add carries the coherent six-word fix. The ordered morph
set starts from a byte-exact round-16 add-only task:

1. round-16 baseline and a known-dead main-source group control;
2. one-word L2 second-source, Common second-source, source-2 DMA, and result
   configuration switches;
3. one-word main and skip L2-row switches;
4. main and skip TileDMA base/row/plane/depth sensitivity probes;
5. result row and plane sensitivity probes.

Every morph is one task, binds the same two coherent CPU-reference inputs,
requests three cold runs, and differs from the baseline in exactly one named
32-bit register. Geometry sensitivity probes intentionally select the former
112-byte class or a one-row base displacement; they do not replace the
ordinary fixed add.

tools/h13-resnet50/round17.py emits the complete side-by-side table, verifies
all one-word deltas, and requires stem, pool, conv1/conv2/conv3, projection,
the H=1 probe, and next-stall projection to be byte-identical to round 16.

## Qualification

The compiler-built kit keeps the full ladder through block 16, global mean,
and logits so hardware review can measure whether correcting the first
two-source boundary also moves the later non-finite and hang ceiling.
Acceptance still requires add-only and one-bottleneck-block relative L2 at
most 0.02 and bit-identical over three cold runs, with zero DART faults and no
hang.
