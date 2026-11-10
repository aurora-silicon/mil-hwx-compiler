# H13 residual linked ring

`mil-hwxc --target H13 --schedule linked-ring` recognizes a residual graph
with a convolution stem, max-pool, basic (3x3,3x3) or bottleneck
(1x1,3x3,1x1) blocks, global mean, and a linear classifier. It emits one
program. The public 224 ResNet-50 is 80 tasks; 192-input ResNet-50 and a
ResNet-18-shaped graph use the same constructors and allocator.

Every task field is constructed from operation geometry and storage semantics.
Block discovery identifies main and projection branches, hardware-capacity
formulas split large kernels, and a circular first-fit allocator keeps live
shortcuts and main values disjoint in the 2 MiB L2 ring. When the live set does
not fit, the same routing constructors emit a bind-sized DRAM spill. No stage,
block, or layer index selects an address, stride, size, control, or selector
word. No per-network task table, HWX payload, or constant payload is stored.
Kernel constants are rebuilt from the supplied licensed MIL blob.
The global-mean fp32 scale is calculated as `1 / (H * W)` from its input, so
the same constructor averages 7x7, 6x6, and other valid final surfaces.

The L2 layout is consumer geometry, not tensor allocation padding. A spatial
row advances across every channel plane and then applies the shared
`l2Skew(stride, width, channels)` bank rotation. The previous residual-only
rule appended one complete channel row instead. Host decoding exposed the
error on the exact `[3,224,224]` external stem input: its channel stride agreed
with the device-proven linked-ring stem, but its row extent included the extra
row and its last extent was zero. External NCHW staging now repeats the whole
pixel-row span in all three extent fields, while resident results use the
consumer's stride-class bank skew. This is one rule for 224/192 ResNet-50 and
ResNet-18; it is not selected by a model or layer name.

For the reference 224 model the result has 80 descriptors, 53,158,912 bytes
of rebuilt constants, and a 12,058,624-byte spill arena. Channel 5 is the
`[1,3,224,224]` input with a 448-byte row and 100,352-byte plane. Channel 4 is
the `[1,1,1,1000]` output with a 2,000-byte physical row.

The optional private parity gate reports per-task field-difference counts and
the constant byte-difference count against the compiler output made from the
same MIL and blob. It is a report, never an encoder input or a public table.
Allocator-chosen L2 bases are allowed to differ. The public unit test rebuilds
224 and 192 bottleneck networks and a basic-block network with deterministic
centre-tap fp16 weights. An independent float64 residual reference is executed
for all three, and every compiled task's input/output geometry, final spatial
extent, and mean scale are checked before the generated ANECs reach the libane
extent gate.

The residual stream preserves each task template's BaseEnable word while it
replaces only the three surface selectors and task-buffer bit. Consequently,
kernel table/execution enables remain set on convolution and linear tasks and
remain clear on pool, add, and mean PE tasks. Every task emits TSE+TSR before
its successor can consume an L2 or spill dependency. The stem-to-pool edge is
the linked-ring initial-alias form: the stem keeps the populated destination
geometry and producer control, while the pool carries the matching source
configuration, consumer control, and channel-3 selectors. A host test decodes
every task of the 224/192 bottleneck programs, ResNet-18, the two-task pool
rung, and the first-block rung; it also checks that the stem result base and
channel/row/plane strides equal the pool source fields, while the terminal
pool plane/depth fields and the separate PE piece-advance word follow their
respective package and output-shape rules.
It additionally checks all sixteen stem kernel-DMA spans, the native sparse
kernel fields, the leading two-cell TensorFlow-SAME padding of the 7x7 stem,
and the exact external-source layout shared with the proven 3x3 linked-ring
stem.

Rounds 8 and 10 showed that both the scratch-fed and resident forms of task 1
produce the same wrong pool columns.  Round 10 also proved that changing
Common `0x2c` from the old role class to the width-derived class, and repeating
the L2 source depth advance in the group-advance field, has no observable
effect: the uint16 output stayed bit-identical.  Those shape rules remain as
guards, but they are no longer claimed as the column fix.  Round 9's other
hard guard also remains: Common `0x34` is the executing PE pipeline form with
named enable and pool stage fields.  Substituting the convolution schedule
there hangs the pool.

Round 11 proves that Common `0x2c` sets the PE output piece width.  The normal
class emits 16-column fp16 pieces; setting the whole word to zero emits
one-column pieces and is much slower.  The result writer groups pieces in
32-column tiles.  It walks one piece across every row before the next piece,
and later tiles can overwrite the same physical slot.

The host model names TileDMA destination `0x17808` as the PE destination
piece-advance word.  On this PE path the hardware decodes half of the stored
word as bytes.  Round 10 put the ordinary 128-byte physical row there, so a
piece advanced 64 bytes even though one 16-column fp16 piece is only 32 bytes.
For width 28, the two pieces consequently land at byte bases 0 and 64, exactly
matching source columns 0 through 15 in place and source columns 16 through 27
at physical columns 32 through 43.  For width 56, tile zero's second piece and
tile one's first piece collide at byte 64; the latter wins, explaining correct
columns 32 through 47.  Tile one's tail lands at byte 128 and overwrites the
next row's 16-column slot: its first eight values are the preceding row's
columns 48 through 55 and its padded half clears columns 8 through 15.
The zero `0x2c` diagnostic emits extra one-column pieces into the second slot;
the last writers leave only column 32 and the preceding row's column 55.  The
unit model reproduces all three measured maps, including write ordering and
the 128-byte row wrap.

The shape rule programs twice the byte size of `min(width, 16)` fp16 columns
into destination `0x17808`.  For the 56-column pool that stored word is 64 and
decodes to the required 32-byte advance.  The model then places tile-zero
pieces at physical columns 0 and 16 and tile-one pieces at 32 and 48, covering
the complete logical row without touching its eight padding columns.  The
same rule is host-tested for 224- and 192-input ResNet-50 and the ResNet-18
shape.

Round 11's separate attempt to clear destination Cfg writer bit 26 is fully
reverted: it timed out with no finish lines and poisoned all subsequent jobs
until reboot.  The round-10 executing writer remains selected for both
external and scratch results.  The new sweep keeps that control and changes
one word per variant: every destination record word has its own candidate,
`0x2c` gets high-nibble classes 5 and 4 plus the low-nibble issue alternative,
and widths 28 and 16 remain explicit.  The best-guess and destination
piece-advance variants change only `0x17808`; the known hanging writer-bit
variant is last in `sweepOrder`.  Tests compare every variant byte-for-byte
with the control, retain the 504-byte KernelDMA grammar and guarded PE pipeline
word, and reject changes outside the named word.  No packaged byte is patched.

The full-stream timeout remains localized to index 1, class pool.  Once it is
correct, the next descriptor is index 2, the first-block projection class.
The ordinary one-block rung includes that class but does not terminate at it,
so the kit now adds a three-task `next-stall-projection` diagnostic with its
own `projection01` reference.  The original 21 ladder rungs and their
qualification order are unchanged; this diagnostic and the field sweep are
separate kit sections.

PE tasks have both kernel-table and kernel-execution enables clear and carry
no constants, but the loader/descriptor walk still requires the 504-byte H13
record grammar.  Round 9's 376-byte form was refused before execution.  The
encoder therefore emits the two leading reserved words, sixteen named disabled
coefficient-DMA configs, and the named disabled bias/post-scale/palette policy
groups.  No path enable is set and no coefficient range is supplied; the
records preserve required descriptor structure rather than enabling kernels.

The dependency-free numeric reference explicitly executes TensorFlow-SAME
max-pool with the odd padding cell after the surface. Its asymmetric test case
would produce a different first window under torchvision-style symmetric
padding, so the padding convention is part of the host gate. The synthetic
float64 residual tests execute the pool and one-block arithmetic topology;
they do not claim to simulate undocumented ANE arithmetic internals.

`--terminate-after <tensor>` makes a named stem, pool, fused block result,
mean, or logits tensor the external result of the linked stream. The encoder rebuilds
that terminal task through the same L2 result, destination DMA, destination
format, and header-event field functions as the full network. The package
manifest records every task's structural layer and role plus the named tensor
at each legal boundary.

`tools/h13-resnet50/kit.py` reads those manifest boundaries and recompiles each
rung; it never edits an ANEC. Its first three ladder rungs are the one-task
stem, a two-task `stem-pool-via-dram` diagnostic that forces a real scratch
spill, and the ordinary resident stem-to-pool pair. Every ladder, next-stall,
and sweep entry has its own package manifest, input, plain float64 NPZ, and a
libane channel-allocation check. `kit.json` keeps the 21 entries under `rungs`,
puts the projection diagnostic under `next_stall`, and lists the one-field
variants under `sweep` with a one-line description and stable sweep key.
