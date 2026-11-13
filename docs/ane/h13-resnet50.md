# H13 residual linked ring

`mil-hwxc --target H13 --schedule linked-ring` recognizes a residual graph
with a convolution stem, max-pool, basic (3x3,3x3) or bottleneck
(1x1,3x3,1x1) blocks, global mean, and a linear classifier. It emits one
program. The public 224 ResNet-50 is 103 tasks; 192-input ResNet-50 and a
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

For the reference 224 model the result has 103 descriptors and a
12,058,624-byte spill arena. Channel 5 is the
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
pool row/plane/depth fields and the Common PE piece-width field follow their
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

Rounds 11 and 12 pin down Common `0x2c` as the PE output piece-width control.
Its low nibble is the width in groups of four columns: low 4 emits 16-column
fp16 pieces and low 2 emits eight-column pieces. Changing the high nibble to
classes 4 or 5 left every output bit-identical. Destination `0x17808` is not a
piece advance: it is the row stride. Setting it to 64 packed rows at `64*y`
and left physical rows 29 through 55 unwritten. Destination `0x1780c` is the
plane stride; setting it to 3584 moved channel `c` to byte `3584*c`.

For a piece of `P` fp16 columns, piece `k` starts at
`roundup(2*P*k, 64)` bytes. With the measured 16-column pieces, width 28 puts
columns 16 through 27 at byte 64. Width 56 puts pieces 1 and 2 at the same
byte 64, so piece 2 overwrites piece 1, and puts the final piece at byte 128
on top of the next row. The host model walks every row of a piece before the
next piece and reproduces the control, 64-byte-row, width-28, low-2, and
3584-byte-plane maps. It also proves that a 256-byte row cannot be the fix:
it prevents the final piece from wrapping into the next row but pieces 1 and
2 still collide at byte 64.

The selected shape rule keeps destination `0x17808` at the physical row stride
and chooses one power-of-two piece covering up to 32 columns. Widths
56/28/14/7 therefore use 32/32/16/8-column pieces (low nibbles 8/8/4/2).
At width 56 the two 64-byte pieces land at byte 0 and byte 64, covering the
complete 128-byte physical row without a collision. The same rule covers PE
add boundaries, whose old unwritten slots retained non-finite BO contents;
that explains why the one-block boundary was merely wrong while block-prefix
02 through 13 reported non-finite external results. The passing three-task
projection at relL2 3.0e-4 independently shows that resident pool arithmetic
and its first consumer were already sound.

The narrow sweep retains the executing writer bit and the 128-byte-row
control, tries 20-, 24-, and 28-column pieces, the selected 32-column piece,
and the coherent 256-byte-row alternative. It includes both control and fixed
width-28 programs, ordered last because the round-12 width-28 cold run
completed but its timed loop hung. Tests restrict every variant to its named
Common or destination-layout words, retain the 504-byte KernelDMA grammar and
guarded PE pipeline word, and reject package byte patching.

Round 12's first q4 hang is the first 7-wide block. In one-based task order,
65 is a 1x1 stride-2 projection from 1024x14x14 to 2048x7x7, 66 and 67 are
the two 256-output 1x1 main slices in the low L2 band (the latter writes base
zero), 68 is the 512-output 3x3 stride-2 task, 69 is the 1x1 expansion to
2048, and 70 is the PE add. The projection, 3x3 task, and expansion were the
first residual descriptors to select OCG 5 (32 interleaved lanes). The passing
`next-stall-projection` is a 64-to-256 1x1 task at width 56 and stays in the
OCG-4 envelope. Round 13 consequently keeps stride-2 packing at sixteen lanes
and slices the 7-wide dense expansion into eight 256-output OCG-4 tasks. The
focused prefix is 77 tasks and the standalone first 7-wide block is 13 tasks;
both are separate kit diagnostics, while the original 21-rung qualification
order remains unchanged.

PE tasks have both kernel-table and kernel-execution enables clear and carry
no constants, but the loader/descriptor walk still requires the 504-byte H13
record grammar.  Round 9's 376-byte form was refused before execution.  The
encoder therefore emits the two leading reserved words, sixteen named disabled
coefficient-DMA configs, and the named disabled bias/post-scale/palette policy
groups.  No path enable is set and no coefficient range is supplied; the
records preserve required descriptor structure rather than enabling kernels.

Round 14 audits the first block's complete access graph in
`h13-resnet50-round14.md`. Scratch consumers now use the decoded
TSE+TSR/DependencyMode-7 handoff on their first slice and the mode-3 reread
form on later slices. Per-task L2 staging allocations are cumulative, so a
scratch source, result, and second source cannot reuse the same staging tile.
The package manifest exposes each task's surface and L2 ranges for the offline
audit tool.

Round 15 separates external surface geometry from the L2 bank phase.  The
standalone 64x56 1x1 task had correct 112-byte NCHW source rows but discarded
the consumer layout's 80-byte L2 bank rotation.  Its deterministic 32-pixel
row walk reproduces the device dump offline; retaining the rotation restores
the 56-pixel walk.  The native stem remains the proven unskewed exception.
Values consumed by a PE add use the PE's 32-byte bank rotation on both branch
layouts instead of the convolution rotation.  The decoded field table and
offline model are in `h13-resnet50-round15.md` and
`tools/h13-resnet50/round15.py`.

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
focused diagnostic, and sweep entry has its own package manifest, input, plain
float64 NPZ, and a libane channel-allocation check. `kit.json` keeps the 21
entries under `rungs`, puts the projection diagnostic under `next_stall`, the
focused prefix and standalone small-spatial block under `diagnostics`, and the
piece/layout variants under `sweep` with stable keys.
