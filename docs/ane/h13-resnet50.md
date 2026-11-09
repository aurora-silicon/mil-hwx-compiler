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
channel/row/plane strides equal the pool source fields and that the terminal
pool DMA row/plane strides equal the package layout.
It additionally checks all sixteen stem kernel-DMA spans, the native sparse
kernel fields, the leading two-cell TensorFlow-SAME padding of the 7x7 stem,
and the exact external-source layout shared with the proven 3x3 linked-ring
stem.

Round 8 showed that both the scratch-fed and resident forms of task 1 produce
the same wrong pool columns.  A host field decode now compares that PE task
with the device-proven linked-ring depthwise 3x3/s2 task having the same
112-to-56 geometry.  Input/output dimensions, kernel and stride, traversal,
group counts, tile height, and L2 source channel/row/depth advances agree.
Round 9 incorrectly reused the convolution Common Cfg word at `0x34` as a PE
column schedule.  Device A/B established that the convolution form hangs a PE
pool; this register stays in the executing PE pipeline form, with named enable
and pool/binary/reduction stage fields.

The real PE column field is Common `0x2c`.  Its low nibble is the four-lane PE
issue class and its high nibble is `ceil(log2(active columns))`, as decoded by
the licensed public PE width families.  The old role-based constructor put
class 4 on every pool and add, limiting the 56-column pool to a 16-column
class; the shape rule puts class 6 there.  The measured under-class model
starts two 16-column pieces late and advances destinations two pieces at a
time: source piece two reaches columns 32 through 47, and source piece three
lands at physical column 64 and wraps into columns zero through seven of the
following 64-column physical row.  Columns 8 through 31 and 48 through 55 are
written as zero.  This reproduces the device dump.  With the complete class,
the model starts at piece zero, advances one piece, and covers every logical
column.  Pool/add derive the class from output width; mean derives it from the
input reduction width.

The same decode found that L2 SourceGroupStride was zero while the proven
112-to-56 task repeats SourceDepthStride there.  Resident and spill-fed PE
sources now construct both advances from the same plane span.  Tests compare
those named fields and the PE column class for the 224 and 192 ResNet-50
programs and ResNet-18.

The full-stream timeout remains localized to index 1, class pool.  Its
task-buffer link points to index 2, its nonterminal TBE is set, and its control
word requests both TSE and TSR.  The column-class and source-advance repair
does not change routing or the dependency chain.  The ladder already has a
one-block rung (first add), every later block prefix, reduce-mean, and full
linear output, so no additional task-class rung is needed.

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
rung; it never edits an ANEC. Its first three rungs are the one-task stem, a
two-task `stem-pool-via-dram` diagnostic that forces a real scratch spill, and
the ordinary resident stem-to-pool pair. Every rung has its own package
manifest, input, and plain float64 NPZ containing exactly its boundary array.
The kit rejects any rung that violates libane's channel-allocation gate.
