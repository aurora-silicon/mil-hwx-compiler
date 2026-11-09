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

For the reference 224 model the result has 80 descriptors, 53,158,912 bytes
of rebuilt constants, and a 12,058,624-byte spill arena. Channel 5 is the
`[1,3,224,224]` input with a 448-byte row and 100,352-byte plane. Channel 4 is
the `[1,1,1,1000]` output with a 2,000-byte physical row.

The optional private parity gate reports per-task field-difference counts and
the constant byte-difference count against the compiler output made from the
same MIL and blob. It is a report, never an encoder input or a public table.
Allocator-chosen L2 bases are allowed to differ. The public unit test rebuilds
224 and 192 bottleneck networks and a basic-block network with deterministic
finite fp16 values; all three generated ANECs pass the libane extent check.

`tools/h13-resnet50/kit.py` builds the full package, runs the same bind-overrun
rule as libane, and can generate deterministic float64 references with
TensorFlow SAME padding. It discovers max-pool, add+ReLU, global-mean, and
block-prefix boundaries from the emitted tasks, then rejects the kit if any
case violates the channel-allocation gate.
