# H13 ResNet-50 linked ring

`mil-hwxc --target H13 --schedule linked-ring` recognizes the exact
coremltools ResNet-50 topology (53 biased convolutions, max-pool, sixteen
residual additions, global mean, and a 2048-to-1000 classifier) and emits one
80-task program.

Every task field is constructed from operation geometry and its semantic role:
the bottleneck parser identifies main and projection branches, the split rule
partitions large 3x3 and projection kernels, the 2 MiB allocator keeps live
shortcuts and main-branch values disjoint, and the router derives the L2 and
TileDMA records for resident, spilled, and boundary edges. No per-network task
word table, HWX payload, or constant payload is stored in the repository.
Kernel constants are rebuilt from the licensed MIL blob by dense and H13
immediate-mask sparse rules. The result for the reference model is:

- task stream: 80 descriptors, 57,204 bytes including inter-task gaps;
- constants: 53,158,912 bytes;
- scratch channel 3: 1,605,632 bytes;
- input channel 5: `[1,3,224,224]`, 448-byte row, 100,352-byte plane;
- output channel 4: `[1,1,1,1000]`, 2,048-byte physical row.

The private parity gate compares every task byte and the entire constant
section against the compiler output made from the same MIL and blob. It is a
test oracle only and is not an input to the encoder. The public unit test
rebuilds the topology with deterministic finite fp16 values and checks
task/constant extents and bind-safe channel allocations.

`tools/h13-resnet50/kit.py` builds the full ANEC package, runs the same
bind-overrun rule as libane, and can generate deterministic float64 references
with TensorFlow SAME padding. It also emits conservative max-pool, one-block,
add+ReLU, global-mean, and block-prefix boundaries from the same rule-built
stream, truncating unused constants and rejecting the kit if any case violates
libane's channel-allocation gate.
