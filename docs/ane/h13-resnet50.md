# H13 ResNet-50 linked ring

`mil-hwxc --target H13 --schedule linked-ring` recognizes the exact
coremltools ResNet-50 topology (53 biased convolutions, max-pool, sixteen
residual additions, global mean, and a 2048-to-1000 classifier) and emits one
80-task program.

The task fields come from sanitized decoded register facts. No HWX payload or
constant bytes are stored in the repository. Kernel constants are rebuilt
from the MIL blob by rules for dense packing, H13 immediate-mask sparse
stride-2 packing, Apple’s reversed wide-kernel slice order, and the linear
head. The result for the reference model is:

- task stream: 80 descriptors, 57,204 bytes including inter-task gaps;
- constants: 53,158,912 bytes;
- scratch channel 3: 1,605,632 bytes;
- input channel 5: `[1,3,224,224]`, 448-byte row, 100,352-byte plane;
- output channel 4: `[1,1,1,1000]`, 2,048-byte physical row.

The private parity gate compares every task byte and the entire constant
section against the compiler output made from the same MIL and blob. The
public unit test rebuilds the topology with deterministic finite fp16 values
and checks task/constant extents and bind-safe channel allocations.

`tools/h13-resnet50/kit.py` builds the full ANEC package, runs the same
bind-overrun rule as libane, and can generate deterministic float64 references
with TensorFlow SAME padding. It deliberately does not synthesize standalone
or prefix boundary DMA forms: those cases must be added only when their task
bytes are independently exact.
