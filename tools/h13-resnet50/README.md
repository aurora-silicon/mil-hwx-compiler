# H13 ResNet-50 linked-ring kit

Build the full rule-built program with:

```sh
python3 tools/h13-resnet50/kit.py /path/to/resnet50.mlmodelc out/resnet50-kit
```

The kit starts with a one-task `stem` rung, then a two-task
`stem-pool-via-dram` diagnostic whose pool reads a real scratch spill instead
of the L2 ring, followed by the ordinary resident `maxpool` rung.  This makes
stem arithmetic, DRAM layout, and the resident handoff independently visible.
It checks every rung against libane's bind-overrun rule before writing
`kit.json` and produces deterministic float64 references using TensorFlow SAME
padding. Each NPZ is plain (no object arrays) and contains exactly one of
`stem`, `pool`, `projection01`, `block01` through `block16`, `mean`, or
`logits`. Stage 1 remains
stride 1 (`[256,56,56]`); only later stages downsample.

Run the programs in `kit.json` order. Max-pool, one bottleneck, the remaining
block prefixes, global mean, and full logits are independently compiled with
`--terminate-after` using tensor names published by the full package manifest.
The DRAM diagnostic uses the compiler's `pool-via-dram` termination mode. No
program bytes are patched by the kit.

The `rungs` and `qualificationOrder` arrays remain the original 21-rung
smallest-first ladder. `next_stall` is a separate three-task
stem/pool/projection diagnostic for the descriptor expected immediately after
the pool is repaired. `sweep` contains a round-10 control and compiler-built
single-word variants of the scratch-fed pool. It isolates every destination
record word, the predicted destination piece advance, and three Common
`0x2c` nibble alternatives. Each entry names the changed word and describes it
in one line; `sweepOrder` places the known hanging writer-bit experiment last.
The 28- and 16-column variants change only the pool's Common output-width word
while retaining the ordinary 56-column output allocation so the moving piece
window remains visible in a full-width dump.
