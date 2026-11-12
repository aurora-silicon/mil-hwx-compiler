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
stem/pool/projection diagnostic. `diagnostics` adds a focused copy of the
first 7x7 prefix and a thirteen-task standalone form of that block; the latter
uses the float64 block-13 result as its input, so it does not repeat the first
64 descriptors.

`subRungs` splits the first bottleneck into standalone `conv1`,
`conv1+conv2`, `conv1+conv2+conv3`, projection, and fused residual-add
programs. The convolution and projection programs use the float64 pool result
as input; add uses separate float64 main3 and projection inputs. Every entry
requests three cold runs compared as raw uint16 arrays before the relL2 check.
The generated `verify_cold.py` performs both checks from three output NPYs.
The task manifests can be audited offline with
`tools/h13-resnet50/task_graph.py`.

`sweep` starts from the measured round-12 control: Common `0x2c` low-nibble 4
emits 16-column pieces, destination `+0x08` is the 128-byte row stride, and
destination `+0x0c` is the plane stride. It tries low-nibble widths 5, 6, and
7 before the chosen low-nibble 8 (one 32-column/64-byte piece), then the
coherent 256-byte-row alternative. Width-28 control and fixed variants are
last because the round-12 width-28 timed loop hung after a successful cold
run. Every entry is compiler-built and bind-checked; the kit never patches an
ANEC.
