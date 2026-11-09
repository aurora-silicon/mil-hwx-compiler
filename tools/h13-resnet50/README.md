# H13 ResNet-50 linked-ring kit

Build the full rule-built program with:

```sh
python3 tools/h13-resnet50/kit.py /path/to/resnet50.mlmodelc out/resnet50-kit
```

The kit checks every rung against libane's bind-overrun rule before writing
`kit.json` and produces deterministic float64 references using TensorFlow SAME
padding. Each NPZ is plain (no object arrays) and contains exactly one of
`pool`, `block01` through `block16`, `mean`, or `logits`. Stage 1 remains
stride 1 (`[256,56,56]`); only later stages downsample.

Run the programs in `kit.json` order. Max-pool, one bottleneck, the remaining
block prefixes, global mean, and full logits are independently compiled with
`--terminate-after` using tensor names published by the full package manifest.
No program bytes are patched by the kit.
