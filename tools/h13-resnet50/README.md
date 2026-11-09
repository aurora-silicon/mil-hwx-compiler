# H13 ResNet-50 linked-ring kit

Build the full rule-built program with:

```sh
python3 tools/h13-resnet50/kit.py /path/to/resnet50.mlmodelc out/resnet50-kit
```

The kit checks libane's bind-overrun rule before writing `kit.json` and
produces a deterministic float64 reference using TensorFlow SAME padding.
The reference records the max-pool output and all sixteen block boundaries;
stage 1 remains stride 1 (`[256,56,56]`) and only later stage entries use
stride 2.
Run the full program only after the qualification ladder in `kit.json` has
passed smallest-first. The full and linear rungs retain the native rule-built
program. Max-pool, one-block, fused add+ReLU, global-mean, and all discovered
block-prefix rungs terminate the same rule-built stream with a conservative
external TileDMA boundary; every emitted ANEC is checked against libane's
surface-extent rule before it enters the kit.
