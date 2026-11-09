# H13 ResNet-50 linked-ring kit

Build the full, byte-exact program with:

```sh
python3 tools/h13-resnet50/kit.py /path/to/resnet50.mlmodelc out/resnet50-kit
```

The kit checks libane's bind-overrun rule before writing `kit.json` and
produces a deterministic float64 reference using TensorFlow SAME padding.
Run the full program only after the qualification ladder in `kit.json` has
safe byte-exact boundary forms. The compiler intentionally does not invent
standalone or prefix DMA descriptors: an approximate boundary task can wedge
the engine even when its arithmetic fields look correct.
