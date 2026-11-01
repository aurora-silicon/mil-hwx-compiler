# H13 linked chains and MobileNetV1

Tools for `--schedule linked` (docs/ane/linked-programs.md). Host side needs
numpy; the device side needs numpy and libane from omarchy-ane.

On the host:

```sh
# any conv chain: C,H,W then cout,k,stride,groups,bias,relu per layer
MILC=build/mil-hwxc python3 tools/h13-mobilenet/mkchain.py cases/c2 64,16,16 '64,1,1,1,1,1;64,1,1,1,1,1'
# MobileNetV1 224 (random weights, or --weights mobilenet_v1_1_224_quant.tflite)
MILC=build/mil-hwxc python3 tools/h13-mobilenet/mobilenet.py cases/mn
python3 tools/h13-mobilenet/mn_refs.py cases/mn 16     # test images + float64 references
python3 tools/h13-mobilenet/prefixes.py cases/mn cases/mn-prefix   # per-layer profile
```

On the M1 (user in the `accel` group):

```sh
gcc -O2 -I$LIBANE -o tools/h13-mobilenet/anechain tools/h13-mobilenet/anechain.c $LIBANE/libane.a -ldrm
python3 tools/h13-mobilenet/run_chain.py cases/mn 1000     # linked vs per-layer, timing, bit agreement
python3 tools/h13-mobilenet/mn_eval.py cases/mn labels.txt # logits vs the CPU reference
tools/h13-mobilenet/run_prefixes.sh cases/mn-prefix 50     # busy time per layer inside the stream
```

`anechain` runs one program, or several with a host copy of each output into
the next input, and prints p50/p99 wall time, inf/s and engine busy time from
`ane_stats`. `-n` times `ane_exec` alone.

`tflite_weights.py` reads the weights out of a TFLite flatbuffer without
TensorFlow. The quantized MobileNetV1 (Apache-2.0) that Mesa's teflon tests
ship dequantizes to float weights with batch norm folded. The ANE network
uses ReLU where the original uses ReLU6, so accuracy is measured against a
float64 CPU reference of the same ReLU network.

For a reproducible round-3 handoff, download the public quantized weights
from the [TensorFlow MobileNetV1 catalog](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)
([archive](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)).
Extract `mobilenet_v1_1.0_224_quant.tflite`; its SHA-256 must be
`ecc3a67c47c5a609ec35f6a58a7d97532834e43df4cb7d3f1204a8164b7d20dd`.
The generated package contains Apache-2.0 attribution, 28 per-layer programs,
the linked program, 16 procedural inputs and float64 references, source and
weight hashes, and `host-stats.json`:

```sh
python3 tools/h13-mobilenet/prepare_l2.py build/mn-l2-r3 \
  --weights mobilenet_v1_1.0_224_quant.tflite --compiler build/mil-hwxc
python3 tools/h13-mobilenet/l2_stats.py build/mn-l2-r3/linked
```

`prepare_l2.py` needs NumPy and a fresh output directory. Byte predictions sum
enabled source/destination TileDMA extents and per-task kernel-DMA lengths;
they include repeated reads by split tasks, exclude cache effects and DRAM
transaction overhead, and are not a latency prediction.

The ANE lead runs the following through the hardware controller on the
explicitly selected M1 Pro, with the device's normal protection and ownership
policy. From the extracted package directory (`$LIBANE` is the public libane
build directory):

```sh
gcc -O2 -I"$LIBANE" -o tools/h13-mobilenet/anechain \
  tools/h13-mobilenet/anechain.c "$LIBANE/libane.a" -ldrm
python3 tools/h13-mobilenet/run_l2.py . --passes 5000 --output timing-r3
```

The runner first checks `probes/wide128`, exercising the new terminal descriptor form before the whole
network. It reuses real pointwise/depthwise weights in an artificial triplet;
it is a transport probe, not a MobileNet accuracy result. Its six cold
comparisons must pass before the runner continues to the whole network.

The runner requires cold exact equality to the per-layer programs and top-1
16/16, then invokes `anechain -w 20 -r 5000 -n` with the last reference input
for exec-only timing against 1641 us. It preserves commands, stdout/stderr,
outputs, cold results and before/after driver counters in a fresh directory.
Any submission failure exits nonzero. The exact bounded 56x56 c128
terminal chain is checked on the host. The actual current-main and branch
manifests have six DRAM edges (21/27 resident), versus the four/23 count in
the earlier summary. The supplied facts do not prove spatial handoffs for
the oversized 112x112 tensors, the nonterminal wide 56x56 form, or a dependent
edge from the split pointwise producer into the split dense classifier.
