# H13 linked chains and MobileNetV1

Tools for `--schedule linked` (docs/ane/linked-programs.md). Host side needs
numpy; the device side needs numpy and libane from omarchy-ane.

On the host:

```sh
# any conv chain: C,H,W then cout,k,stride,groups,bias,relu per layer
MILC=build/mil-hwxc python3 tools/h13-mobilenet/mkchain.py cases/c2 64,16,16 '64,1,1,1,1,1;64,1,1,1,1,1'
# MobileNetV1 224 with fused ReLU6 (random weights, or --weights mobilenet_v1_1_224_quant.tflite)
MILC=build/mil-hwxc python3 tools/h13-mobilenet/mobilenet.py cases/mn
MILC=build/mil-hwxc python3 tools/h13-mobilenet/mobilenet.py cases/mn-native --frame native --no-perop
python3 tools/h13-mobilenet/mn_refs.py cases/mn 16     # test images + float64 references
python3 tools/h13-mobilenet/prefixes.py cases/mn cases/mn-prefix   # per-layer profile
```

Pass `--relu` only for the legacy unbounded-ReLU comparison network.

On the M1 (user in the `accel` group):

```sh
gcc -O2 -I$LIBANE -o tools/h13-mobilenet/anechain tools/h13-mobilenet/anechain.c $LIBANE/libane.a -ldrm
python3 tools/h13-mobilenet/run_chain.py cases/mn 1000     # linked vs per-layer, timing, bit agreement
python3 tools/h13-mobilenet/mn_eval.py cases/mn labels.txt # logits vs the CPU reference
tools/h13-mobilenet/run_prefixes.sh cases/mn-prefix 50     # busy time per layer inside the stream
```

`--frame native` keeps the TensorFlow frame and native stride-2 layers.  It
selects H13 `linked-ring` by default.  `ring_kit.py` packages the native stem,
four stride-2 depthwise rungs, p01..p28, the full network, and float64 refs.

`anechain` runs one program, or several with a host copy of each output into
the next input, and prints p50/p99 wall time, inf/s and engine busy time from
`ane_stats`. `-n` times `ane_exec` alone.

`tflite_weights.py` reads the weights out of a TFLite flatbuffer without
TensorFlow. The quantized MobileNetV1 (Apache-2.0) that Mesa's teflon tests
ship dequantizes to float weights with batch norm folded. The ANE network
uses ReLU where the original uses ReLU6, so accuracy is measured against a
float64 CPU reference of the same ReLU network.
