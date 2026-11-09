# MobileNetV1 224 on H14

Build the native-frame 28-layer graph used by `tools/h13-mobilenet`, with a
rule-built H14 `linked-ring`: native k3/s2 stem and depthwise layers, one early
scratch spill, then a 2 MiB circular L2 allocation. No Apple model, decoded
task prefix, or compiled bytes are bundled. Random fp16 weights are the
default; `--weights model.tflite` uses the existing converter for the public
Apache-2.0 quantized MobileNetV1 weights.

```sh
make build/mil-hwxc
python3 -m pip install numpy
MILC="$PWD/build/mil-hwxc" tools/h14-mobilenet/package.sh build/mobilenet-h14
```

The directory contains `src/model.mil`, generated weights, `params.npz`,
`tfweights.npz`, `linked/manifest.json`, and one 28-task H14 ANEC. Use
`ring_kit.py` for the smallest-first M2 ladder: depthwise stride-2 singles,
stem, prefixes, full ReLU, and full fused ReLU6.

After building the C runner below, execute a generated kit in its declared
smallest-first order with:

```sh
python3 out/h14-ring/run_kit.py out/h14-ring \
  --runner ./h14-mobilenet-run --device 0 --repeat 1
```

On the M2, build `run.c` against the existing MIT libane from
[aurora-silicon/omarchy-ane](https://github.com/aurora-silicon/omarchy-ane).
It needs the ABI-v2 `ane_m2_init_ports`, `ane_m2_send/read`, `ane_query_caps`
and `ane_exec` APIs plus the matching public driver headers. For example,
with an installed libane and its headers:

```sh
cc -O2 -Wall -Wextra -Werror tools/h14-mobilenet/run.c \
  -I/path/to/libane -I/path/to/uapi -I/usr/include/libdrm \
  -L/path/to/libane -lane -o h14-mobilenet-run
python3 tools/h14-mobilenet/run.py build/mobilenet-h14 \
  --runner ./h14-mobilenet-run --device 0 --reference
```

Use the controller's explicit M2 identity and approval policy when running
these commands on hardware. The worker does not run them. The C runner
checks ABI 2, prints the device's reported family, binds the exact input,
output and scratch sizes from the ANEC header, then calls libane. Libane
performs PROG_LOAD/PROC_CREATE/EXEC and releases its objects on exit.

The native k3/s2 forms are intentionally a linked-ring-only lowering. A
separately supplied `perop/manifest.json` can still be selected with
`--compare-perop`; the standard package and kit do not synthesize a different
standalone schedule for comparison.

The default is one execution with **zero warm-ups**. `--input image.npy`
accepts finite `[3,224,224]` floats in [-1,1]; otherwise a deterministic
random image is used. `--reference` compares the same graph in float64 with
fp16 boundaries. Start with cold prefix/per-layer comparisons before timing
repeated passes with `--repeat N`. A cold mismatch must not be hidden with
warm-ups. No runner changes firmware, clocks, protection services or boot
state.
