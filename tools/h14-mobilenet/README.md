# MobileNetV1 224 on H14

Build the same mirrored, space-to-depth stem and 28-layer graph used by
`tools/h13-mobilenet`, with H14 formula-built convolutions and one linked
program. No Apple model or compiled bytes are bundled. Random fp16 weights
are the default; `--weights model.tflite` uses the existing H13 converter
for the public Apache-2.0 quantized MobileNetV1 weights. Its ReLU replacement
for ReLU6 and the host's final average pool are unchanged.

```sh
make build/mil-hwxc
python3 -m pip install numpy
MILC="$PWD/build/mil-hwxc" tools/h14-mobilenet/package.sh build/mobilenet-h14
```

The directory contains `src/model.mil`, original generated weights,
`params.npz`, `tfweights.npz`, `linked/manifest.json`, one H14 ANEC and, unless
`--no-perop` is supplied, per-layer programs for cold comparisons. This is a
host-built package; M2 accuracy and timing still need qualification.

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
  --runner ./h14-mobilenet-run --device 0 --compare-perop --reference
```

Use the controller's explicit M2 identity and approval policy when running
these commands on hardware. The worker does not run them. The C runner
checks ABI 2, prints the device's reported family, binds the exact input,
output and scratch sizes from the ANEC header, then calls libane. Libane
performs PROG_LOAD/PROC_CREATE/EXEC and releases its objects on exit.

`--compare-perop` also runs the 28 per-layer programs once each, using the
explicit ABI-v2 ports, and fails if their logical fp16 output differs bit
for bit from the linked output. Padding bytes are excluded.

The default is one execution with **zero warm-ups**. `--input image.npy`
accepts finite `[3,224,224]` floats in [-1,1]; otherwise a deterministic
random image is used. `--reference` compares the same graph in float64 with
fp16 boundaries. Start with cold prefix/per-layer comparisons before timing
repeated passes with `--repeat N`. A cold mismatch must not be hidden with
warm-ups. No runner changes firmware, clocks, protection services or boot
state.
