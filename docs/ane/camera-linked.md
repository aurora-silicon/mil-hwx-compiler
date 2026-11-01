# Linked camera packages

YuNet has a source-built one-program, 113-task package with package-MIL
simulation passing on all 30 audited portraits. Build it with:

```sh
python3 tools/yunet_linked.py "$CAMERA_MODELS/yunet" --compiler build/mil-hwxc --output "$OUTPUT/yunet"
```

The lowering uses a phase-layout stem, nonnegative convolution max trees,
channel padding, residual sums, native nearest resize and a compact readback.
Image layout and score/head decode remain outside the program and count in
device frame timing. Placement-only concat candidates emit no task only when
the allocator proves an exclusive adjacent producer tree; otherwise they are
refused. No unsupported copy kernel or CPU partition is substituted.

The paired effects branch ships the audited package and host receipt under
`examples/camera-effects/packages/`. It still requires lead device
qualification; no ANE timing or speedup is claimed. Selfie remains on CPU.

`tools/camera_linked.py` attempts the audited camera-effects MIL as **one
H13 program and one dispatch**. It writes `camera-linked.json` only after
the existing ANEC reader accepts the compiler output and the boundary
contract matches the original model. It never substitutes island programs
or host partitions. Both unsuccessful and successful attempts get a
`build.json` and compiler log; each attempt requires a new output directory.

```sh
python3 tools/camera_linked.py "$CAMERA_MODELS/selfie" --output "$OUTPUT/selfie"
python3 tools/camera_linked.py "$CAMERA_MODELS/yunet" --output "$OUTPUT/yunet"
```

The native-layer branch is merged with main's H13 L2 r2 and H14 formula
linking. Native layers keep DRAM edges: the r2 planner does not interpret
their internal tasks as generic convolution geometry. H14 routes generic
convolution chains to the formula linker, and native graphs to the native
linker. The ANEC header and channel contracts are unchanged.

## Original-export refusals and remaining selfie work

The original exports do not compile directly as linked camera graphs.
The original exports give these inventories (constants omitted):

| Model | Operations | First linked refusal |
| --- | --- | --- |
| MediaPipe general selfie | 54 conv, 12 pad, 25 add, 11 clip, 32 mul, 22 relu, 10 avg_pool, 11 sigmoid, 3 upsample_bilinear, 1 conv_transpose | `pad` before the stem |
| YuNet 320 | 53 conv, 15 relu, 4 max_pool, 2 upsample_nearest_neighbor, 2 add, 12 transpose, 12 reshape, 6 sigmoid | twelve boundary results |

Removing the first diagnostic alone is insufficient. The linked frontend
only accepts its qualified conv/native/concat envelope; the exact decoded
native activation and convolution grids are also narrower than these models.
Selfie's squeeze/excitation needs reductions and runtime broadcast multiply;
its exported hard-swish uses a different staged arithmetic expression from
the canonical form recognized by the native fuser. YuNet needs pooling,
residual branches, a multi-head readback contract, and a qualified stem
lowering. The 35x35 camera schedules fix the earlier island rewrite, not
these whole-graph requirements.

The YuNet lowering above resolves its original-export gaps without host NN
partitions. Selfie still requires additional qualified shape/operation support
and numerical work. A host experiment rounding the original selfie arithmetic
to fp16 boundaries passes 28/30 portraits, failing IoU on two, including the
previously sensitive portrait. This diagnostic is not a linked-package parity
result. The gate stays at 0.98 and blur stays on CPU. An unsuccessful build
directory must never be installed as a candidate package.

## Runtime and qualification contract

The effects backend in `aurora-silicon/omarchy-ane` accepts packages under
`models/MODEL/compiled/H13/linked`. `camera-linked.json` hashes the manifest,
ANEC, MIL and weights, records original model identities, and binds input
and output shapes/names. The simulator evaluates the package MIL/weights
with fp16 surface boundaries; it does not read the reference ONNX graph.
It is a numerical model, not a task-descriptor emulator.

`linked/qualify.py` there compares against the unchanged ONNX CPU model on
the audited 30 portraits, then measures one resident handle against CPU.
Device qualification is separate from host simulation and must run through
the controller. A passing device receipt is required before normal native
loading. See that repository's `examples/camera-effects/linked/README.md`.

Generated weights/programs remain install-time artifacts under their model
licences. See [PUBLIC-SOURCES.md](../../PUBLIC-SOURCES.md). Procedural runtime
tests establish ownership and refusal behavior, not camera model parity.
