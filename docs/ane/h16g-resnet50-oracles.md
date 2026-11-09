# H16G ResNet-50 oracle campaign

Task: `t-5611-mil-hwx-compiler-h16g-resnet-50-end-to-end-7x7-s`

This campaign covers the exact public, random-weight ResNet-50 graph exported
by `tools/mil_model_coverage.py`: 23 unique dense biased convolution
geometries, both fused-ReLU states, 3x3/s2 SAME max-pool, residual add+ReLU at
all four stage sizes, the 7x7 global mean, and the 2048-to-1000 linear head.

`research/mint_h16g_resnet50_oracles.py` follows the repository's established
oracle boundary. Apple-compiled HWX exists only in a temporary directory and
is deleted before the script returns. The only durable output is decoded JSON:
source MIL, semantic parameters, hashes, task headers, register writes,
container metadata, and compiler provenance. Never commit an HWX payload or a
private framework binary.

The campaign currently records no ResNet field values. Existing MobileNet
schedule rows do not establish ResNet tile height, source skew, result mode,
output-channel splitting, 7x7 kernel packing, branch lifetime, max-pool, mean,
or the 1000-channel tail. Those values must come from a matching H16G oracle;
they must not be inferred from workload names or guessed from nearby shapes.

The public BSD-3 `freedomtan/coreml_to_ane_hwx` workflow independently records
that an H16 ResNet-50 stem uses input 224x224x3, output 112x112x64, kernel 7x7,
stride 2, and top/left padding 3. That corroborates the semantic geometry only;
it is not sufficient to add a compiler schedule row.

## Capture

On a controller-approved macOS reference source with the existing local oracle
tool:

```sh
python3 research/mint_h16g_resnet50_oracles.py --list
python3 research/mint_h16g_resnet50_oracles.py \
  --oracle-tool /path/to/ane-compile-hwx \
  --output /path/to/decoded/resnet50
```

The expected campaign size is 55: 46 isolated convolution cases, seven
isolated non-conv cases, and two complete bottleneck chains (projection and
identity). Move any binary-analysis notes to the tainted research area. Only
the reviewed decoded facts may be copied into the public repository.

## Full-model capture

The 2026-11-08 J616s capture contains the public random-weight full model, not
the 55 isolated cases. `research/decode_h16g_resnet50_full.py` decodes that
object in private and checks the exact operation/task association against the
public `graph.json` before emitting JSON. The executed cache entry contains
one program, 82 task descriptors, no task decode errors, a
`[1,3,224,224]` input, and a 1000-element output surface.

The stream separates into 53 convolution operations, one max-pool, 16 fused
residual add/ReLU tasks, one global mean, and one linear head. The compiler
schedules projection branches around the main branch and splits these wide
convolutions across adjacent tasks:

- `conv_25`: output-channel slices 64 + 176 + 16;
- `conv_27`: 224 + 672 + 128;
- `conv_29`, `conv_32`, `conv_35`, `conv_38`, and `conv_41`: 32 + 224;
- `conv_43`: 256 + 256.

The public source weights make kernel packing independently checkable. Dense
stride-one packing reproduces the captured sections, as does the H16G sparse
3x3/stride-two phase grammar for `conv_12`. The 7x7 stem uses sixteen
output-lane regions; each region carries the four outputs' biases and three
input-channel repetitions of sixteen phase-mask groups covering all 49 taps.

## Rule-built residual encoder

The public encoder contains no full-network descriptor table. It discovers
basic blocks (`3x3,3x3`) and bottleneck blocks (`1x1,3x3,1x1`) by tensor
compatibility, identifies projection branches independently of operation
names, and preserves the graph's convolution order. Output slicing is chosen
from convolution geometry, so the same constructors cover the wide H16G
tasks and a synthetic ResNet-18 graph.

Each task header and register record is constructed from its input/output
shape, kernel, stride, channel slice, operation role, and storage plan.
Header live-bank masks are computed from the 64-KiB banks occupied in a 2-MiB
L2 arena. A first-fit allocator keeps the active main-branch tensors apart;
residual boundaries use the scratch BAR with explicit TileDMA offsets. Kernel
and surface BAR slots follow the object writer's allocation rather than a
captured program.

Max-pool, add+ReLU, global mean, and the linear head have dedicated packet
constructors. The linear dimensions come from the final block and classifier
bias rather than the ResNet-50 name. ResNet-50 uses a 4-MiB scratch allocation
for two residual spill regions; smaller compatible networks use the same
policy without adding tables.

All constants are formula-built from public model weights. Dense task slices
use one interleave width per compiler slice; stride-two 3x3 planes use padded
phase slots and 16-bit nonzero masks with a 24-bit body length; stride-two
1x1 planes use the alternating phase grammar exposed by the wide projection.
The 1000-output head distributes its tail as 63 lanes in the first eight
regions and 62 in the last eight. Together these formulas reproduce every
convolution and head region byte-for-byte.

Packaged objects follow the object-writer BAR law: input slot 6, output slot
7, `__KERN_0` slot 8, and scratch slot 3. The offline validator bounds every
external DMA extent and rejects unbound or misdirected apertures.

## Admission gate

Before a geometry becomes executable, independently verify every register
field used by the encoder, constant-section packing, relocation slot, external
surface layout, task/record count, and `h16g_validate.py` invariant. For
residual blocks, capture both projection and identity forms so the linker can
prove that the skip value remains live until the add. An isolated primitive
oracle does not establish a safe residual DAG schedule.
