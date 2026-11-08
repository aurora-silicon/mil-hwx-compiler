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

## Admission gate

Before a geometry becomes executable, independently verify every register
field used by the encoder, constant-section packing, relocation slot, external
surface layout, task/record count, and `h16g_validate.py` invariant. For
residual blocks, capture both projection and identity forms so the linker can
prove that the skip value remains live until the add. An isolated primitive
oracle does not establish a safe residual DAG schedule.
