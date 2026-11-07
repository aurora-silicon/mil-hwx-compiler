# Real-model coverage and ResNet CPU reference

`tools/mil_model_coverage.py` exports static, single-result operations from a
Core ML MIL protobuf package, or from a public Python MIL builder. It uses
coremltools 9.0's [MIL protobuf loader](https://github.com/apple/coremltools/blob/9.0/coremltools/converters/mil/frontend/milproto/load.py),
not the platform Core ML compiler or runtime. Exported floating tensors use
the same source-owned BLOBFILE representation as the MobileNet tools. String,
boolean, integer, and small scalar constants retain their types. Symbolic
dimensions, nested blocks, and tuple inputs are refused by the exporter.

Install `coremltools==9.0` and `numpy` in a virtual environment on a build
machine. After building `build/mil-hwxc`:

```sh
python tools/mil_model_coverage.py --compiler build/mil-hwxc \
  --package resnet50=models/resnet50.mlpackage \
  --package bert_block=models/bert_block.mlpackage \
  --builder whisper_tiny_enc=public_models.py:m_whisper_tiny_enc \
  --out build/real-model-coverage
```

The builder entry is a Python file and function supplied by the caller. It must
return a MIL Program. The exporter records the builder file digest, or digests
of the package protobuf and weight files, and the compiler executable digest.
Package weights are imported from the existing package rather than regenerated
from a nominal RNG seed. This matters when several builders share one RNG.

Each model gets `model.mil`, `graph.json`, `constants.npz`, and a weight blob.
The result includes `coverage.json` and `compiler-coverage.md`. Coverage runs:

- both targets, H16G and H17;
- whole-graph `per-op` and `linked` attempts;
- the default H16G staged CLI route as an additional, separately labeled set
  of whole-graph and isolated probes (without `--format` or `--schedule`);
- every nonconstant operation in isolation with its exact constants and
  boundary input types;
- conv+ReLU pairs where the conv has a single ReLU consumer;
- the independent `h16g_validate.py` check on every H16G object the probes emit.

The full operation table preserves source operation names, runtime shapes,
constant parameter shapes and small values, and each named error. Raw stderr
is saved alongside the probes. A refused probe must leave no partial output
directory. An accepted isolated op does not prove the original graph can be
scheduled. In particular, the H16G formula-convolution entry point refuses
nonconvolution operations even when the distinct staged compiler has a
primitive for some other geometry. H17 routes convolution-containing graphs
to that formula entry point; its isolated nonconvolution probes use its own
parity backend. These are route-specific results, not universal statements
about the silicon.

The H16G/H17 formula preflight now emits all independent operation and geometry
refusals before returning. It preserves the existing error codes, excludes a
single-consumer fused ReLU from redundant errors, and includes the output
name and shape. It still emits no package after any refusal and accepts no
additional hardware geometry.

## CPU fp16 chain

```sh
python tools/resnet_fp16_reference.py \
  --model build/real-model-coverage/resnet50 \
  --out build/resnet-refs --images 4 --seed 1234
MILC="$PWD/build/mil-hwxc" python tests/test_model_coverage.py
```

The reference evaluates the exported graph, retaining both live branches of
each residual block. Convolution, linear, and mean accumulate in float64;
every MIL operation writes fp16. SAME padding splits an odd total with the
extra element at the bottom/right. Max pooling pads with negative infinity.
The tests compare dense convolution, including 7x7/s2, against an independent
scalar loop and explicitly check pooling padding and an add-to-mean case
that changes if the intermediate fp16 rounding is omitted. Integration tests
check multi-error preflight on both targets and the absence of refused packages.

`input_NN.bin` uses the external NCHW surface with 64-byte-aligned rows.
`logits_NN.fp16` is a dense `[1,1000]` CPU result, **not a claimed ANE output
surface**. `top1.json` records the four random-input top-1 indices, and
`trace.json` records every operation boundary's shape, range, and SHA-256.
The first image is evaluated twice and must be bitwise deterministic.

These tools do not expand the encoder envelope. ResNet needs measured
convolution schedules and sparse grammar outside MobileNet's shapes, a
pooling encoder, residual addition and branching storage/lifetime support,
global pooling, and the 2048-to-1000 head before it can be a run package.
CPU references, an accepted export, and structural HWX validation each answer
different questions; none establishes numerical device correctness.
