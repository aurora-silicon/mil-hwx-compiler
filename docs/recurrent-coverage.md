# Recurrent results, bounded regions and DFN3 coverage

H14 packages now use `mil-hwxc.h14-anec-package.v2`. As in H13 v2,
`physicalOutputs` names distinct produced tensors, and `logicalResults`
retains function-return order and duplicates with explicit element ranges.
A value such as `c_next` can be both returned and consumed by a later `tanh`
to calculate `h_next`. It stays an output allocation rather than becoming a
scratch tensor. Input passthrough and constant returns are refused. This is
package metadata; it adds no multi-destination DMA instruction or state commit.
Consumers must understand v2 and expose all results only after the complete
invocation succeeds. Runtime transaction/retirement is owned by shared libane.
H17 and H18 retain their existing single-result and byte-parity contracts.

## Decoded regions

`--schedule regions` is opt-in on H13 and H14. It looks for one `add`, `mul` or
constant-weight `matmul` whose result has exactly one use, a shape-preserving
`relu`. The target must admit the existing decoded ReLU epilogue. A returned
producer, another reader, an alias read, unsupported geometry or another
activation prevents fusion. The remaining graph keeps normal per-operation
bindings and dispatches. Every fused pair is recorded in `regions`, naming its
program, erased producer value and surviving ReLU output.

The emitted ANEC of an admitted pair equals the existing isolated `chain`
route. The H13 preflight checks region metadata and the actual epilogue word.
This does not join arbitrary task streams or invent SRAM/DMA routing.
Sigmoid/tanh and general recurrent bridges remain outside the decoded fusion
contract. A ReLU-candidate GRU can save one of 24 dispatches; a tanh GRU and an
LSTM keep 24 and 25 respectively. Native arithmetic and complete-hop timing
still need device qualification.

## Recurrent export and references

`tools/recurrent_lowering.py` reuses the MIT single-step exporter from
[capture-ns](https://github.com/aurora-silicon/capture-ns/blob/f843fd3d053f716222caa7286a2095a9ffb8e762/tools/recurrent.py).
Its [MIT notice](licenses/capture-ns-MIT.txt) also covers the adapted recurrent
reference tests. GRUs distinguish reset-before and reset-after; RNNoise uses a
ReLU candidate. LSTM uses ONNX i,o,f,c ordering and returns hidden then cell
state. State inputs/outputs are explicit. The 512-lane variant zero-pads every
gate separately and supplies bias and +/-1 vectors as fixed model feeds.
Consumers must bind those exact vectors and zero invalid input/state lanes.

The tests compare unpadded recurrent equations against independent ONNX Runtime
operators for 100 steps, compare padded and unpadded state including zero tails,
and compile padded cells for both generations, schedules and artifact formats.
This proves lowering/packaging, not full-model fp16 numerical equivalence.
The CPU MIL reference additionally evaluates nonnegative custom convolution
padding, with independent ONNX comparison. Hardware envelopes are unchanged.

```sh
python3 -m venv .venv
.venv/bin/pip install --only-binary=:all: -r tools/coverage-requirements.txt
make test-recurrent-regions
PATH="$PWD/.venv/bin:$PATH" make test-recurrent-reference
```

## Every-node DFN3 audit

The source is DeepFilterNet revision
`d375b2d8309e0935d165700c91da9de862a99c31`, selecting its MIT licence.
[Its model archive](https://github.com/Rikorose/DeepFilterNet/blob/d375b2d8309e0935d165700c91da9de862a99c31/models/DeepFilterNet3_onnx.tar.gz)
is hash-pinned to
`c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616`.
The [repository licence](https://github.com/Rikorose/DeepFilterNet/blob/d375b2d8309e0935d165700c91da9de862a99c31/LICENSE)
is MIT OR Apache-2.0; no separate model restriction was found in the prior
capture-ns audit. No weights, Apple programs, firmware or audio are included.

```sh
.venv/bin/python tools/dfn3_coverage.py --df-model DeepFilterNet3_onnx.tar.gz \
  --compiler build/mil-hwxc --frames 1 --output build/dfn3-coverage
```

The audit accounts for every node in encoder, ERB decoder and DF decoder,
records graph/archive/compiler hashes, attributes, shapes and constant
provenance, and preserves compiler diagnostics. A deterministic CPU ONNX sample
observes all intermediate shapes for the selected `S` frame count. Observed
runtime values never become model constants. Only initializers, Constant,
shape-specialized Shape and their static descendants are treated as
compile-time values. This is shape specialization, not a streaming proof.

Supported primitive spellings are probed with exact observed shapes and
fp16-rounded source constants, for H13/H14 and per-op/regions scheduling.
GRUs are probed as separately padded single-step cells with explicit state.
Unimplemented ONNX lowering is named per node, including grouped Einsum,
ConvTranspose and temporal/data movement. Parser/blob/tool errors are recorded
as errors, not target coverage refusals, and make the audit fail.

A successful cell or primitive probe does not establish graph-edge routing,
sequence loops, temporal caches, graph import, DSP or a complete model package.
`--prepare-only` replaces `--compiler` when only shape/provenance and MIL
export checks are wanted. It records `compiler_probes_run: false`, a null
compiler hash and `fragment-exported` rather than `fragment-probed`. The
[preparation receipt](dfn3-preparation.json) is this mode at one frame. It
accounts for all 375 nodes: 263 compile-time values, 62 exported probes and
50 nodes requiring additional lowering. It contains no compilation result.

The receipt explicitly records `complete_native_graph`,
`streaming_state_complete` and `full_model_package_validated` as false.

## The 10 ms hop budget

```sh
python3 tools/graph_budget.py --package build/cell-a --package build/cell-b \
  --overhead-ms 0.25 --budget-ms 10 --require-fit
```

The bound counts dispatched programs rather than task descriptors. It needs an
explicit overhead assumption and optional compute/DSP reserve. Equal-to-budget
fails: there is no remaining time for other work. Missing complete-model,
streaming-state and package-validation evidence prevents admission even when
the dispatch lower bound fits. No estimate sets `device_budget_proven` true.

Historical 0.04/0.25 ms per-call assumptions imply the following dispatch-only
bounds. They are arithmetic scenarios, not measurements of this change or of a
particular reference Mac:

| Region set | Dispatches | At 0.04 ms | At 0.25 ms |
| --- | ---: | ---: | ---: |
| Three unfused GRUs | 72 | 2.88 ms | 18 ms |
| Three ReLU-candidate GRUs with bounded regions | 69 | 2.76 ms | 17.25 ms |
| Five DFN3 tanh GRUs | 120 | 4.8 ms | 30 ms |

The remaining work needs a proved general recurrent routing/epilogue contract,
full DFN3 movement and temporal-state lowering, shared-runner transactions, and
complete-hop device measurements. The compiler continues to refuse unproved
instruction forms rather than label repeated per-op programs as a fused cell.
