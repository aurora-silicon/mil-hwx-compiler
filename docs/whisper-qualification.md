# Exact Whisper encoder qualification

The t-3655 [coverage receipt](../tests/whisper/compiler-coverage.json) is now an
executable compiler regression input. It covers public MIT tiny.en, base.en and
small.en encoder exports from coremltools 8.3.0 / CPU PyTorch 2.5.0, MIL program
1.3 / ios18, against compiler `c28fe541697d37d7ec88886bb6b81cef8329dc44`.
The receipt and sixty isolated MIL sources are retained without changing their
bytes. [The fixture index](../tests/whisper/probes.json) pins their SHA-256 hashes.
Model weights are supplied separately and checked against the receipt.

The original result is six full-encoder refusals, 108 isolated refusals, five
isolated compiles, and seven 60-second host timeouts. The source JSON calls the
timeouts `error`; the replay retains that raw field and classifies rc 124 as a
host timeout. Neither a successful coverage command nor a budget refusal means
that a model runs on the ANE.

| Requirement | Exact model geometry | Qualification |
| --- | --- | --- |
| Conv1d | `[1,80,3000]` with `[D,80,3]`, stride 1; `[1,D,3000]` with `[D,D,3]`, stride 2; custom pad `[1,1]`, dilation 1 | Original H13/H14 convolution refusals retained; CPU rank-3 convolution with explicit pads, groups and bias |
| Layout movement | `[1,D,1500]` to `[1,1500,D]`; `[1,1500,H,64]` to/from `[1,H,1500,64]` | Exact physical permutations checked with unique index tags over every element; materialized hardware transposes still refuse |
| Views | `[1,1500,D]` to/from `[1,1500,H,64]`, including inferred `-1` | Producer/reshape/consumer cases distinguish a folded view from an isolated returned-input alias |
| Broadcasting | Positional `[1500,D]` plus `[1,1500,D]`; singleton `[1,1,1,1]` scale `0.353515625` | CPU semantic examples; original hardware constant refusals retained |
| Affine normalization | `[1,1500,D]`, axis -1, gamma/beta `[D]`, epsilon `0.000010013580322265625` | CPU affine reference retains the exported epsilon and applies affine parameters before final fp16 rounding; hardware normalization gate unchanged |
| Attention | Runtime QK with transpose_y; `[1,H,1500,1500]` softmax; scores times V | CPU batched matmul uses a separate runtime RHS per batch and supports batch broadcasting; full 1500-key softmax reductions tested on sampled queries; exact hardware shapes replayed and still refused |
| GELU | `[1,D,3000]`, `[1,D,1500]`, `[1,1500,4D]`, mode EXACT | CPU erf reference; large hardware unary gates unchanged |
| H14 bias | D-to-D, D-to-4D, 4D-to-D with nonzero vector bias | CPU linear+bias example; `h14.linear-bias-unsupported` retained |
| Bounded/fused emission | 1500-token projections and linear/GELU/linear FFN | H13 program materialization budget checked before payload encoding; exact FFN chain retains its named fusion refusal |

Here `D=384/512/768` and `H=6/8/12` for tiny/base/small. CPU semantic tests use independent small examples. Conv1d additionally checks
144 output samples over all six exact input/weight/output geometries, including
borders and positions around 64-element boundaries, with sparse dyadic synthetic
weights and lazy finite fp16 inputs. `conv1d_samples` returns selected outputs
without cropping the original geometry. Affine normalization checks full
384/512/768 widths on two rows. These are not dense numerical comparisons of all
large operations or learned checkpoints. Movement tests cover complete exact
layouts. The softmax test covers full key extent on two queries. No device
numerical parity, WER, RTF, energy or full-encoder execution result is implied.

## Reproduce

Build the public compiler on a Linux build host using the normal toolchain,
then run the device-free checks:

```sh
scripts/verify-linux-compiler.sh build-only
make test-whisper-coverage
python3 tools/whisper_coverage.py --check
```

Recreate checkpoint-matched exports with the pinned
[Whisper exporter](https://github.com/aurora-silicon/whisper.cpp/tree/e65f3c4423baae5898e40458d8cb1d192897172e/models/ane)
and its [export instructions](https://github.com/aurora-silicon/whisper.cpp/blob/e65f3c4423baae5898e40458d8cb1d192897172e/docs/ane/README.md).
Place each `model.mil` and `weights.bin` under `exports/tiny.en`,
`exports/base.en` and `exports/small.en`. Replay into an absent result directory:

```sh
python3 tools/whisper_coverage.py \
  --compiler build/mil-hwxc --models-root exports --output coverage-run
```

Replay checks all selected model hashes before launching any compiler, checks
probe hashes, input/constant/output declarations and exact attributes, and uses a fresh temporary
package directory for every invocation. Each result retains the return code,
diagnostic codes/text, elapsed host time, source hash and emitted program count.
Successful packages also receive a digest over program filenames and SHA-256s.
A success requires complete manifest/program files; a failed compilation that
leaves a package is an error. Logs are checkpointed after every case. Unexpected partial or changed packages are retained under `failures/`. Other
temporary probe packages are discarded after inspection; they are not runnable model
artifacts. This tool only launches the compiler, never the device runner.

`review_passed` means that refusals matched the receipt and prior successful or
timed-out probes were bounded or retained their recorded compile counts. New
support or a changed diagnostic/count requires review. Timeouts, unknown exit
codes and incomplete packages fail the review check. `runnable_full_models` and
`native_qualified` are separate fields.

## H13 program budget

`mil-hwxc --target H13 --max-programs N` bounds the number of materialized
programs. The default is 65,536, including calls through the existing Objective-C
API. The overload taking `maxPrograms:` allows explicit limits. H14/H17/H18 do
not accept this option. Zero, negative, malformed and overflowing values return
CLI usage error 64.

H13 uses the same slice planner for a count-only pass and an encoding pass. An
excess returns 65 with `h13.program-budget` before any payload encoding or
package output. Biased row-linear expansion first checks its two-programs-per-row
lower bound before allocating row operations. The existing encoders, rounding,
constant packing, routing and qualified fusion rules are unchanged. Tests compare
allowed program bytes against the default compilation and exercise exact and
cumulative boundaries.

The replay uses 2,048 for prior successes/timeouts and 65,536 for prior semantic
refusals, recording each effective limit. This retains diagnostic visibility for
unproved ops while bounding the known expensive emission cohort. Adjust
`--max-programs` and `--refusal-max-programs` explicitly when reviewing new
coverage. A low budget can precede an operation's semantic validation; a budget
refusal must never be relabeled unsupported or numerically qualified.

The limit counts temporary per-op materializations used before an existing chain
is fused, rather than only the final fused package count. It is not a wall-time,
total-memory, task-descriptor or constant-byte limit. Host timeout is enforced
separately. Arbitrary Whisper FFN fusion remains outside the proved envelope;
existing measured fusions retain their own checks.

## Sources and remaining work

The reference follows the public MIL definitions for
[normalization](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/normalization.py),
[matmul/linear](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/linear.py)
and [convolution](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/conv.py).
It uses fp32 product/sum rounding and final fp16 outputs. The existing H13
linear reference rounds the matmul result before adding bias, matching its
separate-program lowering; fused arithmetic would need a separate numerical
qualification. CPU reference support does not extend any hardware envelope.

Materialized movement and Conv1d dispatch still need proved encodings and routing.
Affine normalization, batched attention, large GELU and H14 biased linears still
need qualified exact-shape emission. The original H13 D-to-D projections emitted
10,500 programs (tiny) and 3,000 (base); residual adds emitted 9,000/12,000/18,000.
The bounded check does not reduce those costs. A new fused path must prove its
intermediate layout, fp16 rounding and resource use before replacing them. A
complete checkpoint-matched package must then pass existing libane/runtime
qualification before full-model device comparison or performance claims.
