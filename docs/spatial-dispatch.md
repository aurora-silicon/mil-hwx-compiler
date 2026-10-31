# Spatial dispatch on H13 and H14

`--schedule spatial` compiles convolution, resize, concat and the MobileNetV1
head into a sequence of existing ANE kernels. It emits
`mil-hwxc.spatial-package.v1`. Select the schedule explicitly because the
original per-op/chain package readers do not implement this contract.

```sh
build/mil-hwxc --target H14 --schedule spatial --mil model.mil \
  --model-root model --output spatial-package
python3 tools/spatial_package.py spatial-package
python3 tools/spatial_package.py spatial-package --simulate \
  --input x=x.fp16 --output probabilities=probabilities.fp16
```

The second command validates the complete structure and reports the expanded
dispatch count without expanding the loops or opening a device. Simulation
and callback execution need NumPy. The simulator models dispatch arithmetic; it does not model the
accuracy of target LUTs or establish device execution.

## Movement and convolution

A copy is the existing 256-channel elementwise maximum with the same input
on both operands. Each binding names a contiguous dense slice. The runner
packs that slice into the kernel's physical layout, zero-pads the unused
lanes, dispatches the ANEC, then stores the specified output prefix. This is
the same staging operation used by the original sliced matrix packages.
There is no host convolution, interpolation, concatenation or indexed gather
step in the dispatch format.

Convolutions use the existing 64-channel kernels, padding channel tails with
zero weights and inputs. Regular convolution uses 16- or 64-pixel square
patches; depthwise convolution uses 16-pixel patches. Every patch is first
filled by explicit ANE copies from a zero constant, then populated by ANE
row copies. A k3 patch contributes its inner `side - 2` samples. The overlap
is input halo; outputs have a single owner, including partial bottom/right
tiles. Batches are independent dispatch loops. SAME_UPPER and VALID, square
k1/k3, unit dilation, and equal strides 1/2 are supported. Groups are 1 or
k3 depthwise with equal input/output channels. Input channels are at most
1024 and output channels at most 2048. Other forms fail by name.

A stride-2 convolution runs a unit-stride kernel and selects the appropriate
rows and columns. The top/left input origin uses the original stride-2
padding, including its asymmetric SAME_UPPER case. Column selection is a
one-hot matrix passed to the existing 256-by-256 matvec encoder. Independent
64-channel reduction chunks are summed with existing fp16 add kernels.
Bias is added once, after the channel reduction, using an existing add
kernel. ReLU and clip use existing maximum/minimum kernels.

Concat copies each source's contiguous suffix into its output interval for
each outer index. Negative axes, non-unit outer dimensions and multiple
returned values are supported. Interleaving concat is refused.

Nearest resize uses one-hot matrices. Bilinear resize puts the coordinate
weights into matrix kernels and sums the two contributing rows. Both reuse
the existing matvec encoders. Large rows are partitioned into windows whose
input span and output count fit 256 lanes. DEFAULT, STRICT_ALIGN_CORNERS and
UNALIGN_CORNERS coordinates agree with the CPU reference, including clamped
borders and singleton axes. Nearest resize requires integer upscales.
Upsampling accepts the CPU reference's scale and align/half-pixel forms.

## MobileNet head and numerical contract

Spatial mean over axes 2/3 uses constant matrix weights; the spatial extent
must fit 256 elements. Linear accepts up to 1024 input/output features,
padding to an existing 256/512/1024 matvec geometry and dispatching each row
independently. H14 bias is a separate add, as on H13. Softmax over the last
axis accepts up to 1024 lanes, pads excluded lanes with negative infinity,
and crops the result. Padding with zero would give incorrect probabilities.

Movement, regular/depthwise tiles, matrices, sums and activations all use
existing task encoders. No descriptor geometry is extrapolated, and the
existing per-op and chain encoders remain unchanged. H16G/H17/H18 retain
their current schedules.

Arithmetic rounds at each fp16 dispatch boundary. Channel chunk sums,
post-convolution bias, fp16 resize coefficients and the spatial-mean
reciprocal can differ from a single fp32 accumulation followed by one fp16
round. These packages therefore declare `arithmetic: fp16-partials`.
Software tests compare the emitted dispatches against `h13_reference` with
explicit absolute/relative tolerances. Copy/concat, clip and exactly
representable resize examples use exact comparisons. This is software
validation, with no new device accuracy or throughput claim.

## Package and executor contract

The manifest contains:

- `kernels`: unique ANEC files with fixed physical input/output layouts,
  task counts, constant spans and bounded scratch allocations. Bindings are
  supplied by dispatches; scratch allocations must match the ANEC header.
- `tensors`: positive static shapes, element counts and roles (`input`,
  `output`, `intermediate`, `constant`, `scratch`). Scratch is uninitialized.
- `constants`: filenames for read-only dense fp16 tensors.
- `dispatchPlan`: kernel calls and bounded, lexically scoped `loop` nodes.
- `outputs`: the ordered returned tensor names.

A call has a kernel index, one binding per input, and one output binding.
Each binding has `tensor`, `offset`, `count` and `paddingBits`. Offsets are
integers or affine expressions with a nonnegative base and nonnegative
strides for enclosing loop variables. Loop variables start at zero. Counts
are positive, never exceed the physical surface, and must fit their named
tensor for every loop iteration. Input padding is zero or negative infinity;
output padding must be zero. Input and constant tensors cannot be written.
Calls cannot read and write the same tensor. Loop depth, descriptor size,
element arithmetic and expanded dispatch count are bounded.

`Package.execute(inputs, callback)` implements ordered dispatch and physical
buffer staging. The synchronous callback receives the kernel record, its
ANEC bytes and physical input buffers, and returns the physical output
buffer. Actual source reads require initialized elements; results require
complete output coverage. A callback failure propagates immediately and
no later kernel runs. Session approval, device identity, resource ownership,
submission, completion and recovery remain the caller's responsibilities.
This module does not open a device or provide an alternate approval path.
The supplied CLI validates or simulates only.

`Package.simulate` decodes matrix and convolution weights directly from the
emitted constant sections. It models primitive arithmetic and actual slice
routing, without consulting source MIL or carrying duplicate dense weights
in the package. Existing encoder parity tests separately pin task words
and constant packing. Structural validation is not device authorization.

## Acceptance

```sh
make test-spatial-reference
make test-spatial-compiler   # NumPy; numerical and malformed-package tests
make test-spatial-mobilenet # NumPy; exact CSV geometries plus full graphs
make test-h13-parity test-h14-parity test-h17-parity test-h18-parity
```

`tests/data/mobilenet-compiler-gaps.csv` preserves the 120-point acceptance
list from [omarchy-ane PR 1](https://github.com/aurora-silicon/omarchy-ane/pull/1),
including the original diagnostics. The test checks its exact input/output
shapes, compiles every point for its specified target/batch, and compiles
all four complete 27-convolution graphs with bias, ReLU6, mean, linear and
softmax. Generated weights are deterministic test inputs. Results are
written to `build/spatial-acceptance.json`.

This schedule prioritizes explicit, inspectable routing and kernel reuse.
It can expand to many submissions, especially for batch 8 and narrow channel
chunks. Expanded dispatch counts are reported before execution; no model
latency claim follows from successful compilation or simulation.
