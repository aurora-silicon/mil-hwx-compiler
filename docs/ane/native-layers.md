# Native H13/H14 layers

Resize, channel concat, small-channel convolution, 5×5 kernels, sigmoid and
hardswish have whole-layer encodings. No element-by-element dispatch is used
for these operations. The new envelope is defined by decoded compile-only
oracles in `research/oracles/{h13,h14}/layer_*.json`; neighboring shapes are
not inferred from those points. Host field/constant parity is distinct from
device numerical qualification. Device checks of this change are pending.

## Operations and envelope

All new surfaces are static, batch-one, rank-four NCHW fp16. Square resize
points below list `(channels, input spatial size)`.

| Operation | Tasks per layer | New decoded points |
| --- | ---: | --- |
| 2× nearest | 1 | `(1,8)`, `(3,8)`, `(16,16)`, `(32,35)`, `(64,16)`, `(128,16)`, `(24,32)`, `(16,64)`, `(64,10)`, `(64,20)`, `(1,128)` |
| 2× half-pixel bilinear | 1 | Same points |
| Channel concat | 2, or 0 with producer placement | Channel pairs `(3,8)`, `(16,16)`, `(24,32)`, `(64,64)` at 16×16; axis 1 or -3; no interleave |
| Sigmoid | 1 | C = 1/3/16/32/64 at 16×16; C = 16/24/96/128 at 1×1; C = 1 at 256×256 |
| Hardswish | 2 at C1, otherwise 4 | C = 1/3/16/32/64 at 16×16 |
| Small dense/depthwise conv | 1 | Grids below; kernels 1/3/5, stride 1 |

`upsample_nearest_neighbor` and `resize_nearest_neighbor` share the nearest
encoding when both scales are exactly two. Bilinear accepts
`upsample_bilinear(align_corners=false)` or
`resize_bilinear(sampling_mode="UNALIGN_CORNERS")`, with both scales two.
Other coordinate conventions are refused. These coordinate definitions
follow the [Core ML MIL resize specification](https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremltools/converters/mil/mil/ops/defs/iOS15/image_resizing.py).

Hardswish recognizes the canonical MIL expression
`mul(x, real_div(relu6(add(x, fp16(3))), fp16(6)))`, with no escaping
intermediates. It preserves the native program's internal tasks and constant
loads. Direct `hard_swish` probes were rejected by Apple. The accepted
`sigmoid_hard` experiments decoded a slope of 0.2 even with an explicit 1/6
argument; they are retained as **unqualified evidence**, not used as
hardswish. The arithmetic form uses the relu6 clamp and 1/6 scale. See the
[MIL activation definitions](https://github.com/apple/coremltools/blob/a207b1a3452eafe02f2d870f486287834af2c82d/coremltools/converters/mil/mil/ops/defs/iOS15/activation.py).

New convolution points, each with kernels 1/3/5:

- Cin = 3/8/16/24/32 → Cout32, and depthwise C = 3/8/16/24/32, at 8/16/32/64
  pixels per side, with and without bias.
- Cin16 → Cout3/8/16/24 at the same spatial sizes, with and without bias.
- Cin11/56 → Cout32 at 16×16, with and without bias, for consuming the
  corresponding concat results.
- Cin48/64 → Cout32, and depthwise C48/64, at 16×16 without bias.
- Cin96 → Cout24/32/128, kernel 1 at 16×16, and depthwise C96,
  kernel 5 at 32×32, with and without bias.

The prior H13 generic 1×1/3×3 envelope and the prior decoded H14 points
remain available. New decoded convolution points keep their exact scheduling,
L2 layout and core masks inside linked programs. Generic formulas do not
borrow invariant words from the new points. Uncaptured non-power-of-two
inputs require a decoded layout instead of an extrapolated bank skew.

At 24 dense output channels, eight cores interleave two adjacent channels
each and eight carry one each. Depthwise uses round-robin channels with
individually padded core chunks. A 5×5 depthwise C24 therefore has unequal
chunk sizes. Indexed finite-weight probes verify the permutations and bias
placement, not just a uniform coefficient section. Large generic dense
layers choose uniform output slices under the kernel budget: for example,
512→128 3×3 uses two 64-output tasks instead of an unrepresentable
112-output task with mixed lane widths.

Explicit custom leading padding must match the decoded program. Equal
output dimensions do not make different leading pads equivalent. The exact
35×35 dense corrections and their coefficient orders are documented in
[camera-046-reproducer.md](camera-046-reproducer.md).

## Linked graphs and concat placement

`--schedule linked` accepts native layers and convolutions with one or two
boundary inputs and one result. Concat can join internal branches. Intermediates
use separate regions in one DRAM arena; live storage is not reused.

When a concat has two single-task producers, each has exactly one consumer,
and their written spans fill their allocations, the allocator places their
outputs in adjacent channel slices of the concat surface. The concat emits
no task or dummy constant section. Producer offsets, ownership and the
zero-copy decision are recorded in `programs[0].layers`; intermediate alias
regions name their owner with `aliasOf`. Shared producers and incompatible
allocation layouts use the decoded two-task copy instead. A final concat
can be written directly into the program output surface.

H13 keeps the existing producer release and DRAM dependency fields. A read
of the immediately preceding writer uses dependency mode 7; older writers
use mode 3. Native internal L2 tasks and constant-BAR reads are preserved.
The experimental `linked-l2` scheduler only applies its existing qualified
edge forms; new native layers retain DRAM edges.

H14 ANEC BAR slots are explicit roles: input 5/6, output 4, kernel 1, and
scratch 7. The Linux reader merges scratch read/write references into its
scratch buffer id 0x40. This avoids ambiguous extent-based selection of two
equal-sized inputs. The task's original sparse record structure is preserved;
an absent write remains absent. LUT BAR references at 0x19f4 (one word) and
0x1a00 (two words) are relocated along with coefficient and scalar reads.
DRAM dependency fields and scratch cache hints follow the decoded handoffs
in `env_mm_r2br_m1_k2048_n2048_tx0_ty1` and `sched_fused_lut_d512_s512`.

The H14 reader sizes scratch from the largest reference offset plus the
output allocation. The combined output allocation is therefore at least as
large as any intermediate; its logical shape and strides stay unchanged.
This can increase output allocation and host-copy cost. H14 linked HWX is
refused until its resource descriptor mapping is qualified; per-layer HWX
retains Apple's BAR slots for parity. H14 routing is based on the public
[libane reader](https://github.com/aurora-silicon/omarchy-ane/blob/458152b8315a495a14b632e13e334d29cf5113a0/libane/ane_m2.c).

## Refusals and validation limits

- Resize scales other than two, other bilinear coordinate modes, dynamic
  shapes, and batches other than one.
- Uncaptured new native convolution geometries, arbitrary grouping, and
  unsupported dilation. Dense 3×3 stride 2 remains outside the generic form.
  New depthwise 5×5 stride-2 probes remain unqualified: their coefficient
  hashes do not match the stride-1 packing formula.
- The unsafe sparse stride-2 form above eight interleaved lanes remains
  refused. Existing wide 1×1 stride-2 layers keep their proven 128-output split.
- Dense 3×3 writeout with Cin128 and Cout≥256, or 512→128, at output widths
  33–63 requires a qualified decoded layout. The 35×35 gather/mix points are
  qualified; other captured 128→256 points are available where their bias and
  padding match. Unqualified neighbors are refused.
- Concat axes other than channels, interleaving, more than two operands,
  unsupported native concat shapes, and escaping hardswish intermediates.
- Linked graphs outside the native/conv/concat envelope, more than two
  boundary inputs, multiple results, private per-layer DRAM scratch, and
  graph L2 edges without a decoded form. H14 dependencies more than 16 tasks
  back exceed their four-bit distance and are refused. Whisper transpose extension is
  separate from this envelope; existing transpose support is unchanged.

`make test-h13` and `make test-h14-parity` include native task/constant parity,
linked edges, producer placement and indexed packing checks. The independent
float64 references and ties-away surface rounding are checked with
`make test-layer-reference` (NumPy required). Prepare device
readback fixtures without accessing hardware:

```sh
python3 tools/native_layer_case.py build --target H13 --op bilinear --output /tmp/bilinear-h13
python3 tools/native_layer_case.py build --target H14 --op branch-concat --two-inputs --output /tmp/concat-h14
python3 tools/native_layer_case.py compare /tmp/bilinear-h13 /tmp/bilinear-h13/actual.bin
```

The lead executes through the approved device path and returns the output
surface. Nearest/concat fixtures require exact numerical fp16 agreement;
arithmetic fixtures report absolute and relative error against the rounded
float64 reference. Generated inputs are procedural; no model weights or
private images are distributed.

## Reproducing decoded facts

On a macOS host with the Apple compiler framework, build the compile-only
wrapper outside the source tree and select individual probes:

```sh
clang -O2 -fobjc-arc -fblocks research/compile_layer_oracle.m -framework Foundation -o /tmp/ane-compile-hwx
python3 research/mint_layer_probes.py --tool /tmp/ane-compile-hwx --targets h13 h14 --case 'layer_resize*'
python3 research/generate_layer_templates.py --check
python3 research/mint_conv_probes.py --check
```

The driver removes temporary HWX containers and publishes decoded JSON and
hashes only. Compiler errors are retained as bounded classifications;
original diagnostics stay in the untracked `.git/oracle-failures` directory.
The records identify the compiler tool, probe driver and decoder by hash.
An oracle compiling successfully does not by itself qualify its layout:
the coefficient permutation must also reproduce its constant-section hash.
