# Formula-built H16G and H17 convolutions

This is a partial port. All twelve decoded convolution programs per target
(fifteen tasks) have formula-built descriptors. Executable HWX emission
covers the six measured stride-1 standalone forms. **Full MobileNetV1 224 and
executable linked H16/H17 chains remain refused.** No H16/H17 device execution
is claimed by this change.

The source facts are the sanitized `research/oracles/{h13,h16,h17}/conv_desc_*.json`
from `research/h16g-h17-conv-oracles`. They contain decoded fields, source MIL,
metadata and hashes, never Apple-produced binary payloads. Implementation uses
our existing MIT-licensed stride-1 packer. Public register names and packed
record grammar are documented by
[coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h16_register_map.md)
(BSD-3-Clause). No code from unlicensed sources was copied.

## Reproduce

```
make test-h16g test-h17
make test-h13 test-h13-parity test-h17-parity
python3 tools/h16-mobilenet/emit_measured.py build/measured-h16-h17
build/mil-hwxc --target H16G --format hwx --mil model.mil --model-root model --output out
build/mil-hwxc --target H17 --format hwx --mil model.mil --model-root model --output out
```

H16G's existing default staged path remains available; `--format hwx`
selects the new measured conv backend. H17 routes conv graphs to this backend
and preserves its existing elementwise parity path. The new manifest states
`deviceValidated: false`. Output contains a `program-0.hwx` and its surface
layout in `manifest.json`; weights reside in `__KERN_0`, and the relocation
points there as a 64-bit operand. `__TEXT/__const` contains zeros. The H17
convolution path uses H16-style resource metadata with the H17 architecture
and ISA tags, rather than routing its weight section to elementwise constants.
Legacy corpus-based size caps have been replaced with checks against the
32-bit HWX file-offset representation; no macOS software size cap is imposed.

## Shape formulas and measured schedule

External fp16 NCHW rows are `align64(2*W)`, planes are `row*H`, and allocations
are rounded to 16 KiB. Internal L2 rows are distinct: pointwise uses
`2*min(W,64)`; regular/depthwise 3x3 uses `2*W`. Source stride is
`Cin*internalRow + schedule.sourceSkew`. Result base is input rows times this
stride for standalone forms; resident chains use the measured ping-pong
allocation and cumulative kernel bases. Result stride uses the measured
result-row mode plus skew. In particular a resident depthwise result has a
measured `0x80` skew, not the standalone width-56 skew of `0x40`.

| Geometry (Cin→Cout, square input) | TileHeight | Source skew | Common 0x3c | Executable |
| --- | ---: | ---: | ---: | --- |
| 1x1 64→64 @64 | 64 | 0 | 0x210000 | yes |
| 1x1 32→64 @112 | 112 | 0x50 | 0x231100 | yes |
| 1x1 64→128 @112 | 80 | 0x50 | 0x231100 | yes |
| 1x1 64→128 @56 | 56 | 0x50 | 0x240000 | yes |
| dense 3x3 64→64 @112 | 112 | 0 | 0x200000 | yes |
| depthwise 3x3 C32 @112 | 112 | 0x50 | 0x230000 | yes |
| dense 3x3/s2 3→32 @224 | 112 | 0 | 0x230000 | descriptors only |
| depthwise 3x3/s2 C64 @112 | 56 | 0 | 0x210000 | descriptors only |
| depthwise 3x3/s2 C128 @56 | 28 | 0x30 | 0x220000 | descriptors only |

Only square batch-one SAME, unit dilation, no bias, dense or depthwise groups
are covered. ReLU sets NE MAC bit 16. Dense stride-1 chunks are
`align64(2*Cin*k*k*Cout/16)`; depthwise substitutes one for Cin. Sixteen local
chunk offsets follow the cumulative sizes. Reset-default size 64 is omitted.

H17 applies the explicit generation delta: Common `0x4c=0x200000`, mandatory
NE `0x4900=0x10000082` plus sparse bit `0x100`, destination low byte `0x21`,
bit 27 of Common `0x3c` on measured dense 3x3, and the measured middle-task
header flag. H16G uses `0x100000`, omits the dense NE baseline, and writes
destination `0x31`. Packed-record selection recomputes task sizes after the
H17 NE write.

Two resident schedules are descriptor-tested: depthwise C64/s2→pointwise
C128, and pointwise C64→depthwise C64/s2→pointwise C128. External DMA appears
at the endpoints only; task IDs, local chunk offsets, cumulative kernel bases,
L2 bases and chain flags are rebuilt. This is a descriptor API, not a loader
API: it supplies no sparse kernel bytes and cannot emit an executable package.

## Parity and report corrections

Every register (including omitted/default fields), packed record, task size,
ID and known header word is checked against all twelve oracles for each
target. **Only header[1] ExeCycles is excluded**; the encoder writes zero and
makes no runtime claim about that unknown hint. All six applicable kernel
section sizes/hashes must match, and nonuniform input weights must change the
kernel section. Sparse section hashes are never claimed as passing tests.

The campaign uses the legacy `mint_oracles.blob` header layout. As already
explained by `research/mint_conv_probes.py:blob_view`, its resolved constant
starts at file offset zero and includes the synthetic file header before the
0.5 payload. Hash tests reconstruct that source-owned input view, rather than
assuming a flat array of 0.5 values or changing production packing to fit an
incorrect test input. Ordinary correctly formed BLOBFILE inputs are supported;
the nonuniform-weight package test checks that their real payload is used.

The field report counts seventeen tasks; the twelve decoded programs actually
contain fifteen (ten singles, one pair and one triple).

The field report says no CacheDMA words are written. The decoded JSON instead
contains `0x5900`, `0x5904`, `0x5918`, `0x5928`, `0x592c` on final tasks and
stage-tagged `0x5908` on final resident-chain tasks. The encoder preserves
these decoded facts and tests them. The resident depthwise result skew above
is another schedule-specific fact which must not be extrapolated from the
standalone result-skew formula.

## Unknowns and refusals

- Sparse H16/H17 byte grammar. The stride-2 section hashes differ from H13;
  neither H13 sparse bytes nor a zero placeholder can substitute for weights.
  Descriptor-only chunk sizes describe the campaign BLOBFILE view.
- Bias packing/epilogue, later MobileNet channel counts/spatial widths,
  output-channel splitting, general tiling/skew/result-mode selection, VALID,
  custom padding, dilation, rectangular kernels and non-depthwise groups.
- DRAM dependency synchronization. The H13 linker retains its tested producer
  TSE+TSR and consumer TileDMASrc DependencyMode behavior. H16/H17 have a
  different header: header[6] is not H13's control word. The public map places
  source DependencyMode at bits 28–29 and exposes CacheDMA TaskSync, but the
  focused conv corpus has no DRAM producer/consumer edge. Copying H13 control
  bits or enabling a guessed H16 handshake would not be a validated port.
- General L2 capacity and allocator policy, and the meanings of generation
  bits, destination modes, and cost hints beyond the recorded forms.
- Harness/kernel ABI execution. A readable HWX package and field parity do not
  establish ABI compatibility or numerical correctness on a device.

The existing lane-a graph has 28 bias-enabled convolutions, starts with the
space-to-depth 16→32 stem at 112, and uses later 256/512/1024 widths plus
stride-2 pointwise layers. Those are outside the supplied H16/H17 facts.
Reproduce the exact attempt and a layer-by-layer descriptor coverage report:

```
python3 tools/h16-mobilenet/coverage.py build/mobilenet-h16-h17
```

The tool saves the same random-weight graph and parameters as
`tools/h13-mobilenet`, attempts `--schedule linked` on both targets, writes
`coverage.json`, and exits 2 on refusal. It never labels a refusal or an
H13-relabelled object as an H16/H17 package. Wider decoded schedule/bias/sparse
and DRAM-edge evidence is needed before full-model loadable packages can be
emitted.
