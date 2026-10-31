# H16 execute resource metadata

Each compiler-generated H16G artifact in `manifest.json` has an additive
`h16Execute` object (version 1). The HWX bytes and bundle format version 2 are
unchanged. H13, H14, H17 and H18 output is unchanged.

The metadata describes buffer geometry. A consumer must validate the complete
HWX first, cross-check the metadata, then resolve each resource to its own
pinned mapping with the required permissions. It must never treat this JSON
as allocation, mapping, selector or execution authority.

```sh
python3 tools/h16g_validate.py program-0.hwx --manifest manifest.json
```

The validator returns independently derived `execute_resources`. A metadata
mismatch is `h16g.execute-resource-mapping`. `--owner` additionally requires
the original add-128 size and SHA-256 identity.

## Ordinary buffers

`bufferOffset` is `0x60`, `bufferStride` is `0x30`. `buffers` contains only
external `__FVMLIB` surfaces in aperture order. Embedded kernel, scratch and
text allocations are distinct program resources and do not create external
input/output descriptors. No numeric class is guessed for them.

| Metadata field | Meaning | Execute descriptor offset |
| --- | --- | --- |
| `apertureSlot` | Slot in the validated kind-4 aperture table; not an address | Resource lookup only |
| `index` | Index within that class's input/output array | `+0x04`, u32 |
| `class` | 0 input, 1 output, derived from validated tensor direction and section permissions | `+0x08`, u32 |
| `offsetBytes` | Offset into the matched pinned mapping; currently zero | Added to mapping DVA at `+0x18`, u64 |
| `sizeBytes` | Entire validated surface allocation, including padding | `+0x20`, u64 |
| `mappingBytes` | Required mapping extent from the surface segment size | Bounds check only |

The serializer supplies flags 1 at `+0x00`, fixed 3 at `+0x10`, and zeros in
all reserved fields. DVA and span are separate u64 fields. Require
`offsetBytes <= mapping_size`, `sizeBytes > 0`, and
`sizeBytes <= mapping_size - offsetBytes`; reject DVA-range overflow.
`mappingBytes` is not a tensor's logical element count or one-batch span.
The validator checks the tensor's full `N * batch_stride` against that extent.

Aperture order can interleave input and output surfaces. Indices are counted
separately within each class, as in the H16 request's separate input and
output arrays. The owner has three apertures, in this order:

| Aperture slot | Class | Index | Size |
| --- | --- | --- | --- |
| 0 | 0 (input) | 0 | `0x8000` |
| 1 | 1 (output) | 0 | `0x8000` |
| 2 | 0 (input) | 1 | `0x8000` |

This ordering follows the emitted container, independently of the runtime
binding array. Input descriptor order must match input aperture order;
likewise for output. The class/index convention also appears in
[`ANEProvisionedRuntime.mm` at 927013f](https://github.com/joshuaswarren/mil-hwx-compiler/blob/927013f5480768b07c8e62a866e1514c41aba778/lib/Runtime/ANEProvisionedRuntime.mm),
which builds separate input and output index arrays for H16.

## Custom BARs and command length

`customBarOffset` is `0xca8`, `customBarStride` is `0x18`. The count is written
at command `+0xca0`; total command bytes are exactly
`0xca8 + len(customBars) * 0x18`.

| Custom-BAR field | Tail record offset | Width |
| --- | --- | --- |
| `configuration` | `+0x00` | u32 |
| `valueType` | `+0x04` | u32 |
| `bar` | `+0x08` | u32 |
| `bitOffset` | `+0x0c` | u8 |
| `bitWidth` | `+0x0d` | u8 |
| `flags` | `+0x0e` | u16 |
| `value` | `+0x10` | u64 |

The current HWX writer has no validated custom-BAR descriptions, so it emits
`customBars: []` and `commandBytes: 3240` (`0xca8`). Aperture records do not
imply custom BARs. Adding a tail to this output is a refusal, even if the tail
has well-formed fields. Nonempty output requires a defined program-metadata
encoding and proved selector/value semantics first.

`validate_custom_bars` checks only geometry: at most two records, unsigned
field widths, nonzero bit width, bit range within 64 bits, and no overlapping
fields with the same configuration/BAR. It does not authorize selectors or
assign meaning to a value type. `validate_execute` compares a device-free
packet against an already validated resource plan and caller-owned mapping
geometry. It checks every byte, including all fixed reserved space, unused
ordinary slots, lengths and full-width addresses. Synthetic tests exercise
one- and two-record tails without presenting them as valid compiler output.

The corresponding kernel layout is
[`ane_h16_execute.h` at 62221037](https://github.com/aurora-silicon/linux/blob/6222103718b5c694f85012a731222a0126456fb7/drivers/accel/ane/ane_h16_execute.h).
The geometry checker cannot establish runtime ownership, mapping permissions,
completion, DMA synchronization or lifetime. No accelerator is opened by
these tools or tests.
