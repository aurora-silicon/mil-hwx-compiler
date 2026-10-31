# Aurora compiler changes and existing support

The base is `927013f5480768b07c8e62a866e1514c41aba778`, version 0.3.1.
This work extends that compiler. H16G is an existing backend, not a new target.

## Branch review

| Source reviewed | What already exists | Treatment |
| --- | --- | --- |
| `main`, `CHANGELOG.md` through 0.3.1 | H13/M1 and H14/M2 packages, decoded shape tables, CPU reference and package simulation, hardware receipts, H17/H18 parity targets | Reuse existing encoders and tests |
| `agent/t7-h15-h16-oracles`, tip `cbd2ee2` | Initial compile-only H15/H16 oracle collection, raw recording support, section facts and a prologue word of 1 | Use as evidence; do not merge the older branch over main |
| `compiler/h18g-subtype`, tip `44ace49` | Subtype 11 named H18G by the inspector, with unknown ISA | Already on main as `002a84d`; no new H18G execution backend |
| H16G release and SRAM composition history, `9b0dcbd`, `077a3e8`, `01df3e6` | Source-native H16G pipeline, Linux build, measured program composition families | Extend the existing writer and planner boundaries |

The early H15/H16 branch explicitly calls its records compile-only. Subtype
recognition does not establish device execution. Main's H17/H18 parity targets
are also distinct from the H18G inspector entry.

## Existing H16G functionality

`HWXObjectWriter` already writes kind-3 tensors with code `0x34a`, direction
1/2, fp16 element code 5, four dimensions, byte strides, `main` at `0xd38`,
and the tensor symbol at `0xd3d`. It already writes kind-4 linear `0x8a0`
and mixed `0x8c0` descriptors, five 16-byte aperture records at `0x70`,
and input/output/kernel/scratch/text permissions 1/2/1/3/5.

The existing staged compiler has convolution and depthwise encoders, square
ALU and matmul families, matrix-row broadcasting/division, LUTs, reductions,
space/depth transforms, constant packing and relocations. Existing tiling
covers matrix rows and specific layout packet families. Existing composition
includes matmul plus GELU, scalar folds, unary/reduction SRAM chains and row
normalization. These are measured families, not permission to compose any
shape or arbitrary task sequence.

`prepare_alu add 128` already generates a 49,152-byte owner fixture with SHA-256
`231f54c28ea6b14a2867fac78eac1a2cb22aa3b0eeaa28113ed03c520b83a1d3`.
Its identity is preserved. No captured HWX file is added to this repository.

## Additions

`tools/h16g_validate.py` independently checks the initial fp16 SPEC-r3
container contract. It checks all reserved descriptor bytes, strings, spans,
overflow, segment permissions and overlap, resource ordering, references,
symbol bindings, task framing/register ranges and kernel relocations. The
observed initial record/format whitelists are explicit. Values outside them
are refused even when the research writer can emit them.

The H16 writer now rejects identities that cannot be represented safely,
dimensions that truncate to u32, and duplicate relocation sites. Long MIL
names keep their semantic symbols and receive unique NUL-terminated short
references. The descriptor span is one batch stride; allocation padding is
preserved separately. H13/H14/H17/H18 writer behavior is unchanged.

The owner regression rebuilds the fixture and checks its complete hash and
the three CPU fp16 buffer hashes. Synthetic tests mutate descriptors,
apertures, permissions, tables, arithmetic, task records and relocations.

## Consumer disagreement that remains

The public compiler owner fixture starts `__text` with u32 1 followed by
twelve zero bytes. This agrees with the early H15/H16 branch's observations.
The integration driver's current task gate requires sixteen zero bytes.
Those requirements cannot both pass without changing one side.

The validator defaults to the current driver policy and refuses this fixture
with `h16g.task-envelope`. `--task-policy public` is diagnostic: it accepts
the pinned public prologue, applies the remaining checks and returns
`driver_task_gate: false`. It does not authorize execution. The compiler
does not rewrite the fixture to disguise this disagreement.

```sh
make test-h16g-validation
python3 tools/h16g_validate.py program-0.hwx
python3 tools/h16g_validate.py program-0.hwx --owner --task-policy public
```

Structural validation does not replace firmware/preload authentication,
runtime command/completion decoding, ownership, mapping lifetime, DMA sync
or device numerical tests. No Linux H16 inference claim follows from it.

## Spatial prerequisites and limits

`tools/spatial_tiles.py` adds CPU overlap-save geometry for unit-stride SAME
k1/k3 convolution. Tests cover fixed 64-pixel patches for 1920x1080 and
1280x720, partial edge tiles and unique output ownership. Dense-versus-tiled
fp16 reference tests check signed inputs, multiple channels, depthwise
groups, borders and seams.

The CPU MIL reference also evaluates ordinary concat, integer nearest
upsampling, integer-ratio nearest resize and bilinear resize/upsampling.
Bilinear modes are DEFAULT, STRICT_ALIGN_CORNERS and UNALIGN_CORNERS.
Unsupported sampling modes, fractional nearest scales and interleaving
concat refuse by name. Coordinate rules follow the
[Core ML MIL definitions](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/image_resizing.py).

These are CPU references and planning geometry. They do not add a decoded
ANE copy/resize instruction, a new package gather/scatter ABI or full-frame
convolution emission. Existing matrix tiling is retained. Existing hardware
refusals remain until the missing movement contract is proved or an explicit
host-step package contract is implemented.

| ROADMAP family | H13 | H14 | H16G | Missing coverage |
| --- | --- | --- | --- | --- |
| Full-frame denoiser / small UNet | Incomplete | Incomplete | Incomplete | Spatial gather/scatter, halo dispatch, resize/concat and model-specific conv shapes |
| Selfie segmentation | Incomplete | Incomplete | Incomplete | Resize/concat, small-channel and pooling geometries |
| Face detector | Incomplete | Incomplete | Incomplete | Model-specific small-channel and kernel coverage, head movement |
| Decomposed GRU | Individual covered primitives only | Individual covered primitives only | Individual covered primitives only | Exact model shapes and supported routing between gates |
| Whisper encoder | Individual covered primitives only | Individual covered primitives only | Individual covered primitives only | Exact model shapes and routing; existing matmul/GELU/reduction support is reused |

No complete imported model or open weights were supplied for this change.
This table records missing requirements, not a device benchmark or a claim
that every model in a family was tested. H17/H18 retain byte parity only.
A device-run H17 target additionally needs a proved ownership backend,
runtime protocol, validated mappings and completion, then numerical runs.
