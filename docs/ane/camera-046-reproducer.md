# Camera island 046: 35×35 gather and mix

The reported failure involves a 5×5 depthwise layer rewritten as two dense
3×3 convolutions followed by a host stride-2 select/crop. The gather stage
is 128→512, stride 1, pad 1, 35×35 fp16 with zero bias, padded from 96→384
live channels. The mix stage is 512→128, padded from 384→96 live channels.
These stages can be checked independently of the host crop and stride.

Fresh compile-only H13 and H14 oracles are retained under
`research/oracles/{h13,h14}/layer_conv*_s35_*.json`. They contain decoded
fields and hashes, not Apple containers or constant blobs. Indexed finite
weights expose each task's actual coefficient order:

| Stage | Native task order | Coefficient planes per core |
| --- | --- | --- |
| Gather 128→512 | outputs 256–511, then 0–255 | 16, then 16 interleaved channels |
| Mix 512→128 | outputs 96–127, then 0–95 | 2, then 6 interleaved channels |

Each task has its own contiguous 16-core coefficient section. The first
task caches the input; the second reuses it. The two-task streams and their
weight sections now reproduce the complete decoded oracle, including bias
placement, source/result layout, scheduling words, output offsets and
kernel-DMA tables. Both ordinary and indexed-weight records are checked.

The prior generic writeout at 35×35 copied a power-of-two result layout:
Common `0x34=0x210001`, ResultCfg `0x50017a`, and zero result strides. The
native gather uses `0x242201`, `0x500172`, and C/R/D/G strides
`0x40/0x4040/0x4000/0x4000`. Output slicing and coefficient order must agree
with the entire native schedule; the stride correction alone is not the
qualification. The reader and CPU simulation do not exercise these hardware
addressing fields.

A separate generic planner issue rounded a fitting 113-output budget to
112, which mixes 4+2+1-lane planes under one OCG value. It now chooses uniform
power-of-two slices below 256 channels; 512→128 3×3 at 16×16 is tested as two
64-output tasks. Native 35×35 uses the decoded 32+96 schedule above.

## Device-checkable fixtures

Preparation requires NumPy and accesses no hardware. Output directories must
be empty so earlier failures and readbacks are preserved.

```sh
python3 tools/h13-conv-bench/repro_camera046.py build --target H13 --output /tmp/camera046-gather
python3 tools/h13-conv-bench/repro_camera046.py build --target H13 --kind dense --output /tmp/camera046-dense
python3 tools/h13-conv-bench/repro_camera046.py build --target H13 --kind mix --output /tmp/camera046-mix
```

Use `--target H14` to prepare the corresponding M2 packages. Each directory
contains MIL, source-owned weights, one two-task ANEC program, packed input,
float64 reference rounded to fp16 with ties away, and artifact hashes.

The gather uses the four corner shifts over 96 live channels and zeros all
padded channels. Dense mode exercises every one of the 128 input and 512
output channels. Mix mode uses a sparse exact 5×5-to-3×3 coefficient rewrite
with 384 live inputs and 96 live outputs.

On a lead-authorized device run, start with one cold execution and fresh
input through the existing `anechain` runner from
`tools/h13-mobilenet/README.md`, preserving the normal driver health evidence:

```sh
anechain -w 0 -r 1 -i /tmp/camera046-gather/input.bin \
  -o /tmp/camera046-gather/actual.bin /tmp/camera046-gather/pkg/program-0.anec
python3 tools/h13-conv-bench/repro_camera046.py compare \
  /tmp/camera046-gather /tmp/camera046-gather/actual.bin
```

The gather requires exact numerical fp16 agreement (+0 and -0 compare equal).
Dense and mix modes use rel. L2 ≤0.002 and reject nonfinite output. The
comparator reports max absolute error and its channel/row/column. Repeat
with a different seed, then check the original whole island and its host
crop/select separately. Host parity is not a device numerical receipt.

The full task/weight regression is `tests/test_h13_camera_conv_cli.py`, run
for both targets. Unqualified neighboring dense 3×3 writeouts at widths
33–63 are refused instead of reusing the former generic layout. Stride-2
5×5 depthwise experiments use a different coefficient format and remain
refused; their decoded evidence is retained for subsequent derivation.
