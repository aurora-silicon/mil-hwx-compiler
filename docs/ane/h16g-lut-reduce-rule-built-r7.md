# H16G rule-built LUT/reduction round 7

Round 6 proved the exponential table, direct ReLU, and the producer dependency
mask.  This round leaves those byte-for-byte unchanged and assigns the
remaining unary table selectors and reduction phase geometry to named fields.

## Elementwise table contract

Every table contains 33 fp16 interpolation samples.  The clamp/tail pair and
the two control words select the sample domain; treating word `0x54` as a
sample count made every non-exp curve use the wrong range-reduction mode.

| Operation | Clamp domain | Sample domain | Operation control | Range control |
|---|---|---|---:|---:|
| sigmoid | `[-9.9375, 8.3203125]` | `sigmoid(x)`, `x=-8..8`, step 0.5 | `0x00013be6` | `0x01` |
| tanh | `[0, 4]` | `tanh(x)`, `x=0..4`, step 0.125 | `0x00033c00` | `0x60` |
| gelu | `[-4.3984375, +inf]` | exact GELU, `x=-4..4`, step 0.25 | `0x00020000` | `0x01` |
| silu | `[-10.203125, +inf]` | SiLU, `x=-8..8`, step 0.5 | `0x00019c00` | `0x01` |
| log | `[0, +inf]` | `log2(x)`, `x=0.5..1`, step 1/64 | `0x00060000` | `0xc9` |
| sqrt | `[0, +inf]` | eight zero slots, then `sqrt(x)`, `x=0.5..2`, step 1/16 | `0x00040000` | `0xc5` |
| rsqrt | `[0, +inf]` | eight zero slots, then `1/sqrt(x)` over the same domain | `0x00040000` | `0xc6` |
| reciprocal | `[0, +inf]` | `2/x`, `x=1..2`, step 1/32 | `0x00060000` | `0x67` |

Sigmoid, GELU, and SiLU also populate their three named tail-continuation
halves at bytes 74--79.  Reciprocal and rsqrt share the reciprocal-lane
command; their range-control words select different curves.  Exp retains the
round-6 range-reduced table and command, and ReLU retains its table-free path.

## Reduction Common geometry

Common dimensions describe the local phase, not the source tensor globally.
Here `C/H/W` are the task-local input dimensions, `P` is the packed channel
width (`C` up to 64, otherwise 32), and `T=ceil(C*H*W/16384)`.

| Phase | Common `W/H/C` | Output `W/H/C` | Extent `0x24` | Resident `0x30` | Base `0x34` | Kernel/output `0x3c` | Tiles `0x40` | Schedule / limit |
|---|---|---|---:|---:|---:|---:|---:|---|
| axis-3 direct | `W/1/H` | `W/1/H` | absent | absent | 1 | output-row + 16 | absent | `6 / 0` |
| axis-1 max partial | `W/H/C` | `W/H/1` | absent | `0x404000` | H | max tier | 0 | `absent / 1 MiB` |
| axis-1 sum/mean partial | `W/H/C` | `W/H/C` | absent | `0x44000` | H | partial pipeline | `T+1` | `absent / 1 MiB` |
| axis-1 finalize | `C/1/W` | `P/ceil(C/P)/W` | H | absent | 1 | output-row + 16 | absent | `6 / 0` |
| axis-2 reorder | `W/C/H` | `W/C/H` | absent | `0x44000` | C | reorder pipeline | T | `absent / 1 MiB` |
| axis-2 partial | `H/1/C` | `H/1/C` | W | absent | 1 | output-row + 16 | absent | `6 / 0` |
| axis-2 store | `1/C/-` | `1/C/-` | absent | `0x14000` | W | store pipeline | 0 | `absent / 1 MiB` |

The round-6 direct path wrote source H into Common height.  That made a width
reduction look like 64 stacked local planes and explains why max completed but
missed the accuracy bar.  Mean also pre-scaled `0x4910`; the engine already
applies the dimension reciprocal selected by Other `0x4514`, causing the large
mean error.  Sum uses no Other selector, maximum uses `0x4500`, and only the
vector partial/copy phases emit an NE command.

Axis-1 partial/finalize and axis-2 reorder/partial/store each have their own
execution-cycle, output-format, and dispatch-format header words.  Dependency
word 7 remains the round-6 producer mask (`3,3,0` for three tasks), and the
task-control sequence remains `0x00050001, 0x00000001, 0x00040009`.

## Reviewer gate

Both morph gates now use the runnable rule-built-r6 object.  The original
twelve `rNN` cases start from r6 and regress one r6 repair to r4.  The reduce
`nNN` cases start from r6 and apply one r7 field.  The six axis-2 task-2
dependency objects are byte-identical to round 6 and remain LOAD-only.
