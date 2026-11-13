# H16G rule-built residual LOAD grammar

The round-2 ResNet-50 object was structurally accepted by the driver-side
validator but refused by J616s firmware at LOAD with status 0x6e. No task was
executed. This note records the LOAD-visible field audit and the round-3
changes without treating a host check as device qualification.

## Field audit

| Group | Round-2 result | LOAD hypothesis | Round-3 action |
| --- | --- | --- | --- |
| Task length and id (header word 0) | Every descriptor length equals its decoded record body and ids are the contiguous range `[0, task_count)`. Padding is zero and the final descriptor ends at EOF. | Low: the words differ from the earlier object because the rule builder emits a different compact record set, but the chain is self-consistent. | The checker independently walks lengths, ids, padding, and the program task count. |
| Route and engine events (words 2 and 4) | First/last route bits are bounded, but the terminal descriptor used a different engine-control form. | Medium: firmware can validate route/event combinations before execution. | The final route bit is required only on the last task. |
| Dependency/bank mask (word 7) | All 32 mask bits name banks in the linked L2 ring; register-encoded task ids are bounded by the program task count. | Medium: the values differ substantially from the earlier allocator and are a useful header morph arm, but no nonexistent task id remains. | The checker decodes task ids in TileDMA, completion, KernelCfg, and kernel-lane fields. |
| Completion mode (word 8) | The final task was `0005_0008h`, clearing ready while asserting done. The accepted terminal form is `0005_0009h`. | High: this is a direct illegal terminal engine-control candidate and is visible at LOAD. | Terminal tasks keep ready and done asserted; a synthetic negative test locks this rule. |
| Record grammar and flags | Dense/scatter counts are bounded, register addresses are valid, and BAR qword records use slots 3/6/7/8. | Medium: record grouping differs but no record overruns a descriptor. | The checker rejects duplicate writes, truncated records, unbound BAR slots, and non-canonical task ends. |
| BAR and relocation table | Round 2 had 62 kernel-base records and 62 unique relocations; every relocation addend was inside `__KERN_0`. | Low by itself: the table was complete for the emitted tasks. | Round 3 has 64 kernel-base records and requires exact set equality between record patch sites and relocations. |
| Task split and program count | The generic capacity rule emitted 80 tasks. Two 28-to-14 stride-two kernels used two slices where the native sparse phase grammar requires three. Program metadata correctly said 80, so this was not a stale count; it was the wrong grammar. | High: task count, record code, relocation count, and phase boundaries are all LOAD-visible. | The sparse phase rollover is now derived at 11 lanes for 3x3 and 42 lanes for 1x1. ResNet-50 emits 82 tasks, record code 165, 64 relocations, and the accepted kernel-section byte count. |
| Section and scratch layout | Text, const, kernel, symbols, and the 4 MiB scratch declaration are non-overlapping and internally bound. | Low to medium: scratch is larger than the earlier object but all declared DMA extents remain inside it. | Existing container validation remains mandatory before the stronger grammar audit. |

## Offline gate

`tools/h16g_load_grammar.py` first runs `h16g_validate.py`, then independently
checks the linked-task grammar. It passes the accepted 82-task object. It
rejects the round-2 rule-built object specifically at
`h16g.load-grammar.terminal-engine-control`.

The checker proves consistency only. Firmware acceptance and numerical
correctness require the smallest-first J616s ladder, one object per boot after
any refusal or hang.

## Morph order

`tools/h16g_load_morph.py` builds a private set from two compiler outputs. The
review order is BAR/relocations, record values, headers, then the donor task
split control. The first three arms preserve the accepted object's file layout
and patch only aligned task fields. Changed descriptor boundaries cannot be
represented as an in-place one-group patch, so the final arm is explicitly
labeled as the complete donor control.

The final control is byte-identical to the full round-3 ladder rung. The
ladder copy loaded and reached EXECUTE, while the same bytes timed out at LOAD
when run after three other morph loads in one recovery-kernel session. That
contradiction rules out an object-level task-split conclusion from arm 04;
the different result is sequencing or retained runtime state.

## Stem EXECUTE audit

After LOAD was fixed, every residual prefix hung at EXECUTE, including the
three-task stem/probe object. Its first task differed from the accepted main
stem in five named groups only:

| Group | Rule-builder defect | Round-4 action |
| --- | --- | --- |
| Header hints | The generic MAC estimate and a 256-KiB staging extent produced the wrong execution-cost hint and marked four extra L2 banks live. | Use the sparse-stem phase cost and compute the live mask from the compact source and result extents. |
| Common schedule | The generic wide-spatial schedule selected three traversal groups. | The stem selects four issue and traversal groups for the seven-phase walk. |
| Source layout | The generic residual tuple described the external NCHW row inside L2 and placed it at a 64-KiB first-fit boundary. | Keep external DMA geometry unchanged, but use the compact three-plane phase tuple immediately after the guarded result extent. |
| Result layout | The generic resident traversal and 64-byte row skew were used. | Select the sparse-stem result traversal and its 32-byte row skew. |
| Result-span record | The generic packet explicitly wrote two complete-plane spans. | Omit them; the sparse phase engine derives both spans from geometry. |

With those rules, the complete packed first task—not just its decoded values—
is byte-identical to the accepted main stem task. The remaining two probe
tasks and CPU reference are unchanged.

`tools/h16g_stem_execute_morph.py` builds forward cumulative/single and mirror
cumulative/single sets over these groups. Value groups are patched only at
existing operand offsets. The differing result-span write set always comes
from one of the two compiler-built endpoint objects; the tool never edits a
record header or shifts a task by hand. It also requires a byte-exact accepted
stem-task match and runs the container and LOAD-grammar gates on every output.

## Resident probe-head boundary

A prefix ending before max-pool leaves the stem result resident in the linked
L2 layout. The reduction head must consume that resident base and the exact
row stride published by the producer. Treating every prefix boundary as a
scratch spill instead emits a source TileDMA for data which was never written
to scratch; LOAD accepts that framing, but EXECUTE cannot make progress.

The terminal linear task likewise uses the compact resident source/result
record set. Generic plane-span writes and the scratch-interleaved terminal
format are not valid substitutes. Qualification therefore separates the
stem, accepted full object, resident stem/head, old scratch-reading head, and
an external-input head into independent objects before running the prefix
ladder.

## Stem and max-pool materialization

A terminal stem still needs explicit output-DMA row and plane strides. The
direct terminal format does not infer the 112x112 surface geometry from the
output binding; omitting those fields can complete while leaving nearly the
entire surface untouched.

The max-pool consumes the stem's resident sparse-phase tuple without changing
its 3x3 window, stride-two SAME padding, or source layout. Its result starts
immediately after the stem result rather than at the next allocator bank. The
scratch spill is eight-channel blocked, so its DMA row is `16 * output_width`
bytes and its plane is that row times the output height. A compact planar row
at this internal boundary shifts channels and rows even though LOAD and
EXECUTE complete. Diagnostic boundary objects may instead terminate after
the stem, pool, or residual add; those objects materialize an ordinary planar
NCHW surface and are not used by production graphs.
