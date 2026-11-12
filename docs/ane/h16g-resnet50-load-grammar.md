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
