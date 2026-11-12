# H16G rule-built LUT/reduction round 4

Round 3's device matrix did not complete any rule-built unary or reduction
case. Its first morph was also not a no-op: the exp case returned zero and
issued reads at DVA zero, while the reduction case hung. This round fixes the
constructive unary defect and makes the next morph result attributable to one
named field group.

## DVA-zero cause

The old-main exp task uses the KernelDMA lookup stream at `0x5610`, its table
byte count at `0x5614`, and a relocated 64-bit table base at `0x5618`. The r3
morph started with the rule-built record frame, appended the old-only `0x5618`
write, but retained the rule-built relocation at `0x5510`. Consequently the
active KernelDMA table base remained zero after LOAD. Explicit zero writes for
other absent-side fields also changed the L2 and SourceDMA/ResultDMA state, so
none of the r3 single-group cases was isolated.

The rule-built LUT encoder had the same field-selection error: it treated the
`0x5510` compute-kernel operand used by other program families as the lookup
table base. Round 4 names the lookup stream separately and emits:

- `KernelDMA.command` at `0x5500`;
- `KernelDMA.lookup_stream` at `0x5610`;
- `KernelDMA.table_bytes` at `0x5614`;
- relocated `KernelDMA.table_base` at `0x5618`.

The structured encoder test requires all four named fields and requires the
single kernel relocation to point at `table_base`.

## Valid-by-construction morphs

The morph tool no longer builds a union map or assigns reset values to fields
that are absent on one side. It copies complete records from the selected
compiler output, including record flags and relocation ownership, and leaves
every unselected record byte-for-byte old-main.

`c01-framing-noop` and `s01-framing-noop` retain the old task header, field
presence, values, record bytes, BAR flags, and relocation target. Kit creation
fails unless their decoded header and register map equal `00`. On device, c01
must reproduce 00's output bit-exactly with no DART faults before a later
morph result is accepted.

The cumulative and single-group ladders then switch constant, Store, task
header, Other, KernelDMA, SourceDMA, NE, L2, Common, and ResultDMA. Six axis-2
rule-built programs plus three available old-main controls are marked
`execute: false`; those cases are LOAD/unload probes only.

## Host result and device handoff

All 38 r4 matrix programs and all 57 morph programs pass the independent H16G
container and DMA-extent validator. Host validation establishes container,
record, relocation, and binding consistency; the J616s run is still required
to establish completion and numerical accuracy. Run each 00/c01 pair first,
stop if c01 is not bit-exact or produces any DART fault, then run the remaining
morphs before the r4 matrix.
