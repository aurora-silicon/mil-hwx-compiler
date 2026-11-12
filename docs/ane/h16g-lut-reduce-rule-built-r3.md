# H16G rule-built LUT/reduction round 3

Round 2 produced structurally valid one-task LUT and reduction programs, but
the first exp and axis-3 max cases both timed out at EXECUTE on a J616s. Their
old-main programs completed. The exp constant also changed the range-reduction
contract used by the device-proven S=1500 softmax program and caused a separate
accuracy regression.

## Hang hypothesis by field group

The comparison below is between our own old-main and rule-built compiler
outputs. It does not use or reproduce a vendor program.

| Field group | Hang likelihood and reasoning |
|---|---|
| Task size / record framing | Low. Size follows the emitted record count. The morph kit switches framing explicitly because added/removed writes still need isolation. |
| Execution-cycle hint | Low. Both failing one-task pairs keep the same non-semantic hint. |
| Output/record format | Low. Final-output and record-format fields agree for the one-task pairs. |
| Dispatch/dependency class and mask | Low. A single task has no predecessor; the compared dependency words agree. |
| Engine control | Medium. The reduction changes its enabled-engine count, although exp does not, so this cannot alone explain both hangs. |
| Constant region | Accuracy-only for exp. Constants are not task scheduling fields, and reduction has no table. The old exp clamp/tail contract is nevertheless required by softmax and other proven consumers. |
| Store | Low. The compared one-task Store fields agree. |
| KernelDMA | Low-to-medium. It can stall exp table loading, but reduction has no kernel table, so it is not a shared cause. |
| SourceDMA | Medium. Wrong extents or strides can prevent source completion, but the differences are not the same across the two operations. |
| NE | Medium. Opcode and lane-enable differences can prevent engine completion, but the LUT and reduction commands are necessarily different. |
| L2 | High. Both rule-built streams change transfer extents/strides and add output geometry; an invalid resident walk could leave a completion dependency unsatisfied. |
| Common | High. Tile counts and allocation selectors differ substantially and feed several engines. |
| ResultDMA | Highest. Round 2 cleared ResultDMA's named engine-enable bit in both failing streams while Store waited for its completion event. SourceDMA is armed by a separate flags word; ResultDMA is not. |

Round 3 therefore makes ResultDMA enable explicit in `resultEndpoint()` and
uses it in both constructors. This is a best host-side fix, not device proof.
The reviewer should run the morph cases one at a time, smallest/ordered first,
and reboot after every LOAD or EXECUTE failure.

## Constant compatibility

Standalone rule-built LUT operations retain their direct interpolation table.
Composed consumers use a separate compatibility path: exp names and constructs
its input clamp low/high and output below/above-range values, while reciprocal
constructs its reduced `2/x` samples and clamp/tail values. The H=8 S=1500
softmax host test pins the complete object to current main's device-proven
SHA-256; H=6 must remain byte-identical to H=8.

## Private kit shape

`tools/build_h16g_lut_reduce_morph_kit.py` consumes only two compiler outputs.
For exp 128x128 and reduce-max C1x64x64 axis 3 it emits exact old/rule-built
endpoints plus cumulative and single-group morphs. A union record frame makes
added and removed register writes explicit. The order is header/framing,
constant, Store, Other, KernelDMA, SourceDMA, NE, L2, Common, ResultDMA, so the
strongest shared hang candidate is tested last.
