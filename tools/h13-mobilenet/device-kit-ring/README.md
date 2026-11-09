# H13 linked-ring device kit

Build the complete ANECOMP ladder on the host:

```sh
MILC=build/mil-hwxc python3 tools/h13-mobilenet/ring_kit.py out/h13-ring
```

The output contains five native stride-2 single-task cases, all 28 prefixes,
the full native-frame MobileNet program, and deterministic float64 references.
The native k3/s2 references use TensorFlow SAME padding, with the extra row
and column at bottom/right for even input extents.
Generation fails unless all 34 ANEC files pass the same TileDMA extent versus
channel-allocation check used by libane's `ane_bind_overrun`.
The runner starts with the smallest depthwise case and grows through the stem
before attempting prefixes and the full network.
Copy it with `tools/h13-mobilenet` and a libane-built `anechain` to the M1 Pro,
then run `TOOLS=/path/to/tools/h13-mobilenet out/h13-ring/run.sh`.

The script never controls hardware itself.  It is an offline kit for
ANECOMP's controller-owned M1 Pro ladder.
