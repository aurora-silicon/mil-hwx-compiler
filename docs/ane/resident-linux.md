# Resident H13/H14 graphs on Linux

`tools/h13_run_linux.py` now exposes `ResidentGraph` alongside its existing
qualification and benchmark commands. It uses libane from the matching
`omarchy-ane` change that exports `ane_init_target`, `ane_send_checked` and
`ane_read_checked`. Older libraries are refused with a missing-symbol error.
The command name stays compatible with existing H13 callers.

```python
from h13_run_linux import LibANEAdapter, ResidentGraph

# These paths describe one source-compiled package and its original MIL/weights.
# inputs/outputs map tensor names to dense fp16 file paths, as in the CLI.
adapter = LibANEAdapter(library, device=0)
with ResidentGraph(package, mil, model_root, inputs, outputs, adapter,
                   deadline_seconds=60) as native:
    result = native({name: path.read_bytes() for name, path in inputs.items()})
    # Later calls may supply different input values, with the same shapes.
```

The deadline covers the owner's entire device phase, including preparation and
all calls. It is checked before each program and cannot interrupt an in-flight
kernel request. The caller serializes calls and close on a single owner. Use
separate owners for independent streams, subject to the existing device owner,
resource and qualification gates. Constructing this object grants no permission
to access hardware. Keep the existing hardware preflight and failure recovery
procedure around it.

The package inspector checks the complete graph before loading any handle.
Prepared handles receive the exact program bytes the inspector validated, so
later file changes cannot substitute an unvalidated program. Physical buffer
sizes and counts must agree with libane before any transfer. All programs and
host buffers are prepared once. Each call composes fresh intermediates, reuses
the prepared handles, unpacks the final outputs and checks them against the
existing MIL reference. The reference runs per call and is part of call latency;
use the existing benchmark command for separately scoped device timings.

Any preparation, transfer, submission, deadline or correctness failure closes
all owned handles. A failed owner never submits again. Close is idempotent;
cleanup errors preserve the original failure as well as the cleanup failure.
Numerically failed outputs are retained as `<tensor>.failed.fp16` in the package
directory, following the existing qualification runner's convention. Preserve
that directory and the caller's logs before recovery. Read-only package storage
can prevent writing that diagnostic; it still retires and closes the owner.

## Generation and package validation

H13 keeps `mil-hwxc.h13-anec-package.v2` and all existing task, binding, graph,
scratch, dependency and result metadata checks. H14 accepts
`mil-hwxc.h14-anec-package.v1`, per-operation elementwise/unary/scalar programs
with encoder `h14-oracle-parity`, whole fp16 surfaces and no scratch. Multi-program
H14 graphs use the same dependency, layout and intermediate composition checks.
The H14 path reuses `research/h13_td.py`'s aligned H14 parser and decoder,
including scatter records and extended headers. It does not reinterpret H14
streams as H13. Other H14 encoders, composed chains and future targets fail
closed. Extending those encoders requires their own shape and routing checks.

The expected family comes from the validated package target. libane checks
GET_CAPS on each actual load fd before any BO allocation or firmware program
load: ABI 1/family 13 for H13, ABI 2/family 14 for H14. A missing capability ioctl,
short result, ABI disagreement or different family is refused. There is no
family override or legacy fallback. Existing libane callers retain their old
entry points; the new graph path opts into this stricter contract.

The source contracts are `omarchy-ane` base
`1adf2ee392ef7d3d64a24743d293dbc1670f0e8b` and this compiler's base
`8f8fb25f58e05a27db4c503e8eea7d4f13be4219`, specifically
`plugins/H14/H14Program.cpp`, `ANEH14Compiler.mm`, `research/h13_td.py` and
`research/inspect_anec.py`. Host parser tests and fake device ioctls establish
software behavior only. They establish no device correctness or performance.

## Existing CPU fallback

`ResidentGraph` is a callable suitable for the existing `ane.Fallback` native
callback. Its contract here is a mapping of names to dense fp16 byte buffers;
provide a CPU callback with that same contract and the same original weights.
The reference evaluator is a simple open CPU implementation for supported MIL:

```python
from ane import Fallback
from h13_reference import evaluate

cpu = lambda request: evaluate(mil_text, model_root, request)
# native is an already prepared ResidentGraph, owned by the context above.
model = Fallback(cpu, native=native, library=library)
result, call = model.run(dense_inputs)
```

Use `native_factory` when native construction can fail. The existing fallback
records that failure and runs CPU without reopening the native owner. Keep any
successfully constructed owner's context alive around all calls, and close it
on application exit. The fallback does not own native resources. After a native
call fails, the graph closes every handle and the fallback permanently retires
its native callback. CPU recomputes the whole graph from the original request;
no native intermediate or output is returned as a successful result. The
existing fallback counters continue to describe successful native/CPU calls.
The ONNX adapter can also be used with the caller's existing array/byte adapter;
this change adds no second CPU engine or model loader.

## Host validation

Build libane and its Python binding from the paired change, then build this
compiler with its existing Linux toolchain procedure. Set
`H13_LIBANE_LIBRARY` to the built `libane_python.so`, and run:

```sh
python3 tests/test_h13_linux_runtime.py
python3 tests/test_resident_graph.py -v
```

The resident tests compile open two-program add/mul packages for both targets.
A host adapter emulates arithmetic and records handle ownership. They test
changed inputs, handle reuse, partial-load cleanup, failure retirement, CPU
fallback, rejected metadata/streams and preserved numerical failures. No test
accesses an ANE. Run the paired libane ioctl and fallback harnesses as well.
