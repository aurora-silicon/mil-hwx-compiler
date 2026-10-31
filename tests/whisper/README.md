# Public Whisper compiler fixtures

`compiler-coverage.json` is copied byte-for-byte from
[aurora-silicon/whisper.cpp](https://github.com/aurora-silicon/whisper.cpp/blob/e65f3c4423baae5898e40458d8cb1d192897172e/docs/ane/compiler-coverage.json).
The sixty `*.mil` probes are the public-model isolated operations used by that
receipt, produced by its MIT `models/ane/export.py` and `compile.py` workflow.
They contain MIL declarations and references into public open-weight blobs,
not compiled ANE payloads. `probes.json` pins each probe and the source receipt.

OpenAI Whisper and whisper.cpp are MIT licensed. MIL definitions and
coremltools are BSD licensed; no Core ML runtime or Apple compiler is called.
The receipt records the public checkpoint, converted ggml, MIL and fp16 weight
hashes. No weight blobs or native binaries are added as fixtures here.

See [the qualification guide](../../docs/whisper-qualification.md) for replay,
CPU reference scope, budget behavior and the remaining hardware refusals.
