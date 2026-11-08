# H16G transformer-linear references

`make_refs.py` creates deterministic fp16 test vectors for the nine requested
BERT-base and Whisper tiny/base linear geometries. Each geometry has plain and
per-channel-bias cases; the three expansion linears also have fused exact GELU
cases, and the three model-width linears have an fp16 0.125 output-scale case
(`1/sqrt(64)`). That is 24 cases in total.

Run it with a Python environment that provides NumPy:

```sh
python3 tools/h16-linear/make_refs.py build/h16-linear-refs
```

Each case contains logical contiguous `input.bin`, row-major `[Cout,Cin]`
`weights.bin`, optional `bias.bin`, and `expected.bin`. `manifest.json` records
shapes, byte counts, and SHA-256 checksums. Expected values use float32 CPU
accumulation followed by an fp16 store; GELU uses the exact erf definition.

The binaries deliberately contain no physical allocation padding. Pair an
emitted program with the generic Linux runner as:

```sh
python3 ane-run.py --hwx program-0.hwx \
  --case input.bin:expected.bin --runs 20
```

The HWX manifest owns the allocation and stride padding. The reference file
owns only the logical result span compared by the runner.
