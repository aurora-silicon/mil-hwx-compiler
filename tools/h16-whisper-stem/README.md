# H16G Whisper stem tools

`reference.py` generates deterministic CPU references for the two-convolution
Whisper audio stem. It evaluates convolution and exact GELU in float64, rounds
every hardware-visible edge to fp16, and emits both Numpy archives and the
64-byte-row-padded external surfaces consumed by the Linux ANE runner.

```sh
/path/to/python-with-numpy tools/h16-whisper-stem/reference.py build/whisper-stem-reference
```

The generated data is intentionally not checked in. The default output holds
Whisper-tiny (`d_model=384`) and Whisper-base (`d_model=512`) cases with:

- input `[1,80,3000]`;
- biased `1x3` SAME convolution at stride 1 and exact GELU;
- biased `1x3` SAME convolution at stride 2 and exact GELU;
- transpose `[1,d,1500] -> [1,1500,d]`;
- addition of a constant positional tensor `[1,1500,d]`.

The reference generator is public-source-only. H16G descriptor and packing
facts are documented separately; raw vendor-compiler objects never belong in
this directory or repository.
