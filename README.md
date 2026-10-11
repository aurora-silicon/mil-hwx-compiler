# mil-hwx-compiler

`mil-hwxc` compiles textual MIL into programs for the Apple Neural Engine.
MIL is Core ML's intermediate language.
The compiler writes H13 (M1) and H14 (M2) ANEC packages, and it writes H16G (M4) HWX objects.
It never calls Apple's compiler or any Apple tool.

## What it does and who it is for

macOS compiles models onto the ANE with private tools.
Those tools do not exist on Linux.
This compiler is the Linux side of that pipeline.
It reads the MIL that Core ML tools emit.
It writes packages for [omarchy-ane](https://github.com/joshuaswarren/omarchy-ane), the open ANE driver stack, to load and run.
On an M1 running Omarchy, H13 packages from this compiler run on the ANE.

People who study ANE program formats are the other audience.
Each backend matches decoded Apple-compiler output byte for byte.
[docs/ane/README.md](docs/ane/README.md) records what that decoding work found.

Inputs, outputs, and weights are fp16 with static shapes.
The supported operations are add, mul, sub, real_div, relu, clip, conv, matmul, and linear.
Softmax and layer_norm are supported, and so are the three reduce ops.
reshape, squeeze, and expand_dims round out the set.
A shape outside a backend's verified envelope is refused with a named error, such as `h13.conv-outside-envelope`.
The compiler does not guess.

## Supported targets

| Target | Chip | Output | Device runs |
| --- | --- | --- | --- |
| H13 | M1 | ANEC package, or HWX with `--format hwx` | Runs on Linux M1 hardware through omarchy-ane. Eight elementwise programs built by this compiler ran on a 13-inch M1 and an M1 Max through omarchy-ane. Each was bit-exact on 64 of 64 lanes ([receipt](receipts/2026-10-11-h13-elementwise-m1/)). |
| H14 | M2 | ANEC package, or HWX with `--format hwx` | Matches decoded Apple output byte for byte. Eight elementwise programs built by this compiler ran on an M2 Max through omarchy-ane. Each was bit-exact on 512 of 512 lanes ([receipt](receipts/2026-10-10-h14-elementwise-m2/README.md)). Matvec programs at K = N of 256, 512 and 1024 also ran on an M2 Max, every lane inside a 2 ulp or 4 condition-unit band ([receipt](receipts/2026-10-11-h14-matvec-m2/README.md)). Other larger H14 programs have not run from this backend |
| H16G | M4 | HWX object | Research compiler. Exercised by the macOS hardware tests in `tests/` |
| H17 | A18, A18 Pro and M5 | HWX object, oracle-parity targets | Emits Apple's exact HWX for 165 decoded elementwise, unary and scalar-constant points and refuses every other shape by name. Byte parity only; not run on a device |
| H18 | A19 and A19 Pro | HWX object, oracle-parity targets | Same as H17 for H18: 165 decoded points, named refusals, byte parity only; not run on a device |

## Install

### Omarchy package

```sh
omarchy pkg add mil-hwx-compiler
```

The package builds from the release tag.
It installs the compiler as `/usr/bin/mil-hwxc`.

### From source on Linux

Install clang, clang++, lld, cmake, ninja, make, pkg-config, git, and python3.
Add the dev headers for libffi, libxml2, ICU, OpenSSL, and zlib.
Then run:

```sh
git clone https://github.com/joshuaswarren/mil-hwx-compiler.git
cd mil-hwx-compiler
scripts/verify-linux-compiler.sh all
```

The first run builds pinned libobjc2 and GNUstep Base into `~/.local/mil-hwx-gnustep`.
That run takes longer.
Nothing needs root.
Set `GNUSTEP_PREFIX` to put the toolchain somewhere else.
The script builds the compiler and runs the software tests.
It also emits sample HWX files for inspection.

### From source on macOS

Install the Xcode Command Line Tools with `xcode-select --install`.
Then run:

```sh
make build/mil-hwxc    # build the compiler
make test              # run the full macOS suite
```

## Quick start

Write a small MIL program, and compile it for the M1:

```sh
cat > /tmp/add.mil <<'MIL'
program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1,64,1,1]> a, tensor<fp16, [1,64,1,1]> b) {
    tensor<fp16, [1,64,1,1]> y = add(x = a, y = b)[name = string("sum")];
  } -> (y);
}
MIL
mkdir -p build/add
./build/mil-hwxc --target H13 --mil /tmp/add.mil --model-root /tmp --output build/add
```

`build/add` now holds `manifest.json` and one ANEC program, `program-0.anec`.
You can compile the checked-in conv and ReLU fixture the same way.
That path gives you an H16G HWX object:

```sh
./build/mil-hwxc --mil tests/fixtures/conv_relu.mil \
  --model-root tests/models/conv_relu --output build/conv-hwx
```

That writes `build/conv-hwx/program-0.hwx`.
H16G is the default target, so the flag is optional there.

## Usage

```text
usage: mil-hwxc --mil FILE --model-root DIR --output DIR
                [--target H16G|H13|H14|H17|H18] [--format anec|hwx]
                [--schedule per-op|chain]
```

`--target` picks the backend.
The default is H16G.
H13 and H14 write ANEC packages unless `--format hwx` says otherwise.
`--schedule` applies to H13 and H14.
The default, `per-op`, emits one program per operation.
`chain` fuses a supported producer into a following `relu`.
The pair then becomes one program with no intermediate buffer.
`--model-root` is the directory where `BLOBFILE` constants resolve.
For the fixtures, that is their `tests/models` directory.
The output directory must be absent or empty.

Each run writes `manifest.json` plus one `program-N.anec` or `program-N.hwx` per program.
The manifest lists each tensor's shape, byte count, and role.
Roles are input, output, intermediate, or constant.
It also lists the dispatch plan and each program's bindings.
A runner can drive the package without parsing programs.

| Operation | Accepted shapes |
| --- | --- |
| add, mul | Two matching fp16 tensors, or a tensor plus a fp16 constant. Rank-4 NCHW pairs also broadcast per channel, by space, by scalar, or by batch |
| sub, real_div | One runtime input and one same-shape constant. The host folds the constant |
| relu, clip | Any positive static shape. Clip takes fp16-exact finite `alpha` and `beta` |
| conv | Rank-4 NCHW. Kernels 1 and 3, and stride 1 or 2. Groups 1, 4, or depthwise. At stride 1: channels 64 to 1024, space 1 to 64 |
| matmul, linear | Constant weight `[K,N]` or `[N,K]`, or two runtime inputs in attention's shape, on the decoded grids |
| softmax | Constant axis over covered flat, sequence, spatial, and attention-score shapes |
| layer_norm | Constant axes, `epsilon` of 1e-5, no `gamma` or `beta` |
| reduce_sum, reduce_max, reduce_mean | Constant axes, `keep_dims` either way, never over the batch axis |
| reshape, squeeze, expand_dims | Constant shapes with equal element counts. Emits no program |

Anything else fails with a named `*-outside-envelope` error.
The decoded grids that define each envelope live in `plugins/H13/` and `plugins/H14/`.
The parity suites in `tests/` pin them.

### Running on an M1 under Linux

omarchy-ane provides the ANE kernel module and the libane library.
`tools/h13_run_linux.py` checks a package against the MIL and dense fp16 inputs.
Then it runs the package:

```sh
python3 tools/h13_run_linux.py build/add --mil /tmp/add.mil --model-root /tmp \
  --input a=a.fp16 --input b=b.fp16 --output y=sum.fp16 --dry-run
```

`--dry-run` runs every check and prints the dispatch plan.
It does not touch the device.
Native runs drop that flag and add `--deadline-seconds`, which bounds the wait per dispatch.
`--benchmark-json PATH` adds a timing report gated on correct outputs.
`tools/h13_reference.py` evaluates the accepted MIL subset on the host.
You can use it to make reference outputs without a device.

### Tests

| Command | What it checks |
| --- | --- |
| `make test-h13` | H13 encoding, serialization, and the CLI, host-only |
| `make test-h13-parity` | H13 output byte-equal to decoded Apple oracles. 891 cases at v0.1.0 |
| `make test-h14-parity` | H14 output byte-equal to decoded Apple oracles. 718 cases at v0.1.0 |
| `make test-h17-parity`, `make test-h18-parity` | H17/H18 elementwise, unary and scalar-constant HWX byte-equal to decoded Apple oracles, plus named refusals. 165 cases each. Not device-run |
| `scripts/verify-linux-compiler.sh all` | Build, software tests, emission, and hygiene on Linux |
| `make test` | The full macOS suite |

## How it works

The compiler parses MIL into a graph.
It then simplifies the graph in a few passes.
A per-target backend packs each op into the ANE's task descriptors and constant sections.
It writes the programs into an ANEC container or an HWX object.

Apple does not document these formats.
The layouts come from decoding programs that Apple's own compiler produced.
The parity suites recompile every decoded oracle in the corpus.
They require task words, tensor descriptors, and constant sections to match byte for byte.
That is the only correctness bar for formats with no spec.
Geometry the corpus does not cover stays unimplemented.

## Troubleshooting

The Linux bootstrap prints a missing build tool or dev package by name.
Install it, then rerun `scripts/verify-linux-compiler.sh`.

On Linux, run the suites through the verify script.
Parts of the default `make` target need Apple frameworks such as CommonCrypto.
Those exist only on macOS.

A refusal like `h13.matmul-outside-envelope` is a scope statement, not a bug.
The shape falls outside the verified envelope for that operation.

## Contributing

Issues and pull requests are welcome.
Before you submit, run `scripts/verify-linux-compiler.sh all` on Linux.
On macOS, run `make test`.
Cover any new envelope with a parity case.
[docs/ane/README.md](docs/ane/README.md) is the project's ANE knowledge base.
[DISCLAIMER.md](DISCLAIMER.md) states the project's scope.

## License

MIT.
See [LICENSE](LICENSE).
