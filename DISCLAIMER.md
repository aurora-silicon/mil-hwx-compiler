# Project scope

This repository is independent research into the Apple Neural Engine. It is
not affiliated with, endorsed by, or supported by Apple Inc.

## Limits

- It is not a replacement for Core ML, MLX, or Apple's ANE compiler.
- It is not a production SDK or a supported platform API.
- It is not intended for safety-critical, security-critical, or real-time use.
- It is not compatible with App Store distribution in its current form.
- It does not promise support across macOS releases or Apple chip generations.

The tested runtime uses private Apple interfaces. Those interfaces can change
or disappear without notice. The compiler currently covers measured H16G
operation and geometry families on the M4 configuration named in
`docs/VERIFICATION.md`. Unsupported inputs fail closed.

## Copyright and provenance

The MIT License covers original source code, tests, documentation, and data
authored for this project. It cannot grant rights in third-party material.

The repository does not include Apple-compiled HWX containers or
Apple-compiled container skeletons. `research/oracles/` does contain decoded
Task Descriptor words, section sizes and hashes recorded from compiling this
project's own MIL programs with generated weights; the `plugins/*/*Templates.inc`
tables are generated from those decoded words. Regression tests store expected
hashes and decoded field values rather than the corresponding Apple binary
output.

`plugins/H16G/Encoding/*EncoderData.inc` contains measurements recorded from
compiler output for specific operations and geometries. These tables describe
observed interface behavior; users remain responsible for determining whether
their use and distribution are permitted where they live.

No Apple source code was available or used. The production compiler source is
an independent implementation. It does not call `ANECCompile` or copy an Apple
container at compile time.

Apple, Apple Neural Engine, Core ML, and macOS are trademarks of Apple Inc.

## Legal context

The project was developed by studying lawfully available software and hardware
for interoperability. *Atari Games Corp. v. Nintendo of America Inc.*, 975
F.2d 832 (Fed. Cir. 1992), is one of the United States cases that discusses
reverse engineering used to understand unprotected functional elements.

That case does not decide whether every use of this project is lawful. The MIT
License covers this repository's original work. It does not grant rights to
Apple software, services, trademarks, or confidential material. This is
general project context, not legal advice.

## Warranty

The software is provided under the warranty disclaimer in `LICENSE`. Use of
private APIs, generated hardware programs, and system cache provisioning is at
the user's own risk.
