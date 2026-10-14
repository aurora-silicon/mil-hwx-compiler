#!/usr/bin/env python3
"""The scaled exp chain compiles to ONE program only when the MIL is exactly the decoded form, and is refused otherwise.

The decoded form is exp, mul by fp16 0x1b18 (16 K), add fp16 16, mul, mul by fp16 0.0625 over a [1, C, 1, 1] fp16 input with C in {64, 512}. test_h14_parity.py
proves the emitted program equals Apple's oracle word for word. This test proves the matcher: a changed constant, a changed order, an unmatched channel count, a
different operation, and a second consumer of the input each leave the one-program path (the compiler refuses the chain), and both schedules give the one program.
Usage: tests/test_h14_exp_chain_cli.py [path-to-mil-hwxc]"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()


def mil(channels=512, k="0x1.c6p-9", sixteen="0x1p+4", scale="0x1p-4", body=None):
    kind = f"tensor<fp16, [1, {channels}, 1, 1]>"
    lines = body or [
        f'{kind} e = exp(x = x)[name = string("e")];',
        f'{kind} c = mul(x = x, y = k16)[name = string("c")];',
        f'{kind} g = add(x = c, y = sixteen)[name = string("g")];',
        f'{kind} p = mul(x = e, y = g)[name = string("p")];',
        f'{kind} y = mul(x = p, y = sixteenth)[name = string("y")];',
    ]
    consts = [f'fp16 k16 = const()[name = string("k16"), val = fp16({k})];',
              f'fp16 sixteen = const()[name = string("sixteen"), val = fp16({sixteen})];',
              f'fp16 sixteenth = const()[name = string("sixteenth"), val = fp16({scale})];']
    text = "\n".join(f"    {line}" for line in consts + lines)
    return (f"program(1.3)\n[buildInfo = dict<string, string>({{}})]\n{{\n  func main<ios18>({kind} x) {{\n{text}\n  }} -> (y);\n}}\n")


def compile_text(text, schedule="per-op", artifact_format="anec"):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "op.mil").write_text(text)
        (root / "model").mkdir()
        out = root / "out"
        result = subprocess.run([str(COMPILER), "--mil", str(root / "op.mil"), "--model-root", str(root / "model"), "--target", "H14", "--format", artifact_format,
                                 "--schedule", schedule, "--output", str(out)], capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
        manifest = json.loads((out / "manifest.json").read_text()) if result.returncode == 0 else None
        return result.returncode, manifest


def main():
    for channels in (64, 512):
        for schedule in ("per-op", "chain"):
            for artifact_format in ("anec", "hwx"):
                rc, manifest = compile_text(mil(channels), schedule, artifact_format)
                assert rc == 0, f"c{channels} {schedule} {artifact_format}: refused"
                programs = manifest["programs"]
                assert len(programs) == 1 and programs[0]["encoder"] == "apple-parity-expchain" and programs[0]["taskDescriptors"] == 3, (channels, schedule, programs)
                assert len(programs[0]["inputs"]) == 1 and len(programs[0]["outputs"]) == 1
    kind = "tensor<fp16, [1, 512, 1, 1]>"
    refused = {
        "a different 16 K": mil(k="0x1.c8p-9"),
        "a different 16": mil(sixteen="0x1p+3"),
        "a different last scale": mil(scale="0x1p-3"),
        "an unmatched channel count": mil(channels=128),
        "mul and add swapped": mil(body=[
            f'{kind} e = exp(x = x)[name = string("e")];', f'{kind} c = add(x = x, y = k16)[name = string("c")];',
            f'{kind} g = mul(x = c, y = sixteen)[name = string("g")];', f'{kind} p = mul(x = e, y = g)[name = string("p")];',
            f'{kind} y = mul(x = p, y = sixteenth)[name = string("y")];']),
        "the product reads the input, not exp": mil(body=[
            f'{kind} e = exp(x = x)[name = string("e")];', f'{kind} c = mul(x = x, y = k16)[name = string("c")];',
            f'{kind} g = add(x = c, y = sixteen)[name = string("g")];', f'{kind} p = mul(x = x, y = g)[name = string("p")];',
            f'{kind} y = mul(x = p, y = sixteenth)[name = string("y")];']),
        "the product multiplies exp by c, not g": mil(body=[
            f'{kind} e = exp(x = x)[name = string("e")];', f'{kind} c = mul(x = x, y = k16)[name = string("c")];',
            f'{kind} g = add(x = c, y = sixteen)[name = string("g")];', f'{kind} p = mul(x = e, y = c)[name = string("p")];',
            f'{kind} y = mul(x = p, y = sixteenth)[name = string("y")];']),
        "sigmoid instead of exp": mil().replace("exp(x = x)", "sigmoid(x = x)"),
    }
    for label, text in refused.items():
        for schedule in ("per-op", "chain"):
            rc, manifest = compile_text(text, schedule)
            one_program = rc == 0 and len(manifest["programs"]) == 1 and manifest["programs"][0]["encoder"] == "apple-parity-expchain"
            assert not one_program, f"{label} ({schedule}) still compiled to the scaled exp chain program"
    print("scaled exp chain: one program for the decoded form (64 and 512 channels, both schedules and formats); "
          f"{len(refused)} changed forms leave the one-program path")


if __name__ == "__main__":
    main()
