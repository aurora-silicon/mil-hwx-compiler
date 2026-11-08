#!/usr/bin/env python3
"""H16G staged elementwise forms on non-square tensors.

A standalone binary ALU or table-unary operation on contiguous external fp16
tensors runs as the measured square [1,1,N,N] program over the same bytes,
with N the smallest measured size whose N*N covers the tensor. The emitted
object must be byte-identical to the square program, and the manifest must
describe the tensor's logical bytes inside the square allocation.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

COMPILER = str(Path(sys.argv[1]).resolve())
HEADER = 'program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'


def ttype(shape):
    return f'tensor<fp16, [{", ".join(str(d) for d in shape)}]>'


def binary(op, shape):
    t = ttype(shape)
    return (f'{HEADER}  func main<ios18>({t} x, {t} y) {{\n'
            f'    {t} z = {op}(x = x, y = y)[name = string("z")];\n  }} -> (z);\n}}\n')


def unary(op, shape):
    t = ttype(shape)
    extra = ', mode = string("EXACT")' if op == 'gelu' else ''
    return (f'{HEADER}  func main<ios18>({t} x) {{\n'
            f'    {t} z = {op}(x = x{extra})[name = string("z")];\n  }} -> (z);\n}}\n')


def compile_case(root, name, mil):
    source = root / f'{name}.mil'
    source.write_text(mil)
    out = root / name
    result = subprocess.run([COMPILER, '--mil', str(source), '--model-root', str(root),
                             '--output', str(out)], capture_output=True, text=True, timeout=120)
    return result, out


def elements(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def check(failures, cond, message):
    if not cond:
        failures.append(message)


def main():
    failures = []
    cases = 0
    # (builder, op, logical shape, expected square size)
    accepted = [
        (binary, 'add', [1, 128, 768], 512),       # BERT-base residual
        (binary, 'mul', [1, 1500, 384], 1024),     # Whisper-tiny width
        (binary, 'max', [64, 64], 128),
        (binary, 'add', [1, 64, 4, 192], 256),
        (binary, 'add', [1, 1, 512, 512], 512),    # already square: unchanged
        (unary, 'gelu', [1, 128, 3072], 1024),     # BERT-base FFN activation
        (unary, 'gelu', [1, 1500, 2048], 2048),    # Whisper-base FFN activation
        (unary, 'exp', [1, 12, 128, 128], 512),
        (unary, 'sigmoid', [100], 128),
        (unary, 'gelu', [1, 1, 256, 256], 256),
    ]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for build, op, shape, size in accepted:
            name = f'{op}_{"x".join(map(str, shape))}'
            result, out = compile_case(root, name, build(op, shape))
            cases += 1
            if result.returncode:
                failures.append(f'{name}: refused: {result.stderr.strip()}')
                continue
            square = [1, 1, size, size]
            ref_result, ref = compile_case(root, f'{name}_square', build(op, square))
            if ref_result.returncode:
                failures.append(f'{name}: square reference refused: {ref_result.stderr.strip()}')
                continue
            check(failures, (out / 'program-0.hwx').read_bytes() ==
                  (ref / 'program-0.hwx').read_bytes(),
                  f'{name}: object differs from the square [1,1,{size},{size}] program')
            artifact = json.loads((out / 'manifest.json').read_text())['artifacts']
            check(failures, len(artifact) == 1, f'{name}: expected one program')
            for binding in artifact[0]['bindings']:
                check(failures, binding['logicalBytes'] == 2 * elements(shape),
                      f'{name}: {binding["identifier"]} logical bytes {binding["logicalBytes"]}')
                check(failures, binding['allocationBytes'] >= 2 * size * size and
                      binding['allocationBytes'] % 0x4000 == 0,
                      f'{name}: {binding["identifier"]} allocation {binding["allocationBytes"]}')
                check(failures, binding['rowStrideBytes'] == 2 * size,
                      f'{name}: {binding["identifier"]} row stride {binding["rowStrideBytes"]}')

        refused = [
            ('too_large', binary('add', [1, 2049, 2048])),            # > 2048*2048 lanes
            ('shape_mismatch', f'{HEADER}  func main<ios18>({ttype([1, 128, 768])} x, '
             f'{ttype([1, 768, 128])} y) {{\n    {ttype([1, 128, 768])} z = add(x = x, y = y)'
             '[name = string("z")];\n  } -> (z);\n}\n'),
            ('log_view', unary('log', [1, 128, 768])),               # log: [1,1,256,256] only
            ('rank5', unary('gelu', [1, 1, 2, 64, 64])),
        ]
        for name, mil in refused:
            result, _ = compile_case(root, name, mil)
            cases += 1
            check(failures, result.returncode != 0, f'{name}: accepted, expected a named refusal')
            check(failures, result.returncode == 0 or 'error [h16g.' in result.stderr,
                  f'{name}: refusal without an h16g code: {result.stderr.strip()}')
    for failure in failures:
        print('FAIL', failure)
    print(f'H16G elementwise views: {cases} cases, {len(failures)} failures')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
