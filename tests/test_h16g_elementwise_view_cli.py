#!/usr/bin/env python3
"""H16G staged elementwise forms on non-square tensors.

A standalone binary ALU uses its qualified square family. A table-unary
operation uses a rule-built [1,1,N,N] packet with N rounded to 64 lanes. The
emitted object must match an explicit square program, and the manifest must
describe the tensor's logical bytes inside the square allocation.
"""
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile

COMPILER = str(Path(sys.argv[1]).resolve())
VALIDATOR = str(Path(sys.argv[2]).resolve()) if len(sys.argv) > 2 else None
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


def section(image, segment_name, section_name):
    """Return one Mach-O-like HWX section without depending on research code."""
    _, _, _, _, count, _, _, _ = struct.unpack_from('<8I', image)
    cursor = 32
    for _ in range(count):
        command, size = struct.unpack_from('<2I', image, cursor)
        if command == 0x19:
            fields = struct.unpack_from('<2I16s4Q4I', image, cursor)
            segment = fields[2].split(b'\0', 1)[0].decode()
            section_cursor = cursor + 72
            for _ in range(fields[-2]):
                entry = struct.unpack_from('<16s16s2Q8I', image, section_cursor)
                name = entry[0].split(b'\0', 1)[0].decode()
                if segment == segment_name and name == section_name:
                    return image[entry[4]:entry[4] + entry[3]]
                section_cursor += 80
        cursor += size
    return None


def same_program_payload(left, right):
    """Ignore object metadata while retaining descriptor/table parity.

    The staged assembler and direct contiguous-view route use different public
    symbol names at the exact 128/256 unary geometries, but the task stream,
    constant allocation, and kernel table must still be byte-identical.
    """
    names = (('__TEXT', '__text'), ('__TEXT', '__const'),
             ('__KERN_0', '__kern_0'))
    return all(section(left, *name) == section(right, *name) for name in names)


def main():
    failures = []
    cases = 0
    # (builder, op, logical shape, expected square size)
    accepted = [
        (binary, 'add', [1, 128, 768], 512),       # BERT-base residual
        (binary, 'mul', [1, 1500, 384], 1024),     # Whisper-tiny width
        (binary, 'max', [64, 64], 128),
        (binary, 'add', [1, 64, 4, 192], 256),
        (binary, 'add', [1, 64, 1024], 256),       # held-out transformer width
        (binary, 'add', [1, 256, 1024], 512),      # held-out token count
        (binary, 'add', [1, 512, 1024], 1024),     # held-out token count
        (binary, 'add', [1, 1, 512, 512], 512),    # already square: unchanged
        (unary, 'gelu', [1, 128, 3072], 640),      # BERT-base FFN activation
        (unary, 'gelu', [1, 1500, 2048], 1792),    # Whisper-base FFN activation
        (unary, 'exp', [1, 12, 128, 128], 448),
        (unary, 'sigmoid', [100], 128),
        (unary, 'gelu', [1, 64, 1024], 256),       # held-out transformer width
        (unary, 'gelu', [1, 256, 1024], 512),      # held-out token count
        (unary, 'gelu', [1, 512, 1024], 768),      # held-out token count
        (unary, 'gelu', [1, 1, 256, 256], 256),
        (unary, 'log', [1, 128, 768], 320),
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
            if VALIDATOR:
                validation = subprocess.run([
                    sys.executable, VALIDATOR, str(out / 'program-0.hwx'),
                    '--manifest', str(out / 'manifest.json')],
                    capture_output=True, text=True, timeout=30)
                check(failures, validation.returncode == 0,
                      f'{name}: validation refused: '
                      f'{(validation.stderr or validation.stdout).strip()}')
            square = [1, 1, size, size]
            ref_result, ref = compile_case(root, f'{name}_square', build(op, square))
            if ref_result.returncode:
                failures.append(f'{name}: square reference refused: {ref_result.stderr.strip()}')
                continue
            check(failures, same_program_payload(
                  (out / 'program-0.hwx').read_bytes(),
                  (ref / 'program-0.hwx').read_bytes()),
                  f'{name}: descriptor/table payload differs from the square '
                  f'[1,1,{size},{size}] program')
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
