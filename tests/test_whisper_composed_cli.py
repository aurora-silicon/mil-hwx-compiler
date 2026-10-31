#!/usr/bin/env python3
"""Composed exact geometry tests with synthetic public zero weights.

View tests have producers and consumers, so an input-alias refusal cannot be
mistaken for an inability to fold a reshape. FFN chain tests retain the
unproved-fusion gate. Budget refusals are resource results, not op coverage.
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from timeouts import scaled_timeout

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else 'build/mil-hwxc').resolve()


def tensor_type(shape):
    return 'tensor<fp16, [' + ', '.join(map(str, shape)) + ']>'


def view_source(d, h, split):
    source, result = ((1, 1500, d), (1, 1500, h, 64)) if split else ((1, 1500, h, 64), (1, 1500, d))
    literal = list(result)
    if split:
        literal[-1] = -1
    x, y = tensor_type(source), tensor_type(result)
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x} x) {{
    {x} producer = add(x = x, y = x)[name = string("producer")];
    tensor<int32, [{len(result)}]> shape = const()[name = string("shape"), val = tensor<int32, [{len(result)}]>([{', '.join(map(str, literal))}])];
    {y} view = reshape(x = producer, shape = shape)[name = string("view")];
    {y} y = add(x = view, y = view)[name = string("consumer")];
  }} -> (y);
}}
'''


def ffn_source(d):
    x, y = tensor_type((1, 1500, d)), tensor_type((1, 1500, 4*d))
    w1, w2 = tensor_type((4*d, d)), tensor_type((d, 4*d))
    b1, b2 = tensor_type((4*d,)), tensor_type((d,))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x} x) {{
    {w1} w1 = const()[name = string("w1"), val = {w1}(BLOBFILE(path = string("@model_path/w.bin"), offset = uint64(64)))];
    {b1} b1 = const()[name = string("b1"), val = {b1}(BLOBFILE(path = string("@model_path/b1.bin"), offset = uint64(64)))];
    {w2} w2 = const()[name = string("w2"), val = {w2}(BLOBFILE(path = string("@model_path/w.bin"), offset = uint64(64)))];
    {b2} b2 = const()[name = string("b2"), val = {b2}(BLOBFILE(path = string("@model_path/b2.bin"), offset = uint64(64)))];
    string mode = const()[name = string("mode"), val = string("EXACT")];
    {y} hidden = linear(x = x, weight = w1, bias = b1)[name = string("hidden")];
    {y} activated = gelu(x = hidden, mode = mode)[name = string("activated")];
    {x} y = linear(x = activated, weight = w2, bias = b2)[name = string("y")];
  }} -> (y);
}}
'''


def blob(count):
    header = bytearray(128)
    struct.pack_into('<IIQQ', header, 64, 0xDEADBEEF, 1, count*2, 128)
    return header + bytes(count*2)


def main():
    cases = 0
    with tempfile.TemporaryDirectory(prefix='whisper-composed-') as tmp:
        root = Path(tmp)
        for d, h in ((384, 6), (512, 8), (768, 12)):
            (root / 'w.bin').write_bytes(blob(4*d*d))
            (root / 'b1.bin').write_bytes(blob(4*d))
            (root / 'b2.bin').write_bytes(blob(d))
            for name, source in [('split', view_source(d, h, True)),
                                 ('merge', view_source(d, h, False)),
                                 ('ffn', ffn_source(d))]:
                mil = root / f'{d}-{name}.mil'
                mil.write_text(source)
                variants = [('H13', 'per-op', 'h13.program-budget')]
                variants.append(('H14', 'per-op', 'h14.linear-bias-unsupported' if name == 'ffn' else 'h14.outside-parity-envelope'))
                if name == 'ffn':
                    variants.append(('H13', 'chain', 'h13.chain-unrepresentable-edge'))
                for target, schedule, code in variants:
                    out = root / f'{d}-{name}-{target}-{schedule}'
                    command = [str(COMPILER), '--mil', str(mil), '--model-root', str(root),
                               '--target', target, '--schedule', schedule, '--output', str(out)]
                    if target == 'H13':
                        command += ['--max-programs', '2048']
                    run = subprocess.run(command, capture_output=True, text=True, timeout=scaled_timeout(30))
                    assert run.returncode == 65, run.stdout + run.stderr
                    assert f'error [{code}]' in run.stderr, run.stderr
                    assert 'returned-input-alias' not in run.stderr, run.stderr
                    assert not out.exists(), 'refused composition wrote a package'
                    print(json.dumps(dict(d=d, heads=h, case=name, target=target,
                                          schedule=schedule, rc=run.returncode, diagnostic=code)))
                    cases += 1
    assert cases == 21
    print('Whisper composed geometry: PASS (21 cases; no native execution)')


if __name__ == '__main__':
    main()
