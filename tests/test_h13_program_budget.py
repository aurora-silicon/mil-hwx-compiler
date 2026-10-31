#!/usr/bin/env python3
"""Program limits must reject before writing packages and preserve allowed bytes."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from timeouts import scaled_timeout

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else 'build/mil-hwxc').resolve()


def run(root, text, budget=None, target='H13', name='case', extra=()):
    source, out = root / (name + '.mil'), root / name
    source.write_text(text)
    cmd = [str(COMPILER), '--mil', str(source), '--model-root', str(root),
           '--target', target, '--output', str(out), *extra]
    if budget is not None:
        cmd += ['--max-programs', str(budget)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=scaled_timeout(30)), out


def add_source(elements):
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [{elements}]> x, tensor<fp16, [{elements}]> z) {{
    tensor<fp16, [{elements}]> y = add(x = x, y = z)[name = string("y")];
  }} -> (y);
}}
'''


def blob(count):
    header = bytearray(128)
    struct.pack_into('<IIQQ', header, 64, 0xDEADBEEF, 1, count * 2, 128)
    return header + bytes(count * 2)


def linear_source(rows, k, n):
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {rows}, {k}]> x) {{
    tensor<fp16, [{n}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{n}, {k}]>(BLOBFILE(path = string("@model_path/w.bin"), offset = uint64(64)))];
    tensor<fp16, [{n}]> b = const()[name = string("b"), val = tensor<fp16, [{n}]>(BLOBFILE(path = string("@model_path/b.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {rows}, {n}]> y = linear(x = x, weight = w, bias = b)[name = string("y")];
  }} -> (y);
}}
'''


def assert_budget_refusal(result, output):
    assert result.returncode == 65, result.stdout + result.stderr
    assert 'error [h13.program-budget]' in result.stderr, result.stderr
    assert not output.exists(), 'budget refusal wrote a package'


def main():
    with tempfile.TemporaryDirectory(prefix='h13-budget-') as tmp:
        root = Path(tmp)
        text = add_source(128)
        for value in ('0', '-1', '+2', '1.5', '18446744073709551616', 'abc', ' 2'):
            result, out = run(root, text, value, name='bad-' + str(len(value)))
            assert result.returncode == 64 and not out.exists(), result.stderr
        result, out = run(root, text, 2, target='H14', name='wrong-target')
        assert result.returncode == 64 and not out.exists()
        result, out = run(root, text, 1, name='one')
        assert_budget_refusal(result, out)
        default, default_out = run(root, text, name='default')
        allowed, allowed_out = run(root, text, 2, name='boundary')
        assert default.returncode == allowed.returncode == 0, default.stderr + allowed.stderr
        assert len(json.loads((allowed_out / 'manifest.json').read_text())['programs']) == 2
        for name in ('program-0.anec', 'program-1.anec'):
            assert (default_out / name).read_bytes() == (allowed_out / name).read_bytes()
        # Cumulative limit: each operation is within the limit, their sum is not.
        chain = text.replace('} -> (y);', '} -> (out);').replace(
            '  } -> (out);', '    tensor<fp16, [128]> out = add(x = y, y = z)[name = string("out")];\n  } -> (out);')
        result, out = run(root, chain, 3, name='cumulative')
        assert_budget_refusal(result, out)
        # Real exact Whisper linears, including all seven prior timeouts. No
        # package is encoded merely to discover that its count is excessive.
        for d in (384, 512, 768):
            for k, n in ((d, d), (d, 4*d), (4*d, d)):
                (root / 'w.bin').write_bytes(blob(k*n))
                (root / 'b.bin').write_bytes(blob(n))
                result, out = run(root, linear_source(1500, k, n), 2048,
                                  name=f'linear-{k}-{n}')
                assert_budget_refusal(result, out)
        # Exact boundary for a row/reduction decomposition with nonzero bias.
        (root / 'w.bin').write_bytes(blob(1024*128))
        bias = bytearray(blob(128))
        bias[128:] = struct.pack('<128e', *([1.] * 128))
        (root / 'b.bin').write_bytes(bias)
        for limit in (14, 15):
            result, out = run(root, linear_source(3, 1024, 128), limit,
                              name=f'linear-boundary-{limit}')
            if limit == 14:
                assert_budget_refusal(result, out)
            else:
                assert result.returncode == 0, result.stderr
                assert len(json.loads((out / 'manifest.json').read_text())['programs']) == 15
    print('H13 program budget: PASS (boundaries, cumulative, nine exact linears, unchanged bytes)')


if __name__ == '__main__':
    main()
