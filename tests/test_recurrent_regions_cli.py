#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package state retention and decoded ReLU regions, without a device."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from timeouts import scaled_timeout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'build/mil-hwxc').resolve()
T = 'tensor<fp16, [1,512]>'


def source(ops, returns, inputs=('a', 'b')):
    body = '\n'.join(f'    {T} {name} = {op}({args})[name = string("{name}")];'
                     for name, op, args in ops)
    args = ', '.join(f'{T} {name}' for name in inputs)
    return (f'program(1.3)\n[buildInfo = dict<string, string>({{}})]\n{{\n'
            f'  func main<ios18>({args}) {{\n{body}\n  }} -> ({", ".join(returns)});\n}}\n')


def compile_at(root, name, text, target, schedule='per-op', fmt='anec', error=None):
    mil = root / (name + '.mil')
    mil.write_text(text)
    output = root / name
    run = subprocess.run([str(COMPILER), '--mil', str(mil), '--model-root', str(root),
                          '--target', target, '--format', fmt, '--schedule', schedule,
                          '--output', str(output)], capture_output=True, text=True,
                         timeout=scaled_timeout(60))
    if error:
        assert run.returncode == 65 and error in run.stderr, run.stdout + run.stderr
        assert not output.exists(), 'refused compilation wrote a package'
        return None, output
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads((output / 'manifest.json').read_text()), output


def main():
    # A returned cell state must stay live while a later operation reads it.
    ops = [('c_next', 'add', 'x = a, y = b'),
           ('candidate', 'add', 'x = c_next, y = b'),
           ('activated', 'relu', 'x = candidate'),
           ('h_next', 'mul', 'x = activated, y = c_next')]
    text = source(ops, ['h_next', 'c_next', 'c_next'])
    tail = source([('forget', 'mul', 'x = f, y = c'),
                   ('update', 'mul', 'x = i, y = g'),
                   ('c_next', 'add', 'x = forget, y = update'),
                   ('cell_tanh', 'tanh', 'x = c_next'),
                   ('h_next', 'mul', 'x = o, y = cell_tanh')],
                  ['h_next', 'c_next'], inputs=('f', 'c', 'i', 'g', 'o'))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for target in ('H13', 'H14'):
            for fmt in ('anec', 'hwx'):
                perop, _ = compile_at(root, target + fmt + '-perop', text, target, fmt=fmt)
                region, directory = compile_at(root, target + fmt + '-region', text, target,
                                               'regions', fmt)
                assert len(perop['programs']) == 4 and len(region['programs']) == 3
                assert region['dispatchPlan'] == [0, 1, 2]
                assert region['regions'] == [{'program': 1,
                    'operations': ['candidate', 'activated'], 'postOperation': 'relu'}]
                assert 'candidate' not in region['tensors']
                assert region['tensors']['c_next']['role'] == 'output'
                assert 'c_next' not in region['intermediates']
                assert [r['name'] for r in region['logicalResults']] == ['h_next', 'c_next', 'c_next']
                assert {r['tensor'] for r in region['physicalOutputs']} == {'c_next', 'h_next'}
                assert len(region['physicalOutputs']) == 2
                assert region['programs'][2]['inputs'][1]['name'] == 'c_next'
                assert region['programs'][1]['outputs'][0]['name'] == 'activated'
                assert region['logicalResults'][1]['physical'] == {
                    'tensor': 'c_next', 'elementOffset': 0, 'elementCount': 512}
                if fmt == 'anec' and target == 'H13':
                    from test_h13_package_simulation import run_sim
                    from h13_reference import encode_fp16
                    inputs = {name: encode_fp16([((i % 13) - 6) * scale for i in range(512)])
                              for name, scale in (('a', .125), ('b', -.25))}
                    actual, expected, _ = run_sim(directory, root / (target + fmt + '-region.mil'), root, inputs)
                    assert actual == expected, 'decoded region arithmetic/state differs from fp16 MIL'
                if fmt == 'anec':
                    isolated = source([('candidate', 'add', 'x = c_next, y = b'),
                                       ('activated', 'relu', 'x = candidate')],
                                      ['activated'], inputs=('c_next', 'b'))
                    chain, chain_dir = compile_at(root, target + '-isolated', isolated, target, 'chain')
                    assert (directory / region['programs'][1]['file']).read_bytes() == (
                        chain_dir / chain['programs'][0]['file']).read_bytes(), 'region differs from decoded chain'
                lstm, _ = compile_at(root, target + fmt + '-lstm-tail', tail, target, 'regions', fmt)
                assert len(lstm['programs']) == 5 and lstm['regions'] == []
                assert [r['name'] for r in lstm['logicalResults']] == ['h_next', 'c_next']
                assert lstm['tensors']['c_next']['role'] == 'output'
                # Exposing the pre-ReLU value or adding another reader forbids fusion.
                exposed, _ = compile_at(root, target + fmt + '-exposed',
                    source(ops, ['h_next', 'c_next', 'candidate']), target, 'regions', fmt)
                assert len(exposed['programs']) == 4 and not exposed['regions']
                fanout, _ = compile_at(root, target + fmt + '-fanout',
                    source(ops + [('other', 'mul', 'x = candidate, y = h_next')], ['other', 'c_next']),
                    target, 'regions', fmt)
                assert len(fanout['programs']) == 5 and not fanout['regions']
                sigmoid, _ = compile_at(root, target + fmt + '-sigmoid',
                    source([('sum', 'add', 'x = a, y = b'), ('z', 'sigmoid', 'x = sum')], ['z']),
                    target, 'regions', fmt)
                assert len(sigmoid['programs']) == 2 and not sigmoid['regions']
            compile_at(root, target + '-exposed-chain', text, target, 'chain',
                       error=target.lower() + ('.unsupported-chain' if target == 'H14' else '.chain-unrepresentable-edge'))
        # H13 removes view operations but retains alias metadata. An identity
        # view feeding ReLU must not lose its backing tensor during fusion.
        view_text = source([('sum', 'add', 'x = a, y = b'),
                            ('view', 'reshape', 'x = sum, shape = same_shape'),
                            ('activated', 'relu', 'x = view')], ['activated'])
        view_text = view_text.replace('    ' + T + ' sum =',
            '    tensor<int32, [2]> same_shape = const()[name = string("same_shape"), '
            'val = tensor<int32, [2]>([1,512])];\n    ' + T + ' sum =')
        view, view_dir = compile_at(root, 'h13-view-region', view_text, 'H13', 'regions')
        assert len(view['programs']) == 2 and not view['regions']
        assert view['tensors']['view']['aliasOf'] == 'sum'
        from research import inspect_anec
        inspect_anec.load_package(view_dir)
        compile_at(root, 'h14-passthrough', source(ops, ['h_next', 'a']), 'H14',
                   error='h14.unsupported-logical-result-storage')
        compile_at(root, 'h14-dead-tail', source(ops, ['c_next']), 'H14',
                   error='h14.unsupported-chain')
        for target in ('H17', 'H18'):
            # Their package/byte-parity contracts are deliberately unchanged.
            compile_at(root, target + '-multi', text, target, fmt='hwx',
                       error=target.lower() + '.unsupported-chain')
    print('recurrent state returns and bounded ReLU regions: PASS')


if __name__ == '__main__':
    main()
