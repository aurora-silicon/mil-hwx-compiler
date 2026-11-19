#!/usr/bin/env python3
"""H16G and H17 denoise-grid layers: compile each layer of the 960x540 (1080p),
640x360 (720p) and 320x240 (VGA) grids with mil-hwxc and check it against the
decoded macOS compiler output of the same MIL and weights
(research/oracles/{h16,h17}/denoise_grid_[<size>_]*.json): every register,
the task size and header (header[1] ExeCycles stays unmodelled), the tensor
descriptors and the kernel section hash. Also checks that the six-layer
grid network compiles as one linked DRAM chain for each target.

usage: test_h16g_denoise_grid_parity.py build/mil-hwxc
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from mint_oracles import blob, parse_hwx  # noqa: E402

COMPILER = str(Path(sys.argv[1]).resolve())
CHAIN = ['pw4x32', 'dw32', 'pw32x32', 'c32k3', 'dw32b', 'pw32x16']


def registers(task):
    return {int(a, 16): int(v, 16) for block in task['blocks'].values()
            for a, v in block['words'].items()}


def layer_mil(index, case, source):
    ci, co, k, g = case['cin'], case['cout'], case['kernel'], case['groups']
    w, h = case['width'], case['height']
    blobfile = 'BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({}))'
    lines = [
        f'    string pt{index} = const()[name = string("pt{index}"), val = string("same")];',
        f'    tensor<int32, [2]> st{index} = const()[name = string("st{index}"), val = tensor<int32, [2]>([1, 1])];',
        f'    tensor<int32, [4]> pd{index} = const()[name = string("pd{index}"), val = tensor<int32, [4]>([0, 0, 0, 0])];',
        f'    tensor<int32, [2]> dl{index} = const()[name = string("dl{index}"), val = tensor<int32, [2]>([1, 1])];',
        f'    int32 gp{index} = const()[name = string("gp{index}"), val = int32({g})];',
        f'    tensor<fp16, [{co}, {ci // g}, {k}, {k}]> W{index} = const()[name = string("W{index}"), '
        f'val = tensor<fp16, [{co}, {ci // g}, {k}, {k}]>({blobfile.format(case["woff"])})];',
        f'    tensor<fp16, [{co}]> B{index} = const()[name = string("B{index}"), '
        f'val = tensor<fp16, [{co}]>({blobfile.format(case["boff"])})];',
        f'    tensor<fp16, [1, {co}, {h}, {w}]> c{index} = conv(bias = B{index}, dilations = dl{index}, '
        f'groups = gp{index}, pad = pd{index}, pad_type = pt{index}, strides = st{index}, '
        f'weight = W{index}, x = {source})[name = string("conv{index}")];',
    ]
    result = f'c{index}'
    if case['relu']:
        lines.append(f'    tensor<fp16, [1, {co}, {h}, {w}]> r{index} = relu(x = c{index})'
                     f'[name = string("relu{index}")];')
        result = f'r{index}'
    return lines, result


def program(cases):
    body, cur = [], 'x'
    for index, case in enumerate(cases):
        lines, cur = layer_mil(index, case, cur)
        body += lines
    cin, w, h = cases[0]['cin'], cases[0]['width'], cases[0]['height']
    return ('program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
            f'  func main<ios18>(tensor<fp16, [1, {cin}, {h}, {w}]> x) {{\n' + '\n'.join(body) +
            f'\n  }} -> ({cur});\n}}\n')


def weights_blob(cases):
    """One BLOBFILE: per layer a weight record then a bias record, 64-byte aligned."""
    out = bytearray(blob(b''))[:64]
    for case in cases:
        data = (ROOT / case['weights']).read_bytes()
        wbytes = case['cout'] * (case['cin'] // case['groups']) * case['kernel'] ** 2 * 2
        for key, part in (('woff', data[:wbytes]), ('boff', data[wbytes:])):
            meta = len(out)
            case[key] = meta
            out += (0xDEADBEEF).to_bytes(4, 'little') + (1).to_bytes(4, 'little')
            out += len(part).to_bytes(8, 'little') + (meta + 64).to_bytes(8, 'little') + bytes(40)
            out += part + bytes((-len(part)) % 64)
    out[0:8] = (1).to_bytes(4, 'little') + (2).to_bytes(4, 'little')
    return bytes(out)


def compile_(root, name, cases, schedule, target):
    (root / name).mkdir()
    (root / name / 'weights.bin').write_bytes(weights_blob(cases))
    (root / name / 'model.mil').write_text(program(cases))
    result = subprocess.run([COMPILER, '--mil', str(root / name / 'model.mil'),
                             '--model-root', str(root / name), '--target', target,
                             '--format', 'hwx', '--schedule', schedule,
                             '--output', str(root / name / 'out')],
                            capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise SystemExit(f'{name}: compile failed: {result.stderr.strip()[-400:]}')
    objects = sorted((root / name / 'out').rglob('*.hwx'))
    assert len(objects) == 1, objects
    return objects[0].read_bytes()


def main():
    checked = 0
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for target, facts in (('H16G', 'h16'), ('H17', 'h17')):
            for size in ('', '640x360_', '320x240_'):
                oracles = {name: json.loads((ROOT / f'research/oracles/{facts}/denoise_grid_{size}{name}.json').read_text())
                           for name in CHAIN}
                grid = f'{oracles[CHAIN[0]]["case"]["width"]}x{oracles[CHAIN[0]]["case"]["height"]}'
                for name, oracle in oracles.items():
                    data = compile_(root, f'{facts}-{size}{name}', [dict(oracle['case'])], 'per-op', target)
                    ours = parse_hwx(data, facts)
                    want, got = oracle['task_descriptors'][0], ours['task_descriptors'][0]
                    a, b = registers(want), registers(got)
                    bad = {hex(k): (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
                    assert not bad, f'{target} {grid} {name}: register mismatches {bad}'
                    assert want['size_bytes'] == got['size_bytes'], f'{target} {grid} {name}'
                    # Record grammar and order; the BAR slot of relocation records is
                    # the object writer's (the M4 Pro facts number slots one lower).
                    rec = lambda t: [(r['address'], int(r['header'], 16) & 0x807fffff) for r in t['records']]
                    assert rec(want) == rec(got), f'{target} {grid} {name}: records'
                    hw, hg = list(want['header_words']), list(got['header_words'])
                    hw[1] = hg[1] = None
                    assert hw == hg, f'{target} {grid} {name}: header {want["header_words"]} vs {got["header_words"]}'
                    assert ours['tensor_descriptors'] == oracle['tensor_descriptors'], f'{target} {grid} {name}'
                    ks = oracle['kernel_section']
                    kernel = ours['kernel_section']
                    assert kernel['sha256'] == ks['sha256'] and kernel['size'] == ks['size'], \
                        f'{target} {grid} {name}: kernel section differs'
                    checked += len(a) + 1
                    print(f'{target} {grid} {name}: {len(a)} registers, header, tensors and kernel section match')
                # The whole grid network as one linked DRAM chain.
                cases = [dict(oracles[n]['case']) for n in CHAIN]
                data = compile_(root, f'{facts}-{size}chain', cases, 'linked', target)
                chain = parse_hwx(data, facts)
                w, h = cases[0]['width'], cases[0]['height']
                assert len(chain['task_descriptors']) == len(CHAIN)
                assert chain['tensor_descriptors'][0]['shape'] == [1, 4, h, w]
                assert chain['tensor_descriptors'][-1]['shape'] == [1, 16, h, w]
                print(f'{target} {grid} chain: {len(CHAIN)} linked tasks, sha256 {hashlib.sha256(data).hexdigest()[:16]}')
    print(f'denoise grid parity: {checked} facts checked')


if __name__ == '__main__':
    main()
