#!/usr/bin/env python3
"""Audit formula geometry, constants, bounds and refusals over the H14 corpus.

The original test_h14_parity remains the strict byte-parity test. Generic
scheduling may choose smaller output-channel tasks and tiles than Apple.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
sys.path.insert(0, str(ROOT / 'tests'))
from h13_td import split_h14_tasks, decode_task, register_words  # noqa: E402
import mint_conv_probes as conv  # noqa: E402
from test_h14_parity import write_weights  # noqa: E402
from timeouts import scaled_timeout  # noqa: E402

COMPILER = str(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'build/mil-hwxc').resolve())


def expected_generic(p):
    ci, co = p['input_shape'][1], p['output_shape'][1]
    k, kw = p['weight_shape'][2:]
    st, g = p['stride'], p['groups']
    return (ci % 16 == 0 and co % 16 == 0 and k == kw and k in (1, 3) and
            (g == 1 or g == ci == co) and (st == 1 or (st == 2 and k == 1 and g == 1)) and
            not (max(p['input_shape'][2:]) <= 8 and co == 768))


def compile_one(r, root):
    (root / 'model.mil').write_text(r['mil'])
    write_weights(r, root)
    out = root / 'out'
    env = dict(os.environ, MILHWXC_H14_CONV_GENERIC='1')
    run = subprocess.run([COMPILER, '--target', 'H14', '--mil', str(root / 'model.mil'),
                          '--model-root', str(root), '--output', str(out)],
                         env=env, text=True, capture_output=True, timeout=scaled_timeout(120))
    return run, out


def linked_checks(root):
    checked = 0
    for path in sorted((ROOT / 'research/oracles/h14-mobile').glob('*chain*.json')):
        r = json.loads(path.read_text())
        case = root / path.stem
        case.mkdir()
        (case / 'model.mil').write_text(r['mil'])
        write_weights(r, case)
        run = subprocess.run([COMPILER, '--target', 'H14', '--schedule', 'linked',
                              '--mil', str(case / 'model.mil'), '--model-root', str(case),
                              '--output', str(case / 'out')], capture_output=True, text=True,
                             timeout=scaled_timeout(120))
        assert run.returncode == 0, (path.name, run.stderr)
        m = json.loads((case / 'out/manifest.json').read_text())
        assert len(m['programs']) == 1 and m['dispatchPlan'] == [0]
        assert len(m['scratch']['regions']) == r['parameters']['layers'] - 1
        regions = sorted(m['scratch']['regions'].values(), key=lambda x: x['offset'])
        for a, b in zip(regions, regions[1:]):
            assert a['offset'] + a['bytes'] <= b['offset']
        assert regions[-1]['offset'] + regions[-1]['bytes'] == m['scratch']['bytes']
        a = (case / 'out/program-0.anec').read_bytes()
        assert struct.unpack_from('<I', a, 0x28 + 3 * 4)[0] * 0x4000 == m['scratch']['bytes']
        tasks = [register_words(decode_task(t, 'h14')) for t in
                 split_h14_tasks(a[0x1000:0x1000 + struct.unpack_from('<Q', a, 0x10)[0]])]
        assert len(tasks) == m['taskDescriptors']
        previous = -1
        for tr, t in zip(m['tasks'], tasks):
            layer = tr['layer']
            assert (t[0x1100] >> 28) == (1 if layer and layer != previous else 0)
            if layer:
                assert t[0x1108] == 0x4e
            if layer + 1 < r['parameters']['layers']:
                assert t[0x1500] & 0xf0 == 0xf0
            previous = layer
        checked += 1
    assert checked == 6, checked
    return checked


def main():
    records = []
    for directory in ('h14', 'h14-mobile'):
        for path in sorted((ROOT / 'research/oracles' / directory).glob('*.json')):
            r = json.loads(path.read_text())
            if r.get('error') is None and r.get('parameters', {}).get('weight_shape'):
                records.append(r)
    counts = collections.Counter()
    with tempfile.TemporaryDirectory(prefix='h14-conv-formula-') as tmp:
        root = Path(tmp)
        counts['linked_chains'] = linked_checks(root)
        for index, r in enumerate(records):
            p = r['parameters']
            case = root / str(index)
            case.mkdir()
            run, out = compile_one(r, case)
            if not expected_generic(p):
                assert run.returncode != 0, r['case']
                assert 'h14.conv-outside-envelope' in run.stderr, (r['case'], run.stderr)
                counts['explicit_refusal'] += 1
                continue
            assert run.returncode == 0, (r['case'], run.stderr)
            m = json.loads((out / 'manifest.json').read_text())
            assert len(m['programs']) == 1, r['case']
            a = (out / 'program-0.anec').read_bytes()
            td_bytes = struct.unpack_from('<Q', a, 0x10)[0]
            tasks = [register_words(decode_task(t, 'h14')) for t in split_h14_tasks(a[0x1000:0x1000 + td_bytes])]
            co = 0
            for t in tasks:
                assert t[0] == (p['input_shape'][2] << 16) | p['input_shape'][3], r['case']
                assert t[0xc] == p['input_shape'][1], r['case']
                assert t[0x14] == (p['output_shape'][2] << 16) | p['output_shape'][3], r['case']
                co += t[0x10]
                assert t[0x1514] == p['output_shape'][2] * t[0x1510], r['case']
                assert t[0x111c] == p['input_shape'][2] * t[0x1118], r['case']
                if p['stride'] == 2:
                    assert t[0x1c] <= 3, (r['case'], t[0x1c])
                base = t.get(0x1908, 0)
                for i in range(16):
                    assert base + t.get(0x1958 + 4 * i, 0) + t[0x1998 + 4 * i] <= m['constantBytes'], r['case']
                expected_mac = register_words(r['task_descriptors'][0]).get(0xd04)
                if expected_mac is not None:
                    assert t[0xd04] == expected_mac, r['case']
            assert co == p['output_shape'][1], r['case']
            if len(tasks) == 1 and conv.covered(r):
                constants = a[0x1000 + m['constantOffset']:]
                assert hashlib.sha256(constants).hexdigest() == r['constant_section']['sha256'], r['case']
                counts['constant_parity'] += 1
            counts['formula_geometry'] += 1
    assert counts['formula_geometry'] >= 250, counts
    assert counts['constant_parity'] >= 150, counts
    print('H14 convolution corpus audit: PASS', dict(counts), 'records=', len(records))

if __name__ == '__main__':
    main()
