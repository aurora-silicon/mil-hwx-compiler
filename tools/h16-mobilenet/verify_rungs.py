#!/usr/bin/env python3
"""Prove three H16G MobileNetV1 L2 packages against a compiler baseline.

This compares decoded fields and section hashes.  It never consumes or
compares a private object and never performs a raw whole-object comparison.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import register_words  # noqa: E402
from mint_oracles import parse_hwx  # noqa: E402

EDGE_STORAGE = ['l2', 'dram'] + ['l2'] * 25
SCHEDULE = [
    0x230000,0x230000,0x231100,0x211100,0x243300,0x243300,0x243300,
    0x222200,0x222200,0x222200,0x222200,0x233300,0x233300,0x233300,
    0x233300,0x233300,0x233300,0x233300,0x233300,0x233300,0x233300,
    0x233300,0x233300,0x244404,0x244404,0x244404,0x244404,0x244404,
]
COST = [4,11,11,12,2,7,5,6,3,3,6,2,3,1,7,1,7,1,7,1,7,1,7,1,15,1,30,37]
MASK = [
    0x00001fff,0x00001fff,0x01ffffff,0x7fffffff,0x7f001fff,0x01ffffff,0xfffff01f,
    0xff00079f,0x000007ff,0x00007fff,0x003fff80,0x007fc000,0x0060000f,0x000000ff,
    0x000007f8,0x00007f80,0x0003fc00,0x003fc000,0x01fe0000,0x1fe00000,0xff000000,
    0xf000000f,0x8000007f,0x000000f8,0x000000c3,0x0000000f,0x0000003e,0,
]
TASK_BYTES = [
    412,320,400,360,344,356,348,364,348,356,352,364,348,356,
    352,360,352,360,352,360,352,360,352,364,348,356,352,408,
]


def load(package):
    package = Path(package)
    manifest = json.loads((package / 'linked/manifest.json').read_text())
    raw = (package / 'linked/program-0.hwx').read_bytes()
    decoded = parse_hwx(raw, 'h16')
    assert not decoded['task_decode_errors']
    return manifest, decoded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--rung1', required=True)
    ap.add_argument('--rung2', required=True)
    ap.add_argument('--rung3', required=True)
    args = ap.parse_args()
    baseline_manifest, baseline = load(args.baseline)
    assert baseline['program_descriptor']['task_count'] == 28
    assert baseline_manifest['programs'][0]['taskDescriptors'] == 28
    reports = []
    for number, path in enumerate((args.rung1, args.rung2, args.rung3), 1):
        manifest, decoded = load(path)
        tasks = decoded['task_descriptors']
        regs = [register_words(t) for t in tasks]
        assert len(tasks) == 28 and decoded['program_descriptor']['task_count'] == 28
        assert decoded['kernel_section']['sha256'] == baseline['kernel_section']['sha256']
        assert decoded['kernel_section']['size'] == baseline['kernel_section']['size'] == 8506176
        assert decoded['constant_section']['sha256'] == baseline['constant_section']['sha256']
        assert decoded['tensor_descriptors'] == baseline['tensor_descriptors']
        assert [t['blocks'].get('0x04900') for t in tasks] == \
               [t['blocks'].get('0x04900') for t in baseline['task_descriptors']]
        assert [i for i, t in enumerate(tasks) if '0x04d00' in t['blocks']] == [0, 2]
        assert [i for i, t in enumerate(tasks) if '0x05100' in t['blocks']] == [1, 27]
        assert [regs[i][0x4148] for i in range(28)] == \
               [0x500170,0x84a] + [0x140] * 25 + [0x14a]
        assert regs[1][0x414c] == 0x0c5500
        assert [regs[2][a] for a in (0x4110,0x4114,0x4118,0x411c,0x4120)] == \
               [0x188e00,0x80,0x1050,0x1000,0x1000]
        assert regs[27][0x414c] == 0x543f0 and regs[27][0x5100] == 0x011b0031
        assert [regs[i][0x5500] & 0x20 for i in range(28)] == \
               [0x20 if i in (0,3,7,11,23) else 0 for i in range(28)]
        assert manifest['schedule'] == ('linked-l2' if number == 1 else
            'linked-l2-schedule' if number == 2 else 'linked-l2-header')
        assert manifest['scratch']['channel'] == 3
        assert manifest['scratch']['bytes'] == 0xc4000
        assert len(manifest['scratch']['regions']) == 1
        program = manifest['programs'][0]
        assert program['edgeStorage'] == EDGE_STORAGE
        assert program['tileDMAEndpoints'] == 4 and program['rowBandEdges'] == 0
        assert program['scratchChannel'] == 3 and program['scratchRegionBytes'] == 0xc4000
        assert program['sourceSurfaces'] == program['destinationSurfaces'] == 1
        layers = program['layers']
        storage = ['dram' if layer['outputEdge'] == 'dram' else 'l2'
                   for layer in layers[:-1]]
        assert storage == EDGE_STORAGE
        if number >= 2:
            assert [r[0x3c] for r in regs] == SCHEDULE
        headers = [[int(v, 16) for v in t['header_words']] for t in tasks]
        if number == 3:
            assert [h[1] for h in headers] == COST
            assert [h[4] for h in headers] == [0x68] + [0x28] * 27
            assert [h[7] for h in headers] == MASK
            assert [t['size_bytes'] for t in tasks] == TASK_BYTES
        reports.append({
            'rung': number,
            'tasks': len(tasks),
            'edgeStorage': EDGE_STORAGE,
            'tileDMAEndpoints': 4,
            'scratchBytes': 0xc4000,
            'kernelSHA256': decoded['kernel_section']['sha256'],
            'taskBytes': [t['size_bytes'] for t in tasks],
        })
    print(json.dumps({'status': 'PASS', 'rungs': reports}, indent=2))


if __name__ == '__main__':
    main()
