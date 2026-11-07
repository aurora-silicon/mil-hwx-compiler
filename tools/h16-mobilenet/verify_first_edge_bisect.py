#!/usr/bin/env python3
"""Prove the v7 two-task first-edge hybrids differ by one field group."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import register_words  # noqa: E402
from mint_oracles import parse_hwx  # noqa: E402


def load(path):
    path = Path(path)
    manifest = json.loads((path / 'linked/manifest.json').read_text())
    decoded = parse_hwx((path / 'linked/program-0.hwx').read_bytes(), 'h16')
    assert not decoded['task_decode_errors']
    tasks = decoded['task_descriptors']
    assert len(tasks) == 2
    return manifest, decoded, [register_words(task) for task in tasks]


def changes(left, right):
    return {(task, address) for task in range(2)
            for address in set(left[task]) | set(right[task])
            if left[task].get(address) != right[task].get(address)}


def values(left, right, changed):
    return [{'task': task, 'address': hex(address),
             'left': None if left[task].get(address) is None else hex(left[task][address]),
             'right': None if right[task].get(address) is None else hex(right[task][address])}
            for task, address in sorted(changed)]


def semantic_headers(decoded):
    result = []
    for task in decoded['task_descriptors']:
        header = [int(word, 16) for word in task['header_words']]
        header[0] &= 0xffff              # omit packed descriptor byte count
        result.append(header)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('corrected', 'legacy', 'kernel', 'geometry', 'linear'):
        ap.add_argument(f'--{name}', required=True, type=Path)
    args = ap.parse_args()
    loaded = {name: load(getattr(args, name))
              for name in ('corrected', 'legacy', 'kernel', 'geometry', 'linear')}
    manifests = {name: item[0] for name, item in loaded.items()}
    decoded = {name: item[1] for name, item in loaded.items()}
    regs = {name: item[2] for name, item in loaded.items()}

    kernel_hash = decoded['corrected']['kernel_section']['sha256']
    assert all(item['kernel_section']['sha256'] == kernel_hash
               for item in decoded.values())
    assert semantic_headers(decoded['corrected']) == semantic_headers(decoded['linear'])
    for task in range(2):
        assert decoded['corrected']['task_descriptors'][task]['blocks'].get('0x04900') == \
               decoded['linear']['task_descriptors'][task]['blocks'].get('0x04900')

    expected = {
        'legacy': {(1, 0x4148)},
        'kernel': {(0, 0x5500)},
        'geometry': {(0, 0x4110), (0, 0x414c), (0, 0x4154),
                     (1, 0x4110), (1, 0x4118), (1, 0x414c)},
    }
    reports = {}
    for name, want in expected.items():
        changed = changes(regs['corrected'], regs[name])
        assert changed == want, (name, changed, want)
        reports[name] = values(regs['corrected'], regs[name], changed)

    assert regs['corrected'][1][0x4148] == 0x142
    assert regs['legacy'][1][0x4148] == 0x14a
    assert regs['corrected'][0][0x5500] & 0x20
    assert not regs['kernel'][0][0x5500] & 0x20
    assert (regs['geometry'][0].get(0x4110, 0), regs['geometry'][0][0x414c],
            regs['geometry'][0][0x4154]) == (0, 0x49800, 0x1c80)
    assert (regs['geometry'][1][0x4110], regs['geometry'][1][0x4118],
            regs['geometry'][1][0x414c]) == (0x49800, 0x1c80, 0x111000)
    geometry_layers=manifests['geometry']['programs'][0]['layers']
    assert geometry_layers[0]['l2Result'] == 0x49800
    assert geometry_layers[0]['l2ResultBytes'] == 112*0x1c80
    assert geometry_layers[1]['l2Source'] == 0x49800

    linear_changed = changes(regs['corrected'], regs['linear'])
    print(json.dumps({
        'status': 'PASS',
        'schedules': {name: manifest['schedule']
                      for name, manifest in manifests.items()},
        'kernelSHA256': kernel_hash,
        'isolatedChanges': reports,
        'correctedVsLinear': values(regs['corrected'], regs['linear'], linear_changed),
    }, indent=2))


if __name__ == '__main__':
    main()
