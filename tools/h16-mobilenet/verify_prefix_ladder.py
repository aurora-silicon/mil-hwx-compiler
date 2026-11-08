#!/usr/bin/env python3
"""Offline proof for the H16G prefix-1 truncated-task ladder."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import register_words  # noqa: E402
from mint_oracles import parse_hwx  # noqa: E402

RUNGS = (("tasks-0-1", 2), ("tasks-0-2", 3), ("tasks-0-3", 4),
         ("tasks-0-5", 6), ("tasks-0-8", 9), ("tasks-0-14", 15),
         ("tasks-0-27", 28))
TERMINAL_CFG = {2: 0x142, 3: 0x142, 4: 0x14a, 6: 0x14a,
                9: 0x14a, 15: 0x14a, 28: 0x14a}


def load(package):
    package = Path(package)
    manifest = json.loads((package / 'linked/manifest.json').read_text())
    decoded = parse_hwx((package / 'linked/program-0.hwx').read_bytes(), 'h16')
    assert not decoded['task_decode_errors']
    return manifest, decoded


def bar3_addends(tasks):
    result = []
    for task_index, task in enumerate(tasks):
        words = register_words(task)
        for record in task['records']:
            header = int(record['header'], 16)
            address = int(record['address'], 16)
            if not header & 0x20000000 or (header >> 23) & 0x3f != 3:
                continue
            assert address in (0x4d10, 0x5108) and record['count'] == 2
            pair = (words[address], words[address + 4])
            addend = pair[0] | pair[1] << 32
            assert addend < 0x380000, (task_index, hex(address), pair)
            result.append((task_index, hex(address), pair, addend))
    return result


def prove_package(package, count):
    manifest, decoded = load(package)
    program = manifest['programs'][0]
    tasks = decoded['task_descriptors']
    regs = [register_words(task) for task in tasks]
    assert manifest['schedule'] == 'linked-l2-prefix-1'
    assert len(tasks) == count == program['taskDescriptors']
    assert program['edgeStorage'] == ['l2'] + ['dram'] * (count - 2)
    assert program['tileDMAEndpoints'] == 2 + 2 * (count - 2)
    assert [i for i,t in enumerate(tasks) if '0x04d00' in t['blocks']] == \
           [0] + list(range(2,count))
    assert [i for i,t in enumerate(tasks) if '0x05100' in t['blocks']] == \
           list(range(1,count))
    assert regs[-1][0x4148] == TERMINAL_CFG[count]
    if count >= 3:
        assert regs[1][0x4148] == 0x84a and regs[2][0x4104] == 0x842
        assert regs[1][0x5100] == 0x010100f1
        assert regs[2][0x4d00] == 0x20020041
    addends = bar3_addends(tasks)
    if count >= 6:
        assert (3, '0x5108', (0x1c0000, 0), 0x1c0000) in addends
        assert (4, '0x4d10', (0x1c0000, 0), 0x1c0000) in addends
    refs = sorted((Path(package) / 'refs').glob('output_*.bin'))
    inputs = sorted((Path(package) / 'refs').glob('input_*.bin'))
    assert refs and len(refs) == len(inputs)
    output_bytes = program['outputs'][0]['allocationBytes']
    assert all(path.stat().st_size == output_bytes for path in refs)
    return {'name': Path(package).name, 'tasks': count,
            'scratchBytes': manifest['scratch']['bytes'],
            'bar3Addends': addends, 'references': len(refs),
            'outputBytes': output_bytes,
            'kernelSHA256': decoded['kernel_section']['sha256']}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--prefix-1', type=Path)
    ap.add_argument('--rung1', type=Path)
    args = ap.parse_args()
    reports = [prove_package(args.root / name, count) for name, count in RUNGS]
    if args.prefix_1:
        prefix_manifest, prefix = load(args.prefix_1)
        full_manifest, full = load(args.root / RUNGS[-1][0])
        assert prefix_manifest['programs'][0]['taskDescriptors'] == 28
        assert (args.prefix_1 / 'linked/program-0.hwx').read_bytes() == \
               (args.root / RUNGS[-1][0] / 'linked/program-0.hwx').read_bytes()
        assert prefix['kernel_section']['sha256'] == full['kernel_section']['sha256']
    if args.rung1:
        manifest, decoded = load(args.rung1)
        regs = [register_words(task) for task in decoded['task_descriptors']]
        assert manifest['schedule'] == 'linked-l2'
        assert manifest['scratch']['bytes'] == 0xc4000
        assert regs[1][0x4148] == 0x84a and regs[2][0x4104] == 0x842
        bar3_addends(decoded['task_descriptors'])
    print(json.dumps({'status': 'PASS', 'rungs': reports}, indent=2))


if __name__ == '__main__':
    main()
