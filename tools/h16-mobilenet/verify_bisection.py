#!/usr/bin/env python3
"""Offline proof for H16G MobileNet L2 prefix/spill bisection packages."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import register_words  # noqa: E402
from mint_oracles import parse_hwx  # noqa: E402

ELIGIBLE = [0] + list(range(2, 27))
# Deliberately synthetic, disjoint bases.  The proof is about slot selection,
# immutable addends and bounds; it must not depend on a device allocation.
BAR_BASE = {slot: slot << 40 for slot in (3, 6, 7, 8)}
BAR_BYTES = {3: 0x380000, 6: 0x4c000, 7: 0x70000, 8: 0x820000}


def load(path):
    path = Path(path)
    manifest = json.loads((path / 'linked/manifest.json').read_text())
    decoded = parse_hwx((path / 'linked/program-0.hwx').read_bytes(), 'h16')
    assert not decoded['task_decode_errors']
    return manifest, decoded


def source_cfg(edge):
    return 0x120 if edge <= 5 or edge in (7, 8) else 0x104


def bar_trace(tasks):
    """Resolve every dense BAR operand from immutable descriptor addends."""
    trace = []
    for task_index, task in enumerate(tasks):
        words = register_words(task)
        for record in task['records']:
            address = int(record['address'], 16)
            if address not in (0x4d10, 0x5108, 0x5510):
                continue
            header = int(record['header'], 16)
            assert header & 0x20000000
            slot = (header >> 23) & 0x3f
            assert slot in BAR_BASE and record['count'] == 2
            addends = (words[address], words[address + 4])
            assert all(addend < BAR_BYTES[slot] for addend in addends)
            resolved = tuple(BAR_BASE[slot] + addend for addend in addends)
            assert all(value >> 32 != 0x21 for value in resolved)
            trace.append((task_index, address, slot, addends, resolved))
    return trace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True)
    for prefix in (1, 2, 4, 8, 26):
        ap.add_argument(f'--prefix-{prefix}', required=True)
    ap.add_argument('--baseline-spill', required=True)
    args = ap.parse_args()
    _, baseline = load(args.baseline)
    kernel = baseline['kernel_section']['sha256']
    reports = []
    paths = [(prefix, getattr(args, f'prefix_{prefix}'))
             for prefix in (1, 2, 4, 8, 26)]
    paths.append((26, args.baseline_spill))
    for index, (prefix, path) in enumerate(paths):
        manifest, decoded = load(path)
        tasks = decoded['task_descriptors']
        regs = [register_words(t) for t in tasks]
        assert len(tasks) == 28 and decoded['kernel_section']['sha256'] == kernel
        resident = set(ELIGIBLE[:prefix])
        dram = set(range(27)) - resident
        bridges = {edge for edge in dram
                   if edge == 1 or (edge and edge - 1 in resident)}
        expected_storage = ['l2' if edge in resident else 'dram'
                            for edge in range(27)]
        assert manifest['edgeStorage'] == expected_storage
        assert manifest['tileDMAEndpoints'] == 2 + 2 * len(dram)
        measured_prefix = Path(path).name.startswith('prefix-')
        assert manifest['scratch']['bytes'] == (0xc4000 if prefix == 26 and measured_prefix
                                                 else 0x380000)
        assert [i for i, t in enumerate(tasks) if '0x04d00' in t['blocks']] == \
               [0] + [edge + 1 for edge in sorted(dram)]
        assert [i for i, t in enumerate(tasks) if '0x05100' in t['blocks']] == \
               sorted(dram) + [27]
        for edge in resident:
            assert regs[edge][0x4148] == (0x500170 if edge == 0 else 0x140)
            assert regs[edge + 1][0x4104] == source_cfg(edge)
        assert [regs[i][0x5500] & 0x20 for i in range(28)] == \
               [0x20 if i in (0, 3, 7, 11, 23) else 0 for i in range(28)]
        if measured_prefix:
            for edge in bridges:
                assert regs[edge][0x4148] == 0x84a
                assert regs[edge + 1][0x4104] == 0x842
                assert regs[edge][0x5100] == 0x010000f1 | edge << 16
                assert regs[edge + 1][0x4d00] == 0x20000041 | (edge + 1) << 16
                assert regs[edge][0x5138] == 0x08302031
                assert regs[edge + 1][0x4d68] == 0x08002031
            assert regs[1][0x414c] == 0x0c5500
            assert (regs[1][0x5110], regs[1][0x5114]) == (0x700,0x31000)
            assert (regs[2][0x4d18], regs[2][0x4d1c]) == (0x700,0x31000)
            # No task may combine a resident input with the standalone DRAM
            # result form.  That unsupported hybrid is what left task 23's
            # slot-3 source unrelocated on prefix-1's second execution.
            for task in range(1, 27):
                if task - 1 in resident and task not in resident:
                    assert regs[task][0x4148] == 0x84a
            # Resolve the exact same immutable BAR addends twice.  This
            # catches a missing/incorrect slot record and proves that a
            # second EXECUTE needs no cursor or address consumed by the first.
            first = bar_trace(tasks)
            second = bar_trace(tasks)
            assert first == second
        if index == len(paths) - 1:
            assert manifest['schedule'] == 'linked-l2-baseline-spill'
            assert regs[1][0x5100] != 0x010100f1
            assert regs[2][0x4d00] != 0x20020041
            assert regs[1][0x414c] == 0x0c5500 and regs[2][0x4110] == 0x188e00
        reports.append({'name': Path(path).name, 'residentPrefix': prefix,
                        'edgeStorage': expected_storage,
                        'spillBridges': sorted(bridges) if measured_prefix else [],
                        'repeatBarRecords': len(bar_trace(tasks)),
                        'tileDMAEndpoints': 2 + 2 * len(dram),
                        'scratchBytes': manifest['scratch']['bytes']})
    print(json.dumps({'status': 'PASS', 'variants': reports}, indent=2))


if __name__ == '__main__':
    main()
