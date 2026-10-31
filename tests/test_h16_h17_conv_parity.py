#!/usr/bin/env python3
"""All decoded conv facts, independently compared with production formulas.

The ONLY task-field exclusion is ExeCycles (header[1]), named unknown in the
field report. Sparse byte hashes are NOT claimed: those packages are refused.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import decode_task, split_h14_tasks  # noqa: E402
from mint_conv_probes import blob_view  # noqa: E402
from mint_oracles import half_payload  # noqa: E402

DUMP = str(Path(sys.argv[1]).resolve())
TARGET = sys.argv[2]
ISA = {'H16G': 'h16', 'H17': 'h17'}[TARGET]


def stages(oracle):
    p = oracle['parameters']
    if 'chain2' in oracle['case']:
        return [(64, 64, 112, 3, 2, 64, 1), (64, 128, 56, 1, 1, 1, 1)]
    if 'chain3' in oracle['case']:
        return [(32, 64, 112, 1, 1, 1, 1), (64, 64, 112, 3, 2, 64, 1),
                (64, 128, 56, 1, 1, 1, 1)]
    return [(p['input_channels'], p['output_channels'], p['spatial'], p['kernel'],
             p['stride'], p['groups'], int(p['relu']))]


def run(mode, layers, success=True, weights=None):
    result = subprocess.run([DUMP, mode, TARGET, *map(str, sum(map(list, layers), []))],
                            capture_output=True, timeout=30, input=weights)
    assert (result.returncode == 0) == success, result.stderr.decode()
    return result


def main():
    paths = sorted((ROOT / 'research/oracles' / ISA).glob('conv_desc_*.json'))
    assert len(paths) == 12, (ISA, len(paths))
    task_count = fields = packages = 0
    for path in paths:
        oracle = json.loads(path.read_text())
        layers = stages(oracle)
        stream = run('describe', layers).stdout
        actual = [decode_task(t, ISA) for t in split_h14_tasks(stream)]
        expected = oracle['task_descriptors']
        assert len(actual) == len(expected), path.name
        for index, (got, want) in enumerate(zip(actual, expected)):
            gh, wh = got['header_words'], want['header_words']
            # Cost hints are unknown; every size, ID, event and flag is tested.
            assert gh[:1] + gh[2:] == wh[:1] + wh[2:], (path.name, index, gh, wh)
            assert got['blocks'] == want['blocks'], (path.name, index, got['blocks'], want['blocks'])
            assert got['records'] == want['records'], (path.name, index, got['records'], want['records'])
            assert got['size_bytes'] == want['size_bytes'], path.name
            fields += sum(len(b['words']) for b in want['blocks'].values()) + 8
            task_count += 1
        if len(layers) == 1 and layers[0][4] == 1:
            elements = oracle['weights']['payload_bytes'] // 2
            packed = run('pack', layers, weights=blob_view(half_payload(elements), elements)).stdout
            assert len(packed) == oracle['kernel_section']['size'], path.name
            assert hashlib.sha256(packed).hexdigest() == oracle['kernel_section']['sha256'], path.name
            packages += 1
        elif len(layers) == 1:
            elements = oracle['weights']['payload_bytes'] // 2
            refused = run('pack', layers, False, blob_view(half_payload(elements), elements))
            assert b'sparse kernel byte grammar' in refused.stderr
        print(f'PASS {TARGET} {path.stem}')
    # Adjacent unmeasured schedules must fail, even when the H13 encoder works.
    for layer in [(32,64,111,1,1,1,0), (64,256,56,1,1,1,0),
                  (256,256,28,3,1,256,1)]:
        assert b'measured' in run('describe', [layer], False).stderr
    refused = run('describe', [(32,64,112,1,1,1,1), (64,128,112,1,1,1,1)], False)
    assert b'DRAM dependency protocol' in refused.stderr
    print(f'{TARGET} conv parity PASS: {len(paths)} oracles, {task_count} tasks, '
          f'{fields} fields, {packages} stride-1 kernel hashes; sparse packages refused')


if __name__ == '__main__':
    main()
