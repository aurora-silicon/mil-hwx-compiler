#!/usr/bin/env python3
"""All decoded conv facts, independently compared with production formulas.

The ONLY task-field exclusion is ExeCycles (header[1]), named unknown in the
field report. Every single-task oracle is rebuilt through the executable path
from its campaign BLOBFILE view, so dense and sparse (stride-2) kernel sections
and their data-dependent chunk tables are both checked against Apple's hashes.
The round-2 multi-layer chains and the 1024->1024 output-split forms are
checked by research/verify_h16g_h17_conv_fields.py, not here.
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
             p['stride'], p['groups'], int(p['relu']) | (2 if p.get('bias') else 0))]


def campaign_weights(oracle):
    """The campaign's BLOBFILE view: weight, then bias read from the same blob."""
    p = oracle['parameters']
    elements = oracle['weights']['payload_bytes'] // 2
    view = blob_view(half_payload(elements), elements)
    return view + (blob_view(half_payload(elements), p['output_channels']) if p.get('bias') else b'')


def run(mode, layers, success=True, weights=None):
    result = subprocess.run([DUMP, mode, TARGET, *map(str, sum(map(list, layers), []))],
                            capture_output=True, timeout=30, input=weights)
    assert (result.returncode == 0) == success, result.stderr.decode()
    return result


def main():
    # Output-channel split (1024->1024 at 14/28) and the round-2 multi-layer
    # chains are verified by research/verify_h16g_h17_conv_fields.py; the
    # encoder refuses split geometries.
    skipped = {'conv_desc_chain6_mob_c128_s28', 'conv_desc_chain_tail_c512_s14',
               'conv_desc_chain6_dram_pw1024_s28', 'conv_desc_pw_c1024_n1024_s14',
               'conv_desc_pw_c1024_n1024_s28'}
    paths = [p for p in sorted((ROOT / 'research/oracles' / ISA).glob('conv_desc_*.json'))
             if p.stem not in skipped]
    assert len(paths) == 38, (ISA, len(paths))
    task_count = fields = packages = sparse = 0
    for path in paths:
        oracle = json.loads(path.read_text())
        layers = stages(oracle)
        if len(layers) == 1:
            stream = run('exec', layers, weights=campaign_weights(oracle)).stdout
        else:
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
        if len(layers) == 1:
            packed = run('pack', layers, weights=campaign_weights(oracle)).stdout
            assert len(packed) == oracle['kernel_section']['size'], path.name
            assert hashlib.sha256(packed).hexdigest() == oracle['kernel_section']['sha256'], path.name
            packages += 1
            sparse += layers[0][4] == 2
        print(f'PASS {TARGET} {path.stem}')
    # Adjacent unmeasured schedules must fail, even when the H13 encoder works.
    for layer in [(32,64,111,1,1,1,0), (64,256,56,1,1,1,0),
                  (256,256,56,3,1,256,1)]:
        assert b'measured' in run('describe', [layer], False).stderr
    refused = run('describe', [(32,64,112,1,1,1,1), (64,128,112,1,1,1,1)], False)
    assert b'measured resident chains' in refused.stderr
    # Stride-2 sparse kernels exist only for the decoded forms.
    refused = run('pack', [(128,128,56,3,2,32,0)], False, bytes(128*4*9*2))
    assert b'measured' in refused.stderr or b'sparse stride-2 grammar' in refused.stderr
    print(f'{TARGET} conv parity PASS: {len(paths)} oracles, {task_count} tasks, '
          f'{fields} fields, {packages} kernel hashes ({sparse} sparse stride-2)')


if __name__ == '__main__':
    main()
