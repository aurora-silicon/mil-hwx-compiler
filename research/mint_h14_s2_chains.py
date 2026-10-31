#!/usr/bin/env python3
"""Compile-only H14 chains whose consumer is a stride-2 or multi-task layer
reading its producer, on macOS. Retains decoded facts and hashes, like
mint_h14_mobile.py; the compiler is never asked to load or run a program.

These records show how Apple's compiler links a multi-task consumer: the first
slice loads the input (from scratch DRAM with DependencyMode 1, or from L2),
later slices carry no source TileDMA and reuse the staged input."""

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import mint_conv_probes as conv
import mint_oracles as om
from mint_h14_mobile import compile_mil

# (kernel, input channels, output channels, input spatial, stride, groups),
# bias on, no relu.
CHAINS = {
    'dw128_pws2_256': [(3, 128, 128, 56, 1, 128), (1, 128, 256, 56, 2, 1)],
    'pw128_dw_pws2_256': [(1, 128, 128, 56, 1, 1), (3, 128, 128, 56, 1, 128),
                          (1, 128, 256, 56, 2, 1)],
    'pws2_256_dw_pw': [(1, 128, 256, 56, 2, 1), (3, 256, 256, 28, 1, 256),
                       (1, 256, 256, 28, 1, 1)],
    'dw512_pws2_1024': [(3, 512, 512, 14, 1, 512), (1, 512, 1024, 14, 2, 1)],
    'pw512_pws2_2048': [(1, 512, 512, 28, 1, 1), (1, 512, 2048, 28, 2, 1)],
    'pw1024_pws2_2048': [(1, 1024, 1024, 28, 1, 1), (1, 1024, 2048, 28, 2, 1)],
}


def chain_case(name, spec):
    body, current, max_weights = [], 'x', 0
    for i, (k, ci, co, sp, st, g) in enumerate(spec):
        item = conv.conv_case(k, ci, co, sp, st, g, True)
        text = item['mil'].split(') {\n', 1)[1].rsplit('  } ->', 1)[0]
        for n in ('pt', 'st', 'pd', 'dl', 'gp', 'w', 'b', 'y'):
            text = re.sub(r'\b' + n + r'\b', n + str(i), text)
        text = text.replace('x = x)', 'x = ' + current + ')')
        body.extend(text.strip().splitlines())
        current = 'y' + str(i)
        max_weights = max(max_weights, item['weights_description']['payload_bytes'])
    k, ci, co, sp, st, g = spec[0]
    return om.case(f'h14_s2link_{name}', 'h14_s2_link', dict(layers=spec),
                   om.program(f'{om.tensor_type((1, ci, sp, sp))} x', body, current),
                   om.blob(om.half_payload(max_weights // 2)),
                   dict(storage='BLOBFILE', value='fp16(0x1p-1)', payload_bytes=max_weights))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path(__file__).parent / 'oracles/h14-mobile')
    ap.add_argument('--case')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, spec in CHAINS.items():
        item = chain_case(name, spec)
        if args.case and args.case not in item['name']:
            continue
        out = args.output / (item['name'] + '.json')
        if out.exists() and not args.force:
            continue
        record = dict(schema_version=1, case=item['name'], family=item['family'],
                      target='h14', source_commit=om.source_commit(),
                      parameters=item['parameters'], mil=item['mil'],
                      weights=item['weights_description'], compile_only=True,
                      compiler={'platform': platform.platform(), 'target': 'H14',
                                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        try:
            record.update(compile_mil(item['mil'], item['weights']))
            record['error'] = None
        except Exception as exc:
            record['error'] = str(exc)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        print(out.name, 'tasks', len(record.get('task_descriptors') or []), record['error'])


if __name__ == '__main__':
    main()
