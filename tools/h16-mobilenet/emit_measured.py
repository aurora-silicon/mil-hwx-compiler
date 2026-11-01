#!/usr/bin/env python3
"""Emit open-compiled standalone stride-1 packages for lead-run ABI tests.

These six measured cases per target are NOT full MobileNet or runtime proof.
Uses a correctly formed source-owned BLOBFILE with fp16(0.5) weights, rather
than the campaign's legacy header view. No hardware is accessed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--compiler', type=Path, default=ROOT / 'build/mil-hwxc')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = {'fullMobileNet': False, 'deviceValidated': False, 'packages': []}
    for target, isa in (('H16G','h16'), ('H17','h17')):
        for path in sorted((ROOT / 'research/oracles' / isa).glob('conv_desc_*.json')):
            oracle = json.loads(path.read_text())
            if len(oracle['task_descriptors']) != 1 or oracle['parameters']['stride'] != 1:
                continue
            case = args.output / target / path.stem
            case.mkdir(parents=True)
            (case / 'model.mil').write_text(oracle['mil'])
            size = oracle['weights']['payload_bytes']
            weights = bytearray(128 + size)
            struct.pack_into('<II',weights,0,1,2)
            struct.pack_into('<IIQQ',weights,64,0xDEADBEEF,1,size,128)
            weights[128:] = b'\x00\x38' * (size//2)
            (case / 'weights.bin').write_bytes(weights)
            run = subprocess.run([str(args.compiler.resolve()), '--target', target,
                '--format','hwx','--mil',str(case/'model.mil'),'--model-root',str(case),
                '--output',str(case/'package')],capture_output=True,text=True,timeout=120)
            if run.returncode:
                raise RuntimeError(f'{target} {path.stem}: {run.stderr}')
            artifact = case / 'package/program-0.hwx'
            index['packages'].append({'target':target, 'case':path.stem,
                'file':str(artifact.relative_to(args.output)), 'bytes':artifact.stat().st_size,
                'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
                'weights':'fp16(0.5), correctly formed BLOBFILE',
                'validation':'descriptor parity; execution pending'})
    (args.output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(f"{len(index['packages'])} standalone packages emitted; fullMobileNet=false, deviceValidated=false")


if __name__ == '__main__':
    main()
