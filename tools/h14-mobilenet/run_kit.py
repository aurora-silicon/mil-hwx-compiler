#!/usr/bin/env python3
"""Run an H14 linked-ring kit through the ABI-2 C runner, smallest first."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'h13-mobilenet'))
from chain import pack, unpack  # noqa: E402


def cases(kit, order):
    for item in order:
        if item == 'prefixes/p01..p28':
            yield from (kit / 'prefixes' / f'p{index:02d}'
                        for index in range(1, 29))
        else:
            yield kit / item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kit', type=Path)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--max-relative-l2', type=float, default=1e-2)
    args = parser.parse_args()
    kit, runner = args.kit.resolve(), args.runner.resolve()
    index = json.loads((kit / 'kit.json').read_text())
    reports = []
    with tempfile.TemporaryDirectory(prefix='h14-ring-kit-run-') as directory:
        inp, out = Path(directory) / 'input.bin', Path(directory) / 'output.bin'
        for case in cases(kit, index['executionOrder']):
            manifest = json.loads((case / 'linked/manifest.json').read_text())
            program = manifest['programs'][0]
            refs = np.load(case / 'refs.npz', allow_pickle=False)
            if len(refs['inputs']) != 1:
                raise SystemExit(f'{case}: kit runner requires one reference')
            inp.write_bytes(pack(refs['inputs'][0], program['inputs'][0]['nchw'],
                                 program['inputs'][0]['allocationBytes']))
            run = subprocess.run([
                str(runner), str(case / 'linked' / program['file']), str(inp),
                str(out), str(args.repeat), str(args.device)], check=True,
                text=True, capture_output=True, timeout=600)
            actual = unpack(out.read_bytes(), program['outputs'][0]['nchw'])
            expected = refs['outputs_float64'][0]
            relative = float(np.linalg.norm(actual.astype(np.float64) - expected) /
                             max(np.linalg.norm(expected), 1e-30))
            if not np.isfinite(actual).all() or relative > args.max_relative_l2:
                raise SystemExit(f'{case}: relative L2 {relative:.6g} exceeds '
                                 f'{args.max_relative_l2:.6g}')
            report = dict(case=str(case.relative_to(kit)), relative_l2=relative,
                          runtime=json.loads(run.stdout.strip().splitlines()[-1]))
            reports.append(report)
            print(json.dumps(report), flush=True)
    print(json.dumps(dict(status='PASS', cases=len(reports), reports=reports)))


if __name__ == '__main__':
    main()
