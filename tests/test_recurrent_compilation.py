#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile decomposed padded cells, including H14's two-state LSTM return."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from recurrent_lowering import gru, lstm
from graph_budget import estimate

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else 'build/mil-hwxc').resolve()


def main():
    rng = np.random.default_rng(3824)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for kind, reset, activation in (('GRU', 0, 'relu'), ('GRU', 1, 'tanh'), ('LSTM', 0, 'tanh')):
            gates, hidden, inputs = (3 if kind == 'GRU' else 4), 11, 7
            W, R, B = [rng.normal(0, .1, shape).astype(np.float16).astype(np.float32)
                       for shape in ((gates * hidden, inputs), (gates * hidden, hidden), (2 * gates * hidden,))]
            graph = (gru(W, R, B, linear_before_reset=reset, candidate=activation, pad_to=512)
                     if kind == 'GRU' else lstm(W, R, B, pad_to=512))
            source = root / f'{kind}-{reset}-{activation}'
            graph.write(source, runtime_vectors=True)
            for target in ('H13', 'H14'):
                for schedule in ('per-op', 'regions'):
                    for fmt in ('anec', 'hwx'):
                        output = source / f'{target}-{schedule}-{fmt}'
                        run = subprocess.run([str(COMPILER), '--mil', str(source / 'cell.mil'),
                            '--model-root', str(source), '--output', str(output), '--target', target,
                            '--schedule', schedule, '--format', fmt], capture_output=True, text=True, timeout=120)
                        assert run.returncode == 0, run.stdout + run.stderr
                        manifest = json.loads((output / 'manifest.json').read_text())
                        expected_count = 25 if kind == 'LSTM' else 23 if activation == 'relu' and schedule == 'regions' else 24
                        assert len(manifest['dispatchPlan']) == expected_count
                        assert [r['name'] for r in manifest['logicalResults']] == graph.outputs
                        for name in graph.outputs:
                            assert manifest['tensors'][name]['role'] == 'output'
                            assert name not in manifest['intermediates']
                        if kind == 'LSTM':
                            assert len(manifest['physicalOutputs']) == 2
                            readers = [p for p in manifest['programs'] if any(i['name'] == 'c_next' for i in p['inputs'])]
                            assert len(readers) == 1 and readers[0]['operation'] == 'tanh'
                        if target == 'H13' and fmt == 'anec':
                            from research import inspect_anec
                            inspect_anec.load_package(output)
                        bound = estimate([manifest], .25)
                        assert bound['dispatches'] == expected_count and not bound['device_budget_proven']
                        print(f'{target} {schedule} {fmt} {kind} reset={reset} {activation}: {expected_count} dispatches')
    print('decomposed recurrent compilation: PASS')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
