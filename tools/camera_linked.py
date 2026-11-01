#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a single-dispatch camera package, preserving compiler refusals.

Inputs are the audited camera-effects export (model.json/MIL/ONNX/weights).
No island fallback or graph splitting is permitted. Generated weights and
programs are install-time data, outside public source control.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'research'))
import inspect_anec
from h13_reference import Parser


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(model, output, compiler, schedule='linked'):
    model, output, compiler = model.resolve(), output.resolve(), compiler.resolve()
    metadata = json.loads((model/'model.json').read_text())
    if metadata.get('model') not in ('selfie', 'yunet'):
        raise ValueError('expected audited selfie or yunet model')
    for name in ('model.onnx', 'model.mil', 'weights.bin'):
        if metadata['artifacts'].get(name) != sha(model/name):
            raise ValueError(f'model export identity mismatch: {name}')
    # Refuse reuse; retain both failed and successful attempts for comparison.
    output.mkdir(parents=True, exist_ok=False)
    parameters, operations, returns = Parser((model/'model.mil').read_text()).parse()
    result = dict(schema='aurora.camera-linked-build.v1', status='fail', target='H13',
                  model=metadata['model'], compiler_sha256=sha(compiler), schedule=schedule,
                  source_sha256=metadata['source_sha256'], mil_sha256=sha(model/'model.mil'),
                  operation_counts=dict(Counter(op.name for op in operations if op.name != 'const')),
                  boundary_inputs=len(parameters), boundary_outputs=len(returns))
    package = output/'package'
    command = [str(compiler), '--target', 'H13', '--schedule', schedule,
               '--mil', str(model/'model.mil'), '--model-root', str(model), '--output', str(package)]
    try:
        start = time.monotonic()
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 timeout=300)
        (output/'compiler.log').write_bytes(process.stdout)
        result.update(compiler_rc=process.returncode, seconds=time.monotonic()-start)
        if process.returncode:
            result['reason'] = 'compiler refused full model; no linked package produced'
            return result
        manifest, _ = inspect_anec.load_package(package)
        if len(manifest['programs']) != 1 or manifest['dispatchPlan'] != [0]:
            raise ValueError('compiler output is not one program/one dispatch')
        program = manifest['programs'][0]
        if (program.get('operation') != 'linked' or len(program['inputs']) != 1 or
                program['inputs'][0]['shape'] != metadata['input'] or
                [b['name'] for b in program['outputs']] != metadata['mil_outputs'] or
                [b['shape'] for b in program['outputs']] != metadata['output_shapes']):
            raise ValueError('linked camera boundary contract mismatch')
        for name in ('model.mil', 'weights.bin'):
            shutil.copyfile(model/name, package/name)
        files = ['manifest.json', 'model.mil', 'weights.bin', program['file']]
        plan = dict(schema='aurora.camera-linked.v1', target='H13', model=metadata['model'],
                    source_sha256=metadata['source_sha256'], onnx_sha256=sha(model/'model.onnx'),
                    compiler_sha256=sha(compiler), schedule=schedule,
                    input_shape=metadata['input'], output_shapes=metadata['output_shapes'],
                    output_names=metadata['output_names'], mil_outputs=metadata['mil_outputs'],
                    files={name:sha(package/name) for name in files})
        (package/'camera-linked.json').write_text(json.dumps(plan, indent=2)+'\n')
        result.update(status='pass', package_sha256=sha(package/'camera-linked.json'),
                      programs=1, dispatches=1, tasks=program['taskDescriptors'],
                      scratch_bytes=program['scratchBytes'])
        return result
    except subprocess.TimeoutExpired as error:
        (output/'compiler.log').write_bytes(error.stdout or b'')
        result['reason'] = 'compiler exceeded 300-second build deadline'
        return result
    except Exception as error:
        result['reason'] = f'{type(error).__name__}: {error}'
        return result
    finally:
        (output/'build.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, default=Path(__file__).resolve().parents[1]/'build/mil-hwxc')
    parser.add_argument('--schedule', choices=['linked', 'linked-l2'], default='linked')
    args = parser.parse_args()
    try:
        result = build(args.model, args.output, args.compiler, args.schedule)
    except Exception as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
