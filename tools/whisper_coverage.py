#!/usr/bin/env python3
"""Replay the pinned public Whisper coverage without conflating refusals and timeouts.

Device-free. Every run uses a new package directory. Public weights are supplied
by the caller and verified against the receipt before invoking the compiler.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/whisper'
DIAGNOSTIC = re.compile(r'error \[([^\]]+)\]')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_cases():
    from h13_reference import Parser, Call, _resolve, _flatten

    def contains_blob(expression):
        if isinstance(expression, Call):
            return expression.name == 'BLOBFILE' or any(contains_blob(v) for v in
                (*expression.positional, *expression.named.values()))
        return isinstance(expression, list) and any(contains_blob(v) for v in expression)
    index = json.loads((FIXTURES / 'probes.json').read_text())
    if sha256(FIXTURES / 'compiler-coverage.json') != index['receipt_sha256']:
        raise ValueError('coverage receipt hash mismatch')
    receipt = json.loads((FIXTURES / 'compiler-coverage.json').read_text())
    cases = []
    for model, details in receipt['models'].items():
        for target, target_details in details['targets'].items():
            records = target_details['records']
            if len(records) != 21 or len({r['name'] for r in records}) != 21:
                raise ValueError('coverage must retain the full model and 20 unique probes')
            for record in records:
                if record['name'] != 'full-model':
                    rel = str(Path(model) / record['mil'])
                    if sha256(FIXTURES / rel) != index['files'][rel]:
                        raise ValueError(f'probe hash mismatch: {rel}')
                    parameters, ops, returns = Parser((FIXTURES / rel).read_text()).parse()
                    if {name: list(t.shape) for t, name in parameters} != record['inputs']:
                        raise ValueError(f'probe input geometry mismatch: {rel}')
                    constants = {op.result: list(op.result_type.shape) for op in ops if op.name == 'const'}
                    if constants != record['constants'] or len(returns) != 1:
                        raise ValueError(f'probe constant geometry mismatch: {rel}')
                    op = ops[-1]
                    if op.name != record['op'] or list(op.result_type.shape) != record['output'] or returns != [op.result]:
                        raise ValueError(f'probe output geometry mismatch: {rel}')
                    environment = {}
                    for constant in ops:
                        if constant.name == 'const' and not contains_blob(constant.attributes['val']):
                            environment[constant.result] = _resolve(constant.attributes['val'], environment, FIXTURES)
                    for key, expected in record['attributes'].items():
                        actual = _resolve(op.arguments[key], environment, FIXTURES)
                        if _flatten([actual]) != _flatten([expected]):
                            raise ValueError(f'probe attribute mismatch: {rel}/{key}')
                cases.append((model, target, record))
    return receipt, cases


def classify(rc, stderr, program_count):
    codes = sorted(set(DIAGNOSTIC.findall(stderr)))
    if rc == 124:
        return 'timeout', codes
    if rc == 0 and program_count is not None and program_count > 0:
        return 'compiled', codes
    if rc == 65 and codes:
        return ('budget-refused' if codes == ['h13.program-budget'] else 'refused'), codes
    return 'error', codes


def baseline_outcome(record):
    # The source receipt labels rc=124 as "error"; preserve that raw field in
    # results, but compare it as a host timeout, never an unsupported op.
    return 'timeout' if record.get('rc') == 124 else record['outcome']


def compare(baseline, outcome, codes, program_count):
    # Budget and elapsed-time limits describe a compilation resource bound,
    # never missing mathematical support or native correctness.
    if outcome == 'budget-refused':
        return 'bounded', baseline_outcome(baseline) in ('compiled', 'timeout')
    if outcome in ('timeout', 'error'):
        return 'unresolved', False
    if baseline_outcome(baseline) == 'refused':
        matched = outcome == 'refused' and codes == sorted(set(baseline['diagnostic_codes']))
        return ('unchanged-refusal' if matched else 'changed-needs-review'), matched
    if baseline_outcome(baseline) == 'compiled':
        matched = outcome == 'compiled' and program_count == baseline['program_count']
        return ('unchanged-compile' if matched else 'changed-needs-review'), matched
    return 'changed-needs-review', False


def run_case(command, timeout, package):
    started = time.monotonic()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            rc = process.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            rc = 124
    count, package_digest = None, None
    if rc == 0:
        try:
            programs = json.loads((package / 'manifest.json').read_text())['programs']
            names = [p['file'] for p in programs]
            files = {p.name for p in package.glob('program-*') if p.is_file()}
            if len(names) != len(set(names)) or set(names) != files:
                raise ValueError('program files differ from manifest')
            digest = hashlib.sha256()
            for name in names:
                if Path(name).name != name or not name.startswith('program-') or not (package / name).stat().st_size:
                    raise ValueError('invalid or empty program file')
                digest.update(name.encode() + b'\0' + bytes.fromhex(sha256(package / name)))
            count, package_digest = len(programs), digest.hexdigest()
        except (OSError, ValueError, KeyError, TypeError) as error:
            stderr += f'\nqualification package validation: {error}\n'
    outcome, codes = classify(rc, stderr, count)
    if rc != 0 and package.exists():
        outcome = 'error'
        stderr += '\nqualification: failed compilation left a package directory\n'
    return dict(rc=rc, outcome=outcome, diagnostic_codes=codes,
                program_count=count, package_programs_sha256=package_digest,
                seconds=round(time.monotonic() - started, 6), stdout=stdout, stderr=stderr)


def checkpoint(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='check receipt and probe integrity only')
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--models-root', type=Path, help='tiny.en/base.en/small.en export directories')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', action='append', choices=['tiny.en', 'base.en', 'small.en'])
    parser.add_argument('--target', action='append', choices=['H13', 'H14'])
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--max-programs', type=int, default=2048,
                        help='H13 limit for prior successes and timeouts')
    parser.add_argument('--refusal-max-programs', type=int, default=65536,
                        help='H13 limit for prior refusals, to retain their semantic diagnostics')
    args = parser.parse_args(argv)
    receipt, cases = load_cases()
    if args.check:
        print(json.dumps(dict(cases=len(cases), isolated=len(cases) - 6,
                              receipt_sha256=sha256(FIXTURES / 'compiler-coverage.json'))))
        return 0
    if not all((args.compiler, args.models_root, args.output)) or not 0 < args.timeout < float('inf') or args.max_programs <= 0 or args.refusal_max_programs <= 0:
        parser.error('replay needs --compiler, --models-root, --output and positive finite limits')
    compiler = args.compiler.resolve(strict=True)
    cases = [(m, t, r) for m, t, r in cases if (not args.model or m in args.model) and (not args.target or t in args.target)]
    # Verify all selected model bytes before creating a result or starting any compiler.
    for model in sorted({m for m, _, _ in cases}):
        for file, digest in receipt['models'][model]['files'].items():
            if sha256(args.models_root / model / file) != digest:
                raise ValueError(f'public model hash mismatch: {model}/{file}')
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='mil-hwxc.whisper-qualification.v1',
                  baseline_compiler_revision=receipt['compiler_revision'],
                  compiler_sha256=sha256(compiler),
                  receipt_sha256=sha256(FIXTURES / 'compiler-coverage.json'),
                  timeout_s=args.timeout, max_programs=args.max_programs,
                  refusal_max_programs=args.refusal_max_programs,
                  device_calls=False, native_qualified=False, records=[])
    for model, target, record in cases:
        with tempfile.TemporaryDirectory(prefix='whisper-qualification-') as tmp:
            package = Path(tmp) / 'package'
            mil = args.models_root / model / 'model.mil' if record['name'] == 'full-model' else FIXTURES / model / record['mil']
            command = [str(compiler), '--mil', str(mil), '--model-root', str((args.models_root / model).resolve()),
                       '--target', target, '--output', str(package)]
            effective_budget = args.refusal_max_programs if baseline_outcome(record) == 'refused' else args.max_programs
            if target == 'H13':
                command += ['--max-programs', str(effective_budget)]
            actual = run_case(command, args.timeout, package)
            comparison, expected = compare(record, actual['outcome'], actual['diagnostic_codes'], actual['program_count'])
            if not expected and package.exists():
                retained = args.output / 'failures' / model / target / record['name']
                retained.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(package, retained, symlinks=True)
                shutil.copyfile(mil, retained / 'source.mil')
                actual['preserved_package'] = str(retained.relative_to(args.output))
        entry = dict(model=model, target=target, name=record['name'], mil_sha256=sha256(mil), baseline_outcome=baseline_outcome(record), baseline_raw_outcome=record['outcome'],
                     effective_max_programs=effective_budget if target == 'H13' else None,
                     baseline_program_count=record['program_count'], comparison=comparison,
                     expected=expected, **actual)
        result['records'].append(entry)
        # Checkpoint every case, including individual failure logs. Host paths
        # and commands are not part of a shareable numeric coverage record.
        checkpoint(args.output / 'results.json', result)
        print(f"{model} {target} {record['name']}: {actual['outcome']} ({comparison})", flush=True)
    from collections import Counter
    result['summary'] = dict(Counter(r['outcome'] for r in result['records']))
    result['review_passed'] = all(r['expected'] for r in result['records'])
    result['runnable_full_models'] = sum(r['name'] == 'full-model' and r['outcome'] == 'compiled' for r in result['records'])
    checkpoint(args.output / 'results.json', result)
    print(json.dumps({k: result[k] for k in ('summary', 'review_passed', 'runnable_full_models', 'native_qualified')}))
    return 0 if result['review_passed'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f'whisper qualification: {error}', file=sys.stderr)
        sys.exit(1)
