#!/usr/bin/env python3
"""Export public coremltools MIL and measure compiler refusals without a device.

Each non-constant operation is also compiled in isolation with its exact input
types and constants. These probes expose errors hidden behind the whole graph's
first refusal; they do not prove graph composition or device correctness.
Requires coremltools==9.0 and numpy. Never invokes CoreML execution or compilation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def type_info(var):
    from coremltools.converters.mil.mil import types
    dtype = types.builtin_to_string(var.dtype)
    return {'dtype': 'string' if dtype == 'str' else dtype,
            'shape': [int(x) for x in var.shape]}


def mil_type(info):
    if not info['shape']:
        return info['dtype']
    return f"tensor<{info['dtype']}, {info['shape']}>"


def literal(value):
    if isinstance(value, (str, np.str_)):
        return json.dumps(str(value))
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not np.isfinite(value):
        raise ValueError('nonfinite constants need an explicit MIL representation')
    return repr(float(value))


class Exporter:
    def __init__(self, function, directory):
        self.function = function
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.names = {}
        self.variables = {}
        self.constant_lines = {}
        self.constants = {}
        self.arrays = {}
        self.blob = bytearray(struct.pack('<II', 1, 2) + bytes(56))
        for var in function.inputs.values():
            self.register(var)
        self.operations = []
        for op in function.operations:
            if op.blocks or len(op.outputs) != 1:
                raise ValueError(f'{op.name}: nested/multi-result operations need an exporter')
            var = op.outputs[0]
            name = self.register(var)
            if op.op_type == 'const':
                self.constant_lines[name] = self.const_line(var)
                continue
            arguments = {}
            for key, value in op.inputs.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    raise ValueError(f'{op.name}: tuple input {key} needs an exporter')
                arguments[key] = self.names[id(value)]
            self.operations.append({'name': op.name, 'op': op.op_type,
                                    'result': name, 'type': self.variables[name],
                                    'inputs': arguments})
        (self.directory / 'weights').mkdir()
        (self.directory / 'weights/weight.bin').write_bytes(self.blob)
        np.savez(self.directory / 'constants.npz', **self.arrays)
        self.graph = {'inputs': [self.names[id(v)] for v in function.inputs.values()],
                      'outputs': [self.names[id(v)] for v in function.outputs],
                      'variables': self.variables, 'constants': self.constants,
                      'operations': self.operations}
        (self.directory / 'graph.json').write_text(json.dumps(self.graph, indent=2) + '\n')
        (self.directory / 'model.mil').write_text(self.program(self.operations))

    def register(self, var):
        # Stable generated identifiers avoid unsupported characters in CoreML names.
        name = f'v{len(self.names)}'
        self.names[id(var)] = name
        self.variables[name] = type_info(var)
        return name

    def const_line(self, var):
        name = self.names[id(var)]
        info = self.variables[name]
        value = np.asarray(var.val)
        typ = mil_type(info)
        if info['dtype'] == 'fp16' and info['shape']:
            data = value.astype('<f2').tobytes()
            offset = len(self.blob)
            self.blob += struct.pack('<IIQQ', 0xDEADBEEF, 1, len(data), offset + 64)
            self.blob += bytes(40) + data
            self.blob += bytes(-len(self.blob) % 64)
            expression = ('BLOBFILE(path = string("@model_path/weights/weight.bin"), '
                          f'offset = uint64({offset}))')
        elif info['shape']:
            expression = '[' + ', '.join(literal(x) for x in value.flat) + ']'
        else:
            expression = literal(value.item())
        if info['shape']:
            self.arrays[name] = value
            self.constants[name] = {'array': name}
        else:
            self.constants[name] = {'value': value.item()}
        return f'    {typ} {name} = const()[name = string("{name}"), val = {typ}({expression})];'

    @staticmethod
    def operation_line(op):
        args = ', '.join(f'{k} = {v}' for k, v in sorted(op['inputs'].items()))
        return (f"    {mil_type(op['type'])} {op['result']} = {op['op']}({args})"
                f"[name = string({json.dumps(op['name'])})];")

    def program(self, operations, isolated=False):
        if isolated:
            produced = {op['result'] for op in operations}
            operands = {v for op in operations for v in op['inputs'].values()}
            inputs = sorted(operands - produced - self.constant_lines.keys())
            constants = [n for n in self.constant_lines if n in operands]
            outputs = [operations[-1]['result']]
        else:
            inputs = self.graph['inputs']
            constants = list(self.constant_lines)
            outputs = self.graph['outputs']
        params = ', '.join(f'{mil_type(self.variables[n])} {n}' for n in inputs)
        body = [self.constant_lines[n] for n in constants]
        body += [self.operation_line(op) for op in operations]
        return ('program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
                f'  func main<ios18>({params}) {{\n' + '\n'.join(body) +
                '\n  } -> (' + ', '.join(outputs) + ');\n}\n')


def compile_probe(compiler, source, root, output, target, schedule='per-op', staged=False):
    command = [str(compiler), '--mil', str(source), '--model-root', str(root),
               '--target', target, '--output', str(output)]
    if not staged:
        command += ['--format', 'hwx', '--schedule', schedule]
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    errors = [{'code': code, 'message': message}
              for code, message in re.findall(r'error \[([^]]+)\]: ([^\n]+)', result.stderr)]
    if result.returncode and not errors:
        raise RuntimeError(f'compiler failed without a named error: {result.stderr}')
    if result.returncode and output.exists():
        raise RuntimeError(f'refused probe left a partial package: {output}')
    (output.parent / (output.name + '.stderr')).write_text(result.stderr)
    validation = []
    # Every emitted H16G object must pass the independent validator.
    if not result.returncode and target == 'H16G':
        validator = Path(__file__).with_name('h16g_validate.py')
        for program in sorted(output.glob('program-*.hwx')):
            checked = subprocess.run(['python3', str(validator), str(program),
                                      '--manifest', str(output / 'manifest.json')],
                                     capture_output=True, text=True, timeout=60)
            validation.append({'program': program.name, 'rc': checked.returncode,
                               'stdout': checked.stdout, 'stderr': checked.stderr})
            if checked.returncode:
                raise RuntimeError(f'emitted program failed validation: {validation[-1]}')
    return {'rc': result.returncode, 'errors': errors, 'validation': validation}


def measure(exporter, compiler, target, staged=False):
    root = exporter.directory
    probes = root / (target + ('-staged' if staged else ''))
    probes.mkdir()
    whole = {schedule: compile_probe(compiler, root / 'model.mil', root,
                                     probes / f'whole-{schedule}', target, schedule, staged)
             for schedule in (('default',) if staged else ('per-op', 'linked'))}
    rows = []
    for index, op in enumerate(exporter.operations):
        source = probes / f'op-{index:03d}.mil'
        source.write_text(exporter.program([op], isolated=True))
        result = compile_probe(compiler, source, root, probes / f'op-{index:03d}', target,
                               staged=staged)
        parameters = {}
        for key, name in op['inputs'].items():
            if name not in exporter.constants:
                continue
            constant = exporter.constants[name]
            value = exporter.arrays[constant['array']] if 'array' in constant else constant['value']
            value = np.asarray(value)
            parameters[key] = value.tolist() if value.size <= 8 else {'shape': list(value.shape)}
        row = {'index': index, **op, 'parameters': parameters, 'input_types': {
            k: exporter.variables[v] for k, v in op['inputs'].items()}, **result}
        if op['op'] == 'conv' and index + 1 < len(exporter.operations):
            relu = exporter.operations[index + 1]
            users = [o for o in exporter.operations if op['result'] in o['inputs'].values()]
            if relu['op'] == 'relu' and users == [relu]:
                fused_source = probes / f'fused-{index:03d}.mil'
                fused_source.write_text(exporter.program([op, relu], isolated=True))
                row['with_relu'] = compile_probe(compiler, fused_source, root,
                                                probes / f'fused-{index:03d}', target,
                                                staged=staged)
        rows.append(row)
    return {'whole_graph': whole, 'operations': rows,
            'standalone_refusals': sum(bool(row['rc']) for row in rows)}


def report(data):
    lines = ['# Real-model compiler coverage', '',
             'Offline compiler results. No device execution is claimed. Each row probes the',
             'exact operation and constants with runtime predecessors made boundary inputs.',
             'Standalone acceptance does not establish graph linking; standalone ReLU refusals',
             'can coexist with accepted conv+ReLU fusion. Whole-graph per-op and linked results',
             'are recorded separately. H16G-staged additionally measures the default CLI',
             'without --format/--schedule; H16G uses the explicit formula-HWX route.',
             'All named errors are retained in coverage.json.', '']
    for name, model in data['models'].items():
        for target, result in model['targets'].items():
            lines += [f'## {name} / {target}', '']
            for schedule, whole in result['whole_graph'].items():
                first = whole['errors'][0] if whole['errors'] else None
                lines.append(f"- Whole graph `{schedule}`: " + (
                    f"`{first['code']}` — {first['message']}" if first else 'accepted'))
            lines += [f"- Standalone refusals: {result['standalone_refusals']} / "
                      f"{len(result['operations'])} operations.", '',
                      '| # | Op / source name | Runtime input shapes → output | Named refusal / result |',
                      '| --- | --- | --- | --- |']
            for row in result['operations']:
                runtime = [f"{key}={typ['shape']}" for key, typ in row['input_types'].items()
                           if row['inputs'][key] not in model['constants']]
                errors = '; '.join(f"`{e['code']}`: {e['message']}" for e in row['errors']) or 'accepted'
                if 'with_relu' in row:
                    errors += '; conv+ReLU: ' + ('refused' if row['with_relu']['rc'] else 'accepted')
                geometry = ', '.join(runtime) + f" → {row['type']['shape']}"
                geometry += '; ' + ', '.join(f'{k}={v}' for k, v in row['parameters'].items())
                lines.append(f"| {row['index']} | {row['op']} / {row['name']} | {geometry} | {errors.replace('|', '/')} |")
            lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True, type=Path)
    parser.add_argument('--package', action='append', default=[], metavar='NAME=MLPACKAGE')
    parser.add_argument('--builder', action='append', default=[], metavar='NAME=SCRIPT:FUNCTION')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load
    args.out.mkdir(parents=True, exist_ok=False)
    programs = []
    for entry in args.package:
        name, path = entry.split('=', 1)
        path = Path(path)
        model = path / 'Data/com.apple.CoreML/model.mlmodel'
        spec = ct.utils.load_spec(str(path))
        prog = load(spec, spec.specificationVersion,
                    file_weights_dir=str(model.parent / 'weights'))
        provenance = {'kind': 'mlpackage', 'model_sha256': digest(model),
                      'weight_sha256': {p.name: digest(p) for p in sorted((model.parent / 'weights').glob('*'))}}
        programs.append((name, prog, provenance))
    for entry in args.builder:
        name, builder = entry.split('=', 1)
        path, function = builder.rsplit(':', 1)
        spec = importlib.util.spec_from_file_location('public_model_builder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        programs.append((name, getattr(module, function)(),
                         {'kind': 'MIL builder', 'script_sha256': digest(path), 'function': function}))
    if not programs or len({name for name, _, _ in programs}) != len(programs):
        parser.error('provide at least one uniquely named model')
    data = {'coremltools': ct.__version__, 'compiler_sha256': digest(args.compiler), 'models': {}}
    for name, prog, provenance in programs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            parser.error('model names must be simple directory names')
        exporter = Exporter(prog.functions['main'], args.out / name)
        model = {'provenance': provenance, 'constants': list(exporter.constants),
                 'targets': {t: measure(exporter, args.compiler, t) for t in ('H16G', 'H17')}}
        model['targets']['H16G-staged'] = measure(exporter, args.compiler, 'H16G', staged=True)
        data['models'][name] = model
        print(name, {t: v['standalone_refusals'] for t, v in model['targets'].items()}, flush=True)
    (args.out / 'coverage.json').write_text(json.dumps(data, indent=2) + '\n')
    (args.out / 'compiler-coverage.md').write_text(report(data))


if __name__ == '__main__':
    main()
