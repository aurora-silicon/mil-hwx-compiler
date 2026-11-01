#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build dependency-closed prefixes and isolated layers of a camera MIL graph.

Diagnostic packages only: none is a deployable camera model. The source must
use one statement per line, as emitted by yunet_linked.py. Original statement
text, weights and operation order are preserved; dead sibling branches are
pruned so the compiler can expose an intermediate as its sole output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h13_reference import Parser, Ref


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refs(value):
    if isinstance(value, Ref):
        return {value.name}
    if isinstance(value, dict):
        return set().union(set(), *(refs(v) for v in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(set(), *(refs(v) for v in value))
    return set()


class Graph:
    def __init__(self, text):
        self.parameters, self.operations, self.returns = Parser(text).parse()
        if len(self.parameters) != 1:
            raise ValueError('diagnostic graph needs one input')
        self.input = self.parameters[0][1]
        self.nodes = {op.result: op for op in self.operations}
        self.shapes = {self.input: self.parameters[0][0].shape}
        self.shapes.update({op.result: op.result_type.shape for op in self.operations})
        self.lines = {}
        for line in text.splitlines():
            match = re.match(r'^\s*(?:tensor<[^>]+>|\w+)\s+(\w+)\s*=.*;\s*$', line)
            if match:
                self.lines[match[1]] = line
        if set(self.lines) != set(self.nodes):
            raise ValueError('expected complete single-line MIL statements')

    def source(self, result, *, isolated=False):
        if result not in self.nodes or self.nodes[result].name == 'const':
            raise ValueError('checkpoint must be a runtime operation')
        root = self.nodes[result]
        # Preserve the compiler's conv+relu fusion in an isolated probe.
        primitive = self.nodes[root.arguments['x'].name] if root.name == 'relu' else root
        source = primitive.arguments.get('x') if isolated else None
        if isolated and (not isinstance(source, Ref) or primitive.name not in
                         ('conv',)):
            raise ValueError('isolated probe requires conv(+relu)')
        boundary = source.name if isolated else self.input
        needed = set()
        def visit(name):
            if name == boundary or name in needed:
                return
            needed.add(name)
            for dep in refs(self.nodes[name].arguments) | refs(self.nodes[name].attributes):
                visit(dep)
        visit(result)
        shape = list(self.shapes[boundary])
        body = '\n'.join(self.lines[op.result] for op in self.operations if op.result in needed)
        return (f'program(1.3) {{\nfunc main<ios18>(tensor<fp16, {shape}> {boundary}) {{\n' +
                body + f'\n}} -> ({result});\n}}\n'), boundary, len([
                    op for op in self.operations if op.result in needed and op.name not in ('const', 'relu')])


def build(directory, output, compiler, checkpoints=None):
    directory, output, compiler = Path(directory), Path(output), Path(compiler).resolve()
    source_manifest = json.loads((directory/'manifest.json').read_text())
    if len(source_manifest['programs']) != 1:
        raise ValueError('expected one linked source program')
    graph = Graph((directory/'model.mil').read_text())
    layers = source_manifest['programs'][0]['layers']
    selected = set(checkpoints) if checkpoints else {v['result'] for v in layers}
    if selected - {v['result'] for v in layers}:
        raise ValueError('unknown checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    for source,name in ((directory/'LICENSE-MODEL','LICENSE-MODEL'),
                        (Path(__file__).resolve().parents[1]/'LICENSE','LICENSE-COMPILER')):
        if source.is_file(): shutil.copyfile(source,output/name)
    index = dict(schema='aurora.camera-prefixes.v1', diagnostic_only=True,
                 source_files={p.name:sha(p) for p in directory.iterdir() if p.is_file()
                               and p.name in ('model.mil','weights.bin','manifest.json','program-0.anec')},
                 compiler_sha256=sha(compiler), records=[])
    for i, layer in enumerate(layers):
        result = layer['result']
        if result not in selected:
            continue
        for mode in ('prefix', 'isolated'):
            try:
                text, boundary, count = graph.source(result, isolated=mode == 'isolated')
            except ValueError:
                if mode == 'isolated': continue
                raise
            case = output/f'{i:03d}-{result}-{mode}'
            case.mkdir()
            (case/'model.mil').write_text(text)
            shutil.copyfile(directory/'weights.bin', case/'weights.bin')
            schedule = 'linked' if count >= 2 else 'per-op'
            run = subprocess.run([str(compiler),'--target','H13','--schedule',schedule,
                '--mil',str(case/'model.mil'),'--model-root',str(case),'--output',str(case/'package')],
                capture_output=True, text=True, timeout=300)
            (case/'compiler.log').write_text(run.stdout+run.stderr)
            record = dict(layer=i, result=result, mode=mode, directory=case.name,
                          source_first_task=0 if mode=='prefix' else layer['firstTask'],
                          source_tasks=layer['firstTask']+layer['taskDescriptors'] if mode=='prefix' else layer['taskDescriptors'],
                          boundary=boundary, shape=list(graph.shapes[result]), status='fail')
            index['records'].append(record)
            if run.returncode == 0:
                sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
                import inspect_anec
                try:
                    manifest, _ = inspect_anec.load_package(case/'package')
                except ValueError as error:
                    record['error'] = f'ANEC reader: {error}'
                    (output/'index.json').write_text(json.dumps(index, indent=2)+'\n')
                    continue
                if len(manifest['programs']) != 1 or manifest['dispatchPlan'] != [0]:
                    raise ValueError('diagnostic must have exactly one dispatch')
                record.update(status='pass',tasks=manifest['programs'][0]['taskDescriptors'])
                record['files'] = {str(p.relative_to(case)):sha(p) for p in case.rglob('*') if p.is_file()}
            else:
                record['error'] = run.stderr[-1600:]
            (output/'index.json').write_text(json.dumps(index, indent=2)+'\n')
    return index


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('package',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--compiler',required=True,type=Path)
    p.add_argument('--checkpoint',action='append')
    args = p.parse_args()
    result = build(args.package,args.output,args.compiler,args.checkpoint)
    failures = [r for r in result['records'] if r['status'] != 'pass']
    print(json.dumps(dict(probes=len(result['records']),failures=failures),indent=2))
    sys.exit(bool(failures))
