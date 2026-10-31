#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit every DFN3 ONNX node and compile exact-shape primitive/cell probes.

A probe is not a full-model package. This tool never opens an ANE or audio
output. CPU ONNX evaluation supplies observed shapes at the selected frame
count; constant provenance is tracked separately from runtime sample values.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tarfile

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from recurrent_lowering import gru, lstm

DFN3_SHA256 = 'c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616'
DFN3_MEMBERS = {'enc.onnx', 'erb_dec.onnx', 'df_dec.onnx'}


class Unsupported(ValueError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_value(value):
    if isinstance(value, bytes): return value.decode('utf-8')
    if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
    if isinstance(value, onnx.TensorProto):
        a = numpy_helper.to_array(value)
        return {'dtype': str(a.dtype), 'shape': list(a.shape), 'sha256': sha(a.tobytes())}
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def observe(model, frames):
    """One deterministic CPU sample, exposing every output without optimizers.

    Sample values never become constants just because they were observed.
    Only initializers/Constant, fixed Shape results and their pure static
    descendants are marked constant for this shape-specialized audit.
    """
    model = copy.deepcopy(model)
    feeds = {}
    for value in model.graph.input:
        shape = []
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_param == 'S': dimension.dim_value = frames
            if not dimension.HasField('dim_value') or dimension.dim_value <= 0:
                raise Unsupported('coverage.dynamic-input: only the explicit S dimension is specialized')
            shape.append(dimension.dim_value)
        dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        feeds[value.name] = np.zeros(shape, dtype=dtype)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    types = {v.name: v.type.tensor_type.elem_type for v in
             [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
    names = [name for node in model.graph.node for name in node.output if name]
    del model.graph.output[:]
    for name in names:
        dtype = types.get(name)
        if not dtype:
            raise Unsupported('coverage.unknown-type: ' + name)
        model.graph.output.append(helper.make_tensor_value_info(name, dtype, None))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=['CPUExecutionProvider'])
    arrays = dict(zip(names, session.run(names, feeds)))
    arrays.update(feeds)
    arrays.update({v.name: numpy_helper.to_array(v) for v in model.graph.initializer})
    constants = {v.name for v in model.graph.initializer}
    for node in model.graph.node:
        # All ONNX nodes in these pinned graphs are deterministic base-domain
        # operations. Do not constant-fold an unknown domain or random op.
        foldable = node.domain in ('', 'ai.onnx') and node.op_type not in (
            'RandomNormal', 'RandomUniform', 'RandomNormalLike', 'RandomUniformLike', 'Dropout')
        if foldable and (node.op_type in ('Constant', 'Shape') or
                         all(name in constants for name in node.input if name)):
            constants.update(name for name in node.output if name)
    return arrays, constants


class MILProbe:
    def __init__(self, directory):
        self.directory = directory
        self.inputs = []
        self.lines = []
        self.blob = bytearray(64)

    @staticmethod
    def typ(array):
        if array.dtype.kind != 'f' or not array.shape:
            raise Unsupported('coverage.non-float-tensor: runtime probes require ranked floating tensors')
        return 'tensor<fp16, [' + ','.join(map(str, array.shape)) + ']>'

    def operand(self, name, array, constant=False):
        typ = self.typ(array)
        if constant:
            while len(self.blob) % 64: self.blob.append(0)
            offset = len(self.blob)
            payload = np.asarray(array, dtype='<f2').tobytes()
            self.blob.extend(struct.pack('<IIQQ', 0xDEADBEEF, 0, len(payload), offset + 64))
            self.blob.extend(bytes(40))
            self.blob.extend(payload)
            self.lines.append(f'{typ} {name} = const()[name = string("{name}"), val = {typ}'
                f'(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offset})))];')
        else:
            self.inputs.append(f'{typ} {name}')
        return name

    def vector(self, name, values):
        t = f'tensor<int32, [{len(values)}]>'
        self.lines.append(f'{t} {name} = const()[name = string("{name}"), val = {t}({list(values)})];')
        return name

    def write(self, op, args, output):
        self.lines.append(f'{self.typ(output)} result = {op}({args})[name = string("result")];')
        self.directory.mkdir(parents=True, exist_ok=True)
        body = '\n'.join('    ' + line for line in self.lines)
        text = ('program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
                f'  func main<ios18>({", ".join(self.inputs)}) {{\n{body}\n  }} -> (result);\n}}\n')
        (self.directory / 'cell.mil').write_text(text)
        (self.directory / 'weights.bin').write_bytes(self.blob)


def emit_probe(node, arrays, constants, directory):
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if node.domain not in ('', 'ai.onnx'):
        raise Unsupported('coverage.domain: nonstandard ONNX domain')
    op = node.op_type
    if op in ('GRU', 'LSTM'):
        if attrs.get('direction', b'forward') != b'forward' or attrs.get('layout', 0) != 0:
            raise Unsupported('coverage.recurrent-direction: only forward layout-0 cells')
        allowed = {'hidden_size', 'linear_before_reset', 'direction', 'layout'} if op == 'GRU' else {
            'hidden_size', 'direction', 'layout'}
        if set(attrs) - allowed or len(node.input) > (6 if op == 'GRU' else 7):
            raise Unsupported('coverage.recurrent-attributes: nondefault activations/clip/peepholes')
        if any(name not in constants for name in node.input[1:4] if name):
            raise Unsupported('coverage.recurrent-weights: weights must be static')
        W, R = [arrays[name][0] for name in node.input[1:3]]
        B = arrays[node.input[3]][0] if len(node.input) > 3 and node.input[3] else None
        graph = (gru(W, R, B, linear_before_reset=attrs.get('linear_before_reset', 0), pad_to=512)
                 if op == 'GRU' else lstm(W, R, B, pad_to=512))
        graph.write(directory, runtime_vectors=True)
        return {'scope': 'padded-single-step-cell', 'state_explicit': True,
                'padding': 512, 'sequence_loop_packaged': False,
                'valid_sizes': graph.valid_sizes}
    probe = MILProbe(directory)
    if not node.output or len(node.output) != 1:
        raise Unsupported('coverage.multi-result: operation needs explicit lowering')
    output = arrays[node.output[0]]
    unary = {'Relu': 'relu', 'Sigmoid': 'sigmoid', 'Tanh': 'tanh'}
    if op in unary:
        x = probe.operand('x', arrays[node.input[0]])
        probe.write(unary[op], f'x = {x}', output)
    elif op in ('Add', 'Mul', 'MatMul'):
        x = probe.operand('x', arrays[node.input[0]], node.input[0] in constants)
        y = probe.operand('y', arrays[node.input[1]], node.input[1] in constants)
        args = f'x = {x}, y = {y}'
        if op == 'MatMul':
            probe.lines.append('bool f = const()[name = string("f"), val = bool(false)];')
            args += ', transpose_x = f, transpose_y = f'
        probe.write({'Add': 'add', 'Mul': 'mul', 'MatMul': 'matmul'}[op], args, output)
    elif op == 'Conv':
        rank = arrays[node.input[0]].ndim - 2
        if rank != 2 or any(name not in constants for name in node.input[1:] if name):
            raise Unsupported('coverage.conv-weights: rank-4 convolution with static weights required')
        if set(attrs) - {'dilations', 'group', 'kernel_shape', 'pads', 'strides', 'auto_pad'}:
            raise Unsupported('coverage.conv-attributes: unhandled attributes')
        probe.operand('x', arrays[node.input[0]])
        probe.operand('w', arrays[node.input[1]], True)
        args = 'x = x, weight = w'
        if len(node.input) > 2 and node.input[2]:
            probe.operand('b', arrays[node.input[2]], True)
            args += ', bias = b'
        padding = attrs.get('pads', [0] * 4)
        auto = attrs.get('auto_pad', b'NOTSET')
        if auto not in (b'NOTSET', b'VALID', b'SAME_UPPER', b'SAME_LOWER'):
            raise Unsupported('coverage.conv-auto-pad: unknown padding')
        pad_type = {b'VALID': 'valid', b'SAME_UPPER': 'same', b'SAME_LOWER': 'same_lower'}.get(
            auto, 'custom' if any(padding) else 'valid')
        # ONNX: begin_H, begin_W, end_H, end_W. MIL: H_before, H_after, W_before, W_after.
        probe.vector('pad', [padding[0], padding[2], padding[1], padding[3]])
        probe.vector('strides', attrs.get('strides', [1, 1]))
        probe.vector('dilations', attrs.get('dilations', [1, 1]))
        probe.lines += [f'int32 groups = const()[name = string("groups"), val = int32({attrs.get("group", 1)})];',
                        f'string pad_type = const()[name = string("pad_type"), val = string("{pad_type}")];']
        probe.write('conv', args + ', pad = pad, strides = strides, dilations = dilations, '
                    'groups = groups, pad_type = pad_type', output)
    else:
        raise Unsupported('coverage.onnx-' + op.lower() + ': no full-graph lowering/host-step contract')
    return {'scope': 'isolated-exact-shape-primitive', 'weights': 'fp16-rounded-source'}


def compile_probe(compiler, directory, target, schedule):
    output = directory / (target + '-' + schedule)
    run = subprocess.run([str(compiler), '--mil', str(directory / 'cell.mil'),
        '--model-root', str(directory), '--target', target, '--schedule', schedule,
        '--output', str(output)], capture_output=True, text=True, timeout=120)
    (directory / (target + '-' + schedule + '.log')).write_text(run.stdout + run.stderr)
    report = {'target': target, 'schedule': schedule, 'rc': run.returncode,
              'diagnostic': run.stderr.strip(), 'dispatches': None,
              'codes': re.findall(r'\[([^\]]+)\]', run.stderr)}
    report['result'] = ('compiled-fragment' if run.returncode == 0 else
        'refused' if run.returncode == 65 and any(c.startswith(target.lower() + '.') for c in report['codes'])
        else 'error')
    if run.returncode == 0:
        manifest = json.loads((output / 'manifest.json').read_text())
        report.update(dispatches=len(manifest['dispatchPlan']),
                      regions=manifest.get('regions', []),
                      logical_results=manifest.get('logicalResults', []))
    return report


def audit_graph(data, name, frames, compiler, output):
    model = onnx.load_from_string(data)
    arrays, constants = observe(model, frames)
    rows = []
    for index, node in enumerate(model.graph.node):
        attrs = {a.name: json_value(helper.get_attribute_value(a)) for a in node.attribute}
        row = {'index': index, 'name': node.name, 'op': node.op_type, 'attributes': attrs,
               'inputs': [{'name': n, 'shape': list(arrays[n].shape), 'dtype': str(arrays[n].dtype),
                           'constant': n in constants} for n in node.input if n],
               'outputs': [{'name': n, 'shape': list(arrays[n].shape), 'dtype': str(arrays[n].dtype)}
                           for n in node.output if n]}
        if all(n in constants for n in node.output if n):
            row.update(status='compile-time-value', programs=0)
        else:
            directory = output / Path(name).stem / f'node-{index}'
            try:
                row['probe'] = emit_probe(node, arrays, constants, directory)
                row['compilations'] = ([compile_probe(compiler, directory, target, schedule)
                    for target in ('H13', 'H14') for schedule in ('per-op', 'regions')]
                    if compiler else [])
                row['status'] = 'fragment-probed' if compiler else 'fragment-exported'
            except Unsupported as error:
                row.update(status='lowering-required', diagnostic=str(error))
        rows.append(row)
    return {'file': name, 'sha256': sha(data), 'nodes': rows,
            'status_counts': dict(Counter(row['status'] for row in rows)),
            'op_counts': dict(sorted(Counter(node.op_type for node in model.graph.node).items()))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--df-model', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--compiler', type=Path)
    mode.add_argument('--prepare-only', action='store_true',
                      help='observe every node and export probes, without claiming compilation')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=1)
    a = p.parse_args()
    if a.frames <= 0: p.error('--frames must be positive')
    payload = a.df_model.read_bytes()
    if sha(payload) != DFN3_SHA256: p.error('coverage.model-identity: expected audited public DFN3 archive')
    if a.output.exists() and any(a.output.iterdir()): p.error('--output must be empty')
    a.output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'mil-hwxc.dfn3-coverage.v1', 'archive_sha256': sha(payload),
        'compiler_sha256': sha(a.compiler.read_bytes()) if a.compiler else None,
        'compiler_probes_run': bool(a.compiler), 'frames': a.frames,
        'shape_source': 'deterministic CPU ONNX sample; fixed S, not a streaming-state proof',
        'complete_native_graph': False, 'streaming_state_complete': False,
        'full_model_package_validated': False, 'device_execution': False,
        'versions': {'numpy': np.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
        'graphs': []}
    with tarfile.open(a.df_model) as archive:
        members = [m for m in archive.getmembers() if Path(m.name).name in DFN3_MEMBERS]
        if len(members) != 3 or {Path(m.name).name for m in members} != DFN3_MEMBERS:
            p.error('coverage.archive-members: expected exactly the three audited graphs')
        for member in members:
            report['graphs'].append(audit_graph(archive.extractfile(member).read(), member.name,
                a.frames, a.compiler.resolve() if a.compiler else None, a.output))
    report['probe_errors'] = sum(c['result'] == 'error' for g in report['graphs']
        for row in g['nodes'] for c in row.get('compilations', []))
    (a.output / 'coverage.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'compiler_probes_run': bool(a.compiler), 'probe_errors': report['probe_errors'], 'graphs': [{k: g[k] for k in ('file', 'sha256', 'status_counts', 'op_counts')}
                                for g in report['graphs']], 'complete_native_graph': False}, indent=2))


    return 1 if report['probe_errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
