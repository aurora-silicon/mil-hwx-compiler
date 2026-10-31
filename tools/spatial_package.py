#!/usr/bin/env python3
"""Versioned spatial dispatch contract and software interpreter.

The scheduler stages contiguous slices into each existing ANEC surface and
stores a prefix of its result. Every copy, resize and arithmetic step is an
ANE dispatch. The interpreter is a host-only numerical model of those steps;
it decodes weights from the emitted ANEC, rather than evaluating source MIL.
"""
from __future__ import annotations
import argparse
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from inspect_anec import HEADER, HEADER_BYTES, TILE_BYTES, convert_tensor, validate_task_headers
from h13_td import split_h13_tasks, split_h14_tasks, decode_task

SCHEMA = 'mil-hwxc.spatial-package.v1'
LIMIT = 1 << 29


def require(condition, message):
    if not condition:
        raise ValueError('spatial-package.' + message)


def uint(value, upper=LIMIT):
    return type(value) is int and 0 <= value <= upper


def local_file(root, name, limit=64 << 20):
    require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'),
            'invalid-file-name')
    path = root / name
    require(not path.is_symlink() and path.resolve().parent == root.resolve(), 'file-escape')
    require(path.stat().st_size <= limit, 'file-too-large')
    return path.read_bytes()


def physical_elements(layout):
    return math.prod(layout['nchw'][:4])


def validate_layout(layout, index, tiles, layouts):
    require(isinstance(layout, dict) and set(layout) == {'index', 'nchw', 'allocationBytes'},
            'layout-fields')
    dims = layout['nchw']
    require(isinstance(dims, list) and len(dims) == 6 and
            all(uint(n, LIMIT * 64) and n > 0 for n in dims), 'layout-dimensions')
    require(layout['index'] == index and dims == list(layouts[index*6:index*6+6]),
            'header-layout-mismatch')
    n, c, h, w, plane, row = dims
    require(n == 1 and row >= w * 2 and plane >= h * row and row % 64 == 0,
            'layout-strides')
    allocation = ((n * c * plane + TILE_BYTES - 1) // TILE_BYTES) * TILE_BYTES
    require(uint(allocation, LIMIT * 64) and layout['allocationBytes'] == allocation and
            tiles[index] * TILE_BYTES == allocation, 'layout-allocation')


def validate_kernel(root, k, target):
    require(isinstance(k, dict), 'kernel-record')
    kind = k.get('kind')
    require(kind in ('matrix', 'conv', 'softmax', 'add', 'mul', 'maximum', 'minimum'), 'kernel-kind')
    extra = {'matrix': {'size'}, 'conv': {'kernel', 'side', 'groups'}}.get(kind, set())
    require(set(k) == {'kind', 'file', 'bytes', 'inputs', 'output', 'constantOffset',
                       'constantBytes', 'taskDescriptors', 'scratchBytes'} | extra, 'kernel-fields')
    for field in ('bytes', 'constantOffset', 'constantBytes', 'taskDescriptors'):
        require(uint(k[field], 64 << 20), 'kernel-size')
    require(k['taskDescriptors'] > 0 and k['constantOffset'] > 0 and
            k['bytes'] == HEADER_BYTES + k['constantOffset'] + k['constantBytes'], 'kernel-span')
    data = local_file(root, k['file'])
    require(len(data) == k['bytes'] and len(data) >= HEADER_BYTES, 'kernel-file-size')
    fields = HEADER.unpack_from(data)
    size, first, count, task_size, constants, inputs, outputs = fields[:7]
    require(isinstance(k['inputs'], list) and
            (size, count, constants, inputs, outputs) ==
            (len(data) - HEADER_BYTES, k['taskDescriptors'], k['constantBytes'], len(k['inputs']), 1),
            'header-mismatch')
    require(0 < first <= task_size <= k['constantOffset'] and first % 4 == 0 and
            k['constantOffset'] == (task_size + 63) // 64 * 64, 'task-span')
    require(not any(data[HEADER.size:HEADER_BYTES]) and
            not any(data[HEADER_BYTES+task_size:HEADER_BYTES+k['constantOffset']]), 'reserved-bytes')
    section = data[HEADER_BYTES:HEADER_BYTES+task_size]
    if target == 'H13':
        tasks = split_h13_tasks(section, first // 4 - 1, count)
    else:
        require(section[:16] == bytes(16), 'h14-task-prefix')
        tasks = split_h14_tasks(section)
    require(len(tasks) == count and len(tasks[0]) == first, 'task-count')
    for task in tasks:
        decoded = decode_task(task, target.lower())
        require(not any(b['name'].startswith('unknown_') for b in decoded['blocks'].values()), 'unknown-task-register')
    tiles, layouts = fields[7:39], fields[39:]
    require(tiles[0] == (size + TILE_BYTES - 1) // TILE_BYTES, 'command-allocation')
    require(uint(k['scratchBytes'], 64 << 20) and
            tiles[3] * TILE_BYTES == k['scratchBytes'], 'scratch-allocation')
    if target == 'H13':
        validate_task_headers(k, tasks, tiles)
    occupied = {4, *range(5, 5 + inputs)}
    for index in range(32):
        if index not in occupied:
            require(not any(layouts[index*6:index*6+6]), 'unused-layout')
            if index not in (0, 3):
                require(tiles[index] == 0, 'unused-allocation')
    require(inputs == (2 if kind in ('add', 'mul', 'maximum', 'minimum') else 1), 'kernel-input-count')
    for index, item in enumerate(k['inputs'], 5):
        validate_layout(item, index, tiles, layouts)
    validate_layout(k['output'], 4, tiles, layouts)
    if kind == 'matrix':
        n = k['size']
        require(n in (256, 512, 1024) and constants == n*n*2 and
                k['inputs'][0]['nchw'][:4] == [1, 1, 1, n] and
                k['output']['nchw'][:4] == [1, 1, 1, n], 'matrix-geometry')
    elif kind == 'conv':
        require(k['kernel'] in (1, 3) and k['side'] in (16, 64) and
                k['groups'] in (1, 64) and (k['groups'] == 1 or k['side'] == 16), 'conv-geometry')
        require(k['inputs'][0]['nchw'][:4] == k['output']['nchw'][:4] ==
                [1, 64, k['side'], k['side']], 'conv-layout')
        taps = k['kernel'] ** 2
        expected = 16 * math.ceil(4 * taps * 2 / 64) * 64 if k['groups'] == 64 else 64 * 64 * taps * 2
        require(constants == expected, 'conv-constant-size')
    elif kind == 'softmax':
        require(k['inputs'][0]['nchw'][:4] == k['output']['nchw'][:4] == [1, 1024, 1, 1],
                'softmax-geometry')
    else:
        require(all(item['nchw'][:4] == [1,256,1,1] for item in k['inputs'] + [k['output']]),
                'binary-geometry')
    return data


def offset_bounds(expression, scope):
    if type(expression) is int:
        require(uint(expression), 'offset')
        return expression, expression
    require(isinstance(expression, dict) and set(expression) == {'base', 'terms'} and
            uint(expression['base']) and isinstance(expression['terms'], dict), 'affine-offset')
    lo = hi = expression['base']
    for var, stride in expression['terms'].items():
        require(var in scope and uint(stride), 'affine-term')
        hi += (scope[var] - 1) * stride
    require(hi <= LIMIT, 'offset-overflow')
    return lo, hi


def validate_plan(manifest):
    tensors, kernels = manifest['tensors'], manifest['kernels']
    nodes = 0

    def binding(b, physical, scope, output=False):
        require(isinstance(b, dict) and set(b) == {'tensor', 'offset', 'count', 'paddingBits'}, 'binding-fields')
        require(b['tensor'] in tensors, 'binding-tensor')
        t = tensors[b['tensor']]
        require(uint(b['count']) and 0 < b['count'] <= physical and
                uint(b['paddingBits'], 65535), 'binding-size')
        require(b['paddingBits'] in (0, 0xfc00) and (not output or b['paddingBits'] == 0), 'padding-bits')
        require(not output or t['role'] not in ('input', 'constant'), 'write-to-readonly')
        _, high = offset_bounds(b['offset'], scope)
        require(high + b['count'] <= t['elements'], 'binding-range')
        return b['tensor']

    def visit(body, scope, depth):
        nonlocal nodes
        require(isinstance(body, list) and body and depth <= 8, 'plan-body')
        calls = 0
        for node in body:
            nodes += 1
            require(nodes <= 1_000_000 and isinstance(node, dict), 'plan-size')
            if 'loop' in node:
                require(set(node) == {'loop','count','body'} and isinstance(node['loop'], str) and
                        node['loop'] and node['loop'] not in scope and
                        uint(node['count']) and node['count'] > 0, 'loop')
                count = visit(node['body'], {**scope,node['loop']:node['count']}, depth+1)
                calls += count * node['count']
            else:
                require(set(node) == {'kernel','inputs','output'} and uint(node['kernel']) and
                        node['kernel'] < len(kernels), 'dispatch-fields')
                k = kernels[node['kernel']]
                require(isinstance(node['inputs'], list) and len(node['inputs']) == len(k['inputs']), 'dispatch-inputs')
                sources = [binding(b,physical_elements(t),scope) for b,t in zip(node['inputs'], k['inputs'])]
                dest = binding(node['output'],physical_elements(k['output']),scope,True)
                require(dest not in sources, 'in-place-dispatch')
                calls += 1
            require(calls <= 1_000_000_000, 'dispatch-count')
        return calls
    return visit(manifest['dispatchPlan'], {}, 0)


class Package:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        m = self.manifest = json.loads(local_file(self.root, 'manifest.json'))
        require(isinstance(m, dict) and set(m) == {'schema', 'target', 'artifactFormat', 'schedule',
                'kernels', 'dispatchPlan', 'tensors', 'constants', 'outputs', 'arithmetic'}, 'manifest-fields')
        require(m['schema'] == SCHEMA and m['target'] in ('H13','H14') and
                m['artifactFormat'] == 'anec' and m['schedule'] == 'spatial' and
                m['arithmetic'] == 'fp16-partials', 'manifest-contract')
        require(isinstance(m['tensors'], dict) and m['tensors'], 'tensors')
        for name, t in m['tensors'].items():
            require(isinstance(name,str) and name and isinstance(t,dict) and
                    set(t) == {'shape','elements','role'}, 'tensor-fields')
            require(isinstance(t['shape'],list) and 0 < len(t['shape']) <= 4 and
                    all(uint(v) and v > 0 for v in t['shape']) and
                    uint(t['elements']) and t['elements'] == math.prod(t['shape']), 'tensor-shape')
            require(t['role'] in ('input','output','intermediate','constant','scratch'), 'tensor-role')
        require(isinstance(m['kernels'],list) and m['kernels'], 'kernels')
        self.payloads = [validate_kernel(self.root,k,m['target']) for k in m['kernels']]
        require(isinstance(m['constants'],dict) and set(m['constants']) ==
                {n for n,t in m['tensors'].items() if t['role'] == 'constant'}, 'constants')
        self.constants = {n:local_file(self.root,f) for n,f in m['constants'].items()}
        for n, data in self.constants.items():
            require(len(data) == m['tensors'][n]['elements'] * 2, 'constant-size')
        require(isinstance(m['outputs'],list) and m['outputs'] and
                len(set(m['outputs'])) == len(m['outputs']) and set(m['outputs']) ==
                {n for n,t in m['tensors'].items() if t['role'] == 'output'}, 'outputs')
        self.dispatch_count = validate_plan(m)

    def dispatches(self):
        def offset(expr, env):
            return expr if type(expr) is int else expr['base'] + sum(env[v]*s for v,s in expr['terms'].items())
        def bind(b, env):
            return {**b, 'offset':offset(b['offset'],env)}
        def walk(body, env):
            for node in body:
                if 'loop' in node:
                    for i in range(node['count']):
                        yield from walk(node['body'], {**env,node['loop']:i})
                else:
                    yield node['kernel'], [bind(b,env) for b in node['inputs']], bind(node['output'],env)
        yield from walk(self.manifest['dispatchPlan'], {})

    def execute(self, inputs, dispatch, *, max_dispatches=1_000_000_000):
        """Execute with a caller-supplied synchronous ANEC dispatch callback.

        callback(kernel_record, ANEC_bytes, physical_inputs) -> physical_output.
        Device/session/ownership approval belongs to that caller. This module
        opens no device. Buffer contents stay dense between dispatches, as in
        the original runner; only contiguous prefixes are copied on the host.
        """
        import numpy as np
        m = self.manifest
        validate_plan(m)
        require(set(inputs) == {n for n,t in m['tensors'].items() if t['role'] == 'input'}, 'input-names')
        require(self.dispatch_count <= max_dispatches, 'execution-dispatch-budget')
        buffers, written = {}, {}
        for name,t in m['tensors'].items():
            n = t['elements']
            buffers[name] = np.empty(n,dtype='<f2')
            written[name] = np.zeros(n,dtype=np.bool_)
            if t['role'] in ('constant','input'):
                raw = self.constants[name] if t['role'] == 'constant' else inputs[name]
                require(len(raw) == n*2, 'input-size')
                buffers[name][:] = np.frombuffer(raw,dtype='<f2')
                written[name][:] = True
        for index,sources,dest in self.dispatches():
            k = m['kernels'][index]
            physical = []
            for src,layout in zip(sources,k['inputs']):
                name,at,n = src['tensor'],src['offset'],src['count']
                require(written[name][at:at+n].all(), 'read-before-write')
                padded = bytearray(struct.pack('<H',src['paddingBits']) * physical_elements(layout))
                padded[:n*2] = buffers[name][at:at+n].tobytes()
                binding = {**layout,'logicalBytes':len(padded)}
                physical.append(bytes(convert_tensor(binding,padded,True)))
            result = dispatch(k,self.payloads[index],physical)
            layout = k['output']
            require(len(result) == layout['allocationBytes'], 'dispatch-output-size')
            dense = convert_tensor({**layout,'logicalBytes':physical_elements(layout)*2},result,False)
            name,at,n = dest['tensor'],dest['offset'],dest['count']
            buffers[name][at:at+n] = np.frombuffer(dense,dtype='<f2',count=n)
            written[name][at:at+n] = True
        require(all(written[n].all() for n in m['outputs']), 'incomplete-output')
        return {n:buffers[n].tobytes() for n in m['outputs']}

    def simulate(self, inputs, **kwargs):
        """Model primitives with fp32 accumulation / fp16 dispatch boundaries.

        This checks lowering and routing, not the target's LUT accuracy or
        firmware execution. Kernels and constants come from the ANEC bytes.
        """
        import numpy as np
        decoded = {}
        def dispatch(k,payload,physical):
            source = [np.frombuffer(convert_tensor({**t,'logicalBytes':physical_elements(t)*2},b,False),
                                   dtype='<f2').astype(np.float32) for t,b in zip(k['inputs'],physical)]
            kind = k['kind']
            if kind == 'matrix':
                key = k['file']
                if key not in decoded:
                    n = k['size']
                    packed = np.frombuffer(payload,dtype='<f2',offset=HEADER_BYTES+k['constantOffset'])
                    weights = np.empty((n,n),dtype=np.float32)
                    group = min(16,n//16)
                    groups = n//group
                    for col in range(n):
                        plane = col//group
                        dest = (plane%16)*(groups//16)+plane//16
                        weights[col] = packed[dest*n*group+col%group:dest*n*group+col%group+n*group:group]
                    decoded[key] = weights
                result = decoded[key] @ source[0]
            elif kind == 'conv':
                key = k['file']
                side,taps,groups = k['side'],k['kernel']**2,k['groups']
                if key not in decoded:
                    packed = np.frombuffer(payload,dtype='<f2',offset=HEADER_BYTES+k['constantOffset'])
                    weights = np.zeros((64,64//groups,taps),dtype=np.float32)
                    if groups == 64:
                        stride = math.ceil(4*taps*2/64)*32
                        for col in range(64):
                            start = (col%16)*stride + (col//16)*taps
                            weights[col,0] = packed[start:start+taps]
                    else:
                        lanes, reduction = 4,64*taps
                        for col in range(64):
                            start = (col//lanes)*reduction*lanes+col%lanes
                            weights[col] = packed[start:start+reduction*lanes:lanes].reshape(64,taps)
                    decoded[key] = weights
                weights = decoded[key]
                x = source[0].reshape(64,side,side)
                radius = k['kernel']//2
                padded = np.pad(x,((0,0),(radius,radius),(radius,radius)))
                windows = np.lib.stride_tricks.sliding_window_view(padded,(k['kernel'],k['kernel']),axis=(1,2))
                windows = windows.reshape(64,side,side,taps)
                if groups == 64:
                    result = np.einsum('cyxt,ct->cyx',windows,weights[:,0],dtype=np.float32,optimize=False)
                else:
                    result = np.einsum('iyxt,oit->oyx',windows,weights,dtype=np.float32,optimize=False)
            elif kind == 'softmax':
                shifted = source[0] - np.max(source[0])
                e = np.exp(shifted)
                result = e / np.sum(e,dtype=np.float32)
            else:
                fn = {'add':np.add,'mul':np.multiply,'maximum':np.maximum,'minimum':np.minimum}[kind]
                result = fn(*source)
            dense = np.asarray(result,dtype='<f2').reshape(-1).tobytes()
            return bytes(convert_tensor({**k['output'],'logicalBytes':len(dense)},dense,True))
        return self.execute(inputs,dispatch,**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    parser.add_argument('--simulate',action='store_true')
    parser.add_argument('--input',action='append',default=[],metavar='NAME=FILE')
    parser.add_argument('--output',action='append',default=[],metavar='NAME=FILE')
    args = parser.parse_args()
    package = Package(args.package)
    if args.simulate:
        def assignments(items):
            pairs = [v.split('=',1) for v in items]
            require(all(len(p) == 2 for p in pairs) and len({p[0] for p in pairs}) == len(pairs), 'assignments')
            return {n:Path(p) for n,p in pairs}
        inputs,outputs = assignments(args.input),assignments(args.output)
        require(set(outputs) == set(package.manifest['outputs']), 'output-names')
        result = package.simulate({n:p.read_bytes() for n,p in inputs.items()})
        for n,p in outputs.items():
            p.write_bytes(result[n])
    print(json.dumps({'schema':SCHEMA,'target':package.manifest['target'],'deviceCalls':False,
                      'kernels':len(package.manifest['kernels']),'dispatches':package.dispatch_count,
                      'simulated':args.simulate},sort_keys=True))


if __name__ == '__main__':
    main()
