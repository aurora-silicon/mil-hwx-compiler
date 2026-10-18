#!/usr/bin/env python3
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else 'build/mil-hwxc').resolve())
inspector = str(Path(__file__).resolve().parents[1] / "research" / "inspect_anec.py")
hwx_inspector = str(Path(__file__).resolve().parents[1] / "research" / "inspect_hwx.py")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.inspect_anec import h13_task_registers
from timeouts import scaled_timeout


def inspect(package, *args, success=True):
    result = subprocess.run([sys.executable, inspector, str(package), *map(str, args)],
                            capture_output=True, text=True, timeout=scaled_timeout(15), check=False)
    assert result.returncode == (0 if success else 1), result.stderr
    return result.stdout


def tensor_type(shape):
    return f"tensor<fp16, [{', '.join(map(str, shape))}]>"


def source(op='add', shape=(1, 64, 1, 1), y_shape=None, output_shape=None, extra=''):
    x_type = tensor_type(shape)
    y_type = tensor_type(shape if y_shape is None else y_shape)
    output_type = tensor_type(shape if output_shape is None else output_shape)
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x_type} a, {y_type} b) {{
    {output_type} y = {op}(x = a, y = b)[name = string("binary")];
    {extra}
  }} -> (y);
}}
'''
def constant_source(op, value, const_first=False, blob=None, shape=(1, 64, 1, 1)):
    value_type = tensor_type(shape)
    if blob is not None:
        constant_type = value_type
        literal = (f'{value_type}(BLOBFILE(path = string("@model_path/{blob}"), '
                   'offset = uint64(64)))')
    elif isinstance(value, (tuple, list)):
        constant_type = value_type
        literal = f'{value_type}([{", ".join(f"fp16({item})" for item in value)}])'
    else:
        constant_type = 'fp16'
        literal = f'fp16({value})'
    x, y = ('c', 'a') if const_first else ('a', 'c')
    return f'''program(1.3)
 [buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} a) {{
    {constant_type} c = const()[name = string("c"), val = {literal}];
    {value_type} y = {op}(x = {x}, y = {y})[name = string("binary")];
  }} -> (y);
}}
'''



def chain_source(return_values='result', middle='product', shape=(1, 64, 1, 1)):
    value_type = tensor_type(shape)
    constants = ', '.join('fp16(1.0)' for _ in range(math.prod(shape)))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} a, {value_type} b) {{
    {value_type} c = const()[name = string("c"), val = {value_type}([{constants}])];
    {value_type} sum = add(x = a, y = b)[name = string("sum")];
    {value_type} product = mul(x = sum, y = b)[name = string("product")];
    {value_type} result = sub(x = {middle}, y = c)[name = string("result")];
  }} -> ({return_values});
}}
'''

def activation_source(op, shape=(1, 64, 1, 1), alpha=None, beta=None):
    value_type = tensor_type(shape)
    arguments = 'x = a' if op == 'relu' else \
        f'x = a, alpha = fp32({alpha}), beta = fp32({beta})'
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} a) {{
    {value_type} y = {op}({arguments})[name = string("activation")];
  }} -> (y);
}}
'''


def activation_chain_source(op, alpha=None, beta=None, multiply=False,
                            shape=(1, 64, 1, 1)):
    value_type = tensor_type(shape)
    arguments = 'x = sum' if op == 'relu' else \
        f'x = sum, alpha = fp32({alpha}), beta = fp32({beta})'
    tail = (f'    {value_type} result = mul(x = activated, y = b)'
            '[name = string("result")];\n') if multiply else ''
    returned = 'result' if multiply else 'activated'
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} a, {value_type} b) {{
    {value_type} sum = add(x = a, y = b)[name = string("sum")];
    {value_type} activated = {op}({arguments})[name = string("activation")];
{tail}  }} -> ({returned});
}}
'''


def matmul_source(reduction, x_shape=None, output_shape=None, transpose_x=False,
                  transpose_y=True, weight_shape=None):
    x_shape = (1, reduction) if x_shape is None else x_shape
    output_shape = (1, 512) if output_shape is None else output_shape
    if weight_shape is None:
        weight_shape = (512, reduction) if transpose_y else (reduction, 512)
    tx = 't' if transpose_x else 'f'
    ty = 't' if transpose_y else 'f'
    x_type = tensor_type(x_shape)
    weight_type = tensor_type(weight_shape)
    output_type = tensor_type(output_shape)
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x_type} x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    {weight_type} W = const()[name = string("W"), val = {weight_type}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    {output_type} y = matmul(x = x, y = W, transpose_x = {tx}, transpose_y = {ty})[name = string("projection")];
  }} -> (y);
}}
'''

def linear_source(reduction, columns, bias=None, x_shape=None, output_shape=None,
                  runtime_weight=False, runtime_bias=False):
    x_shape = (1, reduction) if x_shape is None else x_shape
    output_shape = (*x_shape[:-1], columns) if output_shape is None else output_shape
    x_type = tensor_type(x_shape)
    weight_type = tensor_type((columns, reduction))
    output_type = tensor_type(output_shape)
    parameters = [f'{x_type} x']
    body = []
    if runtime_weight:
        parameters.append(f'{weight_type} W')
    else:
        body.append(
            f'    {weight_type} W = const()[name = string("W"), val = '
            f'{weight_type}(BLOBFILE(path = string("@model_path/weights.bin"), '
            'offset = uint64(64)))];')
    arguments = ['x = x', 'weight = W']
    if bias is not None:
        bias_type = tensor_type((columns,))
        if runtime_bias:
            parameters.append(f'{bias_type} bias')
        elif bias == 'blob':
            body.append(
                f'    {bias_type} bias = const()[name = string("bias"), val = '
                f'{bias_type}(BLOBFILE(path = string("@model_path/bias.bin"), '
                'offset = uint64(64)))];')
        else:
            values = ', '.join(f'fp16({value})' for value in bias)
            body.append(
                f'    {bias_type} bias = const()[name = string("bias"), val = '
                f'{bias_type}([{values}])];')
        arguments.append('bias = bias')
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({', '.join(parameters)}) {{
{chr(10).join(body)}
    {output_type} y = linear({', '.join(arguments)})[name = string("projection")];
  }} -> (y);
}}
'''


def matmul_activation_chain_source(reduction=256, columns=512):
    x_type = tensor_type((1, reduction))
    value_type = tensor_type((1, columns))
    weight_type = tensor_type((columns, reduction))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x_type} x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    {weight_type} W = const()[name = string("W"), val = {weight_type}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    {value_type} bias = const()[name = string("bias"), val = {value_type}(BLOBFILE(path = string("@model_path/bias.bin"), offset = uint64(64)))];
    {value_type} projection = matmul(x = x, y = W, transpose_x = f, transpose_y = t)[name = string("projection")];
    {value_type} biased = add(x = projection, y = bias)[name = string("biased")];
    {value_type} result = relu(x = biased)[name = string("result")];
  }} -> (result);
}}
'''


def matmul_relu_chain_source():
    value_type = tensor_type((1, 64, 256))
    weight_type = tensor_type((256, 256))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    {weight_type} W = const()[name = string("W"), val = {weight_type}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    {value_type} projection = matmul(x = x, y = W, transpose_x = f, transpose_y = t)[name = string("projection")];
    {value_type} result = relu(x = projection)[name = string("result")];
  }} -> (result);
}}
'''

def h13_tasks(anec):
    first_bytes, count, stream_bytes = struct.unpack_from('<IIQ', anec, 8)
    stream = anec[4096:4096 + stream_bytes]
    tasks = []
    offset = 0
    task_bytes = first_bytes
    for index in range(count):
        task = stream[offset:offset + task_bytes]
        assert len(task) == task_bytes
        tasks.append(task)
        next_offset = struct.unpack_from('<I', task, 28)[0]
        if index + 1 == count:
            assert next_offset == 0
        else:
            task_bytes = (((struct.unpack_from('<I', task, 4)[0] >> 16) & 0x1ff) + 1) * 4
            offset = next_offset
    return tasks


def oracle_registers(descriptor):
    return {int(address, 16): int(value, 16)
            for block in descriptor['blocks'].values()
            for address, value in block['words'].items()}

def binary_matmul_chain_source(reduction=512):
    value_type = tensor_type((reduction,))
    output_type = tensor_type((512,))
    weight_type = tensor_type((512, reduction))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} a, {value_type} b) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    {weight_type} W = const()[name = string("W"), val = {weight_type}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    {value_type} sum = add(x = a, y = b)[name = string("sum")];
    {output_type} result = matmul(x = sum, y = W, transpose_x = f, transpose_y = t)[name = string("result")];
  }} -> (result);
}}
'''


def square_source():
    value_type = tensor_type((64,))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} x) {{
    {value_type} y = mul(x = x, y = x)[name = string("square")];
  }} -> (y);
}}
'''


def alias_source(op='reshape', return_alias=False, produced_input=False):
    source_shape = (1, 64, 1, 1)
    result_shape = (64,)
    source_type = tensor_type(source_shape)
    result_type = tensor_type(result_shape)
    if op == 'reshape':
        parameter = 'shape'
        parameter_value = 'tensor<int32, [1]>([64])'
    elif op == 'squeeze':
        parameter = 'axes'
        parameter_value = 'tensor<int32, [3]>([0, 2, 3])'
    else:
        source_shape, result_shape = (64,), (1, 64, 1, 1)
        source_type, result_type = tensor_type(source_shape), tensor_type(result_shape)
        parameter = 'axes'
        parameter_value = 'tensor<int32, [3]>([0, 2, 3])'
    base = (f'    {source_type} t = add(x = x, y = x)'
            '[name = string("base")];\n') if produced_input else ''
    input_name = 't' if produced_input else 'x'
    tail = '' if return_alias else (
        f'    {result_type} y = add(x = r, y = b)[name = string("sum")];\n')
    second_input = '' if return_alias else f', {result_type} b'
    returned = 'r' if return_alias else 'y'
    parameter_count = 1 if op == 'reshape' else 3
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({source_type} x{second_input}) {{
{base}    tensor<int32, [{parameter_count}]> parameter = const()[name = string("parameter"), val = {parameter_value}];
    {result_type} r = {op}(x = {input_name}, {parameter} = parameter)[name = string("alias")];
{tail}  }} -> ({returned});
}}
'''


def residual_source(reduction=256):
    input_type = tensor_type((1, reduction))
    value_type = tensor_type((1, 512))
    weight_type = tensor_type((512, reduction))
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({input_type} x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool tr = const()[name = string("tr"), val = bool(true)];
    {weight_type} W = const()[name = string("W"), val = {weight_type}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    {value_type} t = matmul(x = x, y = W, transpose_x = f, transpose_y = tr)[name = string("projection")];
    {value_type} u = relu(x = t)[name = string("activation")];
    {value_type} y = add(x = u, y = t)[name = string("residual")];
  }} -> (y);
}}
'''


def norm_source(op, shape=(1, 512, 1, 1), axis=None, axes=None,
                output_shape=None, keep_dims=True, epsilon='0.00001',
                affine=False):
    """A single softmax, layer_norm, or reduce_* operation over one input."""
    input_type = tensor_type(shape)
    result_type = tensor_type(output_shape if output_shape is not None else shape)
    body = []
    if op == 'softmax':
        body.append(f'    int32 axis = const()[name = string("axis"), '
                    f'val = int32({axis})];')
        arguments = 'x = x, axis = axis'
    else:
        axes_type = f'tensor<int32, [{len(axes)}]>'
        values = ', '.join(map(str, axes))
        body.append(f'    {axes_type} axes = const()[name = string("axes"), '
                    f'val = {axes_type}([{values}])];')
        arguments = 'x = x, axes = axes'
        if op == 'layer_norm':
            if affine:
                affine_type = tensor_type((shape[1],))
                ones = ', '.join(['fp16(1.0)'] * shape[1])
                zeros = ', '.join(['fp16(0.0)'] * shape[1])
                body.append(f'    {affine_type} gamma = const()'
                            f'[name = string("gamma"), val = {affine_type}([{ones}])];')
                body.append(f'    {affine_type} beta = const()'
                            f'[name = string("beta"), val = {affine_type}([{zeros}])];')
                arguments += ', beta = beta, gamma = gamma'
            arguments += f', epsilon = fp32({epsilon})'
        else:
            arguments += f', keep_dims = bool({"true" if keep_dims else "false"})'
    parameters = f'{input_type} x'
    lines = '\n'.join(body)
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({parameters}) {{
{lines}
    {result_type} y = {op}({arguments})[name = string("y")];
  }} -> (y);
}}
'''

with tempfile.TemporaryDirectory(prefix='mil-hwx-h13-test-') as directory:
    root = Path(directory)
    mil = root / 'model.mil'

    def compile_text(text, name, success=True, diagnostic="h13.", format=None,
                     schedule=None):
        mil.write_text(text)
        out = root / name
        command = [compiler, '--mil', str(mil), '--model-root', str(root),
                   '--target', 'H13', '--output', str(out)]
        if format is not None:
            command += ['--format', format]
        if schedule is not None:
            command += ['--schedule', schedule]
        run = subprocess.run(command, capture_output=True, text=True, check=False,
                             timeout=scaled_timeout(15))
        assert (run.returncode == 0) == success, run.stdout + run.stderr
        if not success:
            assert diagnostic in run.stderr, run.stderr
            assert not out.exists(), f'failed compilation wrote {out}'
        return out

    first = compile_text(source(), 'add')
    repeated = compile_text(source(), 'repeat')
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in repeated.iterdir()}
    manifest = json.loads((first / 'manifest.json').read_text())
    assert manifest['target'] == 'H13'
    assert manifest['artifactFormat'] == 'anec'
    assert manifest["schema"] == "mil-hwxc.h13-anec-package.v2"
    assert manifest['dispatchPlan'] == [0]
    assert manifest['intermediates'] == []
    assert manifest['tensors'] == {
        'a': {'shape': [1, 64, 1, 1], 'logicalBytes': 128, 'role': 'input'},
        'b': {'shape': [1, 64, 1, 1], 'logicalBytes': 128, 'role': 'input'},
        'y': {'shape': [1, 64, 1, 1], 'logicalBytes': 128, 'role': 'output'},
    }
    assert manifest['inputs'][0]['name'] == 'a'
    assert manifest['inputs'][1]['name'] == 'b'
    assert manifest['outputs'][0]['name'] == 'y'
    data = (first / 'program-0.anec').read_bytes()
    size, td_size, count, task_size, kernel_size, inputs, outputs = struct.unpack_from('<QIIQQII', data)
    assert len(data) == 4096 + size
    assert (td_size, count, task_size, inputs, outputs) == (0x274, 1, 0x274, 2, 1)
    inspection = json.loads(inspect(first))
    assert inspection["manifest"] == manifest
    dense = bytes(range(128))
    raw, padded, unpacked = root / "input.fp16", root / "input.buffer", root / "output.fp16"
    raw.write_bytes(dense)
    inspect(first, "--pack-input", "a", raw, "--output", padded)
    expected = bytearray(16384)
    for channel in range(64):
        expected[channel * 64:channel * 64 + 2] = dense[channel * 2:channel * 2 + 2]
    assert padded.read_bytes() == expected
    inspect(first, "--unpack-output", "y", padded, "--output", unpacked)
    assert unpacked.read_bytes() == dense
    inspect(first, "--pack-input", "a", raw, "--output", padded, success=False)
    assert padded.read_bytes() == expected
    raw.write_bytes(dense[:-1])
    rejected = root / "rejected.buffer"
    inspect(first, "--pack-input", "a", raw, "--output", rejected, success=False)
    assert not rejected.exists()
    inspect(first, "--pack-input", "y", raw, "--output", rejected, success=False)
    manifest["inputs"][0]["index"] = 4
    (repeated / "manifest.json").write_text(json.dumps(manifest))
    inspect(repeated, success=False)
    for op in ('add', 'mul', 'maximum', 'minimum'):
        canonical = first if op == 'add' else compile_text(source(op), f'{op}-canonical')
        canonical_data = (canonical / 'program-0.anec').read_bytes()
        for shape in ((64,), (1, 64), (2, 4, 8)):
            logical = compile_text(source(op, shape), f'{op}-{len(shape)}d')
            assert (logical / 'program-0.anec').read_bytes() == canonical_data
            logical_manifest = json.loads((logical / 'manifest.json').read_text())
            assert logical_manifest['inputs'][0]['shape'] == list(shape)
            assert logical_manifest['inputs'][1]['shape'] == list(shape)
            assert logical_manifest['outputs'][0]['shape'] == list(shape)
    assert (root / 'mul-canonical' / 'program-0.anec').read_bytes() != data

    square = compile_text(square_source(), 'square')
    square_manifest = json.loads((square / 'manifest.json').read_text())
    assert (square / 'program-0.anec').read_bytes() == \
        (root / 'mul-canonical' / 'program-0.anec').read_bytes()
    assert [item['name'] for item in square_manifest['inputs']] == ['x', 'x']
    assert [item['index'] for item in square_manifest['inputs']] == [5, 6]
    assert json.loads(inspect(square))['manifest'] == square_manifest

    reshape = compile_text(alias_source(), 'reshape')
    reshape_manifest = json.loads((reshape / 'manifest.json').read_text())
    assert len(reshape_manifest['programs']) == 1
    assert (reshape / 'program-0.anec').read_bytes() == data
    assert reshape_manifest['intermediates'] == ['r']
    assert reshape_manifest['tensors']['r'] == {
        'shape': [64], 'logicalBytes': 128, 'role': 'intermediate', 'aliasOf': 'x'}
    assert reshape_manifest['tensors']['x']['shape'] == [1, 64, 1, 1]
    assert reshape_manifest['inputs'][0]['name'] == 'x'
    assert reshape_manifest['outputs'][0]['shape'] == [64]
    assert json.loads(inspect(reshape))['manifest'] == reshape_manifest
    invalid_alias_manifest = json.loads(json.dumps(reshape_manifest))
    invalid_alias_manifest['tensors']['r']['shape'] = [32]
    invalid_alias_manifest['tensors']['r']['logicalBytes'] = 64
    (reshape / 'manifest.json').write_text(json.dumps(invalid_alias_manifest))
    inspect(reshape, success=False)
    (reshape / 'manifest.json').write_text(json.dumps(reshape_manifest))
    for shape_op in ('squeeze', 'expand_dims'):
        aliased = compile_text(alias_source(shape_op), f'{shape_op}-alias')
        aliased_manifest = json.loads((aliased / 'manifest.json').read_text())
        assert aliased_manifest['tensors']['r']['aliasOf'] == 'x'
        assert json.loads(inspect(aliased))['manifest'] == aliased_manifest
    returned_alias = compile_text(alias_source(return_alias=True, produced_input=True),
                                  'returned-intermediate-alias')
    returned_alias_manifest = json.loads((returned_alias / 'manifest.json').read_text())
    assert returned_alias_manifest['intermediates'] == []
    assert returned_alias_manifest['tensors']['t'] == {
        'shape': [1, 64, 1, 1], 'logicalBytes': 128, 'role': 'output'}
    assert returned_alias_manifest['tensors']['r'] == {
        'shape': [64], 'logicalBytes': 128, 'role': 'output', 'aliasOf': 't'}
    assert returned_alias_manifest['outputs'][0]['name'] == 't'
    assert returned_alias_manifest['outputs'][0]['shape'] == [1, 64, 1, 1]
    assert returned_alias_manifest['physicalOutputs'] == [{
        'tensor': 't', 'dtype': 'float16', 'shape': [1, 64, 1, 1],
        'logicalBytes': 128}]
    assert returned_alias_manifest['logicalResults'] == [{
        'name': 'r', 'dtype': 'float16', 'shape': [64],
        'physical': {'tensor': 't', 'elementOffset': 0,
                     'elementCount': 64},
        'conversion': 'identity'}]
    assert json.loads(inspect(returned_alias))['manifest'] == returned_alias_manifest
    compile_text(alias_source(return_alias=True), 'returned-input-alias', False,
                 'h13.returned-input-alias')

    tiled_add = compile_text(source(shape=(192,)), 'tiled-add')
    tiled_manifest = json.loads((tiled_add / 'manifest.json').read_text())
    assert tiled_manifest['dispatchPlan'] == [0, 1, 2]
    assert len(tiled_manifest['programs']) == 3
    tiled_reference = (tiled_add / tiled_manifest['programs'][0]['file']).read_bytes()
    assert tiled_reference == data
    for index, program in enumerate(tiled_manifest['programs']):
        assert (tiled_add / program['file']).read_bytes() == tiled_reference
        assert [item['slice'] for item in program['inputs'] + program['outputs']] == [
            {'tensor': name, 'elementOffset': index * 64, 'elementCount': 64,
             'physicalElements': 64}
            for name in ('a', 'b', 'y')]
    assert tiled_manifest['tensors'] == {
        name: {'shape': [192], 'logicalBytes': 384,
               'role': 'output' if name == 'y' else 'input'}
        for name in ('a', 'b', 'y')
    }
    assert json.loads(inspect(tiled_add))['manifest'] == tiled_manifest
    tiled_manifest_path = tiled_add / 'manifest.json'
    invalid_tiled_manifest = json.loads(tiled_manifest_path.read_text())
    invalid_tiled_manifest['tensors']['a']['logicalBytes'] = 128
    tiled_manifest_path.write_text(json.dumps(invalid_tiled_manifest))
    inspect(tiled_add, success=False)
    invalid_tiled_manifest = json.loads(json.dumps(tiled_manifest))
    invalid_tiled_manifest['programs'][1]['outputs'][0]['slice']['elementOffset'] = 96
    tiled_manifest_path.write_text(json.dumps(invalid_tiled_manifest))
    inspect(tiled_add, success=False)
    tiled_manifest_path.write_text(json.dumps(tiled_manifest))

    tiled_dense = bytes(range(256)) + bytes(range(128))
    tiled_raw = root / 'tiled-input.fp16'
    tiled_raw.write_bytes(tiled_dense)
    tiled_buffers = root / 'tiled-input-buffers'
    inspect(tiled_add, '--pack-input', 'a', tiled_raw, '--output', tiled_buffers)
    for index in range(3):
        expected_slice = bytearray(16384)
        third = tiled_dense[index * 128:(index + 1) * 128]
        for element in range(64):
            expected_slice[element * 64:element * 64 + 2] = third[element * 2:element * 2 + 2]
        assert (tiled_buffers / f'program-{index}.a.buffer').read_bytes() == expected_slice
    tiled_outputs = root / 'tiled-output-buffers'
    tiled_outputs.mkdir()
    for index in range(3):
        (tiled_outputs / f'program-{index}.y.buffer').write_bytes(
            (tiled_buffers / f'program-{index}.a.buffer').read_bytes())
    tiled_unpacked = root / 'tiled-output.fp16'
    inspect(tiled_add, '--unpack-output', 'y', tiled_outputs, '--output', tiled_unpacked)
    assert tiled_unpacked.read_bytes() == tiled_dense
    padded_add = compile_text(source(shape=(96,)), 'padded-add')
    padded_manifest = json.loads((padded_add / 'manifest.json').read_text())
    assert padded_manifest['dispatchPlan'] == [0, 1]
    assert all((padded_add / program['file']).read_bytes() == tiled_reference
               for program in padded_manifest['programs'])
    assert [item['slice'] for item in
            padded_manifest['programs'][1]['inputs'] +
            padded_manifest['programs'][1]['outputs']] == [
                {'tensor': name, 'elementOffset': 64, 'elementCount': 32,
                 'physicalElements': 64}
                for name in ('a', 'b', 'y')]
    assert json.loads(inspect(padded_add))['manifest'] == padded_manifest
    padded_manifest_path = padded_add / 'manifest.json'
    invalid_padded_manifest = json.loads(json.dumps(padded_manifest))
    del invalid_padded_manifest['programs'][1]['inputs'][0]['slice']['physicalElements']
    padded_manifest_path.write_text(json.dumps(invalid_padded_manifest))
    inspect(padded_add, success=False)
    padded_manifest_path.write_text(json.dumps(padded_manifest))

    padded_dense = bytes(range(192))
    padded_raw = root / 'padded-input.fp16'
    padded_raw.write_bytes(padded_dense)
    padded_buffers = root / 'padded-input-buffers'
    inspect(padded_add, '--pack-input', 'a', padded_raw, '--output', padded_buffers)
    expected_tail = bytearray(16384)
    for element in range(32):
        expected_tail[element * 64:element * 64 + 2] = \
            padded_dense[128 + element * 2:130 + element * 2]
    assert (padded_buffers / 'program-1.a.buffer').read_bytes() == expected_tail
    padded_outputs = root / 'padded-output-buffers'
    padded_outputs.mkdir()
    for index in range(2):
        (padded_outputs / f'program-{index}.y.buffer').write_bytes(
            (padded_buffers / f'program-{index}.a.buffer').read_bytes())
    padded_unpacked = root / 'padded-output.fp16'
    inspect(padded_add, '--unpack-output', 'y', padded_outputs,
            '--output', padded_unpacked)
    assert padded_unpacked.read_bytes() == padded_dense

    values96 = tuple(float(index % 8 + 1) for index in range(96))
    padded_constant = compile_text(
        constant_source('add', values96, shape=(96,)), 'padded-constant')
    padded_constant_manifest = json.loads(
        (padded_constant / 'manifest.json').read_text())
    assert bytes.fromhex(
        padded_constant_manifest['programs'][1]['constantInputs']['c']) == \
        b''.join(struct.pack('<e', value) for value in values96[64:]) + bytes(64)

    values128 = tuple(float(index + 1) for index in range(128))
    constant128 = bytearray(128 + 256)
    struct.pack_into('<IIQQ', constant128, 64, 0xDEADBEEF, 1, 256, 128)
    constant128[128:] = b''.join(struct.pack('<e', value) for value in values128)
    (root / 'constant128.bin').write_bytes(constant128)
    tiled_sub = compile_text(
        constant_source('sub', None, blob='constant128.bin', shape=(2, 64)),
        'tiled-sub')
    tiled_sub_manifest = json.loads((tiled_sub / 'manifest.json').read_text())
    assert len(tiled_sub_manifest['programs']) == 2
    for index, program in enumerate(tiled_sub_manifest['programs']):
        expected_half = b''.join(
            struct.pack('<e', -value) for value in values128[index * 64:(index + 1) * 64])
        assert bytes.fromhex(program['constantInputs']['c']) == expected_half
    maximum_data = (root / 'maximum-canonical' / 'program-0.anec').read_bytes()
    minimum_data = (root / 'minimum-canonical' / 'program-0.anec').read_bytes()
    relu_reference = None
    for shape in ((64,), (1, 64), (2, 4, 8)):
        relu = compile_text(activation_source('relu', shape), f'relu-{len(shape)}d')
        relu_manifest = json.loads((relu / 'manifest.json').read_text())
        relu_data = (relu / 'program-0.anec').read_bytes()
        relu_reference = relu_reference or relu_data
        assert relu_data == relu_reference != maximum_data
        assert relu_manifest['operation'] == 'relu'
        assert relu_manifest['encoder'] == 'h13-oracle-parity'
        assert relu_manifest['constantInputs'] == {}
        assert len(relu_manifest['inputs']) == 1

    clipped = compile_text(activation_source('clip', alpha=-1, beta=2), 'clip')
    clipped_manifest = json.loads((clipped / 'manifest.json').read_text())
    assert clipped_manifest['dispatchPlan'] == [0, 1]
    assert clipped_manifest['intermediates'] == ['$h13.y.clipped-low']
    assert [program['operation'] for program in clipped_manifest['programs']] == [
        'maximum', 'minimum']
    clip_low = (clipped / 'program-0.anec').read_bytes()
    clip_high = (clipped / 'program-1.anec').read_bytes()
    assert clip_low != clip_high
    assert clip_low == maximum_data and clip_high == minimum_data
    assert bytes.fromhex(
        clipped_manifest['programs'][0]['constantInputs']['$h13.y.alpha']) == \
        struct.pack('<H', 0xbc00) * 64
    assert bytes.fromhex(
        clipped_manifest['programs'][1]['constantInputs']['$h13.y.beta']) == \
        struct.pack('<H', 0x4000) * 64
    assert json.loads(inspect(clipped))['manifest'] == clipped_manifest
    tiled_clip = compile_text(
        activation_source('clip', (128,), alpha=-1, beta=2), 'tiled-clip')
    tiled_clip_manifest = json.loads((tiled_clip / 'manifest.json').read_text())
    assert [program['operation'] for program in tiled_clip_manifest['programs']] == [
        'maximum', 'maximum', 'minimum', 'minimum']
    assert [(tiled_clip / program['file']).read_bytes() for program in
            tiled_clip_manifest['programs']] == [
                clip_low, clip_low, clip_high, clip_high]
    assert json.loads(inspect(tiled_clip))['manifest'] == tiled_clip_manifest
    compile_text(activation_source('clip', alpha=0.1, beta=2), 'clip-inexact', False,
                 'h13.inexact-constant')
    compile_text(activation_source('clip', alpha=2, beta=-1), 'clip-invalid-range', False,
                 'h13.invalid-clip-range')
    compile_text(activation_source('clip', alpha=1, beta=1), 'clip-equal')

    add_relu = compile_text(activation_chain_source('relu'), 'add-relu')
    add_relu_manifest = json.loads((add_relu / 'manifest.json').read_text())
    assert [program['operation'] for program in add_relu_manifest['programs']] == [
        'add', 'relu']
    assert add_relu_manifest['intermediates'] == ['sum']
    assert json.loads(inspect(add_relu))['manifest'] == add_relu_manifest
    fused_add = compile_text(
        activation_chain_source('relu', shape=(1, 512, 1, 1)),
        'fused-add-relu', schedule='chain')
    fused_add_manifest = json.loads((fused_add / 'manifest.json').read_text())
    fused_add_program = fused_add_manifest['programs'][0]
    assert fused_add_manifest['schedule'] == 'chain'
    assert fused_add_manifest['dispatchPlan'] == [0]
    assert fused_add_program['encoder'] == 'composed-chain'
    assert [task['operation'] for task in fused_add_manifest['tasks']] == ['add']
    fused_add_tasks = h13_tasks((fused_add / 'program-0.anec').read_bytes())
    fused_add_oracle = json.loads((Path(__file__).resolve().parents[1] /
        'research/oracles/h13/chain_add_relu_c512.json').read_text())
    base_add_oracle = json.loads((Path(__file__).resolve().parents[1] /
        'research/oracles/h13/binary_add_1x512x1x1.json').read_text())
    assert len(fused_add_tasks) == 1
    fused_add_registers = h13_task_registers(fused_add_tasks[0])
    assert fused_add_registers == oracle_registers(
        fused_add_oracle['task_descriptors'][0])
    base_add_registers = oracle_registers(base_add_oracle['task_descriptors'][0])
    assert {address: (base_add_registers[address], fused_add_registers[address])
            for address in base_add_registers
            if base_add_registers[address] != fused_add_registers[address]} == {
                0x08800: (0x00080000, 0x00080020)}
    assert json.loads(inspect(fused_add))['manifest'] == fused_add_manifest
    fused_manifest_path = fused_add / 'manifest.json'
    for fused in (None, []):
        invalid = json.loads(json.dumps(fused_add_manifest))
        if fused is None:
            invalid['programs'][0].pop('fused')
            invalid.pop('fused')
        else:
            invalid['programs'][0]['fused'] = fused
            invalid['fused'] = fused
        fused_manifest_path.write_text(json.dumps(invalid))
        inspect(fused_add, success=False)
    fused_manifest_path.write_text(json.dumps(fused_add_manifest))
    compile_text(activation_chain_source('relu', multiply=True),
                 'unsupported-three-op-chain', False,
                 'h13.chain-unrepresentable-edge', schedule='chain')
    compile_text(chain_source(), 'unsupported-general-chain', False,
                 'h13.chain-unrepresentable-edge', schedule='chain')


    for op, alpha, beta, operations in (
            ('relu', None, None, ('add', 'relu', 'mul')),
            ('clip', -1, 2, ('add', 'maximum', 'minimum', 'mul'))):
        activation_chain = compile_text(
            activation_chain_source(op, alpha, beta, multiply=True), f'{op}-chain')
        activation_chain_manifest = json.loads(
            (activation_chain / 'manifest.json').read_text())
        assert tuple(program['operation']
                     for program in activation_chain_manifest['programs']) == operations
        assert json.loads(inspect(activation_chain))['manifest'] == activation_chain_manifest

    chain = compile_text(chain_source(), 'chain')
    chain_manifest = json.loads((chain / 'manifest.json').read_text())
    assert set(chain_manifest) == {
        'schema', 'target', 'artifactFormat', 'programs', 'dispatchPlan',
        'intermediates', 'tensors', 'physicalOutputs', 'logicalResults'}
    assert chain_manifest['dispatchPlan'] == [0, 1, 2]
    assert chain_manifest['intermediates'] == ['sum', 'product']
    assert [program['file'] for program in chain_manifest['programs']] == [
        'program-0.anec', 'program-1.anec', 'program-2.anec']
    assert [program['operation'] for program in chain_manifest['programs']] == [
        'add', 'mul', 'add']
    assert (chain / 'program-0.anec').read_bytes() == data
    assert (chain / 'program-1.anec').read_bytes() == \
        (root / 'mul-canonical' / 'program-0.anec').read_bytes()
    assert (chain / 'program-2.anec').read_bytes() == tiled_reference
    assert chain_manifest['programs'][0]['outputs'][0]['role'] == 'intermediate'
    assert chain_manifest['programs'][1]['inputs'][0]['role'] == 'intermediate'
    assert chain_manifest['programs'][1]['outputs'][0]['role'] == 'intermediate'
    assert chain_manifest['programs'][2]['inputs'][0]['role'] == 'intermediate'
    assert bytes.fromhex(chain_manifest['programs'][2]['constantInputs']['c']) == \
        struct.pack('<e', -1.0) * 64
    chain_inspection = json.loads(inspect(chain))
    assert chain_inspection['manifest'] == chain_manifest
    chain_raw, chain_buffer, chain_output = (
        root / 'chain-input.fp16', root / 'chain-input.buffer', root / 'chain-output.fp16')
    chain_raw.write_bytes(dense)
    inspect(chain, '--pack-input', 'b', chain_raw, '--output', chain_buffer)
    assert (chain_buffer / 'program-0.b.buffer').read_bytes() == expected
    assert (chain_buffer / 'program-1.b.buffer').read_bytes() == expected
    inspect(chain, '--unpack-output', 'result', chain_buffer / 'program-0.b.buffer',
            '--output', chain_output)
    assert chain_output.read_bytes() == dense
    chain_constant = root / 'chain-constant.buffer'
    inspect(chain, '--pack-constant', 'c', '--output', chain_constant)
    expected_chain_constant = bytearray(16384)
    for channel in range(64):
        expected_chain_constant[channel * 64:channel * 64 + 2] = struct.pack('<e', -1.0)
    assert chain_constant.read_bytes() == expected_chain_constant
    padded_chain = compile_text(chain_source(shape=(96,)), 'padded-chain')
    padded_chain_manifest = json.loads((padded_chain / 'manifest.json').read_text())
    assert [program['operation'] for program in padded_chain_manifest['programs']] == \
        ['add', 'add', 'mul', 'mul', 'add', 'add']
    assert padded_chain_manifest['programs'][1]['outputs'][0]['slice'] == \
        padded_chain_manifest['programs'][3]['inputs'][0]['slice']
    assert json.loads(inspect(padded_chain))['manifest'] == padded_chain_manifest
    overlapping_producer = json.loads(json.dumps(padded_chain_manifest))
    overlapping_producer['programs'][1]['outputs'][0]['slice']['elementOffset'] = 0
    (padded_chain / 'manifest.json').write_text(json.dumps(overlapping_producer))
    inspect(padded_chain, success=False)

    invalid_chain = compile_text(chain_source(), 'invalid-chain-manifest')
    invalid_chain_manifest = json.loads((invalid_chain / 'manifest.json').read_text())
    invalid_chain_manifest['dispatchPlan'] = [1, 0, 2]
    (invalid_chain / 'manifest.json').write_text(json.dumps(invalid_chain_manifest))
    inspect(invalid_chain, success=False)
    early_return = compile_text(chain_source('sum, result'),
                                'early-intermediate-return')
    early_manifest = json.loads((early_return / 'manifest.json').read_text())
    assert [output['tensor'] for output in early_manifest['physicalOutputs']] == [
        'sum', 'result']
    assert [result['name'] for result in early_manifest['logicalResults']] == [
        'sum', 'result']
    assert [result['physical']['tensor']
            for result in early_manifest['logicalResults']] == ['sum', 'result']
    assert early_manifest['tensors']['sum']['role'] == 'output'
    assert early_manifest['tensors']['product']['role'] == 'intermediate'
    assert json.loads(inspect(early_return))['manifest'] == early_manifest
    invalid_early_manifest = json.loads(json.dumps(early_manifest))
    invalid_early_manifest['dispatchPlan'] = [1, 0, 2]
    (early_return / 'manifest.json').write_text(json.dumps(invalid_early_manifest))
    inspect(early_return, success=False)
    (early_return / 'manifest.json').write_text(json.dumps(early_manifest))
    compile_text(chain_source(middle='a'), 'unused-intermediate', False,
                 'h13.unsupported-chain')
    values = tuple(1.0 + (index % 8) * 0.125 for index in range(64))
    unchanged_constants = b''.join(struct.pack('<e', value) for value in values)
    for op in ('add', 'mul', 'maximum', 'minimum'):
        canonical = first if op == 'add' else root / f'{op}-canonical'
        canonical_bytes = (canonical / 'program-0.anec').read_bytes()
        folded_bytes = None
        for const_first in (False, True):
            folded = compile_text(constant_source(op, values, const_first),
                                  f'{op}-constant-{"first" if const_first else "second"}')
            folded_manifest = json.loads((folded / 'manifest.json').read_text())
            current = (folded / 'program-0.anec').read_bytes()
            folded_bytes = folded_bytes or current
            assert current == folded_bytes == canonical_bytes
            assert bytes.fromhex(folded_manifest['constantInputs']['c']) == unchanged_constants
            assert folded_manifest['inputs'][0]['name'] == 'a'
            assert folded_manifest['inputs'][1]['binding'] == 'constant'
    sub = compile_text(constant_source('sub', values), 'sub-constant')
    sub_manifest = json.loads((sub / 'manifest.json').read_text())
    negated = b''.join(struct.pack('<e', -value) for value in values)
    assert (sub / 'program-0.anec').read_bytes() == tiled_reference
    assert sub_manifest['operation'] == 'add'
    assert sub_manifest['inputs'][1]['binding'] == 'constant'
    assert bytes.fromhex(sub_manifest['constantInputs']['c']) == negated

    packed_constant = root / 'constant.buffer'
    inspect(sub, '--pack-constant', 'c', '--output', packed_constant)
    expected_constant = bytearray(16384)
    for channel in range(64):
        expected_constant[channel * 64:channel * 64 + 2] = negated[channel * 2:channel * 2 + 2]
    assert packed_constant.read_bytes() == expected_constant

    quarters = (0.25,) * 64
    div = compile_text(constant_source('real_div', (4.0,) * 64), 'div-four')
    mul_quarter = compile_text(constant_source('mul', quarters), 'mul-quarter')
    assert (div / 'program-0.anec').read_bytes() == (mul_quarter / 'program-0.anec').read_bytes()
    div_manifest = json.loads((div / 'manifest.json').read_text())
    mul_manifest = json.loads((mul_quarter / 'manifest.json').read_text())
    assert div_manifest['operation'] == 'mul'
    assert div_manifest['constantInputs'] == mul_manifest['constantInputs']
    assert bytes.fromhex(div_manifest['constantInputs']['c']) == struct.pack('<e', 0.25) * 64
    compile_text(constant_source('real_div', (3.0,) * 64), 'div-three', False,
                 'h13.inexact-reciprocal')
    extremes = compile_text(constant_source('real_div', (32768.0, 2.0 ** -14) * 32), 'div-extremes')
    extremes_manifest = json.loads((extremes / 'manifest.json').read_text())
    assert bytes.fromhex(extremes_manifest['constantInputs']['c']) == \
        struct.pack('<ee', 2.0 ** -15, 16384.0) * 32
    compile_text(constant_source('real_div', (2.0 ** -15,) * 64), 'div-subnormal', False,
                 'h13.inexact-reciprocal')
    compile_text(constant_source('sub', values, const_first=True), 'constant-minus-input',
                 False, 'h13.nonfoldable-binary')
    compile_text(constant_source('real_div', values, const_first=True),
                 'constant-divided-by-input', False, 'h13.nonfoldable-binary')
    scalar_shape = (130,)
    scalar_value = 1.25
    for op in ('add', 'sub'):
        scalar = compile_text(
            constant_source(op, scalar_value, shape=scalar_shape), f'scalar-{op}')
        repeated = compile_text(
            constant_source(op, (scalar_value,) * 130, shape=scalar_shape),
            f'repeated-{op}')
        scalar_manifest = json.loads((scalar / 'manifest.json').read_text())
        repeated_manifest = json.loads((repeated / 'manifest.json').read_text())
        assert scalar_manifest == repeated_manifest
        assert [program['inputs'][0]['slice']['elementOffset']
                for program in scalar_manifest['programs']] == [0, 64, 128]
        assert [program['inputs'][0]['slice']['elementCount']
                for program in scalar_manifest['programs']] == [64, 64, 2]
        assert scalar_manifest['tensors']['a']['shape'] == [130]
        assert scalar_manifest['tensors']['y']['shape'] == [130]
        assert [(scalar / f'program-{index}.anec').read_bytes()
                for index in range(3)] == [
                    (repeated / f'program-{index}.anec').read_bytes()
                    for index in range(3)]
    compile_text(constant_source('sub', scalar_value, const_first=True,
                                 shape=scalar_shape),
                 'scalar-constant-minus-input', False, 'h13.nonfoldable-binary')
    compile_text(constant_source('add', 65520.0, shape=scalar_shape),
                 'nonfinite-scalar-result', False, 'h13.invalid-constant-payload')

    scalar_mul = compile_text(constant_source('mul', -1.0), 'scalar-mul')
    tensor_mul = compile_text(constant_source('mul', (-1.0,) * 64), 'tensor-mul')
    assert (scalar_mul / 'program-0.anec').read_bytes() == (tensor_mul / 'program-0.anec').read_bytes()
    scalar_manifest = json.loads((scalar_mul / 'manifest.json').read_text())
    tensor_manifest = json.loads((tensor_mul / 'manifest.json').read_text())
    assert scalar_manifest['constantInputs'] == tensor_manifest['constantInputs']
    assert scalar_manifest['inputs'][1]['shape'] == [1, 64, 1, 1]

    blob = bytearray(128 + len(values) * 2)
    struct.pack_into('<IIQQ', blob, 64, 0xDEADBEEF, 1, len(values) * 2, 128)
    blob[128:] = b''.join(struct.pack('<e', value) for value in values)
    (root / 'constant.bin').write_bytes(blob)
    blob_sub = compile_text(constant_source('sub', None, blob='constant.bin'), 'blob-sub')
    blob_manifest = json.loads((blob_sub / 'manifest.json').read_text())
    assert (blob_sub / 'program-0.anec').read_bytes() == tiled_reference
    assert bytes.fromhex(blob_manifest['constantInputs']['c']) == negated

    invalid_manifest = json.loads((blob_sub / 'manifest.json').read_text())
    struct.pack_into('<H', blob, 128, 0x7C00)
    (root / 'constant.bin').write_bytes(blob)
    compile_text(constant_source('sub', None, blob='constant.bin'),
                 'nonfinite-sub-constant', False, 'h13.nonfinite-constant')
    invalid_manifest['constantInputs']['c'] = invalid_manifest['constantInputs']['c'][:-2]
    invalid = compile_text(constant_source('sub', values), 'invalid-constant-manifest')
    (invalid / 'manifest.json').write_text(json.dumps(invalid_manifest))
    inspect(invalid, success=False)
    compile_text(source(shape=(0, 64)), 'zero-binary-shape', False)
    compile_text(source(shape=(1, 64), y_shape=(64,), output_shape=(1, 64)),
                 'broadcast-binary-shape', False)
    compile_text(source().replace("program(1.3)", "@program(1.3)"),
                 "lexical-error", False, "mil.lex.")
    (root / "empty-output").mkdir()
    compile_text(source(), "empty-output")
    two_input_sub = compile_text(source('sub'), 'two-input-sub')
    two_input_sub_manifest = json.loads((two_input_sub / 'manifest.json').read_text())
    assert two_input_sub_manifest['operation'] == 'sub'
    assert two_input_sub_manifest['encoder'] == 'h13-oracle-parity'
    assert len(two_input_sub_manifest['inputs']) == 2
    compile_text(source('real_div'), 'two-input-real-div', False,
                 'h13.nonfoldable-binary')
    compile_text(source(extra='tensor<fp16, [1, 64, 1, 1]> z = relu(x = y)[name = string("z")];'),
                 'multiple-ops', False)
    compile_text(source().replace('x = a, y = b', 'x = a, y = a'), 'unused-input', False)
    preserved = {p.name: p.read_bytes() for p in first.iterdir()}
    mil.write_text(source())
    blocked = subprocess.run([compiler, "--mil", str(mil), "--model-root", str(root),
                              "--target", "H13", "--output", str(first)],
                             capture_output=True, text=True, timeout=scaled_timeout(15), check=False)
    assert blocked.returncode == 74, blocked.stderr
    assert preserved == {p.name: p.read_bytes() for p in first.iterdir()}
    small_weights = bytearray(128 + 200 * 300 * 2)
    struct.pack_into('<IIQQ', small_weights, 64, 0xDEADBEEF, 1, 200 * 300 * 2, 128)
    for row, column, bits in ((0, 0, 0x3c00), (7, 13, 0x4000),
                              (199, 299, 0x4200)):
        struct.pack_into('<H', small_weights, 128 + (row * 300 + column) * 2, bits)
    (root / 'weights.bin').write_bytes(small_weights)
    small_projection = compile_text(
        matmul_source(200, output_shape=(1, 300), transpose_y=False,
                      weight_shape=(200, 300)), 'projection-200x300')
    small_manifest = json.loads((small_projection / 'manifest.json').read_text())
    assert small_manifest['inputs'][0]['slice'] == {
        'tensor': 'x', 'elementOffset': 0, 'elementCount': 200,
        'physicalElements': 512}
    assert small_manifest['outputs'][0]['slice'] == {
        'tensor': 'y', 'elementOffset': 0, 'elementCount': 300,
        'physicalElements': 512}
    assert json.loads(inspect(small_projection))['manifest'] == small_manifest

    # Two geometries that pad to the same 256x512 physical program must emit
    # identical bytes. The reference stays at 250x400 because 256x512 is inside
    # the decoded Apple envelope and lowers through the parity encoder instead.
    reference_weights = bytearray(128 + 250 * 400 * 2)
    struct.pack_into('<IIQQ', reference_weights, 64, 0xDEADBEEF, 1,
                     250 * 400 * 2, 128)
    for row in range(200):
        source_offset = 128 + row * 300 * 2
        destination_offset = 128 + row * 400 * 2
        reference_weights[destination_offset:destination_offset + 600] = \
            small_weights[source_offset:source_offset + 600]
    (root / 'weights.bin').write_bytes(reference_weights)
    reference_projection = compile_text(
        matmul_source(250, output_shape=(1, 400), transpose_y=False,
                      weight_shape=(250, 400)),
        'projection-200x300-reference')
    assert (small_projection / 'program-0.anec').read_bytes() == \
        (reference_projection / 'program-0.anec').read_bytes()
    transposed_weights = bytearray(128 + 300 * 200 * 2)
    struct.pack_into('<IIQQ', transposed_weights, 64, 0xDEADBEEF, 1,
                     300 * 200 * 2, 128)
    for row in range(200):
        for column in range(300):
            source_offset = 128 + (row * 300 + column) * 2
            destination_offset = 128 + (column * 200 + row) * 2
            transposed_weights[destination_offset:destination_offset + 2] = \
                small_weights[source_offset:source_offset + 2]
    (root / 'weights.bin').write_bytes(transposed_weights)
    transposed_projection = compile_text(
        matmul_source(200, output_shape=(1, 300), weight_shape=(300, 200)),
        'projection-200x300-transposed')
    assert (transposed_projection / 'program-0.anec').read_bytes() == \
        (small_projection / 'program-0.anec').read_bytes()
    bias_blob = bytearray(128 + 300 * 2)
    struct.pack_into('<IIQQ', bias_blob, 64, 0xDEADBEEF, 1, 300 * 2, 128)
    (root / 'bias.bin').write_bytes(bias_blob)
    odd_chain = compile_text(matmul_activation_chain_source(200, 300),
                             'matmul-add-relu-odd')
    odd_manifest_path = odd_chain / 'manifest.json'
    odd_manifest = json.loads(odd_manifest_path.read_text())
    assert [program['operation'] for program in odd_manifest['programs']] == (
        ['matmul'] + ['add'] * 5 + ['maximum'] * 5)
    assert odd_manifest['dispatchPlan'] == list(range(11))
    assert odd_manifest['programs'][0]['outputs'][0]['slice'] == {
        'tensor': 'projection', 'elementOffset': 0, 'elementCount': 300,
        'physicalElements': 512}
    assert json.loads(inspect(odd_chain))['manifest'] == odd_manifest
    invalid_odd_manifest = json.loads(json.dumps(odd_manifest))
    invalid_odd_manifest['dispatchPlan'][:2] = [1, 0]
    odd_manifest_path.write_text(json.dumps(invalid_odd_manifest))
    inspect(odd_chain, success=False)
    odd_manifest_path.write_text(json.dumps(odd_manifest))
    wide_weights = bytearray(128 + 256 * 1000 * 2)
    struct.pack_into('<IIQQ', wide_weights, 64, 0xDEADBEEF, 1, 256 * 1000 * 2, 128)
    for row, column, bits in ((0, 0, 0x3c00), (7, 700, 0x4000),
                              (255, 999, 0x4200)):
        struct.pack_into('<H', wide_weights, 128 + (row * 1000 + column) * 2, bits)
    (root / 'weights.bin').write_bytes(wide_weights)
    wide_projection = compile_text(
        matmul_source(256, output_shape=(1, 1000), transpose_y=False,
                      weight_shape=(256, 1000)), 'projection-256x1000')
    wide_manifest = json.loads((wide_projection / 'manifest.json').read_text())
    assert len(wide_manifest['programs']) == 2
    assert [program['outputs'][0]['slice'] for program in wide_manifest['programs']] == [
        {'tensor': 'y', 'elementOffset': 0, 'elementCount': 512,
         'physicalElements': 512},
        {'tensor': 'y', 'elementOffset': 512, 'elementCount': 488,
         'physicalElements': 512}]
    assert (wide_projection / 'program-0.anec').read_bytes() != \
        (wide_projection / 'program-1.anec').read_bytes()
    assert json.loads(inspect(wide_projection))['manifest'] == wide_manifest

    # 1000x512 stays outside the decoded Apple envelope, so it keeps exercising
    # the chunked reduction plan; 1024x512 now lowers as one parity program.
    # The 512-wide accumulate lowers through the native 64-element-sliced
    # encoder, and the tail chunk's canonical equality is covered by
    # reduction-700 below.
    reduction_weights = b''.join(
        struct.pack('<H', (row * 17 + column * 31) & 0x3bff)
        for row in range(512) for column in range(1000))
    reduction_blob = bytearray(128 + len(reduction_weights))
    struct.pack_into('<IIQQ', reduction_blob, 64, 0xDEADBEEF, 1,
                     len(reduction_weights), 128)
    reduction_blob[128:] = reduction_weights
    (root / 'weights.bin').write_bytes(reduction_blob)
    wide_reduction = compile_text(matmul_source(1000), 'reduction-1000')
    wide_reduction_manifest = json.loads(
        (wide_reduction / 'manifest.json').read_text())
    assert [program['operation'] for program in
            wide_reduction_manifest['programs'][:2]] == ['matmul', 'matmul']
    assert {program['operation'] for program in
            wide_reduction_manifest['programs'][2:]} == {'add'}
    assert [program['inputs'][0]['slice'] for program in
            wide_reduction_manifest['programs'][:2]] == [
        {'tensor': 'x', 'elementOffset': 0, 'elementCount': 512,
         'physicalElements': 512},
        {'tensor': 'x', 'elementOffset': 512, 'elementCount': 488,
         'physicalElements': 512}]
    assert wide_reduction_manifest['tensors']['y']['accumulation'] == 'chunked-fp16'
    assert [program['outputs'][0]['name'] for program in
            wide_reduction_manifest['programs'][:2]] == [
        '$h13.y.partial0', '$h13.y.partial1']
    multirow_reduction = compile_text(
        matmul_source(1000, x_shape=(2, 1000), output_shape=(2, 512)),
        'reduction-1000-multirow')
    multirow_reduction_manifest = json.loads(
        (multirow_reduction / 'manifest.json').read_text())
    assert [program['inputs'][0]['slice']['elementOffset'] for program in
            multirow_reduction_manifest['programs'][:4]] == [0, 1000, 512, 1512]
    assert json.loads(inspect(multirow_reduction))['manifest'] == \
        multirow_reduction_manifest
    assert json.loads(inspect(wide_reduction))['manifest'] == wide_reduction_manifest
    wide_reduction_path = wide_reduction / 'manifest.json'
    invalid_accumulation = json.loads(json.dumps(wide_reduction_manifest))
    invalid_accumulation['tensors']['y']['accumulation'] = 'fp32'
    wide_reduction_path.write_text(json.dumps(invalid_accumulation))
    inspect(wide_reduction, success=False)
    wide_reduction_path.write_text(json.dumps(wide_reduction_manifest))

    tail_values = b''.join(struct.pack('<H', index & 0x3bff) for index in range(700))
    tail_blob = bytearray(128 + len(tail_values))
    struct.pack_into('<IIQQ', tail_blob, 64, 0xDEADBEEF, 1, len(tail_values), 128)
    tail_blob[128:] = tail_values
    (root / 'weights.bin').write_bytes(tail_blob)
    tail_reduction = compile_text(
        matmul_source(700, output_shape=(1, 1), weight_shape=(1, 700)),
        'reduction-700')
    tail_manifest = json.loads((tail_reduction / 'manifest.json').read_text())
    assert tail_manifest['programs'][1]['inputs'][0]['slice'] == {
        'tensor': 'x', 'elementOffset': 512, 'elementCount': 188,
        'physicalElements': 512}
    tail_canonical_values = tail_values[1024:]
    tail_canonical_blob = bytearray(128 + len(tail_canonical_values))
    struct.pack_into('<IIQQ', tail_canonical_blob, 64, 0xDEADBEEF, 1,
                     len(tail_canonical_values), 128)
    tail_canonical_blob[128:] = tail_canonical_values
    (root / 'weights.bin').write_bytes(tail_canonical_blob)
    tail_canonical = compile_text(
        matmul_source(188, output_shape=(1, 1), weight_shape=(1, 188)),
        'reduction-700-tail-reference')
    assert (tail_reduction / 'program-1.anec').read_bytes() == \
        (tail_canonical / 'program-0.anec').read_bytes()
    assert json.loads(inspect(tail_reduction))['manifest'] == tail_manifest

    linear_weight_values = b''.join(
        struct.pack('<H', (row * 13 + column * 7) & 0x3bff)
        for row in range(300) for column in range(256))
    linear_weights = bytearray(128 + len(linear_weight_values))
    struct.pack_into('<IIQQ', linear_weights, 64, 0xDEADBEEF, 1,
                     len(linear_weight_values), 128)
    linear_weights[128:] = linear_weight_values
    (root / 'weights.bin').write_bytes(linear_weights)
    bias_values = tuple(float(index % 16) for index in range(300))
    linear = compile_text(linear_source(256, 300, bias_values), 'linear-bias')
    linear_manifest = json.loads((linear / 'manifest.json').read_text())
    assert [program['operation'] for program in linear_manifest['programs']] == \
        ['matmul'] + ['add'] * 5
    expected_bias = b''.join(struct.pack('<e', value) for value in bias_values)
    assert b''.join(bytes.fromhex(program['constantInputs']['$h13.y.bias'])[
                        :program['inputs'][1]['logicalBytes']]
                    for program in linear_manifest['programs'][1:]) == expected_bias
    direct = compile_text(
        matmul_source(256, output_shape=(1, 300), weight_shape=(300, 256)),
        'linear-direct-reference')
    assert (linear / 'program-0.anec').read_bytes() == \
        (direct / 'program-0.anec').read_bytes()
    assert json.loads(inspect(linear))['manifest'] == linear_manifest

    linear_without_bias = compile_text(linear_source(256, 300), 'linear-no-bias')
    assert len(json.loads((linear_without_bias / 'manifest.json').read_text())[
        'programs']) == 1
    assert (linear_without_bias / 'program-0.anec').read_bytes() == \
        (direct / 'program-0.anec').read_bytes()
    bias_blob = bytearray(128 + len(expected_bias))
    struct.pack_into('<IIQQ', bias_blob, 64, 0xDEADBEEF, 1, len(expected_bias), 128)
    bias_blob[128:] = expected_bias
    (root / 'bias.bin').write_bytes(bias_blob)
    linear_blob_bias = compile_text(linear_source(256, 300, 'blob'),
                                    'linear-blob-bias')
    linear_blob_manifest = json.loads((linear_blob_bias / 'manifest.json').read_text())
    assert [program['constantInputs'] for program in linear_blob_manifest['programs'][1:]] == \
        [program['constantInputs'] for program in linear_manifest['programs'][1:]]
    assert json.loads(inspect(linear_blob_bias))['manifest'] == linear_blob_manifest
    batched_linear = compile_text(
        linear_source(256, 300, 'blob', x_shape=(2, 256)),
        'linear-batched-bias')
    batched_linear_manifest = json.loads(
        (batched_linear / 'manifest.json').read_text())
    assert [program['operation'] for program in batched_linear_manifest['programs']] == \
        (['matmul'] + ['add'] * 5) * 2
    assert [program['inputs'][0]['slice']['elementOffset'] for program in
            batched_linear_manifest['programs'] if program['operation'] == 'matmul'] == \
        [0, 256]
    assert json.loads(inspect(batched_linear))['manifest'] == batched_linear_manifest
    compile_text(linear_source(256, 300, runtime_weight=True),
                 'linear-runtime-weight', False, 'h13.linear-nonconstant-weight')
    compile_text(linear_source(256, 300, (), runtime_bias=True),
                 'linear-runtime-bias', False, 'h13.linear-nonconstant-bias')
    for reduction, physical in ((1, 512), (257, 512)):
        boundary_weights = bytearray(128 + reduction * 2)
        struct.pack_into('<IIQQ', boundary_weights, 64, 0xDEADBEEF, 1,
                         reduction * 2, 128)
        (root / 'weights.bin').write_bytes(boundary_weights)
        boundary = compile_text(
            matmul_source(reduction, output_shape=(1, 1), weight_shape=(1, reduction)),
            f'projection-boundary-{reduction}')
        boundary_manifest = json.loads((boundary / 'manifest.json').read_text())
        assert boundary_manifest['inputs'][0]['slice']['physicalElements'] == physical
        assert boundary_manifest['outputs'][0]['slice']['physicalElements'] == 512
        assert json.loads(inspect(boundary))['manifest'] == boundary_manifest

    padded_chain_weights = bytearray(128 + 512 * 96 * 2)
    struct.pack_into('<IIQQ', padded_chain_weights, 64, 0xDEADBEEF, 1,
                     512 * 96 * 2, 128)
    (root / 'weights.bin').write_bytes(padded_chain_weights)
    compile_text(binary_matmul_chain_source(96), 'padded-binary-matmul', False,
                 'h13.unsupported-chain')

    for reduction in (256, 512):
        weights = bytearray(128 + 512 * reduction * 2)
        struct.pack_into('<IIQQ', weights, 64, 0xDEADBEEF, 1, 512 * reduction * 2, 128)
        for row in range(512):
            struct.pack_into(f"<{reduction}H", weights, 128 + row * reduction * 2,
                             *((row * 17 + column * 31) & 0x3fff for column in range(reduction)))
        struct.pack_into('<H', weights, 128, 0x3C00)
        struct.pack_into("<H", weights, 128 + (7 * reduction + 13) * 2, 0x4200)
        (root / 'weights.bin').write_bytes(weights)
        matmul = matmul_source(reduction)
        matmul_binary = matmul.replace(
            '  } -> (y);',
            '    tensor<fp16, [1, 512]> z = add(x = y, y = y)[name = string("z")];\n'
            '  } -> (z);')
        matmul_binary_package = compile_text(
            matmul_binary, f'matmul-binary-{reduction}')
        matmul_binary_manifest = json.loads(
            (matmul_binary_package / 'manifest.json').read_text())
        assert [program['operation']
                for program in matmul_binary_manifest['programs'][:1]] == ['matmul']
        assert {program['operation']
                for program in matmul_binary_manifest['programs'][1:]} == {'add'}
        assert json.loads(inspect(matmul_binary_package))['manifest'] == \
            matmul_binary_manifest
        projection = compile_text(matmul, f'projection-{reduction}')
        payload = (projection / 'program-0.anec').read_bytes()
        # rows==1 constant-weight matvecs lower through the native
        # source-qualified encoder at every reduction; only multirow forms
        # keep the decoded parity program.
        record = json.loads((projection / 'manifest.json').read_text())
        assert record['inputs'][0]['shape'] == [1, reduction]
        assert record['outputs'][0]['shape'] == [1, 512]
        assert record['programs'][0]['encoder'] == 'h13-source-qualified'
        assert struct.unpack_from('<Q', payload, 24)[0] == record['constantBytes']
        inspection = json.loads(inspect(projection))
        assert inspection["manifest"] == record
        multirow = compile_text(
            matmul_source(reduction, (2, reduction), (2, 512)),
            f'multirow-{reduction}')
        multirow_manifest = json.loads((multirow / 'manifest.json').read_text())
        assert len(multirow_manifest['programs']) == 1
        multirow_program = multirow_manifest['programs'][0]
        assert multirow_program['encoder'] == 'apple-parity-matvec'
        assert multirow_program['inputs'][0].get('slice') is None
        assert multirow_program['outputs'][0].get('slice') is None
        assert multirow_program['inputs'][0]['shape'] == [2, reduction]
        assert multirow_program['outputs'][0]['shape'] == [2, 512]
        multirow_payload = (multirow / 'program-0.anec').read_bytes()
        assert multirow_payload != payload
        assert json.loads(inspect(multirow))['manifest'] == multirow_manifest
        batched = compile_text(
            matmul_source(reduction, (2, 1, reduction), (2, 1, 512)),
            f'broadcast-batch-{reduction}')
        assert (batched / 'program-0.anec').read_bytes() == multirow_payload
        compile_text(matmul_source(reduction, (reduction, 2), (2, 512), True),
                     f'transposed-multirow-{reduction}', False,
                     'h13.transpose-x-multirow')
        if reduction == 256:
            pair_payload = struct.pack('<e', 0.5) * (256 * 256)
            pair_blob = bytearray(128 + len(pair_payload))
            struct.pack_into('<IIQQ', pair_blob, 64, 0xDEADBEEF, 1,
                             len(pair_payload), 128)
            pair_blob[128:] = pair_payload
            (root / 'weights.bin').write_bytes(pair_blob)
            chain = compile_text(matmul_relu_chain_source(),
                                 'matmul-relu-chain', schedule='chain')
            chain_manifest = json.loads((chain / 'manifest.json').read_text())
            chain_program = chain_manifest['programs'][0]
            assert chain_manifest['schedule'] == 'chain'
            assert chain_manifest['dispatchPlan'] == [0]
            assert len(chain_manifest['programs']) == 1
            assert chain_program['encoder'] == 'composed-chain'
            assert [task['operation'] for task in chain_manifest['tasks']] == [
                'matmul', 'matmul']
            chain_tasks = h13_tasks((chain / 'program-0.anec').read_bytes())
            fused_oracle = json.loads((Path(__file__).resolve().parents[1] /
                'research/oracles/h13/chain_pair_mm_relu_d256_s64.json').read_text())
            base_oracle = json.loads((Path(__file__).resolve().parents[1] /
                'research/oracles/h13/chain_base_matmul_d256_s64.json').read_text())
            assert len(chain_tasks) == 2
            decoded = [h13_task_registers(task) for task in chain_tasks]
            assert decoded == [oracle_registers(task) for task in
                               fused_oracle['task_descriptors']]
            base = [oracle_registers(task) for task in
                    base_oracle['task_descriptors']]
            differences = {(index, address): (base[index][address],
                                               decoded[index][address])
                           for index in range(len(base))
                           for address in base[index]
                           if base[index][address] != decoded[index][address]}
            assert differences == {
                (1, 0x0c804): (0x00101c00, 0x00111c00)}
            assert not decoded[0].get(0x0c804, 0) & 0x00010000
            assert decoded[1][0x0c804] & 0x00010000
            assert json.loads(inspect(chain))['manifest'] == chain_manifest
            (root / 'weights.bin').write_bytes(weights)
            bias_blob = bytearray(128 + 1024)
            struct.pack_into('<IIQQ', bias_blob, 64, 0xDEADBEEF, 1, 1024, 128)
            bias_blob[128:] = struct.pack('<e', 1.0) * 512
            (root / 'bias.bin').write_bytes(bias_blob)
            projected = compile_text(matmul_activation_chain_source(), 'matmul-add-relu')
            projected_manifest = json.loads((projected / 'manifest.json').read_text())
            assert len(projected_manifest['programs']) == 3
            assert [program['operation'] for program in projected_manifest['programs']] == \
                ['matmul', 'add', 'relu']
            assert projected_manifest['dispatchPlan'] == list(range(3))
            projected_manifest_path = projected / 'manifest.json'
            invalid_projected_manifest = json.loads(json.dumps(projected_manifest))
            invalid_projected_manifest['dispatchPlan'][:2] = [1, 0]
            projected_manifest_path.write_text(json.dumps(invalid_projected_manifest))
            inspect(projected, success=False)
            invalid_projected_manifest = json.loads(json.dumps(projected_manifest))
            invalid_projected_manifest['programs'][1]['taskDescriptors'] = 0
            projected_manifest_path.write_text(json.dumps(invalid_projected_manifest))
            inspect(projected, success=False)
            projected_manifest_path.write_text(json.dumps(projected_manifest))
            assert projected_manifest['intermediates'] == ['projection', 'biased']
            assert json.loads(inspect(projected))['manifest'] == projected_manifest
            residual = compile_text(residual_source(), 'residual')
            residual_manifest_path = residual / 'manifest.json'
            residual_manifest = json.loads(residual_manifest_path.read_text())
            assert [program['operation'] for program in residual_manifest['programs'][:2]] == \
                ['matmul', 'relu']
            assert {program['operation'] for program in residual_manifest['programs'][2:]} == \
                {'add'}
            assert residual_manifest['dispatchPlan'] == \
                list(range(len(residual_manifest['programs'])))
            assert residual_manifest['intermediates'] == ['t', 'u']
            assert all(any(item['name'] == 't' for item in program['inputs'])
                       for program in residual_manifest['programs'][2:])
            assert json.loads(inspect(residual))['manifest'] == residual_manifest
            invalid_residual_manifest = json.loads(json.dumps(residual_manifest))
            invalid_residual_manifest['dispatchPlan'][0], \
                invalid_residual_manifest['dispatchPlan'][1] = 1, 0
            residual_manifest_path.write_text(json.dumps(invalid_residual_manifest))
            inspect(residual, success=False)
            residual_manifest_path.write_text(json.dumps(residual_manifest))
        if reduction == 512:
            binary_matmul = compile_text(binary_matmul_chain_source(), 'binary-matmul')
            binary_matmul_manifest = json.loads(
                (binary_matmul / 'manifest.json').read_text())
            assert {program['operation']
                    for program in binary_matmul_manifest['programs'][:-1]} == {'add'}
            assert binary_matmul_manifest['programs'][-1]['operation'] == 'matmul'
            assert binary_matmul_manifest['programs'][-1]['inputs'][0].get('slice') is None
            assert json.loads(inspect(binary_matmul))['manifest'] == binary_matmul_manifest
        unrelated = matmul.replace(
            f"val = tensor<fp16, [512, {reduction}]>",
            "val = fp16(1.0), debug_payload = tensor<fp16, [512, " + str(reduction) + "]>")
        compile_text(unrelated, f"wrong-constant-{reduction}", False)
        metadata = matmul.replace(
            "val = tensor<",
            'debug_payload = BLOBFILE(path = string("@model_path/not-a-weight.bin"), offset = uint64(64)), val = tensor<')
        unchanged = compile_text(metadata, f"metadata-{reduction}")
        assert (unchanged / "program-0.anec").read_bytes() == payload
        for case, x_shape, output_shape, transpose_x in (
                ('vector', (reduction,), (512,), False),
                ('singleton-batches', (1, 1, reduction), (1, 1, 512), False),
                ('transposed', (reduction, 1), (1, 512), True),
                ('transposed-batches', (1, 1, reduction, 1), (1, 1, 1, 512), True)):
            logical = compile_text(
                matmul_source(reduction, x_shape, output_shape, transpose_x),
                f'{case}-{reduction}')
            logical_manifest = json.loads((logical / 'manifest.json').read_text())
            logical_program = logical_manifest['programs'][0]
            if transpose_x:
                # A transposed x binds as the flattened [K] physical surface
                # and lowers through the same native encoder as the
                # untransposed forms.
                assert logical_program['encoder'] == 'h13-source-qualified', case
                assert logical_program['inputs'][0]['nchw'] == \
                    [1, 512, 1, 1, 64, 64], case
            assert (logical / 'program-0.anec').read_bytes() == payload
            assert logical_manifest['inputs'][0]['shape'] == list(x_shape)
            assert logical_manifest['outputs'][0]['shape'] == list(output_shape)
        compile_text(matmul_source(reduction, (reduction,), (512,), True),
                     f'rank-one-transpose-{reduction}', False)
        compile_text(matmul_source(reduction, (0, 1, reduction), (0, 1, 512)),
                     f'zero-batch-{reduction}', False)
        compile_text(matmul_source(reduction, weight_shape=(1, 512, reduction)),
                     f'rank-three-weight-{reduction}', False)
        normal_weights = bytearray(weights)
        for row in range(512):
            for column in range(reduction):
                old = 128 + (row * reduction + column) * 2
                new = 128 + (column * 512 + row) * 2
                normal_weights[new:new + 2] = weights[old:old + 2]
        (root / "weights.bin").write_bytes(normal_weights)
        normal = matmul.replace(f"[512, {reduction}]", f"[{reduction}, 512]")
        normal = normal.replace("transpose_y = t", "transpose_y = f")
        equivalent = compile_text(normal, f"normal-weights-{reduction}")
        assert (equivalent / "program-0.anec").read_bytes() == payload
        weights[128:130] = struct.pack('<H', 0x4000)
        (root / 'weights.bin').write_bytes(weights)
        changed = compile_text(matmul, f'changed-weights-{reduction}')
        assert (changed / 'program-0.anec').read_bytes() != payload

    compile_text(source(), 'unsupported-format', False,
                 'h13.unsupported-format', format='bogus')
    explicit_anec = compile_text(source(), 'explicit-anec', format='anec')
    hwx = compile_text(source(), 'add-hwx', format='hwx')
    hwx_manifest = json.loads((hwx / 'manifest.json').read_text())
    assert hwx_manifest['schema'] == manifest['schema']
    assert hwx_manifest['artifactFormat'] == 'hwx'
    assert [program['file'] for program in hwx_manifest['programs']] == ['program-0.hwx']
    assert hwx_manifest['programs'][0]['bytes'] == (hwx / 'program-0.hwx').stat().st_size
    comparable_hwx = json.loads(json.dumps(hwx_manifest))
    comparable_anec = json.loads((explicit_anec / 'manifest.json').read_text())
    comparable_hwx['artifactFormat'] = 'anec'
    for hwx_program, anec_program in zip(comparable_hwx['programs'],
                                         comparable_anec['programs']):
        hwx_program['file'] = anec_program['file']
        hwx_program['bytes'] = anec_program['bytes']
    if 'file' in comparable_hwx:
        comparable_hwx['file'] = comparable_anec['file']
        comparable_hwx['bytes'] = comparable_anec['bytes']
    assert comparable_hwx == comparable_anec
    extracted = root / 'extracted.anec'
    extraction = subprocess.run(
        [sys.executable, hwx_inspector, '--extract-anec',
         str(hwx / 'program-0.hwx'), str(extracted)],
        capture_output=True, text=True, timeout=scaled_timeout(15), check=False)
    assert extraction.returncode == 0, extraction.stdout + extraction.stderr
    assert extracted.read_bytes() == (explicit_anec / 'program-0.anec').read_bytes()
    inspected_hwx = subprocess.run(
        [sys.executable, hwx_inspector, str(hwx / 'program-0.hwx')],
        capture_output=True, text=True, timeout=scaled_timeout(15), check=False)
    assert inspected_hwx.returncode == 0, inspected_hwx.stdout + inspected_hwx.stderr
    assert 'architecture subtype=0x0004 name=H13 isa=7' in inspected_hwx.stdout
    assert 'h13_anec_content' in inspected_hwx.stdout

    assert '__TEXT/__text' in inspected_hwx.stdout

    # The kernel-table addends belong to the source-qualified matvec encoder;
    # 257x512 keeps it because only the decoded 256/512/1024 grid is parity.
    relocation_weights = bytearray(128 + 512 * 257 * 2)
    struct.pack_into('<IIQQ', relocation_weights, 64, 0xDEADBEEF, 1,
                     512 * 257 * 2, 128)
    (root / 'weights.bin').write_bytes(relocation_weights)
    relocation_anec = compile_text(matmul_source(257), 'matmul-relocations-anec')
    relocation_hwx = compile_text(
        matmul_source(257), 'matmul-relocations-hwx', format='hwx')
    relocation_inspection = subprocess.run(
        [sys.executable, hwx_inspector,
         str(relocation_hwx / 'program-0.hwx')],
        capture_output=True, text=True, timeout=scaled_timeout(15), check=False)
    assert relocation_inspection.returncode == 0, (
        relocation_inspection.stdout + relocation_inspection.stderr)
    assert relocation_inspection.stdout.count('relocation[') == 16
    assert 'address=0x74 info=0x05000004' in relocation_inspection.stdout
    assert 'address=0xb0 info=0x05000004' in relocation_inspection.stdout
    relocation_extracted = root / 'matmul-relocations.anec'
    relocation_result = subprocess.run(
        [sys.executable, hwx_inspector, '--extract-anec',
         str(relocation_hwx / 'program-0.hwx'), str(relocation_extracted)],
        capture_output=True, text=True, timeout=scaled_timeout(15), check=False)
    assert relocation_result.returncode == 0, (
        relocation_result.stdout + relocation_result.stderr)
    assert relocation_extracted.read_bytes() == \
        (relocation_anec / 'program-0.anec').read_bytes()

    # Softmax, layer_norm, and the reductions lower straight to Apple's own
    # multi-task programs: one program per operation, no elementwise
    # decomposition, and the constant section the decoded oracle carries.
    softmax_package = compile_text(norm_source(
        'softmax', axis=1, shape=(1, 512, 1, 1)), 'softmax-c512')
    softmax_manifest = json.loads((softmax_package / 'manifest.json').read_text())
    assert softmax_manifest['programs'] == [softmax_manifest['programs'][0]]
    assert softmax_manifest['encoder'] == 'apple-parity-norm'
    assert softmax_manifest['operation'] == 'softmax'
    assert softmax_manifest['taskDescriptors'] == 5
    assert softmax_manifest['constantBytes'] == 256
    assert softmax_manifest['inputs'][0]['nchw'] == [1, 512, 1, 1, 64, 64]
    assert softmax_manifest['outputs'][0]['nchw'] == [1, 512, 1, 1, 64, 64]
    assert softmax_manifest['inputs'][0].get('slice') is None
    assert softmax_manifest['outputs'][0].get('slice') is None
    # research/inspect_anec.py is not extended here: its operation whitelist
    # and channel-layout model cover neither these programs nor the spatial
    # elementwise packages that already exist, so the package reader stays a
    # separate piece of work.
    softmax_payload = (softmax_package / 'program-0.anec').read_bytes()
    # The exponential table sits at the start of the section and the
    # reciprocal table in its final 128 bytes; a channel-axis softmax needs
    # both, so neither block is zero.
    constant_start = 4096 + softmax_manifest['constantOffset']
    assert struct.unpack_from('<H', softmax_payload, constant_start)[0] == 0xce40
    assert struct.unpack_from(
        '<H', softmax_payload, constant_start + 256 - 128)[0] == 0x0000
    assert struct.unpack_from(
        '<H', softmax_payload, constant_start + 256 - 126)[0] == 0x7c00

    # A last-axis softmax needs no reciprocal table, so its section is the
    # 128-byte exponential block alone.
    last_axis = compile_text(norm_source(
        'softmax', axis=-1, shape=(1, 512, 1, 1)), 'softmax-last-axis')
    last_axis_manifest = json.loads((last_axis / 'manifest.json').read_text())
    assert last_axis_manifest['constantBytes'] == 128
    assert last_axis_manifest['taskDescriptors'] == 5
    assert (last_axis / 'program-0.anec').read_bytes() != softmax_payload

    # The rank-2 form a transformer emits reaches the same program as the
    # rank-4 [1, 1, 1, C] surface it collapses onto.
    flat_softmax = compile_text(norm_source(
        'softmax', axis=-1, shape=(1, 512)), 'softmax-rank2')
    nchw_softmax = compile_text(norm_source(
        'softmax', axis=3, shape=(1, 1, 1, 512)), 'softmax-rank4')
    assert (flat_softmax / 'program-0.anec').read_bytes() == \
        (nchw_softmax / 'program-0.anec').read_bytes()

    layer_norm_package = compile_text(norm_source(
        'layer_norm', axes=(1,), shape=(1, 512, 1, 1)), 'layer-norm-c512')
    layer_norm_manifest = json.loads(
        (layer_norm_package / 'manifest.json').read_text())
    assert layer_norm_manifest['encoder'] == 'apple-parity-norm'
    assert layer_norm_manifest['taskDescriptors'] == 5
    assert layer_norm_manifest['constantBytes'] == 16384

    for operation, tasks in (('reduce_sum', 1), ('reduce_max', 1),
                             ('reduce_mean', 1)):
        spatial = compile_text(norm_source(
            operation, axes=(2, 3), shape=(1, 64, 8, 8),
            output_shape=(1, 64, 1, 1)), f'{operation}-spatial')
        spatial_manifest = json.loads((spatial / 'manifest.json').read_text())
        assert spatial_manifest['encoder'] == 'apple-parity-norm'
        assert spatial_manifest['operation'] == operation
        assert spatial_manifest['taskDescriptors'] == tasks
        assert spatial_manifest['inputs'][0]['nchw'] == [1, 64, 8, 8, 512, 64]
        assert spatial_manifest['outputs'][0]['nchw'] == [1, 64, 1, 1, 64, 64]
        assert spatial_manifest['tensors']['y']['shape'] == [1, 64, 1, 1]
        # keep_dims = false keeps the reduction but writes Apple's dense
        # [1, 1, 1, N] surface instead of the channel-major one.
        flat = compile_text(norm_source(
            operation, axes=(2, 3), shape=(1, 64, 8, 8), output_shape=(1, 64),
            keep_dims=False), f'{operation}-flat')
        flat_manifest = json.loads((flat / 'manifest.json').read_text())
        assert flat_manifest['encoder'] == 'apple-parity-norm'
        assert flat_manifest['outputs'][0]['nchw'] == [1, 1, 1, 64, 128, 128]
        assert (flat / 'program-0.anec').read_bytes() != \
            (spatial / 'program-0.anec').read_bytes()

    # Channel-axis reduce_sum and reduce_mean differ: the mean needs a second
    # task for its scale, so the operation is not interchangeable.
    channel_sum = compile_text(norm_source(
        'reduce_sum', axes=(1,), shape=(1, 64, 8, 8),
        output_shape=(1, 1, 8, 8)), 'reduce-sum-channel')
    channel_mean = compile_text(norm_source(
        'reduce_mean', axes=(1,), shape=(1, 64, 8, 8),
        output_shape=(1, 1, 8, 8)), 'reduce-mean-channel')
    assert json.loads(
        (channel_sum / 'manifest.json').read_text())['taskDescriptors'] == 2
    assert json.loads(
        (channel_mean / 'manifest.json').read_text())['taskDescriptors'] == 2
    assert (channel_sum / 'program-0.anec').read_bytes() != \
        (channel_mean / 'program-0.anec').read_bytes()

    # Outside the decoded envelope the encoder refuses rather than guessing.
    compile_text(norm_source('softmax', axis=1, shape=(1, 777, 1, 1)),
                 'softmax-off-envelope', success=False,
                 diagnostic='h13.norm-outside-envelope')
    compile_text(norm_source('layer_norm', axes=(1,), shape=(1, 512, 1, 1),
                             epsilon='0.001'),
                 'layer-norm-epsilon', success=False,
                 diagnostic='h13.norm-outside-envelope')
    compile_text(norm_source('layer_norm', axes=(1,), shape=(1, 512, 1, 1),
                             affine=True),
                 'layer-norm-affine', success=False,
                 diagnostic='h13.norm-outside-envelope')
    # A reduction over the batch axis has no decoded surface form.
    compile_text(norm_source('reduce_sum', axes=(0,), shape=(1, 512, 1, 1),
                             output_shape=(1, 512, 1, 1)),
                 'reduce-batch-axis', success=False,
                 diagnostic='h13.norm-outside-envelope')

    # Convolution lowers to Apple's own single-task program with the packed
    # weight, and optional bias, as its whole constant section.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
    import mint_conv_probes as conv_probes  # noqa: E402
    import mint_oracles as oracles  # noqa: E402

    def conv_case(kernel, inputs, outputs, spatial, stride, groups, bias,
                  pad_type='same'):
        case = conv_probes.conv_case(kernel, inputs, outputs, spatial, stride,
                                     groups, bias, pad_type)
        (root / 'weights.bin').write_bytes(case['weights'])
        return case

    for kernel, groups, bias, pad_type, section in (
            (1, 1, False, 'same', 8192), (1, 1, True, 'same', 9216),
            (3, 1, False, 'valid', 73728), (3, 64, True, 'same', 2048),
            (1, 1, False, 'same', 11136)):
        stride = 2 if section == 11136 else 1
        case = conv_case(kernel, 64, 64, 16, stride, groups, bias, pad_type)
        package = compile_text(case['mil'], case['name'])
        manifest = json.loads((package / 'manifest.json').read_text())
        assert manifest['encoder'] == 'apple-parity-conv', case['name']
        assert manifest['operation'] == 'conv', case['name']
        assert manifest['taskDescriptors'] == 1, case['name']
        assert manifest['constantBytes'] == section, \
            (case['name'], manifest['constantBytes'])
        assert manifest['inputs'][0]['nchw'] == [1, 64, 16, 16, 1024, 64]
        expected_side = 8 if stride == 2 else (14 if pad_type == 'valid' else 16)
        assert manifest['outputs'][0]['nchw'] == \
            [1, 64, expected_side, expected_side, 64 * expected_side, 64], \
            case['name']
        assert manifest['tensors']['w']['role'] == 'constant', case['name']

    # The packed section is the weight permuted, not copied: the same weight
    # bytes under a different output count give a different section, and a
    # bias only ever adds a row per plane.
    plain = conv_case(1, 64, 64, 16, 1, 1, False)
    plain_package = compile_text(plain['mil'], 'conv-plain')
    biased = conv_case(1, 64, 64, 16, 1, 1, True)
    biased_package = compile_text(biased['mil'], 'conv-biased')
    assert (plain_package / 'program-0.anec').read_bytes() != \
        (biased_package / 'program-0.anec').read_bytes()

    # Outside the decoded envelope the encoder refuses rather than guessing:
    # a channel count no oracle covers, and a stride-2 3x3 kernel, whose
    # zero-skipping packing the campaign does not derive.
    off_envelope = conv_case(1, 64, 48, 16, 1, 1, False)
    compile_text(off_envelope['mil'], 'conv-off-envelope', success=False,
                 diagnostic='h13.conv-outside-envelope')
    strided_taps = conv_case(3, 64, 64, 16, 2, 1, False)
    compile_text(strided_taps['mil'], 'conv-strided-taps', success=False,
                 diagnostic='h13.conv-outside-envelope')
    (root / 'weights.bin').write_bytes(oracles.blob(oracles.half_payload(64)))

print('H13 MIL-to-ANEC/HWX CLI: PASS (device-free)')
