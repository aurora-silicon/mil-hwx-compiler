#!/usr/bin/env python3
"""Throwaway: simulate packaged H13 programs in pure Python; compare to h13_reference.

Simulates each program's arithmetic on physical buffers with fp16 rounding,
using manifest slices/constants. No hardware.
"""
from __future__ import annotations
import json, math, re, struct, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'research')]
from research import inspect_anec
from h13_reference import fp16, add_fp16, decode_fp16, encode_fp16, evaluate, dot_fp32

COMPILER = ROOT / 'build/mil-hwxc'
STRIDE = 64  # bytes per physical lane


def f16_bits(x: float) -> int:
    return struct.unpack('<H', struct.pack('<e', float(x)))[0]


def bits_f16(b: int) -> float:
    return struct.unpack('<e', struct.pack('<H', b))[0]


def surface(binding, dense: bytes) -> bytes:
    """Dense fp16 -> the binding's physical surface (64-byte lanes or dense rows)."""
    return bytes(inspect_anec.convert_tensor(binding, dense, True))


def dense_of_surface(binding, buffer: bytes) -> bytes:
    return bytes(inspect_anec.convert_tensor(binding, buffer, False))


def pack_dense(dense: bytes, count: int, alloc: int) -> bytearray:
    """Dense fp16[count] -> allocationBytes buffer, 64-byte channel stride, zero pad."""
    out = bytearray(alloc)
    assert len(dense) >= count * 2
    for i in range(count):
        out[i * STRIDE:i * STRIDE + 2] = dense[i * 2:i * 2 + 2]
    return out


def unpack_physical(buf: bytes, count: int) -> bytes:
    out = bytearray(count * 2)
    for i in range(count):
        out[i * 2:i * 2 + 2] = buf[i * STRIDE:i * STRIDE + 2]
    return bytes(out)


def lane_values(buf: bytes, n: int) -> list[float]:
    return [bits_f16(struct.unpack_from('<H', buf, i * STRIDE)[0]) for i in range(n)]


def write_lanes(vals: list[float], alloc: int) -> bytearray:
    out = bytearray(alloc)
    for i, v in enumerate(vals):
        struct.pack_into('<e', out, i * STRIDE, float(v))
    return out


def dense_values(dense: bytes) -> list[float]:
    return [bits_f16(value) for (value,) in struct.iter_unpack('<H', dense)]


def dense_bytes(values: list[float]) -> bytes:
    out = bytearray(len(values) * 2)
    for index, value in enumerate(values):
        struct.pack_into('<e', out, index * 2, float(value))
    return bytes(out)


def binary_op(op: str, a: float, b: float) -> float:
    if op == 'add':
        return add_fp16(a, b)
    if op == 'mul':
        product = fp16(float(a) * float(b))
        # The measured H13 datapath produces unsigned zero products (the
        # reference models the same rule; see test_h13_mul_zero_sign).
        return 0.0 if product == 0.0 else product
    if op == 'maximum':
        # match IEEE-ish via fp16 path used in reference
        return fp16(max(float(a), float(b))) if not (math.isnan(float(a)) or math.isnan(float(b))) else float('nan')
    if op == 'minimum':
        return fp16(min(float(a), float(b))) if not (math.isnan(float(a)) or math.isnan(float(b))) else float('nan')
    raise ValueError(op)


def simulate_binary(op: str, in0: bytes, in1: bytes, lanes: int, alloc: int) -> bytearray:
    a = lane_values(in0, lanes)
    b = lane_values(in1, lanes)
    return write_lanes([binary_op(op, x, y) for x, y in zip(a, b)], alloc)


def pack_weights_block_col_row(weights_tx: bytes, reduction: int) -> bytes:
    """Match H13Program.cpp packWeights for transposeY=True layout [512, K] row-major."""
    output_rows, rows_per_block, block_bytes = 512, 32, 0x8000
    dma_blocks = output_rows // rows_per_block
    packed = bytearray(dma_blocks * block_bytes)
    for block in range(dma_blocks):
        block_offset = block * (block_bytes // 2)
        for column in range(reduction):
            column_offset = block_offset + column * rows_per_block
            for row in range(rows_per_block):
                src = ((block * rows_per_block + row) * reduction + column) * 2
                dst = (column_offset + row) * 2
                packed[dst:dst+2] = weights_tx[src:src+2]
    return bytes(packed)


def simulate_matvec(xin: bytes, weight_const: bytes, reduction: int, out_alloc: int) -> bytearray:
    """xin: physical lanes; weight_const: packWeights output; reduction 256|512."""
    x = lane_values(xin, reduction)
    output_rows, rows_per_block, block_bytes = 512, 32, 0x8000
    W = [[0.0] * reduction for _ in range(output_rows)]
    for block in range(output_rows // rows_per_block):
        block_offset = block * (block_bytes // 2)
        for column in range(reduction):
            column_offset = block_offset + column * rows_per_block
            for row in range(rows_per_block):
                dst = (column_offset + row) * 2
                val = bits_f16(struct.unpack_from('<H', weight_const, dst)[0])
                W[block * rows_per_block + row][column] = val
    out = [dot_fp32(x, W[r]) for r in range(output_rows)]
    return write_lanes(out, out_alloc)


def parity_weights(kernel: bytes, rows: int, reduction: int,
                   columns: int) -> list[list[float]]:
    """Invert ane::h13::packMatvecWeights: Apple's constant-section
    permutation, whose row-group size depends on the columns and, above 128 x
    rows, on the reduction."""
    group = min(16, columns // 16)
    if rows > 128:
        group = min(group, 32768 // reduction)
    group = max(1, group)
    groups = columns // group
    weights = [[0.0] * reduction for _ in range(columns)]
    for column in range(columns):
        plane = column // group
        destination = (plane % 16) * (groups // 16) + plane // 16
        base = destination * reduction * group + column % group
        for index in range(reduction):
            bits = struct.unpack_from('<H', kernel, (base + index * group) * 2)[0]
            weights[column][index] = bits_f16(bits)
    return weights


def matrix_from_surface(binding, buffer: bytes, rows: int,
                        columns: int) -> list[list[float]]:
    """One dense-row surface as `rows` lists of `columns` values."""
    values = dense_values(dense_of_surface(binding, buffer))
    assert len(values) >= rows * columns, (len(values), rows, columns)
    return [values[row * columns:(row + 1) * columns] for row in range(rows)]


def simulate_runtime_matmul(x_binding, y_binding, out_binding,
                            x_surface: bytes, y_surface: bytes,
                            transpose_x: bool, transpose_y: bool) -> bytearray:
    """Both operands runtime, the attention form. A square geometry leaves the
    surfaces alone ambiguous, so the flags come from the MIL."""
    rows, columns = out_binding['nchw'][2], out_binding['nchw'][3]
    reduction = x_binding['nchw'][2] if transpose_x else x_binding['nchw'][3]
    x = matrix_from_surface(x_binding, x_surface,
                            reduction if transpose_x else rows,
                            rows if transpose_x else reduction)
    y = matrix_from_surface(y_binding, y_surface,
                            columns if transpose_y else reduction,
                            reduction if transpose_y else columns)
    values = []
    for row in range(rows):
        lhs = [x[index][row] for index in range(reduction)] if transpose_x \
            else x[row]
        for column in range(columns):
            rhs = y[column] if transpose_y \
                else [y[index][column] for index in range(reduction)]
            values.append(dot_fp32(lhs, rhs))
    return bytearray(surface(out_binding, dense_bytes(values)))


def matmul_transposes(text: str) -> tuple[bool, bool]:
    """The transpose flags one MIL matmul carries, resolved through its bool
    constants."""
    constants = {name: value == 'true' for name, value in re.findall(
        r'bool (\w+) = const\(\)\[name = string\("\w+"\), '
        r'val = bool\((true|false)\)\]', text)}
    match = re.search(r'matmul\(transpose_x = (\w+), transpose_y = (\w+)', text)
    if not match:
        raise ValueError('MIL has no matmul with explicit transpose flags')
    return (constants[match.group(1)], constants[match.group(2)])


def broadcast_index(shape, position):
    """The flat index a broadcast operand contributes to one result position."""
    index = 0
    for extent, coordinate in zip(shape, position):
        index = index * extent + (0 if extent == 1 else coordinate)
    return index


def simulate_broadcast(op: str, out_binding, operands) -> bytearray:
    """`operands` are (binding, surface) pairs, or (None, per-channel values)
    for the constant Apple folds into the section's bias and scale blocks."""
    shape = out_binding['nchw'][:4]
    values = []
    resolved = []
    for binding, data in operands:
        if binding is None:
            resolved.append(([1, shape[1], 1, 1], data))
        else:
            resolved.append((binding['nchw'][:4],
                             dense_values(dense_of_surface(binding, data))))
    positions = [(batch, channel, row, column)
                 for batch in range(shape[0]) for channel in range(shape[1])
                 for row in range(shape[2]) for column in range(shape[3])]
    for position in positions:
        operand_values = [operand[broadcast_index(operand_shape, position)]
                          for operand_shape, operand in resolved]
        values.append(binary_op(op, operand_values[0], operand_values[1]))
    return bytearray(surface(out_binding, dense_bytes(values)))


def simulate_parity_matvec(in_binding, out_binding, x_surface: bytes,
                           kernel: bytes) -> bytearray:
    """Apple's two-task form: `rows` dense x rows against every output column."""
    rows, reduction = in_binding['nchw'][2], in_binding['nchw'][3]
    columns = out_binding['nchw'][3]
    x = dense_values(dense_of_surface(in_binding, x_surface))
    weights = parity_weights(kernel, rows, reduction, columns)
    values = []
    for row in range(rows):
        lane = x[row * reduction:(row + 1) * reduction]
        values.extend(dot_fp32(lane, weights[column]) for column in range(columns))
    return bytearray(surface(out_binding, dense_bytes(values)))


def intermediate_compose(binding, tensors, regions):
    return __import__('h13_run_linux', fromlist=['x'])._intermediate_buffer(binding, tensors, regions)


def dense_of(tensors, name):
    return bytearray(tensors[name]['logicalBytes'])


def run_sim(package: Path, mil: Path, model_root: Path, inputs: dict[str, bytes]):
    manifest, _ = inspect_anec.load_package(package)
    chain = manifest.get('schedule') == 'chain'
    region_indices = {region['program'] for region in manifest.get('regions', [])}
    chain_tasks = []
    if chain:
        program = manifest['programs'][0]
        assert manifest['dispatchPlan'] == [0]
        assert program['encoder'] == inspect_anec.CHAIN_ENCODER
        assert len(manifest['tasks']) == program['taskDescriptors']
        assert [task['index'] for task in manifest['tasks']] == \
            list(range(program['taskDescriptors']))
        assert len({task['operation'] for task in manifest['tasks']}) == 1
        assert manifest['scratch']['regions'] == {}
        data = inspect_anec.local_file(package, program['file'], program['bytes'])
        _, first_words, count, stream_bytes = struct.unpack_from('<QIIQ', data)
        chain_tasks = inspect_anec.split_h13_tasks(
            data[inspect_anec.HEADER_BYTES:inspect_anec.HEADER_BYTES + stream_bytes],
            first_words // 4 - 1, count)
    dense = {}
    tensors = manifest['tensors']
    for name, t in tensors.items():
        if t['role'] == 'input' and 'aliasOf' not in t:
            dense[name] = bytearray(inputs[name])
        else:
            dense[name] = bytearray(t['logicalBytes'])

    intermediate_regions = {}
    output_regions = {}

    for pi in manifest['dispatchPlan']:
        program = manifest['programs'][pi]
        op = manifest['tasks'][0]['operation'] if chain else program['operation']
        encoder = program['encoder']
        parity_matvec = encoder == inspect_anec.PARITY_MATVEC or \
            (chain and op == 'matmul')
        in_bufs = []
        for binding in program['inputs']:
            name = binding['name']
            if binding.get('binding') == 'constant':
                # constant hex is dense physical-count fp16 words (64 for elemwise)
                raw = bytes.fromhex(program['constantInputs'][name])
                logical = binding['logicalBytes']
                in_bufs.append(surface(binding, raw[:logical]))
            elif tensors[name]['role'] == 'intermediate':
                in_bufs.append(intermediate_compose(binding, tensors, intermediate_regions.get(name, [])))
            else:
                sliced = inspect_anec.dense_slice(bytes(dense[name]), binding, tensors)
                in_bufs.append(surface(binding, sliced))

        # produce output physical
        out_binding = program['outputs'][0]
        out_phys = inspect_anec.binding_interval(out_binding, tensors)[3]
        section = None
        if program['constantBytes']:
            anec = inspect_anec.local_file(package, program['file'], program['bytes'])
            start = inspect_anec.HEADER_BYTES + program['constantOffset']
            section = anec[start:start + program['constantBytes']]
        if encoder == inspect_anec.PARITY_BROADCAST:
            operands = list(zip(program['inputs'], in_bufs))
            if len(operands) == 1 and program['constantBytes'] == \
                    out_binding['nchw'][1] * 4:
                # Apple folds a per-channel constant into the section's bias
                # and scale blocks; `add` keeps it as the bias, `mul` as the
                # scale.
                channels = out_binding['nchw'][1]
                block = section[:channels * 2] if op == 'add' \
                    else section[channels * 2:]
                operands.append((None, dense_values(block)))
            out_buf = simulate_broadcast(op, out_binding, operands)
        elif encoder == inspect_anec.PARITY_MATMUL:
            transpose_x, transpose_y = matmul_transposes(Path(mil).read_text())
            out_buf = simulate_runtime_matmul(program['inputs'][1],
                                              program['inputs'][0], out_binding,
                                              in_bufs[1], in_bufs[0],
                                              transpose_x, transpose_y)
        elif op == 'relu':
            values = dense_values(dense_of_surface(program['inputs'][0], in_bufs[0]))
            out_buf = bytearray(surface(
                out_binding, dense_bytes([fp16(max(value, 0.0)) for value in values])))
        elif op in ('add', 'mul', 'maximum', 'minimum'):
            assert len(in_bufs) == 2
            out_buf = simulate_binary(op, in_bufs[0], in_bufs[1], out_phys, out_binding['allocationBytes'])
        elif op == 'matmul':
            if parity_matvec:
                out_buf = simulate_parity_matvec(program['inputs'][0], out_binding,
                                                 in_bufs[0], section)
            else:
                in_phys = inspect_anec.binding_interval(program['inputs'][0], tensors)[3]
                out_buf = simulate_matvec(in_bufs[0], section, in_phys, out_binding['allocationBytes'])
        else:
            raise ValueError(op)

        if chain or pi in region_indices:
            if chain:
                tasks = chain_tasks
            else:
                data = inspect_anec.local_file(package, program['file'], program['bytes'])
                _, first_words, count, stream_bytes = struct.unpack_from('<QIIQ', data)
                tasks = inspect_anec.split_h13_tasks(
                    data[inspect_anec.HEADER_BYTES:inspect_anec.HEADER_BYTES + stream_bytes],
                    first_words // 4 - 1, count)
            registers = [inspect_anec.h13_task_registers(task) for task in tasks]
            address, bit = ((0x0c804, 0x00010000) if op == 'matmul'
                            else (0x08800, 0x20))
            assert not any(task.get(address, 0) & bit for task in registers[:-1])
            assert registers[-1].get(address, 0) & bit
            values = dense_values(dense_of_surface(out_binding, bytes(out_buf)))
            out_buf = bytearray(surface(out_binding, dense_bytes(
                [fp16(max(value, 0.0)) for value in values])))

        name, offset, count, physical = inspect_anec.binding_interval(out_binding, tensors)
        if tensors[name]['role'] == 'intermediate':
            intermediate_regions.setdefault(name, []).append((out_binding, bytes(out_buf)))
        else:
            # write into dense via the binding's own physical layout
            dense[name][offset * 2:(offset + count) * 2] = \
                dense_of_surface(out_binding, bytes(out_buf))
            output_regions.setdefault(name, []).append((out_binding, bytes(out_buf)))

    # unpack outputs like linux runner for multi-slice
    from h13_run_linux import _unpack_outputs
    # Prefer dense for single-writer; for multi-slice outputs use regions
    actual = {}
    for name, t in tensors.items():
        if t['role'] != 'output':
            continue
        target = t.get('aliasOf', name)
        if target in output_regions and len(output_regions[target]) > 1:
            actual[name] = _unpack_outputs(manifest, {target: output_regions[target]}, [target])[target]
        else:
            actual[name] = bytes(dense[target])
    expected = evaluate(mil.read_text(), model_root, {
        k: bytes(v) for k, v in inputs.items()
    })
    return actual, expected, manifest


def compile_mil(text: str, model_root: Path, out: Path,
                blobs: dict[str, bytes] | None = None, schedule: str | None = None):
    mil = model_root / 'model.mil'
    mil.write_text(text)
    if blobs:
        for n, b in blobs.items():
            (model_root / n).write_bytes(b)
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        p.unlink()
    command = [str(COMPILER), '--mil', str(mil), '--model-root', str(model_root),
               '--target', 'H13', '--output', str(out)]
    if schedule:
        command += ['--schedule', schedule]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return mil


def blobfile(payload: bytes) -> bytes:
    out = bytearray(128 + len(payload))
    struct.pack_into('<IIQQ', out, 64, 0xDEADBEEF, 1, len(payload), 128)
    out[128:] = payload
    return bytes(out)
def check_fused_chains():
    failures = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        add_mil = '''program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 512, 1, 1]> a,
                   tensor<fp16, [1, 512, 1, 1]> b) {
    tensor<fp16, [1, 512, 1, 1]> sum = add(x = a, y = b)[name = string("sum")];
    tensor<fp16, [1, 512, 1, 1]> y = relu(x = sum)[name = string("y")];
  } -> (y);
}
'''
        package = root / 'add-relu'
        mil_path = compile_mil(add_mil, root, package, schedule='chain')
        a = encode_fp16([fp16((index % 9) - 4.0) for index in range(512)])
        b = encode_fp16([fp16(0.5 - index % 3) for index in range(512)])
        actual, expected, manifest = run_sim(
            package, mil_path, root, {'a': a, 'b': b})
        if actual['y'] != expected['y']:
            failures.append('actual add-relu task stream differs from reference')
        elif min(decode_fp16(actual['y'])) < 0:
            failures.append('actual add-relu task stream did not clamp negatives')
        else:
            print('OK fused add-relu actual task stream 512')
        if manifest['intermediates'] or manifest['scratch']['regions']:
            failures.append('fused add-relu retained an intermediate surface')

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        matmul_mil = '''program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 64, 256]> x) {
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    tensor<fp16, [256, 256]> W = const()[name = string("W"), val = tensor<fp16, [256, 256]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 64, 256]> product = matmul(x = x, y = W, transpose_x = f, transpose_y = t)[name = string("product")];
    tensor<fp16, [1, 64, 256]> y = relu(x = product)[name = string("y")];
  } -> (y);
}
'''
        weights = encode_fp16([0.5] * (256 * 256))
        package = root / 'matmul-relu'
        mil_path = compile_mil(matmul_mil, root, package,
                               {'weights.bin': blobfile(weights)}, schedule='chain')
        rows = []
        for row in range(64):
            if row % 3 == 0:
                rows.extend([-1.0] * 128 + [1.0] * 128)
            elif row % 3 == 1:
                rows.extend([-1.0] * 256)
            else:
                rows.extend([1.0] * 256)
        actual, expected, manifest = run_sim(
            package, mil_path, root, {'x': encode_fp16(rows)})
        if actual['y'] != expected['y']:
            failures.append('actual matmul-relu task stream differs from reference')
        else:
            values = decode_fp16(actual['y'])
            if values[0] != 0.0 or values[256] != 0.0 or values[512] != 128.0:
                failures.append('matmul relu was not applied after the full K sum')
            else:
                print('OK fused matmul-relu actual two-task stream 64x256x256')
        if manifest['intermediates'] or manifest['scratch']['regions']:
            failures.append('fused matmul-relu retained an intermediate surface')
    return failures

def check_ops():
    failures = []
    ops = [
        ('add', lambda x, y: add_fp16(x, y)),
        ('mul', lambda x, y: fp16(x * y)),
        ('maximum', lambda x, y: fp16(max(x, y))),
        ('minimum', lambda x, y: fp16(min(x, y))),
        ('sub', lambda x, y: add_fp16(x, fp16(-y))),  # folded
        ('real_div', None),  # special
        ('scalar_mul', None),  # scalar constant y, native encoder
    ]
    for op, _ in ops:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            if op in ('sub', 'real_div', 'scalar_mul'):
                # fold constant y
                if op == 'sub':
                    mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [32]> a) {{
    tensor<fp16, [32]> c = const()[name = string("c"), val = tensor<fp16, [32]>([{', '.join('fp16(2.0)' for _ in range(32))}])];
    tensor<fp16, [32]> y = sub(x = a, y = c)[name = string("y")];
  }} -> (y);
}}
'''
                elif op == 'scalar_mul':
                    mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [32]> a) {{
    fp16 c = const()[name = string("c"), val = fp16(-1.0)];
    tensor<fp16, [32]> y = mul(x = a, y = c)[name = string("y")];
  }} -> (y);
}}
'''
                else:
                    mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [32]> a) {{
    tensor<fp16, [32]> c = const()[name = string("c"), val = tensor<fp16, [32]>([{', '.join('fp16(4.0)' for _ in range(32))}])];
    tensor<fp16, [32]> y = real_div(x = a, y = c)[name = string("y")];
  }} -> (y);
}}
'''
                a = encode_fp16([float(i - 8) for i in range(32)])
                inputs = {'a': a}
            else:
                mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [80]> a, tensor<fp16, [80]> b) {{
    tensor<fp16, [80]> y = {op}(x = a, y = b)[name = string("y")];
  }} -> (y);
}}
'''
                a = encode_fp16([0.25 * i for i in range(80)])
                b = encode_fp16([1.5 if i % 2 == 0 else -0.5 for i in range(80)])
                inputs = {'a': a, 'b': b}
            pkg = d / 'pkg'
            mil_path = compile_mil(mil, d, pkg)
            actual, expected, man = run_sim(pkg, mil_path, d, inputs)
            # for folded ops, sim uses manifest operation (add/mul)
            if actual['y'] != expected['y']:
                av, ev = decode_fp16(actual['y']), decode_fp16(expected['y'])
                bad = [(i, ev[i], av[i]) for i in range(len(ev)) if abs(ev[i]-av[i]) > 1e-6][:8]
                failures.append(f'{op} mismatch {bad} manifest_ops={[p["operation"] for p in man["programs"]]}')
            else:
                print(f'OK {op}')
    return failures


def check_matmul_grid():
    failures = []
    Ks = [1, 200, 256, 257, 512, 700, 1024]
    Ns = [1, 64, 300, 512, 1000]
    for K in Ks:
        for N in Ns:
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                # weights [N,K] transpose_y=true
                # use deterministic small values to avoid inf
                w = []
                for r in range(N):
                    for c in range(K):
                        w.append(((r * 3 + c * 5) % 17) * 0.125)
                payload = encode_fp16(w)
                blobs = {'weights.bin': blobfile(payload)}
                mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {K}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    tensor<fp16, [{N}, {K}]> W = const()[name = string("W"), val = tensor<fp16, [{N}, {K}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {N}]> y = matmul(x = x, y = W, transpose_x = f, transpose_y = t)[name = string("y")];
  }} -> (y);
}}
'''
                pkg = d / 'pkg'
                try:
                    mil_path = compile_mil(mil, d, pkg, blobs)
                except Exception as e:
                    failures.append(f'compile K={K} N={N}: {e}')
                    continue
                x = encode_fp16([((i % 7) - 3) * 0.25 for i in range(K)])
                try:
                    actual, expected, man = run_sim(pkg, mil_path, d, {'x': x})
                except Exception as e:
                    failures.append(f'sim K={K} N={N}: {e}')
                    continue
                chunked = man['tensors']['y'].get('accumulation') == 'chunked-fp16'
                av, ev = decode_fp16(actual['y']), decode_fp16(expected['y'])
                if len(av) != len(ev):
                    failures.append(f'K={K} N={N} len {len(av)} vs {len(ev)}')
                    continue
                bad = []
                for i, (a, e) in enumerate(zip(av, ev)):
                    if chunked:
                        if not math.isclose(a, e, rel_tol=0.01, abs_tol=0.03125):
                            bad.append((i, e, a))
                    else:
                        if a != e:
                            bad.append((i, e, a))
                if bad:
                    failures.append(f'K={K} N={N} chunked={chunked} bad={bad[:5]} nprog={len(man["programs"])}')
                else:
                    print(f'OK matmul K={K} N={N} programs={len(man["programs"])} chunked={chunked}')
    return failures


def check_reader_vs_linux_intermediate():
    """matmul->add->relu: intermediate physical composition vs convert_tensor packing."""
    failures = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        K, N = 256, 96
        w = [0.125] * (N * K)
        blobs = {'weights.bin': blobfile(encode_fp16(w))}
        bias = encode_fp16([0.5] * N)
        blobs['bias.bin'] = blobfile(bias)
        mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {K}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    tensor<fp16, [{N}, {K}]> W = const()[name = string("W"), val = tensor<fp16, [{N}, {K}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {N}]> bias = const()[name = string("bias"), val = tensor<fp16, [1, {N}]>(BLOBFILE(path = string("@model_path/bias.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {N}]> projection = matmul(x = x, y = W, transpose_x = f, transpose_y = t)[name = string("projection")];
    tensor<fp16, [1, {N}]> biased = add(x = projection, y = bias)[name = string("biased")];
    tensor<fp16, [1, {N}]> result = relu(x = biased)[name = string("result")];
  }} -> (result);
}}
'''
        pkg = d / 'pkg'
        mil_path = compile_mil(mil, d, pkg, blobs)
        x = encode_fp16([1.0] * K)
        actual, expected, man = run_sim(pkg, mil_path, d, {'x': x})
        if actual['result'] != expected['result']:
            av, ev = decode_fp16(actual['result']), decode_fp16(expected['result'])
            bad = [(i, ev[i], av[i]) for i in range(len(ev)) if ev[i] != av[i]][:8]
            failures.append(f'matmul-add-relu mismatch {bad}')
        else:
            print(f'OK matmul-add-relu N={N} programs={len(man["programs"])}')

        # Byte-for-byte: for each intermediate consumer, compose from producer physical
        # equals packing of dense slice through convert_tensor
        # Run sim collecting intermediate regions
        manifest, _ = inspect_anec.load_package(pkg)
        # Use linux helper composition after producing via sim path already validated
    return failures


def check_macos_packing_rule():
    """Cite run_h13.mm packing: only logical count lanes, 64-byte stride, zero fill."""
    # Replicate writePhysical/readPhysical
    def write_physical(dense, offset, count, alloc):
        dest = bytearray(alloc)
        src = dense[offset*2:]
        for e in range(count):
            dest[e*64:e*64+2] = src[e*2:e*2+2]
        return dest
    def read_physical(phys, offset, count, dense_len):
        dense = bytearray(dense_len)
        for e in range(count):
            dense[(offset+e)*2:(offset+e)*2+2] = phys[e*64:e*64+2]
        return dense
    dense = encode_fp16(list(range(100)))
    # slice offset 64 count 36 physical would be 64 for elemwise
    phys = write_physical(dense, 64, 36, 64*64)
    # padding lanes 36..63 must be zero
    assert all(phys[i]==0 for i in range(36*64, 64*64))
    back = read_physical(phys, 64, 36, 200)
    assert back[64*2:100*2] == dense[64*2:100*2]
    # agree with inspect_anec.convert_tensor
    binding = {'logicalBytes': 72, 'allocationBytes': 4096}
    p2 = bytes(inspect_anec.convert_tensor(binding, dense[64*2:100*2], True))
    assert phys[:4096] == p2
    print('OK macOS writePhysical/readPhysical matches inspect_anec.convert_tensor (run_h13.mm writePhysical/readPhysical)')
    return []


def norm_source(op, shape, result_shape, axis=None, axes=None, keep_dims=True):
    def kind(dimensions):
        return f"tensor<fp16, [{', '.join(map(str, dimensions))}]>"
    if op == 'softmax':
        declaration = (f'    int32 axis = const()[name = string("axis"), '
                       f'val = int32({axis})];')
        arguments = 'x = x, axis = axis'
    else:
        axes_kind = f'tensor<int32, [{len(axes)}]>'
        declaration = (f'    {axes_kind} axes = const()[name = string("axes"), '
                       f'val = {axes_kind}([{", ".join(map(str, axes))}])];')
        arguments = 'x = x, axes = axes'
        arguments += (', epsilon = fp32(0.00001)' if op == 'layer_norm' else
                      f', keep_dims = bool({"true" if keep_dims else "false"})')
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({kind(shape)} x) {{
{declaration}
    {kind(result_shape)} y = {op}({arguments})[name = string("y")];
  }} -> (y);
}}
'''


def broadcast_source(operation, x_shape, y_shape, constant):
    def kind(dimensions):
        return f"tensor<fp16, [{', '.join(map(str, dimensions))}]>"
    out_shape = [max(pair) for pair in zip(x_shape, y_shape)]
    if constant:
        declaration = (
            f'    {kind(y_shape)} z = const()[name = string("z"), val = '
            f'{kind(y_shape)}(BLOBFILE(path = string("@model_path/bias.bin"), '
            'offset = uint64(64)))];')
        arguments = f'{kind(x_shape)} x'
    else:
        declaration = ''
        arguments = f'{kind(x_shape)} x, {kind(y_shape)} z'
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({arguments}) {{
{declaration}
    {kind(out_shape)} y = {operation}(x = x, y = z)[name = string("y")];
  }} -> (y);
}}
'''


def check_broadcast_packages():
    """Per-channel and spatial broadcasts package as one program, and their
    arithmetic on the packaged surfaces matches h13_reference."""
    failures = []
    cases = [
        ('add', (1, 64, 8, 8), (1, 64, 1, 1), True),
        ('mul', (1, 64, 8, 8), (1, 64, 1, 1), True),
        ('add', (1, 768, 8, 8), (1, 768, 1, 1), True),
        ('add', (1, 64, 8, 8), (1, 64, 1, 1), False),
        ('mul', (1, 64, 8, 8), (1, 1, 8, 8), False),
        ('add', (1, 64, 16, 16), (1, 1, 1, 1), False),
        ('add', (2, 64, 8, 8), (1, 64, 1, 1), False),
        ('mul', (8, 64, 8, 8), (1, 64, 1, 1), False),
    ]
    for operation, x_shape, y_shape, constant in cases:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            label = (f'{operation} {"x".join(map(str, x_shape))} '
                     f'{"const" if constant else "runtime"} '
                     f'{"x".join(map(str, y_shape))}')
            values = [fp16(0.25 * (index % 9) - 1.0)
                      for index in range(math.prod(y_shape))]
            blobs = {'bias.bin': blobfile(encode_fp16(values))} if constant else None
            source = broadcast_source(operation, x_shape, y_shape, constant)
            package = directory / 'pkg'
            mil_path = compile_mil(source, directory, package, blobs)
            manifest = json.loads((package / 'manifest.json').read_text())
            if len(manifest['programs']) != 1:
                failures.append(f'{label}: {len(manifest["programs"])} programs')
                continue
            program = manifest['programs'][0]
            if program['encoder'] != inspect_anec.PARITY_BROADCAST:
                failures.append(f'{label}: encoder {program["encoder"]}')
            if any(item.get('slice') for item in
                   program['inputs'] + program['outputs']):
                failures.append(f'{label}: whole-tensor program was sliced')
            if len(program['inputs']) != (1 if constant else 2):
                failures.append(f'{label}: {len(program["inputs"])} inputs')
            inputs = {'x': encode_fp16([fp16(0.5 + (index % 5) * 0.25)
                                        for index in range(math.prod(x_shape))])}
            if not constant:
                inputs['z'] = encode_fp16(values)
            actual, expected, _ = run_sim(package, mil_path, directory, inputs)
            if actual['y'] != expected['y']:
                got, want = decode_fp16(actual['y']), decode_fp16(expected['y'])
                bad = [(index, want[index], got[index])
                       for index in range(len(want))
                       if want[index] != got[index]][:5]
                failures.append(f'{label}: broadcast mismatch {bad}')
            elif not failures:
                print(f'OK broadcast {label} ({program["taskDescriptors"]} tasks, '
                      f'{len(want if False else expected["y"]) // 2} elements)')
    return failures


def runtime_matmul_source(rows, reduction, columns, transpose_y):
    x_shape = (1, rows, reduction)
    y_shape = (1, columns, reduction) if transpose_y else (1, reduction, columns)
    def kind(dimensions):
        return f"tensor<fp16, [{', '.join(map(str, dimensions))}]>"
    flag = 'true' if transpose_y else 'false'
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({kind(x_shape)} x, {kind(y_shape)} w) {{
    bool tx = const()[name = string("tx"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool({flag})];
    {kind((1, rows, columns))} y = matmul(transpose_x = tx, transpose_y = ty, x = x, y = w)[name = string("y")];
  }} -> (y);
}}
'''


def check_runtime_matmul_packages():
    """Attention's two shapes -- scores and context -- package as one program
    with both operands as runtime surfaces on channels 5 and 6."""
    failures = []
    cases = [(64, 64, 64, False), (64, 64, 64, True), (64, 64, 128, False),
             (128, 128, 128, True)]
    for rows, reduction, columns, transpose_y in cases:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            label = f'M{rows} K{reduction} N{columns} ty{int(transpose_y)}'
            source = runtime_matmul_source(rows, reduction, columns, transpose_y)
            package = directory / 'pkg'
            mil_path = compile_mil(source, directory, package)
            manifest = json.loads((package / 'manifest.json').read_text())
            if len(manifest['programs']) != 1:
                failures.append(f'{label}: {len(manifest["programs"])} programs')
                continue
            program = manifest['programs'][0]
            if program['encoder'] != inspect_anec.PARITY_MATMUL:
                failures.append(f'{label}: encoder {program["encoder"]}')
            channels = [item['index'] for item in program['inputs']]
            names = [item['name'] for item in program['inputs']]
            if channels != [5, 6] or names != ['w', 'x']:
                failures.append(f'{label}: channels {channels} carry {names}, '
                                'expected the second operand on 5 and x on 6')
            if program['constantBytes'] != 16384:
                failures.append(f'{label}: {program["constantBytes"]} constant bytes')
            x = encode_fp16([fp16(0.25 * ((index % 7) - 3))
                             for index in range(rows * reduction)])
            w = encode_fp16([fp16(0.125 * ((index % 5) - 2))
                             for index in range(reduction * columns)])
            actual, expected, _ = run_sim(package, mil_path, directory,
                                          {'x': x, 'w': w})
            if actual['y'] != expected['y']:
                got, want = decode_fp16(actual['y']), decode_fp16(expected['y'])
                bad = [(index, want[index], got[index])
                       for index in range(len(want))
                       if want[index] != got[index]][:5]
                failures.append(f'{label}: runtime matmul mismatch {bad}')
            elif not failures:
                print(f'OK runtime-runtime matmul {label} '
                      f'({program["taskDescriptors"]} tasks)')
    return failures


def check_norm_packages():
    """Softmax, layer_norm, and the reductions package as one whole-tensor
    program per operation.

    Their arithmetic is not simulated: Apple's task streams evaluate the
    exponential and reciprocal through hardware lookup tables that this
    file's fp16 simulators do not model. What is checked is the part the
    runtime depends on — one program, no slicing, the physical surfaces the
    decoded oracle declares, and dense host bytes that survive the pack and
    unpack round trip — plus that h13_reference evaluates the same MIL.
    """
    failures = []
    cases = [
        ('softmax', (1, 512, 1, 1), (1, 512, 1, 1), dict(axis=1),
         [1, 512, 1, 1, 64, 64], 5),
        ('softmax', (1, 1, 1, 128), (1, 1, 1, 128), dict(axis=-1),
         [1, 1, 1, 128, 256, 256], 5),
        ('layer_norm', (1, 512, 1, 1), (1, 512, 1, 1), dict(axes=(1,)),
         [1, 512, 1, 1, 64, 64], 5),
        ('reduce_sum', (1, 64, 8, 8), (1, 64, 1, 1), dict(axes=(2, 3)),
         [1, 64, 1, 1, 64, 64], 1),
        ('reduce_mean', (1, 64, 8, 8), (1, 1, 8, 8), dict(axes=(1,)),
         [1, 1, 8, 8, 512, 64], 2),
        ('reduce_max', (1, 64, 8, 8), (1, 64), dict(axes=(2, 3), keep_dims=False),
         [1, 1, 1, 64, 128, 128], 1),
    ]
    for op, shape, result_shape, options, output_layout, tasks in cases:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mil = norm_source(op, shape, result_shape, **options)
            package = directory / 'pkg'
            mil_path = compile_mil(mil, directory, package)
            manifest = json.loads((package / 'manifest.json').read_text())
            label = f'{op} {"x".join(map(str, shape))}'
            if len(manifest['programs']) != 1:
                failures.append(f'{label}: {len(manifest["programs"])} programs')
                continue
            program = manifest['programs'][0]
            binding = program['inputs'][0]
            output_binding = program['outputs'][0]
            if program['encoder'] != 'apple-parity-norm':
                failures.append(f'{label}: encoder {program["encoder"]}')
            if program['taskDescriptors'] != tasks:
                failures.append(
                    f'{label}: {program["taskDescriptors"]} tasks, expected {tasks}')
            if output_binding['nchw'] != output_layout:
                failures.append(
                    f'{label}: output layout {output_binding["nchw"]}')
            if binding.get('slice') or output_binding.get('slice'):
                failures.append(f'{label}: whole-tensor program was sliced')
            elements = math.prod(shape)
            dense = encode_fp16([fp16(0.5 + index % 7) for index in range(elements)])
            packed = bytes(inspect_anec.convert_tensor(binding, dense, True))
            if len(packed) != binding['allocationBytes']:
                failures.append(f'{label}: packed {len(packed)} bytes')
            if bytes(inspect_anec.convert_tensor(binding, packed, False)) != dense:
                failures.append(f'{label}: input pack/unpack round trip differs')
            expected = evaluate(mil_path.read_text(), directory, {'x': dense})
            if len(expected['y']) != math.prod(result_shape) * 2:
                failures.append(f'{label}: reference produced the wrong size')
            round_trip = bytes(inspect_anec.convert_tensor(
                output_binding,
                bytes(inspect_anec.convert_tensor(output_binding, expected['y'], True)),
                False))
            if round_trip != expected['y']:
                failures.append(f'{label}: output pack/unpack round trip differs')
            if not failures:
                print(f'OK {label} -> {"x".join(map(str, result_shape))} '
                      f'({tasks} task{"s" if tasks > 1 else ""}, reference agrees '
                      f'on {len(expected["y"]) // 2} elements)')
    return failures


def main():
    all_f = []
    all_f += check_macos_packing_rule()
    all_f += check_fused_chains()
    all_f += check_ops()
    all_f += check_reader_vs_linux_intermediate()
    all_f += check_matmul_grid()
    all_f += check_norm_packages()
    all_f += check_broadcast_packages()
    all_f += check_runtime_matmul_packages()
    if all_f:
        print('FAILURES:')
        for f in all_f:
            print(' -', f)
        sys.exit(1)
    print('ALL SIM CHECKS PASSED')

if __name__ == '__main__':
    main()
