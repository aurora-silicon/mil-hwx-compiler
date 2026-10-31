#!/usr/bin/env python3
"""Inspect compiler ANEC packages and pack/unpack fp16 bytes without a device."""

import argparse
import json
import math
import struct
from pathlib import Path
try:
    from .h13_td import split_h13_tasks, split_h14_tasks, decode_task
except ImportError:
    from h13_td import split_h13_tasks, split_h14_tasks, decode_task

HEADER = struct.Struct('<QIIQQII32I192Q')
HEADER_BYTES = 0x1000
TASK_BYTES = 0x274
CONSTANT_OFFSET = 0x280
PARITY_MATVEC = 'apple-parity-matvec'
PARITY_MATMUL = 'apple-parity-matmul'
PARITY_BATCHED_MATMUL = 'apple-parity-batched-matmul'
PARITY_BATCHED_MATVEC = 'apple-parity-batched-matvec'
PARITY_BOOLEAN = 'apple-parity-boolean'
PARITY_BROADCAST = 'apple-parity-broadcast'
TILE_BYTES = 0x4000
PARITY_ORACLE = 'h13-oracle-parity'
CHAIN_ENCODER = 'composed-chain'


def h13_task_registers(task):
    """One decoded H13 task image as {register byte address: word}: ten header
    words, one extra word when header[9] bit 1 is set, then register records
    headed by `((count - 1) << 26) | byte address`."""
    require(len(task) >= 40 and len(task) % 4 == 0, 'invalid H13 task word count')
    words = struct.unpack(f'<{len(task) // 4}I', task)
    index = 10 + (1 if words[9] & 3 == 3 else 0)
    require(index <= len(words), 'missing H13 task header extension')
    registers = {}
    while index < len(words):
        header = words[index]
        count = (header >> 26) + 1
        base = header & 0x03FFFFFF
        require(index + count < len(words), 'truncated H13 register record')
        for offset in range(count):
            registers[base + offset * 4] = words[index + 1 + offset]
        index += 1 + count
    return registers


# Task DMA surface wiring decoded across the H13 oracle corpus: reads run
# through Src1/Src2 (the input channels, or the matvec's L2 staging read),
# writes through Dst to the output surface or stay in L2. Any other DMA
# config in a composed chain would be an unrouted surface.
H13_DMA_DISABLED = 0x00008880
H13_SRC_ENABLE = (0x00033881, 0x00033880, 0x00048880)
H13_DST_ENABLE = (0x040000c1, 0x000000c0)
H13_DMA_SELECTORS = ((0x13800, 0), (0x13804, 6), (0x17800, 12))
H13_ALLOCATED_CHANNELS = (3, 4, 5, 6)


def validate_chain_routing(program, tasks):
    """Validates the actual emitted DMA routing of a fused composed chain."""
    fused = program.get('fused')
    outputs = program.get('outputs')
    require(isinstance(fused, list) and len(fused) == 1 and
            isinstance(outputs, list) and len(outputs) == 1 and
            isinstance(outputs[0], dict) and
            fused == [outputs[0].get('name')],
            'composed chain must name its sole fused output')
    output_tasks = []
    clamp_tasks = []
    for index, task in enumerate(tasks):
        registers = h13_task_registers(task)
        for address, legal in ((0x13800, (H13_DMA_DISABLED,) + H13_SRC_ENABLE),
                               (0x13804, (H13_DMA_DISABLED,) + H13_SRC_ENABLE),
                               (0x17800, (H13_DMA_DISABLED,) + H13_DST_ENABLE)):
            value = registers.get(address)
            if value is not None:
                require(value in legal,
                        'composed chain carries an unrouted DMA surface config')
        if registers.get(0x17800) == 0x040000c1:
            output_tasks.append(index)
        ne = registers.get(0x0c804)
        pe = registers.get(0x08800)
        if (ne is not None and ne & 0x00010000) or \
                (pe is not None and pe & 0x20):
            clamp_tasks.append(index)
    final = len(tasks) - 1
    require(output_tasks == [final],
            'composed chain must write the output surface only from its final task')
    require(clamp_tasks == [final],
            'composed chain relu must apply only to the final accumulation task')


def validate_task_headers(program, tasks, tiles):
    """Validate the request NID and each enabled DMA surface selector."""
    for index, task in enumerate(tasks):
        words = struct.unpack(f'<{len(task) // 4}I', task)
        require((words[0] >> 16) & 0xff == 0x40,
                f'H13 task[{index}] NID is not the Linux request NID')
        registers = h13_task_registers(task)
        for address, shift in H13_DMA_SELECTORS:
            if registers.get(address, H13_DMA_DISABLED) == H13_DMA_DISABLED:
                continue
            channel = (words[8] >> shift) & 0x1f
            if channel == 0:
                continue
            if channel == 1:
                if program.get('encoder') in (PARITY_BATCHED_MATMUL,
                                              PARITY_BATCHED_MATVEC):
                    require(address in (0x13800, 0x13804) and
                            program.get('constantBytes', 0) > 0,
                            f'H13 task[{index}] selects noncanonical channel 1')
                    continue
                # The canonical broadcast channel-1 source is the folded
                # constant read through the 0x17800 destination words.
                destination = registers.get(0x17800, H13_DMA_DISABLED)
                destination_channel = (words[8] >> 12) & 0x1f
                if program.get('encoder') == PARITY_BOOLEAN and \
                        program.get('constantBytes', 0) > 0:
                    # The boolean encoder reads its folded constant/blob
                    # surface on channel 1; runtime operands stay on 4-7.
                    require(address in (0x13800, 0x13804),
                            f'H13 task[{index}] selects noncanonical channel 1')
                    continue
                if program.get('encoder') == PARITY_ORACLE and \
                        address in (0x13800, 0x13804) and \
                        program.get('operation') in ('add', 'mul') and \
                        len(program.get('inputs', ())) == 1 and \
                        program.get('constantBytes', 0) > 0:
                    # The oracle constant-blob twins (the decoded exported
                    # fixtures): the second source reads the operation's
                    # constant straight from the constant section.
                    continue
                require(address in (0x13800, 0x13804) and
                        program.get('encoder') == PARITY_BROADCAST and
                        len(program.get('inputs', ())) == 1 and
                        program.get('constantBytes', 0) > 0 and
                        destination == 0x000000c0 and destination_channel == 0,
                        f'H13 task[{index}] selects noncanonical channel 1')
                continue
            # Channel 3 is the scratch surface: selectable only when the
            # program backs it (the boolean multi-task forms stage through
            # it; every other encoder leaves tiles[3] at zero).
            allocated = set(H13_ALLOCATED_CHANNELS if program.get('encoder') !=
                            PARITY_BOOLEAN else {4, 5, 6, 7}) | \
                ({3} if tiles[3] > 0 else set())
            require(channel in allocated,
                    f'H13 task[{index}] selects noncanonical channel {channel}')
            require(tiles[channel] > 0,
                    f'H13 task[{index}] selects unallocated channel {channel}')


PROGRAM_FIELDS = (
    'file', 'bytes', 'taskDescriptors', 'encoder', 'operation', 'inputs',
    'constantInputs', 'outputs', 'constantOffset', 'constantBytes', 'scratchBytes')


def surface_layout(shape, bool_elements=False):
    """The physical layout H13 lays an NCHW surface out as: rows padded up
    to a 64-byte stride over the element size (two bytes for fp16, one for
    the bool compare/cond surfaces), one plane per channel, and the batch
    only in the shape.
    """
    batch, channels, height, width = shape
    element = 1 if bool_elements else 2
    row = max(64, (width * element + 63) // 64 * 64)
    return [batch, channels, height, width, row * height, row]


def check_layout(binding):
    """Validates one binding's declared physical layout and returns the
    element count that layout holds."""
    nchw = binding.get('nchw') if isinstance(binding, dict) else None
    require(isinstance(nchw, list) and len(nchw) == 6 and
            all(type(value) is int and value > 0 for value in nchw) and
            nchw == surface_layout(nchw[:4],
                                   binding.get('dtype') == 'bool'),
            'incorrect physical layout')
    return math.prod(nchw[:4])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_file(directory, name, limit):
    require(isinstance(name, str) and name not in ('', '.', '..') and
            Path(name).name == name, 'package filename must be a basename')
    require(type(limit) is int and 0 <= limit, 'package file limit must be nonnegative')
    directory = Path(directory).resolve()
    path = (directory / name).resolve()
    require(path.parent == directory and path.is_file(),
            'package file must be a regular file within the package')
    require(path.stat().st_size <= limit, 'package file exceeds format limits')
    return path.read_bytes()


def check_tensor(name, tensor):
    require(isinstance(name, str) and name, 'tensor needs a non-empty name')
    fields = set(tensor) if isinstance(tensor, dict) else set()
    require(fields in ({'shape', 'logicalBytes', 'role'},
                       {'shape', 'logicalBytes', 'role', 'aliasOf'},
                       {'shape', 'logicalBytes', 'role', 'accumulation'},
                       {'shape', 'logicalBytes', 'role', 'dtype'},
                       {'shape', 'logicalBytes', 'role', 'aliasOf', 'dtype'}),
            'tensor record has incorrect fields')
    require('accumulation' not in tensor or
            tensor['accumulation'] == 'chunked-fp16',
            'unsupported tensor accumulation')
    shape = tensor.get('shape')
    require(isinstance(shape, list) and shape and
            all(type(n) is int and n > 0 for n in shape),
            'tensor shape must have positive static dimensions')
    require(type(tensor.get('logicalBytes')) is int and
            tensor['logicalBytes'] == math.prod(shape) *
            (1 if tensor.get('dtype') == 'bool' else 2),
            'tensor logical byte count does not match its shape')
    require(tensor.get('role') in ('input', 'output', 'intermediate', 'constant'),
            'unsupported tensor role')
    require('aliasOf' not in tensor or
            isinstance(tensor['aliasOf'], str) and tensor['aliasOf'],
            'tensor aliasOf must be a non-empty name')


def binding_interval(binding, tensors):
    slice_record = binding.get('slice')
    if slice_record is None:
        tensor_name = binding['name']
        require(tensor_name in tensors, 'binding references an unknown tensor')
        tensor = tensors[tensor_name]
        require('aliasOf' not in tensor,
                'bindings must reference an underlying tensor')
        require(binding['shape'] == tensor['shape'] and
                binding['logicalBytes'] == tensor['logicalBytes'],
                'whole-tensor binding differs from its tensor')
        count = tensor['logicalBytes'] // (1 if tensor.get('dtype') == 'bool' else 2)
        return tensor_name, 0, count, count
    fields = set(slice_record) if isinstance(slice_record, dict) else set()
    require(fields == {'tensor', 'elementOffset', 'elementCount',
                       'physicalElements'}, 'slice has incorrect fields')
    tensor_name = slice_record.get('tensor')
    offset, count = slice_record.get('elementOffset'), slice_record.get('elementCount')
    physical = slice_record.get('physicalElements')
    require(tensor_name == binding['name'] and tensor_name in tensors,
            'slice references an unknown or mismatched tensor')
    require('aliasOf' not in tensors[tensor_name],
            'bindings must reference an underlying tensor')
    require(type(offset) is int and type(count) is int and
            type(physical) is int and offset >= 0 and count > 0 and physical >= count,
            'slice offsets, counts, and physical elements must be valid integers')
    element = 1 if tensors[tensor_name].get('dtype') == 'bool' else 2
    elements = tensors[tensor_name]['logicalBytes'] // element
    require(count == binding['logicalBytes'] // element and offset <= elements - count,
            'slice exceeds its tensor or differs from its binding')
    return tensor_name, offset, count, physical


def require_covering(bindings, tensors, message, exact=True):
    """Every logical element of the bound tensor must be covered: with
    `exact` the bindings tile it once in order, otherwise their union covers
    it with no hole and no overrun."""
    intervals = sorted(binding_interval(item, tensors)[1:3] for _, item in bindings)
    cursor = 0
    for offset, count in intervals:
        require(offset == cursor if exact else offset <= cursor, message)
        cursor = cursor + count if exact else max(cursor, offset + count)
    first = tensors[binding_interval(bindings[0][1], tensors)[0]]
    require(bindings and
            cursor == first['logicalBytes'] //
            (1 if first.get('dtype') == 'bool' else 2),
            message)


def logical_elements(binding):
    shape = binding.get('shape', [])
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def check_binding(binding, index, tiles, layouts, tensors):
    require(isinstance(binding, dict), 'binding must be an object')
    require(isinstance(binding.get('name'), str) and binding['name'],
            'binding needs a non-empty name')
    require(binding.get('dtype') in ('float16', 'bool'), 'binding must use float16')
    physical_elements = check_layout(binding)
    layout = binding['nchw']
    shape = binding.get('shape')
    require(isinstance(shape, list) and shape and
            all(type(n) is int and 0 < n <= physical_elements for n in shape),
            'logical shape must have positive static dimensions within its layout')
    logical_elements = math.prod(shape)
    require(type(binding.get('index')) is int and binding['index'] == index,
            'binding index does not match ANEC channel')
    require(list(layouts[index * 6:(index + 1) * 6]) == layout,
            'manifest layout differs from ANEC header')
    require(type(binding.get('logicalBytes')) is int and
            binding['logicalBytes'] == logical_elements *
            (1 if binding.get('dtype') == 'bool' else 2),
            'incorrect logical byte count')
    span = layout[0] * layout[1] * layout[4]
    allocation = -(-span // TILE_BYTES) * TILE_BYTES
    require(type(binding.get('allocationBytes')) is int and
            binding['allocationBytes'] == allocation, 'incorrect allocation size')
    require(tiles[index] * TILE_BYTES == allocation,
            'allocation differs from ANEC header')
    require(binding.get('role') in (None, 'intermediate'),
            'unsupported binding role')
    _, _, count, physical = binding_interval(binding, tensors)
    require(count == logical_elements and physical == physical_elements,
            'slice physical elements differ from its binding layout')
    return physical_elements


def validate_program(directory, program, tensors, target="H13", program_data=None):
    h14 = target == "H14"
    require(isinstance(program, dict), 'program must be an object')
    operation = program.get('operation')
    require(operation in ('chain', 'add', 'mul', 'maximum', 'minimum', 'sub',
                          'real_div',
                          'matmul', 'abs', 'exp', 'gelu', 'leaky_relu', 'relu',
                          'rsqrt', 'sigmoid', 'silu', 'sqrt', 'tanh',
                          'softmax', 'layer_norm', 'reduce_sum', 'reduce_max',
                          'reduce_mean', 'conv',
                          'less', 'floor', 'select', 'floor_div',
                          'tile'),
            'unsupported operation')
    inputs, outputs = program.get('inputs'), program.get('outputs')
    require(isinstance(inputs, list) and isinstance(outputs, list),
            'input and output bindings must be arrays')
    constant_inputs = program.get('constantInputs')
    require(isinstance(constant_inputs, dict), 'constantInputs must be an object')
    matmul = operation == 'matmul'
    require((not inputs and operation in ('floor',)) or
            1 <= len(inputs) <= (3 if operation == 'select' else 2),
            'incorrect input/output count')
    require(len(outputs) == 1, 'incorrect output count')
    for key in ('bytes', 'constantBytes', 'constantOffset', 'taskDescriptors'):
        require(type(program.get(key)) is int and program[key] > 0 or
                (key == 'constantBytes' and program.get(key) == 0),
                f'incorrect {key}')
    constants, offset = program['constantBytes'], program['constantOffset']
    require(program['bytes'] == HEADER_BYTES + offset + constants,
            'incorrect bytes')
    data = local_file(directory, program.get('file'), program['bytes'])
    if program_data is not None:
        program_data[program['file']] = data
    require(len(data) == program['bytes'], 'ANEC file length differs from manifest')
    fields = HEADER.unpack_from(data)
    size, td_size, td_count, task_size, kernel_size, source_count, dest_count = fields[:7]
    require((size, td_count, kernel_size, source_count, dest_count) ==
            (offset + constants, program['taskDescriptors'], constants,
             len(inputs), len(outputs)), 'ANEC header differs from manifest')
    require(td_size % 4 == 0 and 0 < td_size <= task_size <= offset and
            offset == (task_size + 0x3f) & ~0x3f,
            'ANEC task stream does not fit its constant offset')
    require(not any(data[HEADER.size:HEADER_BYTES]) and
            not any(data[HEADER_BYTES + task_size:HEADER_BYTES + offset]),
            'nonzero reserved padding')
    if h14:
        # Initial H14 graph support uses per-operation elementwise programs.
        # Other encoders require their own shape/routing validation.
        require(program.get('encoder') == 'h14-oracle-parity' and
                operation in ('add', 'mul', 'maximum', 'minimum', 'sub',
                              'real_div', 'relu', 'tanh', 'sigmoid', 'silu',
                              'gelu', 'sqrt', 'rsqrt', 'exp', 'abs', 'leaky_relu'),
                'H14 runner supports per-operation elementwise programs only')
        require(program.get('firstTaskBytes') == td_size,
                'H14 firstTaskBytes differs from ANEC header')
        tasks = split_h14_tasks(data[HEADER_BYTES:HEADER_BYTES + task_size])
        require(tasks and len(tasks[0]) == td_size,
                'H14 first task size differs from ANEC header')
        for task in tasks:
            decoded = decode_task(task, 'h14')
            require(not any(block['name'].startswith('unknown_')
                            for block in decoded['blocks'].values()),
                    'H14 register outside source-backed blocks')
        require(program.get('scratchBytes', 0) == 0,
                'H14 elementwise programs cannot allocate scratch')
        require(all(item.get('dtype') == 'float16' and 'slice' not in item
                    for item in inputs + outputs),
                'H14 runner requires whole fp16 surfaces')
    else:
        tasks = split_h13_tasks(data[HEADER_BYTES:HEADER_BYTES + task_size],
                                td_size // 4 - 1, td_count)
    require(len(tasks) == program['taskDescriptors'],
            'task list differs from ANEC task stream')
    if operation == 'chain':
        require(program.get('encoder') == CHAIN_ENCODER,
                'chain program must use the composed encoder')
        validate_chain_routing(program, tasks)
    tiles, layouts = fields[7:39], fields[39:]
    require(tiles[0] == (size + TILE_BYTES - 1) // TILE_BYTES,
            'incorrect command allocation')
    scratch = program.get('scratchBytes', 0) if h14 else program.get('scratchBytes')
    require((h14 or 'scratchBytes' in program) and type(scratch) is int and
            0 <= scratch <= 0xffffffff * TILE_BYTES,
            'invalid scratchBytes allocation size')
    require(tiles[3] == (scratch + TILE_BYTES - 1) // TILE_BYTES,
            'scratch allocation differs from ANEC header')
    # Most encoders bind the result on 4 with operands on 5..7; the
    # blob-x boolean twin binds its result on 5 and no operand surface.
    require(outputs[0].get('index') in (4, 5),
            'output binding selects an unallocated channel')
    require([item.get('index') for item in inputs] ==
            list(range(5, 5 + len(inputs))),
            'input bindings must fill channels 5.. in order')
    occupied = {outputs[0]['index'],
                *range(5, 5 + len(inputs))}
    for index in range(32):
        if index not in occupied:
            require(not any(layouts[index * 6:(index + 1) * 6]),
                    'unexpected tensor channel')
            if index not in (0, 3):
                require(tiles[index] == 0, 'unexpected channel allocation')
    if not h14:
        validate_task_headers(program, tasks, tiles)
    encoder = program.get('encoder')
    require(isinstance(encoder, str) and encoder, 'program needs an encoder name')
    for index, item in enumerate(inputs, start=5):
        check_binding(item, index, tiles, layouts, tensors)
    check_binding(outputs[0], outputs[0]['index'], tiles, layouts, tensors)
    if matmul and encoder not in (PARITY_MATVEC, PARITY_MATMUL,
                                  PARITY_BATCHED_MATMUL, PARITY_BATCHED_MATVEC):
        # The chunked encoder pads every surface to its 256- or 512-lane tile.
        require(binding_interval(inputs[0], tensors)[3] in (256, 512) and
                binding_interval(outputs[0], tensors)[3] == 512,
                'unsupported matvec physical size')
    if encoder in (PARITY_MATMUL, PARITY_BATCHED_MATMUL):
        require(matmul and len(inputs) == 2,
                'a runtime-operand matmul binds both operands as inputs')
    if encoder in (PARITY_BATCHED_MATMUL, PARITY_BATCHED_MATVEC):
        require(program['taskDescriptors'] % 26 == 0 and
                program['taskDescriptors'] >= 52,
                'a batched matmul carries 26 tasks per batch')
    if operation == 'tile':
        # A materialized tile runs one task over its whole surfaces: one
        # runtime input, whole-tensor bindings, and the output replicating
        # the input by an integer factor.
        require(len(inputs) == 1, 'a tile program binds one runtime input')
        require(all('slice' not in item for item in inputs + outputs),
                'a tile program covers its whole surfaces')
        require(math.prod(outputs[0]['shape']) >= math.prod(inputs[0]['shape']) and
                math.prod(outputs[0]['shape']) % math.prod(inputs[0]['shape']) == 0,
                'a tile output replicates its input')
    if operation not in ('matmul', 'chain', 'tile') and not inputs:
        # Only the blob-x floor twin folds its whole operand: the operand
        # rides the constant section, so there is no runtime shape to check.
        require(operation == 'floor' and encoder == PARITY_BOOLEAN and
                program.get('constantBytes', 0) > 0,
                'a program with no inputs must be the blob-x floor twin')
    if operation not in ('matmul', 'chain', 'tile') and inputs:
        output_is_returned_alias = any(
            tensor.get('aliasOf') == outputs[0]['name'] and
            tensor['role'] == 'output' and tensor['shape'] == outputs[0]['shape']
            for tensor in tensors.values())
        shapes = [item['shape'] for item in inputs]
        require(len({len(shape) for shape in shapes}) == 1,
                'elementwise operands must share a rank')
        # Identical shapes are the degenerate broadcast; any other operand
        # axis must be 1 or the result's extent.
        broadcast = [max(extents) for extents in zip(*shapes)]
        require(all(extent in (1, result) for shape in shapes
                    for extent, result in zip(shape, broadcast)),
                'elementwise operands must broadcast against each other')
        require(outputs[0]['shape'] == broadcast or
                (output_is_returned_alias and
                 math.prod(outputs[0]['shape']) == math.prod(broadcast)),
                'elementwise operation shapes must match')
    require(outputs[0]['name'] not in {item['name'] for item in inputs},
            'program output must differ from its inputs')
    marked_constants = {item['name'] for item in inputs
                        if item.get('binding') == 'constant'}
    require(all(item.get('binding') in (None, 'constant') for item in inputs),
            'unsupported input binding kind')
    require(not any(item.get('binding') for item in outputs),
            'output bindings cannot be constants')
    require(set(constant_inputs) == marked_constants,
            'constantInputs must match constant bindings')
    for item in inputs:
        if item['name'] not in marked_constants:
            continue
        encoded = constant_inputs[item['name']]
        physical_elements = binding_interval(item, tensors)[3]
        require(isinstance(encoded, str) and len(encoded) == physical_elements * 4 and
                all(character in '0123456789abcdefABCDEF' for character in encoded),
                'constant input must be physical hexadecimal fp16 bytes')
        payload = bytes.fromhex(encoded)
        require(not any(payload[item['logicalBytes']:]),
                'constant input padding must be zero')
    command_bytes = ((program['bytes'] - HEADER_BYTES + TILE_BYTES - 1)
                     // TILE_BYTES) * TILE_BYTES
    input_bytes = sum(item['allocationBytes'] for item in inputs)
    output_bytes = sum(item['allocationBytes'] for item in outputs)
    scratch_allocation = tiles[3] * TILE_BYTES
    return {
        'file': program['file'], 'commandAndConstantsBytes': command_bytes,
        'inputBytes': input_bytes, 'outputBytes': output_bytes,
        'scratchBytes': scratch_allocation,
        'totalBytes': command_bytes + input_bytes + output_bytes + scratch_allocation,
    }


def check_result_metadata(manifest, tensors, produced_outputs):
    physical_outputs = manifest.get('physicalOutputs')
    logical_results = manifest.get('logicalResults')
    require(isinstance(physical_outputs, list) and physical_outputs,
            'physicalOutputs must be a non-empty array')
    require(isinstance(logical_results, list) and logical_results,
            'logicalResults must be a non-empty ordered array')

    physical_by_name = {}
    for output in physical_outputs:
        fields = set(output) if isinstance(output, dict) else set()
        require(fields == {'tensor', 'dtype', 'shape', 'logicalBytes'},
                'physical output has incorrect fields')
        name = output.get('tensor')
        shape = output.get('shape')
        require(isinstance(name, str) and name and name not in physical_by_name,
                'physical output tensor names must be unique and non-empty')
        require(output.get('dtype') in ('float16', 'bool'),
                'physical output must use float16 or bool')
        require(isinstance(shape, list) and shape and
                all(type(n) is int and n > 0 for n in shape),
                'physical output shape must have positive static dimensions')
        element = 1 if output.get('dtype') == 'bool' else 2
        require(type(output.get('logicalBytes')) is int and
                output['logicalBytes'] == math.prod(shape) * element,
                'physical output byte count does not match its shape')
        require(name in tensors and 'aliasOf' not in tensors[name] and
                tensors[name]['role'] == 'output' and
                tensors[name]['shape'] == shape and
                tensors[name]['logicalBytes'] == output['logicalBytes'],
                'physical output differs from its produced tensor')
        physical_by_name[name] = output

    referenced = set()
    for result in logical_results:
        fields = set(result) if isinstance(result, dict) else set()
        require(fields == {'name', 'dtype', 'shape', 'physical', 'conversion'},
                'logical result has incorrect fields')
        require(isinstance(result.get('name'), str) and result['name'],
                'logical result needs a non-empty name')
        shape = result.get('shape')
        require(isinstance(shape, list) and shape and
                all(type(n) is int and n > 0 for n in shape),
                'logical result shape must have positive static dimensions')
        require(result.get('conversion') == 'identity',
                'unsupported logical result conversion')
        physical = result.get('physical')
        physical_fields = set(physical) if isinstance(physical, dict) else set()
        require(physical_fields == {'tensor', 'elementOffset', 'elementCount'},
                'logical result physical mapping has incorrect fields')
        tensor_name = physical.get('tensor')
        require(tensor_name in physical_by_name,
                'logical result references an unknown physical output')
        require(result.get('dtype') == physical_by_name[tensor_name]['dtype'],
                'identity result dtype differs from physical output dtype')
        offset = physical.get('elementOffset')
        count = physical.get('elementCount')
        storage_elements = math.prod(physical_by_name[tensor_name]['shape'])
        require(type(offset) is int and type(count) is int and offset >= 0 and
                count == math.prod(shape) and offset + count <= storage_elements,
                'logical result mapping exceeds physical output storage')
        referenced.add(tensor_name)

    require(referenced == set(physical_by_name) == set(produced_outputs),
            'physicalOutputs must exactly match produced and referenced outputs')


def load_package(directory, *, allow_h14=False, program_data=None):
    directory = Path(directory).resolve()
    manifest = json.loads(local_file(directory, 'manifest.json', 64 << 20))
    require(isinstance(manifest, dict), 'manifest must be an object')
    target = manifest.get('target')
    h14 = allow_h14 and target == 'H14'
    schema = 'mil-hwxc.h14-anec-package.v1' if h14 else 'mil-hwxc.h13-anec-package.v2'
    require(manifest.get('schema') == schema, 'unsupported package schema')
    require((target == 'H13' or h14) and manifest.get('artifactFormat') == 'anec',
            'package must target supported H13/H14 ANEC' if allow_h14 else
            'package must target H13 ANEC')
    if h14:
        require('schedule' not in manifest, 'H14 chained programs are not supported')
        require('physicalOutputs' not in manifest and 'logicalResults' not in manifest,
                'H14 v1 does not carry H13 v2 result metadata')
    programs = manifest.get('programs')
    dispatch = manifest.get('dispatchPlan')
    intermediates = manifest.get('intermediates')
    tensors = manifest.get('tensors')
    require(isinstance(programs, list) and programs, 'programs must be a non-empty array')
    require(isinstance(dispatch, list) and
            all(type(index) is int for index in dispatch) and
            sorted(dispatch) == list(range(len(programs))),
            'dispatchPlan must contain every program index exactly once')
    require(isinstance(intermediates, list) and
            all(isinstance(name, str) and name for name in intermediates) and
            len(set(intermediates)) == len(intermediates),
            'intermediates must contain unique non-empty names')
    require(isinstance(tensors, dict) and tensors, 'tensors must be a non-empty object')
    for name, tensor in tensors.items():
        check_tensor(name, tensor)
    alias_names = {name for name, tensor in tensors.items() if 'aliasOf' in tensor}
    for name in alias_names:
        target = tensors[name]['aliasOf']
        require(target != name and target in tensors and 'aliasOf' not in tensors[target],
                'tensor aliasOf must name an underlying tensor')
        require(tensors[name]['logicalBytes'] == tensors[target]['logicalBytes'],
                'tensor alias element count differs from its underlying tensor')
    intermediate_set = {name for name, tensor in tensors.items()
                        if tensor['role'] == 'intermediate'}
    require(set(intermediates) == intermediate_set,
            'intermediates must match tensors with intermediate role')
    files = [program.get('file') for program in programs if isinstance(program, dict)]
    require(len(files) == len(programs) and len(set(files)) == len(files),
            'program files must be unique')
    if len(programs) == 1:
        for key in (*PROGRAM_FIELDS, *(['firstTaskBytes'] if h14 else [])):
            require(manifest.get(key) == programs[0].get(key),
                    f'legacy top-level {key} must match the sole program')
    else:
        require(not any(key in manifest for key in (*PROGRAM_FIELDS, *(['firstTaskBytes'] if h14 else []))),
                'multi-program packages must omit legacy top-level program fields')

    chain = manifest.get('schedule') == 'chain'
    if chain:
        require(len(programs) == 1 and dispatch == [0] and
                programs[0].get('operation') == 'chain',
                'chain schedule must contain one dispatched chain program')
        tasks = manifest.get('tasks')
        require(isinstance(tasks, list) and len(tasks) ==
                programs[0]['taskDescriptors'],
                'chain tasks must name every task descriptor')
        for index, task in enumerate(tasks):
            require(isinstance(task, dict) and set(task) == {'index', 'operation'} and
                    task['index'] == index and isinstance(task['operation'], str) and
                    task['operation'] not in ('', 'chain', 'const'),
                    'chain task record is invalid')
        scratch = manifest.get('scratch')
        regions = scratch.get('regions') if isinstance(scratch, dict) else None
        scratch_bytes = scratch.get('bytes') if isinstance(scratch, dict) else None
        storage_names = {name for name in intermediate_set if name not in alias_names}
        require(set(scratch or {}) == {'bytes', 'regions'} and
                type(scratch_bytes) is int and scratch_bytes >= 0 and
                isinstance(regions, dict) and set(regions) == storage_names,
                'chain scratch map must cover every storage intermediate')
        for name, region in regions.items():
            require(isinstance(region, dict) and set(region) == {'offset', 'bytes'} and
                    type(region['offset']) is int and region['offset'] >= 0 and
                    region['bytes'] == tensors[name]['logicalBytes'] and
                    region['offset'] + region['bytes'] <= scratch_bytes,
                    'chain scratch region exceeds its allocation')
    else:
        require('schedule' not in manifest and 'tasks' not in manifest and
                'scratch' not in manifest,
                'per-operation packages must omit chain metadata')

    allocations = [validate_program(directory, program, tensors, target, program_data) for program in programs]
    storage_intermediates = [] if chain else [
        name for name in intermediates if name not in alias_names]
    storage_outputs = [name for name, tensor in tensors.items()
                       if tensor['role'] == 'output' and name not in alias_names]
    produced_storage = storage_intermediates + storage_outputs
    producers = {name: [] for name in produced_storage}
    consumers = {name: [] for name in produced_storage}
    output_tensors = {name: producers[name] for name in storage_outputs}
    referenced = set()
    for program_index, program in enumerate(programs):
        for item in program['outputs']:
            name, _, _, _ = binding_interval(item, tensors)
            referenced.add(name)
            role = tensors[name]['role']
            if role == 'intermediate':
                require(item.get('role') == 'intermediate',
                        'intermediate output must have intermediate role')
            else:
                require(role == 'output' and item.get('role') is None,
                        'non-intermediate output must have output tensor role')
            producers[name].append((program_index, item))
        for item in program['inputs']:
            name, _, _, _ = binding_interval(item, tensors)
            referenced.add(name)
            role = tensors[name]['role']
            if role == 'intermediate':
                require(item.get('role') == 'intermediate' and
                        item.get('binding') is None,
                        'intermediate input must have intermediate role')
                consumers[name].append((program_index, item))
            elif role == 'output':
                require(item.get('role') is None and item.get('binding') is None,
                        'returned output input must have no binding role')
                consumers[name].append((program_index, item))
            elif role == 'constant':
                require(item.get('role') is None and item.get('binding') == 'constant',
                        'constant tensor input must have constant binding')
            else:
                require(role == 'input' and item.get('role') is None and
                        item.get('binding') is None,
                        'runtime input must have input tensor role')
    require(all(name in referenced or tensor['role'] == 'constant' or
                name in alias_names or chain and tensor['role'] == 'intermediate'
                for name, tensor in tensors.items()),
            'only chain scratch, constants, and aliases may be unreferenced')
    positions = {program_index: position for position, program_index in enumerate(dispatch)}
    for name in produced_storage:
        require_covering(producers[name], tensors,
                         'produced tensor slices must exactly tile the tensor')
        if not consumers[name]:
            continue
        require_covering(consumers[name], tensors,
                         'produced tensor consumer slices must cover the tensor',
                         exact=False)
        produced_ranges = sorted(
            (offset, offset + physical, program_index)
            for program_index, item in producers[name]
            for _, offset, _, physical in [binding_interval(item, tensors)])
        previous_end = 0
        for start, end, _ in produced_ranges:
            require(start >= previous_end,
                    'produced tensor physical writes overlap')
            previous_end = end
        for consumer_index, consumed in consumers[name]:
            _, consumed_offset, _, consumed_physical = binding_interval(consumed, tensors)
            consumed_end = consumed_offset + consumed_physical
            cursor = consumed_offset
            for produced_offset, produced_end, producer_index in produced_ranges:
                if produced_end <= cursor:
                    continue
                if produced_offset > cursor:
                    break
                if produced_offset < consumed_end and consumed_offset < produced_end:
                    require(positions[producer_index] < positions[consumer_index],
                            'dispatchPlan violates a produced tensor dependency')
                    cursor = max(cursor, produced_end)
                if cursor >= consumed_end:
                    break
            # Programs read in 64-element lanes, so a consumer may reach
            # the end of the lane containing the last covered element —
            # the readable surface padding — but never beyond it.
            require(cursor >= consumed_end or
                    consumed_end <= (cursor + 63) // 64 * 64,
                    'produced tensor consumer range exceeds producer writes')
    if not h14:
        check_result_metadata(manifest, tensors, output_tensors)
    return manifest, allocations


def find_bindings(manifest, name, direction, constants=False):
    matches = []
    for program_index, program in enumerate(manifest['programs']):
        for item in program[direction]:
            tensor_name, _, _, _ = binding_interval(item, manifest['tensors'])
            if tensor_name != name:
                continue
            if constants != (item.get('binding') == 'constant'):
                continue
            matches.append((program_index, program, item))
    require(matches, 'no such constant input' if constants else
            ('no such input' if direction == 'inputs' else 'no such output'))
    return matches


def convert_tensor(binding, data, pack):
    """Packs dense fp16 into, or reads it back from, the physical surface.

    The elementwise surface holds one element per 64-byte lane; the parity
    matvec surface holds dense rows, so the row stride comes from the binding.
    """
    logical, physical = binding['logicalBytes'], binding['allocationBytes']
    require(len(data) == (logical if pack else physical), 'incorrect tensor byte count')
    nchw = binding.get('nchw')
    width, row = (nchw[3], nchw[5]) if nchw else (1, 64)
    result = bytearray(physical if pack else logical)
    for element in range(logical // 2):
        dense = element * 2
        offset = (element // width) * row + (element % width) * 2
        if pack:
            result[offset:offset + 2] = data[dense:dense + 2]
        else:
            result[dense:dense + 2] = data[offset:offset + 2]
    return result


def dense_slice(data, binding, tensors):
    name, offset, count, _ = binding_interval(binding, tensors)
    require(len(data) == tensors[name]['logicalBytes'], 'incorrect dense tensor byte count')
    return data[offset * 2:(offset + count) * 2]


def binding_buffer_name(program_index, binding):
    name = binding['name']
    require(Path(name).name == name and name not in ('.', '..'),
            'binding name cannot form a safe buffer filename')
    return f'program-{program_index}.{name}.buffer'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--pack-input', nargs=2, metavar=('NAME', 'RAW_FP16'))
    mode.add_argument('--pack-constant', metavar='NAME')
    mode.add_argument('--unpack-output', nargs=2, metavar=('NAME', 'PHYSICAL_BUFFER'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    conversion = args.pack_input or args.pack_constant or args.unpack_output
    if bool(conversion) != bool(args.output):
        parser.error('a conversion and --output must be supplied together')
    try:
        manifest, allocations = load_package(args.package)
        if conversion:
            if args.pack_constant:
                name = args.pack_constant
                matches = find_bindings(manifest, name, 'inputs', constants=True)
                encoded = {program['constantInputs'][item['name']]
                           for _, program, item in matches}
                require(len(encoded) == 1,
                        'constant name has inconsistent payloads across programs')
                source = bytes.fromhex(encoded.pop())
                item = matches[0][2]
                data = convert_tensor(item, source[:item['logicalBytes']], True)
                with args.output.open('xb') as destination:
                    destination.write(data)
                written = len(data)
            elif args.pack_input:
                name, source_path = args.pack_input
                matches = find_bindings(manifest, name, 'inputs')
                source = Path(source_path).read_bytes()
                buffers = [(program_index, item,
                            convert_tensor(item, dense_slice(source, item, manifest['tensors']), True))
                           for program_index, _, item in matches]
                if len(buffers) == 1:
                    with args.output.open('xb') as destination:
                        destination.write(buffers[0][2])
                else:
                    args.output.mkdir()
                    for program_index, item, data in buffers:
                        (args.output / binding_buffer_name(program_index, item)).write_bytes(data)
                written = sum(len(data) for _, _, data in buffers)
            else:
                name, source_path = args.unpack_output
                matches = find_bindings(manifest, name, 'outputs')
                if len(matches) == 1:
                    item = matches[0][2]
                    data = convert_tensor(item, Path(source_path).read_bytes(), False)
                else:
                    require_covering([(program_index, item) for program_index, _, item in matches],
                                     manifest['tensors'],
                                     'output slices must exactly tile the tensor')
                    data = bytearray(manifest['tensors'][name]['logicalBytes'])
                    source_directory = Path(source_path).resolve()
                    require(source_directory.is_dir(),
                            'multi-binding output source must be a directory')
                    for program_index, _, item in matches:
                        physical = local_file(source_directory,
                                              binding_buffer_name(program_index, item),
                                              item['allocationBytes'])
                        dense = convert_tensor(item, physical, False)
                        _, offset, count, _ = binding_interval(item, manifest['tensors'])
                        data[offset * 2:(offset + count) * 2] = dense
                with args.output.open('xb') as destination:
                    destination.write(data)
                written = len(data)
            print(f'wrote {written} bytes to {args.output}; no device execution')
        else:
            command_bytes = sum(item['commandAndConstantsBytes'] for item in allocations)
            scratch_bytes = sum(item['scratchBytes'] for item in allocations)
            input_bindings = {}
            output_bindings = {}
            for program in manifest['programs']:
                for item in program['inputs']:
                    name, offset, count, physical = binding_interval(
                        item, manifest['tensors'])
                    if manifest['tensors'][name]['role'] != 'intermediate':
                        input_bindings.setdefault((name, offset, count, physical), item)
                for item in program['outputs']:
                    name, offset, count, physical = binding_interval(
                        item, manifest['tensors'])
                    output_bindings.setdefault((name, offset, count, physical), item)
            input_bytes = sum(item['allocationBytes'] for item in input_bindings.values())
            output_bytes = sum(item['allocationBytes'] for item in output_bindings.values())
            print(json.dumps({
                'validation': 'container, task routing, tensor, binding, slice, and dispatch consistency only',
                'manifest': manifest,
                'bufferAllocation': {
                    'programs': allocations,
                    'commandAndConstantsBytes': command_bytes,
                    'inputBytes': input_bytes, 'outputBytes': output_bytes,
                    'scratchBytes': scratch_bytes,
                    'totalBytes': command_bytes + input_bytes + output_bytes + scratch_bytes,
                    'scope': 'encoded buffers only; shared tensor slices counted once; excludes driver and runtime overhead',
                },
            }, indent=2, sort_keys=True))
    except (OSError, ValueError, struct.error) as error:
        parser.exit(1, f'ANEC package error: {error}\n')


if __name__ == '__main__':
    main()
