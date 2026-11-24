# SPDX-License-Identifier: MIT
"""Exact H14 elementwise contracts from the compiler's existing template source.

No DMA formulas are guessed here. Reuse the generator's reconstruction and
constant hash checks; cache once during preparation, never per graph call.
"""
from functools import lru_cache
import struct

try:
    from . import generate_h14_templates as templates
except ImportError:
    import generate_h14_templates as templates


@lru_cache(maxsize=1)
def elementwise_contracts():
    contracts = {}
    for oracle in templates.selected_oracles('h14'):
        key = templates.classify(oracle)
        stream, runs, size, *_ = templates.template_fields(oracle)
        constants = bytearray(size)
        for start, bits, count in runs:
            for index in range(start, start + count):
                struct.pack_into('<H', constants, index * 2, bits)
        value = (struct.pack(f'<{len(stream)}I', *stream), bytes(constants))
        if contracts.setdefault(key, value) != value:
            raise ValueError(f'inconsistent H14 source template: {key}')
    return contracts


def validate_elementwise(program, task_stream, constants):
    inputs, output = program['inputs'], program['outputs'][0]
    operation = program['operation']
    shape = output['nchw'][:4]
    if (shape[0] != 1 or any(item['nchw'][0] != 1 for item in inputs) or
            inputs[0]['nchw'] != output['nchw']):
        raise ValueError('H14 template requires matching batch-one primary surfaces')
    if operation in templates.BINARY_OPERATIONS:
        kind = 'BinaryRuntime' if len(inputs) == 2 else 'BinaryScalar'
        operand = inputs[1]['nchw'][1:4] if len(inputs) == 2 else shape[1:4]
        operations = [operation]
    else:
        if len(inputs) != 1:
            raise ValueError('H14 unary template requires one input')
        kind, operand = 'Unary', shape[1:4]
        operations = ([operation, 'gelu_sigmoid_approximation', 'gelu_tanh_approximation']
                      if operation == 'gelu' else [operation])
    for name in operations:
        key = (kind, name, tuple(shape[1:4]), tuple(operand))
        if elementwise_contracts().get(key) == (task_stream, constants):
            return
    raise ValueError('H14 descriptor/constants differ from the source template for its geometry')
