#!/usr/bin/env python3
"""Build the H14 compact-task padding-minimum device probe kit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
H13 = HERE.parent / 'h13-mobilenet'
sys.path.insert(0, str(H13))
sys.path.insert(0, str(ROOT / 'research'))
sys.path.insert(0, str(ROOT / 'tests'))
from chain import chain_ref, compile_chain, conv_mil, save_params  # noqa: E402
from h13_td import split_h14_tasks  # noqa: E402
from h14_libane_overrun import overruns  # noqa: E402
import mobilenet as M  # noqa: E402


PADDING_MINIMA = (0, 128, 256, 312)
MINIMUM_OPTION = '--h14-research-compact-task-minimum-bytes'


def task_sizes(path):
    data = path.read_bytes()
    text_bytes = struct.unpack_from('<QIIQQ', data)[3]
    return [len(task) for task in
            split_h14_tasks(data[4096:4096 + text_bytes])]


def layer_form(index, layer):
    if index == 0:
        family = 'stem'
    elif layer['g'] == layer['cout']:
        family = 'depthwise'
    else:
        family = 'pointwise'
    return f'{family} k{layer["k"]}/s{layer["s"]}'


def references(path, layers, params):
    value = np.random.default_rng(6186).uniform(
        -1, 1, (3, 224, 224)).astype(np.float16)
    output = chain_ref(value, layers, params)[-1]
    np.savez(path, inputs=value[None],
             outputs_float64=np.asarray(output[None], dtype=np.float64))


def build_case(root, name, source_program, layers, params, compiler):
    case = root / name
    mil, _, _ = conv_mil(layers, (3, 224, 224), case / 'src',
                         weights=params)
    save_params(case / 'params.npz', layers, params, (3, 224, 224))
    references(case / 'refs.npz', layers, params)
    variants = []
    for minimum in PADDING_MINIMA:
        variant = case / f'pad-{minimum:03d}'
        compile_chain(
            compiler, mil, case / 'src', variant / 'linked', 'linked-ring',
            compiler_options=(MINIMUM_OPTION, str(minimum)))
        shutil.copy2(case / 'refs.npz', variant / 'refs.npz')
        program = variant / 'linked' / 'program-0.anec'
        failures = overruns(program.read_bytes())
        if failures:
            raise RuntimeError(f'{program}: libane bind overrun {failures}')
        variants.append({
            'case': f'{name}/pad-{minimum:03d}',
            'sourceProgram': source_program,
            'paddingMinimumBytes': minimum,
            'taskActualBytes': task_sizes(program),
        })
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--weights')
    args = parser.parse_args()
    out = args.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    os.environ['MILHWXC_TARGET'] = 'H14'
    compiler = os.environ.get('MILC', 'build/mil-hwxc')
    weights = (M.random_tf_weights(5983) if not args.weights else
               __import__('tflite_weights').mobilenet_v1_weights(args.weights))
    layers = M.ane_layers('ane', 'native', relu6=False)
    params = M.to_ane_params(weights, 'ane', 'native')

    variants = []
    for name, source_program, count in (
            ('prefixes/p04', 'p04', 4),
            ('prefixes/p08', 'p08', 8),
            ('full-relu', 'full-relu', len(layers))):
        variants.extend(build_case(out, name, source_program,
                                   layers[:count], params[:count], compiler))

    full = {entry['paddingMinimumBytes']: entry['taskActualBytes']
            for entry in variants if entry['sourceProgram'] == 'full-relu'}
    task_table = [{
        'taskIndex': index,
        'layerForm': layer_form(index, layer),
        'semanticBytes': full[0][index],
        'paddedBytes': full[312][index],
    } for index, layer in enumerate(layers)]
    if not any(row['semanticBytes'] < 312 for row in task_table):
        raise RuntimeError('full ReLU ring has no semantic task below 312 bytes')

    (out / 'kit.json').write_text(json.dumps({
        'target': 'H14',
        'schedule': 'linked-ring',
        'purpose': 'compact-task padding-minimum acceptance probe',
        'inputShape': [1, 3, 224, 224],
        'paddingMinimaBytes': list(PADDING_MINIMA),
        'fullReluTaskSizes': task_table,
        'executionOrder': [entry['case'] for entry in variants],
        'variants': variants,
        'results': 'to be filled from device runs',
    }, indent=2) + '\n')
    shutil.copy2(HERE / 'run.c', out / 'run.c')
    shutil.copy2(HERE / 'run.py', out / 'run.py')
    shutil.copy2(HERE / 'run_kit.py', out / 'run_kit.py')
    print(out)


if __name__ == '__main__':
    main()
