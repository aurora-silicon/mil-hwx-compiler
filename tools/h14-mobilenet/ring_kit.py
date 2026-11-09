#!/usr/bin/env python3
"""Build the smallest-first M2 kit for H14 `linked-ring`."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
H13 = HERE.parent / 'h13-mobilenet'
sys.path.insert(0, str(H13))
sys.path.insert(0, str(HERE.parent.parent / 'tests'))
from chain import chain_ref, compile_chain, conv_mil, save_params  # noqa: E402
from h14_libane_overrun import overruns  # noqa: E402
import mobilenet as M  # noqa: E402

RELU6_SINGLE_FORMS = (0, 1, 2, 3, 5, 7, 13, 25, 26, 27)


def references(case, layers, params, inshape, count):
    rng = np.random.default_rng(5983)
    inputs, outputs = [], []
    for _ in range(count):
        value = rng.uniform(-1, 1, inshape).astype(np.float16)
        inputs.append(value)
        outputs.append(chain_ref(value, layers, params)[-1])
    np.savez(case / 'refs.npz', inputs=np.asarray(inputs),
             outputs_float64=np.asarray(outputs, dtype=np.float64))


def single(root, name, layer, param, inshape, count, compiler):
    case = root / 'singles' / name
    mil, _, _ = conv_mil([layer], inshape, case / 'src', weights=[param])
    save_params(case / 'params.npz', [layer], [param], inshape)
    compile_chain(compiler, mil, case / 'src', case / 'linked', 'linked-ring')
    references(case, [layer], [param], inshape, count)


def verify(root):
    programs = sorted(root.rglob('*.anec'))
    failures = []
    for program in programs:
        for channel, need, have in overruns(program.read_bytes()):
            failures.append(f'{program.relative_to(root)}: channel {channel} '
                            f'needs {need}, has {have}')
    if failures:
        raise RuntimeError('libane bind overrun:\n' + '\n'.join(failures))
    return len(programs)


def write_hash_list(path, root, programs):
    lines = []
    for program in sorted(programs):
        digest = hashlib.sha256(program.read_bytes()).hexdigest()
        lines.append(f'{digest}  {program.relative_to(root)}')
    path.write_text('\n'.join(lines) + '\n')


def prefixes(root, layers, params, count, compiler):
    for prefix in range(1, len(layers) + 1):
        case = root / f'p{prefix:02d}'
        mil, _, _ = conv_mil(layers[:prefix], (3, 224, 224), case / 'src',
                             weights=params[:prefix])
        save_params(case / 'params.npz', layers[:prefix], params[:prefix],
                    (3, 224, 224))
        compile_chain(compiler, mil, case / 'src', case / 'linked',
                      'linked-ring')
    rng = np.random.default_rng(5983)
    inputs = [rng.uniform(-1, 1, (3, 224, 224)).astype(np.float16)
              for _ in range(count)]
    outputs = [chain_ref(value, layers, params) for value in inputs]
    for prefix in range(1, len(layers) + 1):
        np.savez(root / f'p{prefix:02d}' / 'refs.npz',
                 inputs=np.asarray(inputs),
                 outputs_float64=np.asarray(
                     [sample[prefix - 1] for sample in outputs],
                     dtype=np.float64))


def input_shapes(layers, initial=(3, 224, 224)):
    shapes = []
    channels, height, width = initial
    for layer in layers:
        shapes.append((channels, height, width))
        channels = layer['cout']
        height = (height + layer['s'] - 1) // layer['s']
        width = (width + layer['s'] - 1) // layer['s']
    return shapes


def write_relu6_field_changes(root, layers):
    clipped = [index for index, layer in enumerate(layers)
               if layer.get('relu6')]
    fields = [
        'taskModeKernelClipDescriptor (k3) / taskMode zero (k1)',
        'macRelu', 'macFiniteClip',
        'kernelLaneOffset', 'kernelLaneSize',
        'clipLower0/Upper0', 'clipLower1/Upper1',
        'clipUnity0/Unity1', 'clipScale', 'clipEnable']
    unchanged = ['neLutEnable', 'neSparsePhase']
    programs = [{
        'program': 'full-relu6/linked/program-0.anec',
        'clippedLayers': clipped, 'changedFields': fields,
        'unchangedFields': unchanged,
        'rkR7ChangedTasks': [1],
        'rkR7ChangedFields': ['kernelLaneSize[0..15]'],
    }]
    programs.extend({
        'program': f'prefixes6/p{prefix:02d}/linked/program-0.anec',
        'clippedLayers': [index for index in clipped if index < prefix],
        'changedFields': fields, 'unchangedFields': unchanged,
        'rkR7ChangedTasks': [1] if prefix >= 4 else [],
        'rkR7ChangedFields': (['kernelLaneSize[0..15]']
                              if prefix >= 4 else []),
    } for prefix in range(1, len(layers) + 1))
    programs.extend({
        'program': f'singles/r6-L{index:02d}/linked/program-0.anec',
        'clippedLayers': [index] if layers[index].get('relu6') else [],
        'changedFields': fields if layers[index].get('relu6') else [],
        'unchangedFields': unchanged,
        'rkR7ChangedTasks': [], 'rkR7ChangedFields': [],
    } for index in RELU6_SINGLE_FORMS)
    (root / 'relu6-field-changes.json').write_text(json.dumps({
        'referenceEmitter': 'H14 linked convolution clip epilogue',
        'programs': programs,
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--weights')
    parser.add_argument('--refs', type=int, default=1)
    args = parser.parse_args()
    out = args.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    os.environ['MILHWXC_TARGET'] = 'H14'
    compiler = os.environ.get('MILC', 'build/mil-hwxc')
    weights = (M.random_tf_weights(5983) if not args.weights else
               __import__('tflite_weights').mobilenet_v1_weights(args.weights))
    relu = M.ane_layers('ane', 'native', relu6=False)
    relu6 = M.ane_layers('ane', 'native', relu6=True)
    params = M.to_ane_params(weights, 'ane', 'native')

    for name, layers in (('full-relu', relu), ('full-relu6', relu6)):
        case = out / name
        mil, _, _ = conv_mil(layers, (3, 224, 224), case / 'src',
                             weights=params)
        save_params(case / 'params.npz', layers, params, (3, 224, 224))
        compile_chain(compiler, mil, case / 'src', case / 'linked',
                      'linked-ring')
        references(case, layers, params, (3, 224, 224), args.refs)

    prefixes(out / 'prefixes', relu, params, args.refs, compiler)
    prefixes(out / 'prefixes6', relu6, params, args.refs, compiler)
    for index, spatial in ((3, 112), (7, 56), (11, 28), (23, 14)):
        single(out, f'dw-k3-s2-{spatial}', relu[index], params[index],
               (relu[index]['g'], spatial, spatial), args.refs, compiler)
    single(out, 'stem-k3-s2-224', relu[0], params[0], (3, 224, 224),
           args.refs, compiler)
    shapes = input_shapes(relu6)
    for index in RELU6_SINGLE_FORMS:
        single(out, f'r6-L{index:02d}', relu6[index], params[index],
               shapes[index], args.refs, compiler)
    write_relu6_field_changes(out, relu6)

    checked = verify(out)
    programs = sorted(out.rglob('*.anec'))
    relu_programs = [out / 'full-relu/linked/program-0.anec']
    relu_programs.extend(sorted((out / 'prefixes').rglob('*.anec')))
    relu_programs.extend(sorted(
        program for program in (out / 'singles').rglob('*.anec')
        if not program.parent.parent.name.startswith('r6-')))
    write_hash_list(out / 'sha256-programs.txt', out, programs)
    write_hash_list(out / 'sha256-relu.txt', out, relu_programs)
    order = [f'singles/dw-k3-s2-{size}' for size in (14, 28, 56, 112)] + [
        'singles/stem-k3-s2-224', 'prefixes/p01..p28', 'full-relu',
        'singles/r6-L02', 'singles/r6-L05', 'prefixes6/p04',
        'prefixes6/p01..p28', 'full-relu6']
    (out / 'kit.json').write_text(json.dumps({
        'target': 'H14', 'schedule': 'linked-ring', 'tasks': 28,
        'executionOrder': order,
        'reference': 'float64 convolution with fp16 boundaries',
        'libaneBindOverrunProgramsChecked': checked,
    }, indent=2) + '\n')
    shutil.copy2(HERE / 'run.c', out / 'run.c')
    shutil.copy2(HERE / 'run.py', out / 'run.py')
    shutil.copy2(HERE / 'run_kit.py', out / 'run_kit.py')
    print(out)


if __name__ == '__main__':
    main()
