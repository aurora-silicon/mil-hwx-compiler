#!/usr/bin/env python3
"""Build the ANECOMP M1 Pro ladder for H13 `linked-ring`.

usage: ring_kit.py OUTDIR [--weights mobilenet.tflite] [--refs N]

The kit contains native k3-s2 stem/depthwise single-task cases, p01..p28,
the full network, deterministic inputs and float64-per-layer references.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'tests'))
import mobilenet as M  # noqa: E402
from chain import chain_ref, compile_chain, conv_mil, save_params  # noqa: E402
import libane_overrun  # noqa: E402


def refs(case, layers, params, inshape, count):
    rng = np.random.default_rng(5964)
    inputs, outputs = [], []
    for _ in range(count):
        x = rng.uniform(-1, 1, inshape).astype(np.float16)
        inputs.append(x)
        outputs.append(chain_ref(x, layers, params)[-1])
    np.savez(case / 'refs.npz', inputs=np.asarray(inputs),
             outputs_float64=np.asarray(outputs, dtype=np.float64))


def prefix_refs(root, layers, params, count):
    rng = np.random.default_rng(5964)
    inputs = [rng.uniform(-1, 1, (3, 224, 224)).astype(np.float16)
              for _ in range(count)]
    outputs = [chain_ref(x, layers, params) for x in inputs]
    for prefix in range(1, 29):
        np.savez(root / f'p{prefix:02d}' / 'refs.npz',
                 inputs=np.asarray(inputs),
                 outputs_float64=np.asarray(
                     [sample[prefix - 1] for sample in outputs],
                     dtype=np.float64))


def single(root, name, layer, param, inshape, count, milc):
    case = root / 'singles' / name
    mil, _, _ = conv_mil([layer], inshape, str(case / 'src'), weights=[param])
    save_params(case / 'params.npz', [layer], [param], inshape)
    compile_chain(milc, mil, str(case / 'src'), str(case / 'linked'),
                  'linked-ring')
    refs(case, [layer], [param], inshape, count)


def verify_bind_allocations(root):
    """Apply libane's ane_bind_overrun rule to every program in the kit."""
    programs = sorted(root.rglob('*.anec'))
    if len(programs) != 34:
        raise RuntimeError(f'expected 34 linked-ring ANEC programs, found {len(programs)}')
    failures = []
    for program in programs:
        for channel, need, have in libane_overrun.overruns(program.read_bytes()):
            failures.append(
                f'{program.relative_to(root)}: channel {channel} needs '
                f'{need} bytes but is allocated {have}')
    if failures:
        raise RuntimeError('libane bind overrun:\n' + '\n'.join(failures))
    return len(programs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--weights')
    ap.add_argument('--refs', type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    milc = os.environ.get('MILC', 'build/mil-hwxc')
    tfw = (M.random_tf_weights(5964) if not args.weights else
           __import__('tflite_weights').mobilenet_v1_weights(args.weights))
    layers = M.ane_layers('ane', 'native', relu6=False)
    params = M.to_ane_params(tfw, 'ane', 'native')

    full = out / 'full'
    mil, _, _ = conv_mil(layers, (3, 224, 224), str(full / 'src'), weights=params)
    save_params(full / 'params.npz', layers, params, (3, 224, 224))
    compile_chain(milc, mil, str(full / 'src'), str(full / 'linked'),
                  'linked-ring')
    refs(full, layers, params, (3, 224, 224), args.refs)

    subprocess.run([sys.executable, str(HERE / 'prefixes.py'), str(full),
                    str(out / 'prefixes')], check=True,
                   env=dict(os.environ, MILC=milc))
    prefix_refs(out / 'prefixes', layers, params, args.refs)
    single(out, 'stem-k3-s2-224', layers[0], params[0], (3, 224, 224),
           args.refs, milc)
    for index, spatial in ((3, 112), (7, 56), (11, 28), (23, 14)):
        single(out, f'dw-k3-s2-{spatial}', layers[index], params[index],
               (layers[index]['g'], spatial, spatial), args.refs, milc)

    checked = verify_bind_allocations(out)

    manifest = {
        'schedule': 'linked-ring', 'target': 'H13', 'tasks': 28,
        'cases': ['singles/dw-k3-s2-14', 'singles/dw-k3-s2-28',
                  'singles/dw-k3-s2-56', 'singles/dw-k3-s2-112',
                  'singles/stem-k3-s2-224',
                  'prefixes/p01..p28', 'full'],
        'reference': 'float64 convolution with fp16 ties-away at every edge',
        'libaneBindOverrunProgramsChecked': checked,
    }
    (out / 'kit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(HERE / 'device-kit-ring' / 'run.sh', out / 'run.sh')
    os.chmod(out / 'run.sh', 0o755)
    print(out)


if __name__ == '__main__':
    main()
