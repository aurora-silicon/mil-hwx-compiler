#!/usr/bin/env python3
"""H14 native ring: TF-SAME, bind safety, fused clip, and a 192 rule build."""
import json
import os
from pathlib import Path
import sys
import tempfile

try:
    import numpy as np
except ModuleNotFoundError:
    print('SKIP H14 linked-ring (NumPy unavailable)')
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / 'tools/h13-mobilenet'
sys.path.insert(0, str(MOBILE))
sys.path.insert(0, str(ROOT / 'tests'))
from chain import chain_ref, compile_chain, conv_mil, conv_ref  # noqa: E402
from h14_libane_overrun import overruns  # noqa: E402
import mobilenet as M  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / 'build/mil-hwxc').resolve()


def compile_size(root, size, relu6):
    layers = M.ane_layers('ane', 'native', relu6=relu6)
    params = M.to_ane_params(M.random_tf_weights(5983 + size), 'ane', 'native')
    mil, _, output = conv_mil(layers, (3, size, size), root / 'src',
                              weights=params)
    manifest = compile_chain(str(COMPILER), mil, root / 'src', root / 'linked',
                             'linked-ring')
    assert manifest['target'] == 'H14'
    assert manifest['schedule'] == 'linked-ring'
    assert manifest['programs'][0]['taskDescriptors'] == 28
    assert len(manifest['tasks']) == 28
    if relu6:
        assert sum(task['operation'] == 'conv+clip'
                   for task in manifest['tasks']) == 27
    anec = (root / 'linked' / manifest['programs'][0]['file']).read_bytes()
    assert not overruns(anec)
    return layers, params, output


def main():
    os.environ['MILHWXC_TARGET'] = 'H14'
    # TensorFlow SAME puts the odd extra pad at bottom/right on an even input.
    x = np.arange(16, dtype=np.float16).reshape(1, 4, 4)
    w = np.ones((1, 1, 3, 3), dtype=np.float16)
    same = conv_ref(x, w, None, 2, 1, 'tf_same', False)
    np.testing.assert_equal(same, [[[45, 39], [66, 50]]])

    with tempfile.TemporaryDirectory(prefix='h14-linked-ring-') as directory:
        root = Path(directory)
        compile_size(root / 'native224', 224, True)
        layers, params, output = compile_size(root / 'native192', 192, False)
        # Execute the full float64 reference, including fp16 boundaries, rather
        # than merely checking that its helper imports.
        sample = np.random.default_rng(5983).uniform(-1, 1, (3, 192, 192))
        reference = chain_ref(sample, layers, params)[-1]
        assert reference.shape == tuple(output) and np.isfinite(reference).all()
        report = dict(status='PASS', sizes=[224, 192], tasks=28,
                      reference_shape=list(reference.shape),
                      reference_l2=float(np.linalg.norm(
                          reference.astype(np.float64))))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
