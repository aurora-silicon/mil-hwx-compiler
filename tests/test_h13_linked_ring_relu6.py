#!/usr/bin/env python3
"""Native TF-SAME MobileNet compiles as one 28-task ReLU6 ring."""
import json
import struct
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    print('SKIP native TF-SAME ReLU6 linked-ring (NumPy unavailable)')
    sys.exit(0)

ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / 'tools/h13-mobilenet'
sys.path.insert(0, str(MOBILE))
sys.path.insert(0, str(ROOT / 'tests'))
from chain import compile_chain, conv_mil, conv_ref  # noqa: E402
import libane_overrun  # noqa: E402
import mobilenet as M  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / 'build/mil-hwxc').resolve()


def main():
    # TensorFlow SAME puts the single extra k3/s2 pad row and column at the
    # bottom/right for an even input.  Symmetric padding samples a different
    # phase and was the old device-kit reference bug.
    x = np.arange(16, dtype=np.float16).reshape(1, 4, 4)
    w = np.ones((1, 1, 3, 3), dtype=np.float16)
    same = conv_ref(x, w, None, 2, 1, 'tf_same', False)
    np.testing.assert_equal(same, [[[45, 39], [66, 50]]])
    assert not np.array_equal(same, conv_ref(x, w, None, 2, 1, 1, False))

    layers = M.ane_layers('ane', 'native', relu6=True)
    params = M.to_ane_params(M.random_tf_weights(5964), 'ane', 'native')
    with tempfile.TemporaryDirectory(prefix='h13-ring-relu6-') as directory:
        root = Path(directory)
        mil, _, _ = conv_mil(layers, (3, 224, 224), root / 'src',
                             weights=params)
        source = Path(mil).read_text()
        assert source.count('val = string("same")') == 14
        manifest = compile_chain(str(COMPILER), mil, root / 'src',
                                 root / 'package', 'linked-ring')
        program = manifest['programs'][0]
        assert program['taskDescriptors'] == 28
        assert len(program['layers']) == 28
        assert [layer['operation'] for layer in program['layers']].count(
            'conv+clip') == 27
        anec = (root / 'package' / program['file']).read_bytes()
        assert not libane_overrun.overruns(anec)
        tiles = struct.unpack_from('<32I', anec, 40)
        assert tiles[4] == 28 and tiles[5] == 28
        assert json.loads((root / 'package/manifest.json').read_text()) == manifest
    print('PASS native TF-SAME ReLU6 linked-ring: 28 tasks, 27 fused clips, bind-safe')


if __name__ == '__main__':
    main()
