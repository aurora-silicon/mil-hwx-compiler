#!/usr/bin/env python3
"""H14 native ReLU6 ring: TF-SAME, bind safety, and 224/192 references."""
import json
import struct
import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise RuntimeError('H14 linked-ring reference test requires NumPy') from error

ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / 'tools/h13-mobilenet'
sys.path.insert(0, str(MOBILE))
sys.path.insert(0, str(ROOT / 'tests'))
from chain import chain_ref, compile_chain, conv_mil, conv_ref  # noqa: E402
from h14_libane_overrun import overruns  # noqa: E402
from h14_fields import H14Fields  # noqa: E402
sys.path.insert(0, str(ROOT / 'research'))
from h13_td import decode_task, split_h14_tasks  # noqa: E402
import mobilenet as M  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else
                ROOT / 'build/mil-hwxc').resolve()

# ANEC v1 text begins at its fixed page boundary.  Packed-record fields and
# TileDMA register addresses come from H14Fields.h rather than a test copy.
ANEC_TEXT_OFFSET = 4096
FIELDS = H14Fields(ROOT)
SOURCE_DMA_BASE = FIELDS.register('sourceDmaBase')
RESULT_DMA_BASE = FIELDS.register('resultDmaBase')
# ABI-2's public port contract: input BAR 4, scratch BAR 3, output BAR 5.
PROGRAM_INPUT_BAR = 4
SCRATCH_BAR = 3
PROGRAM_OUTPUT_BAR = 5


def io_bars(anec):
    _, _, count, text_bytes, _ = struct.unpack_from('<QIIQQ', anec)
    found = set()
    for raw in split_h14_tasks(
            anec[ANEC_TEXT_OFFSET:ANEC_TEXT_OFFSET + text_bytes]):
        for record in decode_task(raw, 'h14')['records']:
            header, address = int(record['header'], 16), int(record['address'], 16)
            if (header & FIELDS.field('barPresent') and
                    address in (SOURCE_DMA_BASE, RESULT_DMA_BASE)):
                found.add((address, FIELDS.get('barSlot', header)))
    return found


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
    # Same port ABI as the linked emitter and the ABI-2 runner: program input
    # on BAR 4, final output on BAR 5, scratch on BAR 3.
    assert io_bars(anec) == {
        (SOURCE_DMA_BASE, PROGRAM_INPUT_BAR), (SOURCE_DMA_BASE, SCRATCH_BAR),
        (RESULT_DMA_BASE, SCRATCH_BAR), (RESULT_DMA_BASE, PROGRAM_OUTPUT_BAR)}
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
        layers = M.ane_layers('ane', 'native', relu6=True)
        params = M.to_ane_params(M.random_tf_weights(5983), 'ane', 'native')
        refused = root / 'standalone-L02'
        mil, _, _ = conv_mil([layers[2]], (32, 112, 112), refused / 'src',
                             weights=[params[2]])
        ring_out = refused / 'ring'
        run = subprocess.run([
            str(COMPILER), '--target', 'H14', '--schedule', 'linked-ring',
            '--mil', str(mil), '--model-root', str(refused / 'src'),
            '--output', str(ring_out)], text=True, capture_output=True,
            env=dict(os.environ, MILHWXC_H14_CONV_GENERIC='1'))
        assert run.returncode != 0
        assert 'h14.linked-ring-outside-arena' in run.stderr
        assert 'source=802816' in run.stderr
        assert 'result=1609216' in run.stderr
        assert 'scratch=0' in run.stderr
        assert 'coefficient-prefetch=' in run.stderr
        assert '--schedule linked' in run.stderr
        assert not ring_out.exists()
        linked = compile_chain(str(COMPILER), mil, refused / 'src',
                               refused / 'linked', 'linked')
        assert linked['schedule'] == 'linked'

        references = []
        for size in (224, 192):
            layers, params, output = compile_size(
                root / f'native{size}-relu6', size, True)
            # Execute the full float64 reference with fp16 layer boundaries.
            sample = np.random.default_rng(5983 + size).uniform(
                -1, 1, (3, size, size))
            reference = chain_ref(sample, layers, params)[-1]
            assert (reference.shape == tuple(output) and
                    np.isfinite(reference).all())
            references.append(dict(
                size=size, shape=list(reference.shape),
                l2=float(np.linalg.norm(reference.astype(np.float64)))))
        report = dict(status='PASS', activation='ReLU6', sizes=[224, 192],
                      tasks=28, references=references)
        print(json.dumps(report))


if __name__ == '__main__':
    main()
