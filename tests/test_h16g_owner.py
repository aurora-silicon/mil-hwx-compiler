#!/usr/bin/env python3
"""Reproduce the owner identity from source, never from checked-in HWX bytes."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from h16g_validate import OWNER_SHA256, validate, validate_resources
from timeouts import scaled_timeout


def main():
    prepare = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'build/prepare_alu').resolve()
    with tempfile.TemporaryDirectory(prefix='h16g-owner-') as folder:
        run = subprocess.run([str(prepare), 'add', '128', folder], capture_output=True,
                             text=True, timeout=scaled_timeout(60), check=False)
        assert run.returncode == 0, run.stdout + run.stderr
        data = (Path(folder) / 'bundle/program-0.hwx').read_bytes()
        assert len(data) == 49152
        assert hashlib.sha256(data).hexdigest() == OWNER_SHA256
        result = validate(data, owner=True)
        assert result['tasks'] == 1 and result['tensors'] == 3 and result['owner_fixture']
        assert result['driver_task_gate']
        manifest = json.loads((Path(folder) / 'bundle/manifest.json').read_text())
        metadata = manifest['artifacts'][0]['h16Execute']
        validate_resources(metadata, result['execute_resources'])
        assert [(b['index'], b['class']) for b in metadata['buffers']] == [(0, 0), (0, 1), (1, 0)]
        assert metadata['customBars'] == [] and metadata['commandBytes'] == 0xca8
        # Exercise the supported CLI path on the actual emitted pair.
        subprocess.run([sys.executable, str(ROOT / 'tools/h16g_validate.py'),
                        str(Path(folder) / 'bundle/program-0.hwx'), '--owner',
                        '--manifest', str(Path(folder) / 'bundle/manifest.json')], check=True)
    # CPU fp16 reference inputs are exact binary fractions. No target access.
    a = [(i % 17 - 8) * .125 for i in range(16384)]
    b = [(i % 11 - 5) * .25 for i in range(16384)]
    expected = [x + y for x, y in zip(a, b)]
    for items, digest in (
        (a, '3a944e8c891c9e4b00d1cd12bbec5f17cde432fafc1fce0df5809b3deb513e03'),
        (b, '779fff95d69316770cfff014c8bab4249c2f797c673a4518af80ab87b8d86a05'),
        (expected, 'd113d9ae6734b7a7e233458229f64f382da99a3f93868eef3abb06a29f7f4ce8'),
    ):
        assert hashlib.sha256(struct.pack('<16384e', *items)).hexdigest() == digest
    print('H16G owner identity and fp16 reference: PASS; corrected prologue and resource metadata verified')


if __name__ == '__main__':
    main()
