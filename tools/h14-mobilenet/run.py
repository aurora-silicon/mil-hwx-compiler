#!/usr/bin/env python3
"""M2 lead-side runner. One cold execution by default, no device access at build time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/h13-mobilenet'))
from mobilenet import preprocess  # noqa: E402
from chain import pack, unpack, chain_ref, load_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('package', type=Path)
    ap.add_argument('--runner', type=Path, required=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--input', type=Path, help='Numpy CHW [3,224,224] float image in [-1,1]')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--reference', action='store_true', help='compare the same graph in float64')
    args = ap.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package / 'linked/manifest.json').read_text())
    if manifest['target'] != 'H14' or manifest['schedule'] != 'linked' or len(manifest['programs']) != 1:
        raise SystemExit('requires one linked H14 program')
    p = manifest['programs'][0]
    image = (np.load(args.input, allow_pickle=False) if args.input else
             np.random.default_rng(args.seed).uniform(-1, 1, (3, 224, 224)))
    if image.shape != (3, 224, 224) or not np.isfinite(image).all():
        raise SystemExit('input must be finite [3,224,224]')
    x = preprocess(image)
    with tempfile.TemporaryDirectory(prefix='h14-mobilenet-') as tmp:
        inp, out = Path(tmp) / 'input.bin', Path(tmp) / 'output.bin'
        inp.write_bytes(pack(x, p['inputs'][0]['nchw'], p['inputs'][0]['allocationBytes']))
        run = subprocess.run([str(args.runner.resolve()), str(package / 'linked' / p['file']),
                              str(inp), str(out), str(args.repeat), str(args.device)],
                             check=True, text=True, capture_output=True, timeout=600)
        y = unpack(out.read_bytes(), p['outputs'][0]['nchw']).astype(np.float64)
    if not np.isfinite(y).all():
        raise SystemExit('nonfinite output')
    logits = y.mean(axis=(1, 2))[:1001]
    report = dict(runtime=json.loads(run.stdout.strip().splitlines()[-1]),
                  top5=np.argsort(-logits)[:5].tolist(), finite=True)
    if args.reference:
        layers, params, _ = load_params(package / 'params.npz')
        ref = chain_ref(x, layers, params)[-1]
        report['relative_l2'] = float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30))
        report['reference_top1'] = int(ref.mean(axis=(1, 2))[:1001].argmax())
    print(json.dumps(report))

if __name__ == '__main__':
    main()
