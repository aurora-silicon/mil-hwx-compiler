#!/usr/bin/env python3
"""Try the existing lane-a MobileNetV1 graph without widening H16/H17 facts.

Writes the exact graph/weights, attempted linked compiler output, and each
layer's descriptor coverage. A refusal is recorded as a refusal, never as a
loadable package. No hardware is used. Requires numpy, like h13-mobilenet.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/h13-mobilenet'))
from mobilenet import ane_layers, random_tf_weights, to_ane_params  # noqa: E402
from chain import conv_mil, save_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--compiler', type=Path, default=ROOT / 'build/mil-hwxc')
    ap.add_argument('--descriptor-tool', type=Path, default=ROOT / 'build/h16_conv_dump')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    layers = ane_layers('ane')
    weights = to_ane_params(random_tf_weights(args.seed), 'ane')
    source = args.output / 'src'
    mil, params, shape = conv_mil(layers, (16,112,112), str(source), weights=weights)
    save_params(str(args.output / 'params.npz'), layers, params, (16,112,112))
    report = {'graph': 'h13-mobilenet lane-a mirrored-space-to-depth, classifier-before-host-pool',
              'seed': args.seed, 'layers': len(layers), 'outputShape': shape, 'targets': {}}
    for target in ('H16G', 'H17'):
        command = [str(args.compiler.resolve()), '--target', target, '--format', 'hwx',
                   '--schedule', 'linked', '--mil', mil, '--model-root', str(source),
                   '--output', str(args.output / target)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        detail = {'linkedExitCode': run.returncode, 'diagnostic': run.stderr.strip(),
                  'loadableLinkedPackage': run.returncode == 0, 'layers': []}
        ci, width = 16,112
        for i, layer in enumerate(layers):
            result = subprocess.run([str(args.descriptor_tool.resolve()), 'describe', target,
                *map(str, [ci,layer['cout'],width,layer['k'],layer['s'],layer['g'],int(layer['relu'])])],
                capture_output=True, timeout=30)
            detail['layers'].append({'index': i, 'input': [ci,width,width], **layer,
                'descriptorCoveredWithoutBias': result.returncode == 0,
                'biasCovered': False if layer['bias'] else True,
                'descriptorRefusal': result.stderr.decode().strip()})
            ci, width = layer['cout'], (width+layer['s']-1)//layer['s']
        report['targets'][target] = detail
    (args.output / 'coverage.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({t: {'linkedExitCode': d['linkedExitCode'],
                         'descriptorCoveredWithoutBias': sum(l['descriptorCoveredWithoutBias'] for l in d['layers']),
                         'loadableLinkedPackage': d['loadableLinkedPackage']} for t,d in report['targets'].items()},indent=2))
    return 0 if all(d['loadableLinkedPackage'] for d in report['targets'].values()) else 2


if __name__ == '__main__':
    sys.exit(main())
