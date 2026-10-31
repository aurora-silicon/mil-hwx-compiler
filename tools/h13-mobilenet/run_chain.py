"""Run a linked chain package and its per-layer programs on an H13 device.

usage: run_chain.py CASEDIR [repeat] [--no-perop] [--no-ref]

Prints one JSON line: linked vs per-layer bit agreement, accuracy of the
linked output against the float64 reference (fp16 ties-away between layers),
and wall/busy time for one linked submit and for the per-layer sequence with
host copies between programs.
"""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain import pack, unpack, f16_ties_away, stats, chain_ref, load_params   # noqa: E402

case = sys.argv[1]
repeat = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 300
flags = set(a for a in sys.argv[2:] if a.startswith('--'))
anechain = os.environ.get('ANECHAIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anechain'))
tmp = os.environ.get('ANEBENCH_TMP', '/var/tmp/h13-chain'); os.makedirs(tmp, exist_ok=True)
layers, params, inshape = load_params(os.path.join(case, 'params.npz'))
lm = json.load(open(os.path.join(case, 'linked', 'manifest.json')))
lp = lm['programs'][0]
x = np.random.default_rng(7).uniform(-1, 1, inshape).astype(np.float16)
inb = lp['inputs'][0]; outb = lp['outputs'][0]
open(f'{tmp}/in.bin', 'wb').write(pack(x, inb['nchw'], inb['allocationBytes']))

def wedged():
    try: return open('/sys/class/accel/accel0/device/wedged').read().strip()
    except OSError: return '?'

def run(progs, out, extra=()):
    r = subprocess.run([anechain, '-r', str(repeat), *extra, '-i', f'{tmp}/in.bin', '-o', out, *progs],
                       capture_output=True, text=True, timeout=600)
    if r.returncode:
        print(json.dumps(dict(case=case, error=(r.stdout + r.stderr).strip()[-600:], wedged=wedged())))
        sys.exit(1)
    return json.loads(r.stdout.strip().splitlines()[-1])

res = dict(case=os.path.basename(case.rstrip('/')), layers=len(layers), tasks=lp['taskDescriptors'])
res['linked'] = run([os.path.join(case, 'linked', lp['file'])], f'{tmp}/linked.bin')
res['linked_exec_only'] = run([os.path.join(case, 'linked', lp['file'])], f'{tmp}/linked2.bin', ('-n',))
res['wedged_after_linked'] = wedged()
dev = unpack(open(f'{tmp}/linked.bin', 'rb').read(), outb['nchw'])
dev2 = unpack(open(f'{tmp}/linked2.bin', 'rb').read(), outb['nchw'])
res['linked_repeatable'] = bool((dev.view(np.uint16) == dev2.view(np.uint16)).all())
if '--no-perop' not in flags:
    pm = json.load(open(os.path.join(case, 'perop', 'manifest.json')))
    progs = [os.path.join(case, 'perop', pm['programs'][i]['file']) for i in pm['dispatchPlan']]
    res['perop'] = run(progs, f'{tmp}/perop.bin')
    res['wedged_after_perop'] = wedged()
    ref_dev = unpack(open(f'{tmp}/perop.bin', 'rb').read(), pm['programs'][pm['dispatchPlan'][-1]]['outputs'][0]['nchw'])
    same = dev.view(np.uint16) == ref_dev.view(np.uint16)
    res['linked_equals_perop_pct'] = round(100 * float(same.mean()), 4)
if '--no-ref' not in flags:
    ref = chain_ref(x, layers, params)[-1]
    res['vs_float64'] = stats(dev, ref)
print(json.dumps(res))
