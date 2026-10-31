"""Cold-run check for a linked package: every run is a fresh process with no
warm-up (-w 0 -r 1), and the input changes between runs, so a task that reads
its input before the producer's writes land shows up as a mismatch instead of
converging on stale data from an earlier pass.

usage: coldcheck.py CASEDIR [RUNS]   (CASEDIR from mkchain.py / mobilenet.py)

For each run: a new input, the linked program cold, the per-layer programs
cold, bit agreement between the two and accuracy against the float64
reference. Exit status 1 on any disagreement.
"""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain import pack, unpack, stats, chain_ref, load_params   # noqa: E402

case = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
exe = os.environ.get('ANECHAIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anechain'))
tmp = os.environ.get('ANEBENCH_TMP', '/var/tmp/h13-chain'); os.makedirs(tmp, exist_ok=True)
layers, params, inshape = load_params(os.path.join(case, 'params.npz'))
lm = json.load(open(os.path.join(case, 'linked', 'manifest.json')))
pm = json.load(open(os.path.join(case, 'perop', 'manifest.json')))
lp = lm['programs'][0]
perop = [os.path.join(case, 'perop', pm['programs'][i]['file']) for i in pm['dispatchPlan']]
last = pm['programs'][pm['dispatchPlan'][-1]]['outputs'][0]['nchw']

def wedged():
    return open('/sys/class/accel/accel0/device/wedged').read().strip()

def cold(progs, out):
    r = subprocess.run([exe, '-w', '0', '-r', '1', '-i', f'{tmp}/cold_in.bin', '-o', out, *progs],
                       capture_output=True, text=True, timeout=120)
    if r.returncode:
        print(json.dumps(dict(error=r.stderr.strip()[-300:], wedged=wedged())))
        sys.exit(2)

bad = 0
for run in range(runs):
    x = np.random.default_rng(100 + run).uniform(-1, 1, inshape).astype(np.float16)
    open(f'{tmp}/cold_in.bin', 'wb').write(pack(x, lp['inputs'][0]['nchw'], lp['inputs'][0]['allocationBytes']))
    cold([os.path.join(case, 'linked', lp['file'])], f'{tmp}/cold_linked.bin')
    cold(perop, f'{tmp}/cold_perop.bin')
    a = unpack(open(f'{tmp}/cold_linked.bin', 'rb').read(), lp['outputs'][0]['nchw'])
    b = unpack(open(f'{tmp}/cold_perop.bin', 'rb').read(), last)
    same = float((a.view(np.uint16) == b.view(np.uint16)).mean())
    ref = chain_ref(x, layers, params)[-1]
    sa, sb = stats(a, ref), stats(b, ref)
    bad += same != 1.0
    print(json.dumps(dict(run=run, linked_equals_perop_pct=round(100 * same, 4),
                          linked_rel_l2=sa['rel_l2'], perop_rel_l2=sb['rel_l2'], wedged=wedged())), flush=True)
print(json.dumps(dict(case=os.path.basename(case.rstrip('/')), runs=runs, mismatched_runs=bad)))
sys.exit(1 if bad else 0)
