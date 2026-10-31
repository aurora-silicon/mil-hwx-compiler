"""Device side: run the linked MobileNet on each reference input and compare
logits with the float64 CPU reference (refs.npz from mn_refs.py).

usage: mn_eval.py NETDIR [labels.txt]
"""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain import pack, unpack   # noqa: E402
net = sys.argv[1]
labels = open(sys.argv[2]).read().split('\n') if len(sys.argv) > 2 else None
exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anechain')
tmp = '/var/tmp/h13-chain'; os.makedirs(tmp, exist_ok=True)
m = json.load(open(os.path.join(net, 'linked', 'manifest.json')))
p = m['programs'][0]; inb, outb = p['inputs'][0], p['outputs'][0]
z = np.load(os.path.join(net, 'refs.npz'))
rows = []
for i, x in enumerate(z['inputs']):
    open(f'{tmp}/mn_in.bin', 'wb').write(pack(x, inb['nchw'], inb['allocationBytes']))
    r = subprocess.run([exe, '-r', '1', '-w', '0', '-i', f'{tmp}/mn_in.bin', '-o', f'{tmp}/mn_out.bin',
                        os.path.join(net, 'linked', p['file'])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    dev = unpack(open(f'{tmp}/mn_out.bin', 'rb').read(), outb['nchw']).astype(np.float64)
    logits = dev.mean(axis=(1, 2))[:1001]
    ref = z['logits_relu'][i]; tf = z['logits_tf_relu6'][i]
    top5 = set(np.argsort(-logits)[:5]); rtop5 = set(np.argsort(-ref)[:5])
    rows.append(dict(i=i, top1=int(logits.argmax()), ref_top1=int(ref.argmax()), tf_relu6_top1=int(tf.argmax()),
                     top5_overlap=len(top5 & rtop5), max_abs=float(np.abs(logits - ref).max()),
                     rel_l2=float(np.linalg.norm(logits - ref) / np.linalg.norm(ref)),
                     finite=bool(np.isfinite(dev).all())))
    if labels:
        rows[-1]['label'] = labels[rows[-1]['top1']]
    print(json.dumps(rows[-1]), flush=True)
agree = sum(r['top1'] == r['ref_top1'] for r in rows)
print(json.dumps(dict(images=len(rows), top1_agree=agree, top5_mean_overlap=np.mean([r['top5_overlap'] for r in rows]),
                      worst_rel_l2=max(r['rel_l2'] for r in rows),
                      tf_relu6_top1_agree=sum(r['top1'] == r['tf_relu6_top1'] for r in rows),
                      wedged=open('/sys/class/accel/accel0/device/wedged').read().strip())))
