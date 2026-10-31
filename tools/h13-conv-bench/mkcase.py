"""Compile single conv layers: mkcase.py OUTROOT name:C,H,W,Cout,k,s,g,bias,relu[,pad] ..."""
import sys, os, subprocess, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from milgen import conv_mil
root = sys.argv[1]; env = dict(os.environ)
MILC = os.environ.get('MILC', 'build/mil-hwxc')
for spec in sys.argv[2:]:
    name, v = spec.split(':'); f = v.split(',')
    C, H, W, Co, k, s, g, bias, relu = map(int, f[:9]); pad = f[9] if len(f) > 9 else 'same'
    d = os.path.join(root, name); shutil.rmtree(d, ignore_errors=True)
    mil, params, _ = conv_mil([dict(cout=Co, k=k, s=s, g=g, bias=bias, relu=relu, pad=pad)], (C, H, W), d + '/src', seed=sum(map(ord, name)) % 1000)
    r = subprocess.run([MILC, '--target', 'H13', '--mil', mil, '--model-root', d + '/src', '--output', d + '/pkg'], capture_output=True, text=True)
    if r.returncode: print(name, 'COMPILE FAIL', r.stderr.strip()[-300:]); continue
    P = params[0]
    np.savez(os.path.join(d, 'pkg', 'case.npz'), w=P['w'], b=P['b'] if P['b'] is not None else np.zeros(0, np.float16), s=s, g=g, pad=((k-1)//2 if pad == 'same' else 0), relu=relu, inshape=np.array([C, H, W]))
    import json; m = json.load(open(os.path.join(d, 'pkg', 'manifest.json')))
    print(name, 'programs', len(m['programs']), [p['encoder'] for p in m['programs']][:3], 'tasks', [p['taskDescriptors'] for p in m['programs']][:3])
