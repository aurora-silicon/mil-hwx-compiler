"""Compile every prefix of the MobileNet chain as its own linked program, so
the device can time each layer inside a linked stream (busy(k) - busy(k-1)).

usage: prefixes.py NETDIR OUTDIR    (NETDIR from mobilenet.py)
"""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain import conv_mil, compile_chain, load_params   # noqa: E402
net, out = sys.argv[1], sys.argv[2]
layers, params, inshape = load_params(os.path.join(net, 'params.npz'))
milc = os.environ.get('MILC', 'build/mil-hwxc')
shutil.rmtree(out, ignore_errors=True)
index = []
for k in range(1, len(layers) + 1):
    d = os.path.join(out, f'p{k:02d}')
    mil, _, shape = conv_mil(layers[:k], inshape, os.path.join(d, 'src'), weights=params[:k])
    m = compile_chain(milc, mil, os.path.join(d, 'src'), os.path.join(d, 'pkg'), 'linked' if k > 1 else None)
    L = layers[k - 1]
    index.append(dict(k=k, dir=d, tasks=m['programs'][0]['taskDescriptors'], file=m['programs'][0]['file'],
                      layer=f"{'dw' if L['g'] > 1 else ('pw' if L['k'] == 1 else 'conv')} k{L['k']} s{L['s']} -> {shape}"))
json.dump(index, open(os.path.join(out, 'index.json'), 'w'), indent=1)
print(len(index), 'prefixes')
