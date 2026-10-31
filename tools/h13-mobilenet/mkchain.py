"""Compile a conv chain both as one linked program and as per-layer programs.

usage: mkchain.py OUTDIR C,H,W 'cout,k,s,g,bias,relu;...' [seed]

OUTDIR/linked holds the one-program package, OUTDIR/perop the per-layer
programs (both formula-built, so they must agree bit for bit on the device),
OUTDIR/params.npz the weights and geometry for the reference.
"""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain import conv_mil, parse_layers, compile_chain, save_params   # noqa: E402

out = sys.argv[1]
inshape = tuple(map(int, sys.argv[2].split(',')))
layers = parse_layers(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
milc = os.environ.get('MILC', 'build/mil-hwxc')
shutil.rmtree(out, ignore_errors=True)
mil, params, outshape = conv_mil(layers, inshape, os.path.join(out, 'src'), seed=seed)
save_params(os.path.join(out, 'params.npz'), layers, params, inshape)
m = compile_chain(milc, mil, os.path.join(out, 'src'), os.path.join(out, 'linked'), os.environ.get('LINKED_SCHEDULE', 'linked'))
p = compile_chain(milc, mil, os.path.join(out, 'src'), os.path.join(out, 'perop'), None)
print(f"{out}: linked {m['programs'][0]['taskDescriptors']} tasks, "
      f"{m['programs'][0]['bytes']} bytes, scratch {m['scratch']['bytes']}; "
      f"per-op {len(p['programs'])} programs; output {outshape}")
