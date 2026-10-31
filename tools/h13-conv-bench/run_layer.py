"""Run a compiled single-program conv package on the device; compare with a float64 CPU reference.
usage: run_layer.py PKGDIR  (PKGDIR/case.npz holds w, b, k, s, g, pad, relu, inshape)"""
import json, os, subprocess, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surf import *
def conv_ref(x, w, b, s, g, pad, relu):
    C, H, W = x.shape; Co, cig, k, _ = w.shape
    xp = np.pad(x.astype(np.float64), ((0,0),(pad,pad),(pad,pad)))
    Ho = (H + 2*pad - k)//s + 1; Wo = (W + 2*pad - k)//s + 1
    out = np.zeros((Co, Ho, Wo)); cog = Co // g; wd = w.astype(np.float64)
    for gi in range(g):
        xs = xp[gi*cig:(gi+1)*cig]; ws = wd[gi*cog:(gi+1)*cog]
        for dy in range(k):
            for dx in range(k):
                patch = xs[:, dy:dy+s*Ho:s, dx:dx+s*Wo:s]
                if g == 1: out += np.einsum('oc,chw->ohw', ws[:,:,dy,dx], patch, optimize=True)
                else: out[gi*cog:(gi+1)*cog] += ws[:,:,dy,dx][:, :, None, None].sum(1) * patch if cig == 1 else np.einsum('oc,chw->ohw', ws[:,:,dy,dx], patch)
    if b is not None: out += b.astype(np.float64)[:, None, None]
    return np.maximum(out, 0) if relu else out
def cpu_time(x, w, b, s, g, pad, relu, reps=20):
    """fp32 numpy timing of the same layer (im2col-free einsum for 1x1, loop for kxk)."""
    xf = x.astype(np.float32); wf = w.astype(np.float32)
    t = []
    for _ in range(reps):
        a = time.perf_counter()
        if w.shape[2] == 1 and g == 1:
            y = np.tensordot(wf[:,:,0,0], xf[:, ::s, ::s], axes=(1, 0))
        else:
            C, H, W = xf.shape; k = w.shape[2]; xp = np.pad(xf, ((0,0),(pad,pad),(pad,pad)))
            Ho = (H + 2*pad - k)//s + 1; Wo = (W + 2*pad - k)//s + 1
            if g == 1:
                cols = np.stack([xp[:, dy:dy+s*Ho:s, dx:dx+s*Wo:s] for dy in range(k) for dx in range(k)], 1).reshape(C*k*k, Ho*Wo)
                y = (wf.reshape(w.shape[0], -1) @ cols).reshape(-1, Ho, Wo)
            else:
                y = np.zeros((C, Ho, Wo), np.float32)
                for dy in range(k):
                    for dx in range(k): y += wf[:, 0, dy, dx][:, None, None] * xp[:, dy:dy+s*Ho:s, dx:dx+s*Wo:s]
        if b is not None: y += b.astype(np.float32)[:, None, None]
        if relu: np.maximum(y, 0, out=y)
        t.append(time.perf_counter() - a)
    return float(np.median(t)) * 1e6
if __name__ == '__main__':
    pkg = sys.argv[1]; repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    z = np.load(os.path.join(pkg, 'case.npz'), allow_pickle=True)
    w = z['w']; b = z['b'] if z['b'].size else None
    s, g, pad, relu = int(z['s']), int(z['g']), int(z['pad']), bool(z['relu'])
    C, H, W = [int(v) for v in z['inshape']]
    m = json.load(open(os.path.join(pkg, 'manifest.json')))
    assert len(m['programs']) == 1
    p = m['programs'][0]; inb = p['inputs'][0]; outb = p['outputs'][0]
    x = np.random.default_rng(7).uniform(-1, 1, (C, H, W)).astype(np.float16)
    ref = f16_ties_away(conv_ref(x, w, b, s, g, pad, relu))
    tmp = os.environ.get('ANEBENCH_TMP', '/var/tmp/h13-conv-bench'); os.makedirs(tmp, exist_ok=True)
    open(f'{tmp}/in0.bin', 'wb').write(pack(x, inb['nchw'], inb['allocationBytes']))
    r = subprocess.run([os.environ.get('ANEBENCH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'anebench')), '-r', str(repeat), '-i', f'0:{tmp}/in0.bin', '-o', f'0:{tmp}/out0.bin', os.path.join(pkg, p['file'])], capture_output=True, text=True, timeout=120)
    if r.returncode: print(json.dumps(dict(pkg=pkg, error=(r.stdout + r.stderr).strip()[-400:]))); sys.exit(1)
    timing = json.loads(r.stdout.strip().splitlines()[-1])
    dev = unpack(open(f'{tmp}/out0.bin', 'rb').read(), outb['nchw'])
    st = stats(dev, ref)
    Co, cig, k, _ = w.shape; Ho, Wo = dev.shape[1:]
    macs = Co * Ho * Wo * cig * k * k
    print(json.dumps(dict(pkg=os.path.basename(pkg.rstrip('/')), layer=f'{C}x{H}x{W}->{Co}x{Ho}x{Wo} k{k} s{s} g{g} bias={b is not None} relu={relu}',
        encoder=p['encoder'], tasks=p['taskDescriptors'], exact_pct=st['exact_pct'], within1ulp_pct=st['within1ulp_pct'], max_ulp=st['max_ulp'], max_abs=st['max_abs'], rel_l2=st['rel_l2'], nonfinite=st['nonfinite'],
        ane_p50_us=timing['p50_us'], ane_busy_us=timing['busy_us_per_exec'], cpu_fp32_us=round(cpu_time(x, w, b, s, g, pad, relu), 1), gmac_s=round(macs / timing['p50_us'] / 1e3, 1))))
