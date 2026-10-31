"""Dense NCHW fp16 <-> H13 surface helpers and comparison stats."""
import json, sys, subprocess, numpy as np
def pack(dense, nchw, alloc):
    n,c,h,w,plane,row = nchw
    buf = np.zeros(alloc, dtype=np.uint8)
    d = dense.astype(np.float16).reshape(c,h,w)
    for ci in range(c):
        for y in range(h):
            o = ci*plane + y*row
            buf[o:o+2*w] = d[ci,y].view(np.uint8)
    return buf.tobytes()
def unpack(raw, nchw):
    n,c,h,w,plane,row = nchw
    a = np.frombuffer(raw, dtype=np.uint8)
    out = np.zeros((c,h,w), dtype=np.float16)
    for ci in range(c):
        for y in range(h):
            o = ci*plane + y*row
            out[ci,y] = a[o:o+2*w].view(np.float16)
    return out
def f16_ties_away(x64):
    """Round float64 to fp16 with ties away from zero (the ANE's rounding)."""
    x64 = np.asarray(x64, dtype=np.float64)
    r = x64.astype(np.float16)  # ties-to-even
    # find ties: x exactly halfway between two fp16 neighbours
    lo = np.nextafter(r, np.float16(-np.inf)); hi = np.nextafter(r, np.float16(np.inf))
    for nb in (lo, hi):
        mid = (r.astype(np.float64) + nb.astype(np.float64)) / 2
        tie = (mid == x64) & np.isfinite(nb)
        away = np.abs(nb.astype(np.float64)) > np.abs(r.astype(np.float64))
        r = np.where(tie & away, nb, r)
    return r
def stats(dev, ref):
    a = dev.astype(np.float16).ravel(); e = ref.astype(np.float16).ravel()
    ai = a.view(np.int16).astype(np.int32); ei = e.view(np.int16).astype(np.int32)
    # map to monotone ordering for ulp distance
    ai = np.where(ai < 0, -32768 - ai, ai); ei = np.where(ei < 0, -32768 - ei, ei)
    ulp = np.abs(ai - ei)
    eq = (a == e)  # +0 == -0
    af, ef = a.astype(np.float64), e.astype(np.float64)
    return dict(n=int(a.size), exact=int(eq.sum()), exact_pct=round(100*eq.mean(),3),
                within1ulp_pct=round(100*(eq | (ulp<=1)).mean(),3), max_ulp=int(np.where(eq,0,ulp).max()),
                max_abs=float(np.abs(af-ef).max()), rel_l2=float(np.linalg.norm(af-ef)/max(np.linalg.norm(ef),1e-30)),
                nonfinite=int((~np.isfinite(af)).sum()))
