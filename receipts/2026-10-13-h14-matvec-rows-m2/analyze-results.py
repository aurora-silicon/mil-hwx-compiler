"""Analyze results.txt of the row-variant run: refusal rule vs outcome, call-time medians per row count, prediction checks, markdown rows."""
import re
import sys
from pathlib import Path

res = Path(sys.argv[1]).read_text().splitlines()
shape = None
info = {}
for ln in res:
    m = re.match(r"== (m\d+k\d+n\d+) ", ln)
    if m:
        shape = m.group(1)
        info[shape] = {}
        continue
    if shape is None:
        continue
    m = re.match(r"rc_cal\((\w+)\)=(\d+) median_ms=([\d.]+)", ln)
    if m:
        info[shape]["rc_cal"] = int(m.group(2))
        info[shape]["cal_median"] = float(m.group(3))
    m = re.match(r"exec ms over (\d+) calls: min ([\d.]+) p10 ([\d.]+) p25 ([\d.]+) median ([\d.]+)", ln)
    if m and int(m.group(1)) > 1000:
        info[shape]["long_calls"] = int(m.group(1))
        info[shape]["long_median"] = float(m.group(5))
        info[shape]["long_min"] = float(m.group(2))
    m = re.match(r"rc_long\((\w+)\)=(\d+) repeat=(\d+) wall_s=([\d.]+) loop_duty_pct=([\d.]+)", ln)
    if m:
        info[shape]["wall"] = float(m.group(4))
        info[shape]["duty"] = float(m.group(5))
        info[shape]["mean_us"] = float(m.group(4)) / int(m.group(3)) * 1e6

tiles = lambda nbytes: -(-nbytes // 16384)  # noqa: E731
mismatch = 0
failed, passed = [], []
for s, d in sorted(info.items(), key=lambda kv: tuple(int(x) for x in re.findall(r"\d+", kv[0]))):
    m, k, n = map(int, re.findall(r"\d+", s))
    rule_fails = tiles(m * k * 2) != tiles(m * n * 2)
    actual_fails = d["rc_cal"] != 0
    mismatch += rule_fails != actual_fails
    (failed if actual_fails else passed).append(s)
    print(f"{s:14} in tiles {tiles(m * k * 2)} out tiles {tiles(m * n * 2)} rule predicts refusal {rule_fails!s:5} actual {actual_fails}")
print(f"RULE (input tiles != output tiles) vs outcome: {mismatch} mismatches over {len(info)} shapes; failed={len(failed)} passed={len(passed)}")
print()
for mm in (2, 8, 64):
    meds = [d["long_median"] for s, d in info.items() if "long_median" in d and int(re.findall(r"\d+", s)[0]) == mm]
    means = [d["mean_us"] for s, d in info.items() if "mean_us" in d and int(re.findall(r"\d+", s)[0]) == mm]
    print(f"M={mm}: {len(meds)} loops, median call ms {min(meds)} to {max(meds)}; mean call us {min(means):.1f} to {max(means):.1f}")
print("duty per loop:", min(d["duty"] for d in info.values() if "duty" in d), "to", max(d["duty"] for d in info.values() if "duty" in d))
for kn in ("k256n256", "k512n512", "k1024n1024"):
    a, b = info[f"m2{kn}"]["long_median"], info[f"m64{kn}"]["long_median"]
    print(f"{kn}: M=2 median {a}, M=64 median {b}, difference {b - a:.3f} ms")
over = [(s, d["long_median"]) for s, d in info.items() if "long_median" in d and not 0.30 <= d["long_median"] <= 0.50]
print("long medians outside 0.30-0.50 ms:", over)
