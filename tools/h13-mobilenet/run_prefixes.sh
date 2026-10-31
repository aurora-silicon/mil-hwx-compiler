#!/bin/bash
# usage: run_prefixes.sh PREFIXDIR [repeat]   (on the device)
d=$1; r=${2:-50}; here=$(dirname "$0")
python3 - "$d" "$r" "$here/anechain" <<'PY'
import json, subprocess, sys
d, r, exe = sys.argv[1], sys.argv[2], sys.argv[3]
idx = json.load(open(f'{d}/index.json'))
prev = 0.0
for e in idx:
    out = subprocess.run([exe, '-n', '-r', r, f"{e['dir']}/pkg/{e['file']}"], capture_output=True, text=True)
    j = json.loads(out.stdout.strip().splitlines()[-1])
    w = open('/sys/class/accel/accel0/device/wedged').read().strip()
    print(f"{e['k']:2d} {e['layer']:<40} tasks {e['tasks']:3d} p50 {j['p50_us']:8.1f} busy {j['busy_us_per_pass']:8.1f} layer +{j['busy_us_per_pass']-prev:7.1f} wedged {w}")
    prev = j['busy_us_per_pass']
PY
