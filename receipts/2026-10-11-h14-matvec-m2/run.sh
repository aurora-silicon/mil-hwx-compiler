#!/bin/bash
# run.sh : M2 ANE ticket. fp16 matvec programs built by mil-hwx-compiler (--target H14), run through ane-run --check matvec.
# K x N = 256x256 (the fixture shape: the first 4672 bytes of the program equal the checked-in fixture), 512x512, 1024x1024.
# --check matvec accepts a 2 ulp band (the device accumulation order is unknown) and prints the largest ulp difference.
# Rerun-safe; one flock'd ane-run call per size (<= 90 s); DRY=1 makes no ANE call.
set -u
DRY=${DRY:-0}
ROOT=/var/tmp/m2slot/mil-h14-matvec
TOOLS=/var/tmp/m2slot/tree/tools
REC=$ROOT/receipts.log
OUT=$ROOT/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
exec > "$OUT/results.txt" 2>&1
echo "== mil-hwx-compiler H14 matvec programs on the M2 $(date -u +%FT%TZ) dry=$DRY kernel=$(uname -r) module=$(cat /sys/module/ane_t6021/version 2>/dev/null)"
cat "$ROOT/PROVENANCE.txt"
pass=0
fail=0
for s in k256n256 k512n512 k1024n1024; do
	d=$ROOT/$s
	[ -f "$d/out/program-0.anec" ] || { echo "$s: program missing"; fail=$((fail + 1)); continue; }
	if [ "$DRY" = 1 ]; then
		echo "DRY: would run ane-run --anec $d/out/program-0.anec --in 0=$d/x.f16 --out 0=$OUT/$s.y.f16 --weights $d/w.raw.f16 --check matvec --repeat 20"
		continue
	fi
	echo "-- $s program sha256 $(sha256sum "$d/out/program-0.anec" | cut -c1-16)"
	flock -w 600 /var/tmp/ane-run.lock timeout 90 "$TOOLS/ane-run" --anec "$d/out/program-0.anec" --in "0=$d/x.f16" \
		--out "0=$OUT/$s.y.f16" --weights "$d/w.raw.f16" --check matvec --repeat 20
	rc=$?
	echo "rc($s)=$rc"
	if [ "$rc" -eq 0 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
done
echo "SUMMARY pass=$pass fail=$fail dry=$DRY"
echo "$(date -u +%FT%TZ) pass=$pass fail=$fail dry=$DRY out=$OUT" >> "$REC"
[ "$fail" -eq 0 ]
