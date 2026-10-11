#!/bin/bash
# run.sh : M2 ANE ticket. Run programs that mil-hwx-compiler wrote with --target H14 through ane-run --check and report per op.
# Positive control: add (already proven on this chip). New proofs asked for: mul, relu, add-scalar, mul-scalar, real-div-scalar, clip-low, clip-high.
# Rerun-safe; each op is one flock'd ane-run call (<= 60 s); DRY=1 makes no ANE call.
set -u
DRY=${DRY:-0}
ROOT=/var/tmp/m2slot/mil-h14
TOOLS=/var/tmp/m2slot/tree/tools
WORK=/var/tmp/m2slot/work
REC=$ROOT/receipts.log
OUT=$ROOT/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
exec > "$OUT/results.txt" 2>&1
echo "== mil-hwx-compiler H14 programs on the M2 $(date -u +%FT%TZ) dry=$DRY kernel=$(uname -r) module=$(cat /sys/module/ane_t6021/version 2>/dev/null)"
cat "$ROOT/PROVENANCE.txt"
[ -f "$WORK/in-1-0.f16" ] || python3 "$TOOLS/gen-sized-inputs.py" 1 || exit 1
pass=0
fail=0
for op in add mul relu add-scalar mul-scalar real-div-scalar clip-low clip-high; do
	pk=$ROOT/$op/program-0.anec
	[ -f "$pk" ] || { echo "$op: program missing"; fail=$((fail + 1)); continue; }
	case $op in
	add | mul) ins=(--in "0=$WORK/in-1-0.f16" --in "1=$WORK/in-1-1.f16") ;;
	*) ins=(--in "0=$WORK/in-1-0.f16") ;;
	esac
	if [ "$DRY" = 1 ]; then
		echo "DRY: would run ane-run --anec $pk ${ins[*]} --check $op --repeat 20"
		continue
	fi
	echo "-- $op program sha256 $(sha256sum "$pk" | cut -c1-16)"
	flock -w 600 /var/tmp/ane-run.lock timeout 60 "$TOOLS/ane-run" --anec "$pk" "${ins[@]}" \
		--out "0=$OUT/$op.out.f16" --check "$op" --repeat 20
	rc=$?
	echo "rc($op)=$rc"
	if [ "$rc" -eq 0 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
done
echo "SUMMARY pass=$pass fail=$fail dry=$DRY"
echo "$(date -u +%FT%TZ) pass=$pass fail=$fail dry=$DRY out=$OUT" >> "$REC"
[ "$fail" -eq 0 ]
