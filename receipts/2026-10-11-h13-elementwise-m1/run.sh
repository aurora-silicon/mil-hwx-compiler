#!/bin/bash
# run.sh : H13 ANE ticket. The 8 elementwise programs mil-hwx-compiler writes with --target H13 ([1,64,1,1] fp16), each run through
# ane-run and checked on all 64 lanes against an exact reference (ties away from zero, exact compares), padding lanes zero.
# Control for the compiler: add and the others come from the newer 'h13-source-qualified' and the older 'h13-oracle-parity' encoders.
# Rerun-safe; one flock'd ane-run per op (<= 60 s). DRY=1 makes no ANE call. RUNNER overrides the ane-run binary.
set -u
DRY=${DRY:-0}
ROOT=/var/tmp/m1slot/mil-h13
RUNNER=${RUNNER:-/var/tmp/m1slot/tree/tools/ane-run}
REC=$ROOT/receipts.log
OUT=$ROOT/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
exec > "$OUT/results.txt" 2>&1
echo "== mil-hwx-compiler H13 programs $(date -u +%FT%TZ) dry=$DRY kernel=$(uname -r) chip_dt=$(tr -d '\0' < /proc/device-tree/compatible 2>/dev/null | cut -c1-40)"
echo "runner $RUNNER sha256 $(sha256sum "$RUNNER" | cut -c1-16)"
cat "$ROOT/PROVENANCE.txt"
if [ "$DRY" = 1 ]; then
	echo "DRY: would run python3 $ROOT/gate_h13.py $ROOT $RUNNER $OUT --repeat 5"
	python3 -c "import numpy; print('numpy', numpy.__version__)"
	echo "$(date -u +%FT%TZ) rc=0 dry=1 out=$OUT" >> "$REC"
	exit 0
fi
python3 "$ROOT/gate_h13.py" "$ROOT" "$RUNNER" "$OUT" --repeat 5
rc=$?
echo "gate rc=$rc"
echo "$(date -u +%FT%TZ) rc=$rc dry=$DRY out=$OUT $(grep -m1 SUMMARY "$OUT/results.txt")" >> "$REC"
exit "$rc"
