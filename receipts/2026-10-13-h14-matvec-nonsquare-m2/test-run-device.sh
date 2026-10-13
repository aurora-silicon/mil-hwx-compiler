#!/usr/bin/env bash
# Offline test of run-device.sh with a fake runner and fake ane_stats. Touches no device and no real directory.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/h14nonsq-test.XXXXXX)
fails=0
ok()  { echo "PASS $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
for s in k256n512 k256n1024 k512n256 k512n1024 k1024n256 k1024n512; do
  mkdir -p "$T/root/$s/out"; echo prog > "$T/root/$s/out/program-0.anec"; echo x > "$T/root/$s/x.f16"; echo w > "$T/root/$s/w.raw.f16"
done
echo "busy_ns 1000" > "$T/stats"
# fake runner: median 2.0 ms; calibration (--repeat 20) rc 0; FAIL_SHAPE makes that shape's calibration rc 1; log the repeat
cat > "$T/runner" <<'EOF'
#!/usr/bin/env bash
rep=0; prog=""
while [ $# -gt 0 ]; do case "$1" in --repeat) rep=$2; shift;; --anec) prog=$2; shift;; esac; shift; done
echo "repeat=$rep prog=$prog" >> "$FAKE_LOG"
case "$prog" in *"$FAIL_SHAPE"*) [ "$rep" = 20 ] && { echo "boom"; exit 1; };; esac
echo "exec ms over $rep calls: min 1.0 p10 1.0 p25 1.0 median 2.000 p75 2.0 p90 2.0 p99 2.0 max 3.0"
echo "matvec M=1 K=256 N=512: 512/512 lanes within 2 ulp or 4 cond-units (480 bit-exact)"
exit 0
EOF
chmod +x "$T/runner"
run() { FAKE_LOG="$T/log" FAIL_SHAPE="${1:-none}" ROOT="$T/root" RUNNER="$T/runner" STATS="$T/stats" TARGET_S="${2:-110}" bash "$HERE/run-device.sh"; }

: > "$T/log"; run none 110; rc=$?
[ "$rc" -eq 0 ] && ok "all six shapes pass, rc 0" || bad "rc=$rc"
out=$(ls -d "$T"/root/runs/run-*/ | head -1)
grep -q "SUMMARY pass=6 fail=0" "$out/results.txt" && ok "summary pass=6 fail=0" || bad "summary line"
# 110 s / 2.0 ms = 55000 calls
[ "$(grep -c 'repeat=55000 ' "$T/log")" -eq 6 ] && ok "long loop sized to 55000 calls (110 s at 2.0 ms) for each shape" || bad "repeat sizing: $(grep -c 'repeat=55000 ' "$T/log")"
[ "$(grep -c 'repeat=20 ' "$T/log")" -eq 6 ] && ok "six 20-call calibrations" || bad "calibration count"
grep -q "ticket_duty_pct=" "$out/results.txt" && ok "duty line present" || bad "duty line"
[ -s "$T/root/receipts.log" ] && ok "receipt log written" || bad "receipts.log"

: > "$T/log"; run k512n256 110; rc=$?
[ "$rc" -ne 0 ] && ok "a failed calibration makes the cell fail (rc $rc)" || bad "failed calibration still rc 0"
out2=$(ls -t -d "$T"/root/runs/run-*/ | head -1)
[ "$out2" != "$out" ] && ok "a run started in the same second gets its own output directory" || bad "second run reused $out"
grep -q "SUMMARY pass=6 fail=0" "$out/results.txt" && ok "the first run's results were not overwritten" || bad "first run results overwritten"
grep -q "SUMMARY pass=5 fail=1" "$out2/results.txt" && ok "summary pass=5 fail=1" || bad "summary after failure"
[ "$(grep -c 'k512n256.*repeat=[0-9][0-9][0-9][0-9][0-9]' "$T/log")" -eq 0 ] && ok "no long loop for the shape whose calibration failed" || bad "long loop ran after failed calibration"

: > "$T/log"; TARGET_S=0 run none 0 >/dev/null; 
grep -q 'repeat=200 ' "$T/log" && ok "repeat floor of 200 holds" || bad "floor"
[ "$(ls -d "$T"/root/runs/run-*/ | wc -l)" -eq 3 ] && ok "three back-to-back runs left three directories" || bad "run directory count"
rm -rf "$T" 2>/dev/null
echo "failures: $fails"; exit $((fails > 0))
