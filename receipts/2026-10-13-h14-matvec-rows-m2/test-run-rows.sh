#!/usr/bin/env bash
# Offline test of run-rows.sh with a fake runner and fake ane_stats. Touches no device and no real directory.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/h14rows-test.XXXXXX)
fails=0
ok()  { echo "PASS $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
for m in 2 8 64; do for k in 256 512 1024; do for n in 256 512 1024; do
  s="m${m}k${k}n${n}"; mkdir -p "$T/root/$s/out"; echo prog > "$T/root/$s/out/program-0.anec"; echo x > "$T/root/$s/x.f16"; echo w > "$T/root/$s/w.raw.f16"
done; done; done
echo "busy_ns 1000" > "$T/stats"
cat > "$T/runner" <<'EOF'
#!/usr/bin/env bash
rep=0; prog=""
while [ $# -gt 0 ]; do case "$1" in --repeat) rep=$2; shift;; --anec) prog=$2; shift;; esac; shift; done
echo "repeat=$rep prog=$prog" >> "$FAKE_LOG"
case "$prog" in *"$FAIL_SHAPE"*) [ "$rep" = 20 ] && { echo "boom"; exit 1; };; esac
echo "exec ms over $rep calls: min 1.0 p10 1.0 p25 1.0 median 2.000 p75 2.0 p90 2.0 p99 2.0 max 3.0"
echo "matvec: all lanes within 2 ulp or 4 cond-units: PASS"
exit 0
EOF
chmod +x "$T/runner"
run() { FAKE_LOG="$T/log" FAIL_SHAPE="${1:-none}" ROOT="$T/root" RUNNER="$T/runner" STATS="$T/stats" TARGET_S="${2:-40}" bash "$HERE/run-rows.sh"; }
latest() { find "$T/root/runs" -maxdepth 1 -name 'run-*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-; }

: > "$T/log"; run none 40; rc=$?
out1=$(latest)
[ "$rc" -eq 0 ] && ok "all 27 shapes pass, rc 0" || bad "rc=$rc"
grep -q "SUMMARY pass=27 fail=0 skipped_loops=0" "$out1/results.txt" && ok "summary pass=27 fail=0" || bad "summary line"
# 40 s / 2.0 ms = 20000 calls
[ "$(grep -c 'repeat=20000 ' "$T/log")" -eq 27 ] && ok "long loop sized to 20000 calls for each of the 27 shapes" || bad "repeat sizing: $(grep -c 'repeat=20000 ' "$T/log")"
[ "$(grep -c 'repeat=20 ' "$T/log")" -eq 27 ] && ok "27 twenty-call calibrations" || bad "calibration count"
grep -q "ticket_duty_pct=" "$out1/results.txt" && ok "duty line present" || bad "duty line"

: > "$T/log"; run m8k512n256 40; rc=$?
out2=$(latest)
[ "$rc" -ne 0 ] && ok "a failed calibration makes the cell fail (rc $rc)" || bad "failed calibration still rc 0"
[ "$out2" != "$out1" ] && ok "a run started in the same second gets its own directory" || bad "second run reused the directory"
grep -q "SUMMARY pass=27 fail=0" "$out1/results.txt" && ok "the first run's results were not overwritten" || bad "first run results overwritten"
grep -q "SUMMARY pass=26 fail=1 skipped_loops=1" "$out2/results.txt" && ok "summary pass=26 fail=1 skipped_loops=1" || bad "summary after failure"
[ "$(grep -c 'm8k512n256.*repeat=2[0-9][0-9][0-9][0-9]' "$T/log")" -eq 0 ] && ok "no long loop for the shape whose calibration failed" || bad "long loop ran after failed calibration"
[ "$(find "$T/root/runs" -maxdepth 1 -name 'run-*' | wc -l)" -eq 2 ] && ok "two back-to-back runs left two directories" || bad "run directory count"

: > "$T/log"; SHAPES="m2k256n256" run none 0 >/dev/null
grep -q 'repeat=200 ' "$T/log" && ok "repeat floor of 200 holds" || bad "floor"
rm -rf "$T" 2>/dev/null
echo "failures: $fails"; exit $((fails > 0))
