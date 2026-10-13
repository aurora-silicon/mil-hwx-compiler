#!/usr/bin/env bash
# Offline test of run-unary.sh with a fake runner and fake ane_stats. Touches no device and no real directory.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/h14unary-test.XXXXXX)
fails=0
ok()  { echo "PASS $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
ALL="unary_abs_c64 unary_abs_c128 unary_abs_c256 unary_abs_c512 unary_abs_c1024 unary_abs_c2048 unary_abs_c4096 unary_exp_c64 unary_exp_c512 unary_gelu_c64 unary_gelu_c512 unary_leaky_relu_c64 unary_leaky_relu_c512 unary_rsqrt_c64 unary_rsqrt_c512 unary_sigmoid_c64 unary_sigmoid_c512 unary_silu_c64 unary_silu_c512 unary_sqrt_c64 unary_sqrt_c512 unary_tanh_c64 unary_tanh_c512"
for s in $ALL; do mkdir -p "$T/root/$s/out"; echo prog > "$T/root/$s/out/program-0.anec"; echo x > "$T/root/$s/x.f16"; done
echo "busy_ns 1000" > "$T/stats"
cat > "$T/runner" <<'EOF'
#!/usr/bin/env bash
rep=0; prog=""; outfile=""
while [ $# -gt 0 ]; do case "$1" in --repeat) rep=$2; shift;; --anec) prog=$2; shift;; --out) outfile=${2#0=}; shift;; esac; shift; done
echo "repeat=$rep prog=$prog" >> "$FAKE_LOG"
case "$prog" in *"$FAIL_SHAPE"*) [ "$rep" = 20 ] && { echo "boom"; exit 1; };; esac
echo "output" > "$outfile"
echo "exec ms over $rep calls: min 1.0 p10 1.0 p25 1.0 median 2.000 p75 2.0 p90 2.0 p99 2.0 max 3.0"
exit 0
EOF
chmod +x "$T/runner"
run() { FAKE_LOG="$T/log" FAIL_SHAPE="${1:-none}" ROOT="$T/root" RUNNER="$T/runner" STATS="$T/stats" TARGET_S="${2:-30}" bash "$HERE/run-unary.sh"; }
latest() { find "$T/root/runs" -maxdepth 1 -name 'run-*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-; }

: > "$T/log"; run none 30; rc=$?
out1=$(latest)
[ "$rc" -eq 0 ] && ok "all 23 programs run, rc 0" || bad "rc=$rc"
grep -q "SUMMARY pass=23 fail=0 skipped_loops=0" "$out1/results.txt" && ok "summary pass=23 fail=0" || bad "summary line"
# 30 s / 2.0 ms = 15000 calls
[ "$(grep -c 'repeat=15000 ' "$T/log")" -eq 23 ] && ok "long loop sized to 15000 calls for each of the 23 programs" || bad "repeat sizing: $(grep -c 'repeat=15000 ' "$T/log")"
[ "$(grep -c 'repeat=20 ' "$T/log")" -eq 23 ] && ok "23 twenty-call calibrations" || bad "calibration count"
[ "$(find "$out1" -name '*.cal.y.f16' | wc -l)" -eq 23 ] && [ "$(find "$out1" -name '*.long.y.f16' | wc -l)" -eq 23 ] && ok "both output files kept for every program" || bad "output files"
grep -q "ticket_duty_pct=" "$out1/results.txt" && ok "duty line present" || bad "duty line"
[ -s "$T/root/receipts.log" ] && ok "receipt log written" || bad "receipts.log"

: > "$T/log"; run unary_silu_c512 30; rc=$?
out2=$(latest)
[ "$rc" -ne 0 ] && ok "a failed calibration makes the cell fail (rc $rc)" || bad "failed calibration still rc 0"
[ "$out2" != "$out1" ] && ok "a run started in the same second gets its own directory" || bad "second run reused the directory"
grep -q "SUMMARY pass=23 fail=0" "$out1/results.txt" && ok "the first run's results were not overwritten" || bad "first run results overwritten"
grep -q "SUMMARY pass=22 fail=1 skipped_loops=1" "$out2/results.txt" && ok "summary pass=22 fail=1 skipped_loops=1" || bad "summary after failure"
[ "$(grep -c 'unary_silu_c512.*repeat=1[0-9][0-9][0-9][0-9]' "$T/log")" -eq 0 ] && ok "no long loop for the program whose calibration failed" || bad "long loop ran after failed calibration"

: > "$T/log"; CASES="unary_abs_c64" run none 0 >/dev/null
grep -q 'repeat=200 ' "$T/log" && ok "repeat floor of 200 holds" || bad "floor"
rm -rf "$T" 2>/dev/null
echo "failures: $fails"; exit $((fails > 0))
