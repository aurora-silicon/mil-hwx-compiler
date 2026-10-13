#!/usr/bin/env bash
# H14 unary programs on the M2 Max (abs, exp, gelu, leaky_relu, rsqrt, sigmoid, silu, sqrt, tanh; 23 programs built by mil-hwx-compiler).
# Per program: a 20-call calibration with --time, then a back-to-back loop sized to about TARGET_S seconds from the calibration median.
# The output of both runs is kept for compare-unary.py, which judges it off the device. A program whose calibration fails gets no long loop.
# Duty = busy_ns delta over wall time from the driver's ane_stats.
set -uo pipefail
ROOT=${ROOT:-/var/tmp/w7h-h14-unary}
RUNNER=${RUNNER:-/var/tmp/m2slot/tree/tools/ane-run}
STATS=${STATS:-/sys/class/accel/accel0/device/ane_stats}
TARGET_S=${TARGET_S:-30}
MAX_REPEAT=3000000
MIN_REPEAT=200
CASES=${CASES:-"unary_abs_c64 unary_abs_c128 unary_abs_c256 unary_abs_c512 unary_abs_c1024 unary_abs_c2048 unary_abs_c4096 \
unary_exp_c64 unary_exp_c512 unary_gelu_c64 unary_gelu_c512 unary_leaky_relu_c64 unary_leaky_relu_c512 \
unary_rsqrt_c64 unary_rsqrt_c512 unary_sigmoid_c64 unary_sigmoid_c512 unary_silu_c64 unary_silu_c512 \
unary_sqrt_c64 unary_sqrt_c512 unary_tanh_c64 unary_tanh_c512"}
OUT="$ROOT/runs/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$ROOT/runs" && mkdir "$OUT" || exit 2
exec >"$OUT/results.txt" 2>&1
busy() { awk '$1=="busy_ns"{print $2}' "$STATS"; }
printf 'H14 unary cell start=%s kernel=%s target_s=%s\n' "$(date -u +%FT%TZ)" "$(uname -r)" "$TARGET_S"
printf 'chip=%s runner_sha256=%s\n' "$(tr -d '\0' </proc/device-tree/compatible 2>/dev/null | cut -c1-40)" "$(sha256sum "$RUNNER" | cut -c1-16)"
b0=$(busy); t0=$(date +%s.%N)
pass=0; fail=0; skipped=0
for case in $CASES; do
  pkg="$ROOT/$case"
  printf '\n== %s program_sha256=%s ==\n' "$case" "$(sha256sum "$pkg/out/program-0.anec" | cut -c1-16)"
  common=(--anec "$pkg/out/program-0.anec" --in "0=$pkg/x.f16" --time)
  timeout 90 "$RUNNER" "${common[@]}" --out "0=$OUT/$case.cal.y.f16" --repeat 20 >"$OUT/$case.cal.txt" 2>&1
  rc_cal=$?
  med=$(sed -n 's/.*median \([0-9.]*\).*/\1/p' "$OUT/$case.cal.txt" | head -1)
  tail -n 2 "$OUT/$case.cal.txt"
  printf 'rc_cal(%s)=%s median_ms=%s\n' "$case" "$rc_cal" "${med:-none}"
  if [ "$rc_cal" -ne 0 ] || [ -z "${med:-}" ]; then
    fail=$((fail + 1)); skipped=$((skipped + 1)); printf 'SKIP long loop: calibration run failed\n'; continue
  fi
  rep=$(awk -v t="$TARGET_S" -v m="$med" -v lo="$MIN_REPEAT" -v hi="$MAX_REPEAT" 'BEGIN{r=int(t*1000/m); if(r<lo)r=lo; if(r>hi)r=hi; print r}')
  s0=$(date +%s.%N); bs0=$(busy)
  timeout $((TARGET_S * 3 + 90)) "$RUNNER" "${common[@]}" --out "0=$OUT/$case.long.y.f16" --repeat "$rep" >"$OUT/$case.long.txt" 2>&1
  rc_long=$?
  s1=$(date +%s.%N); bs1=$(busy)
  tail -n 2 "$OUT/$case.long.txt"
  printf 'rc_long(%s)=%s repeat=%s wall_s=%s loop_duty_pct=%s\n' "$case" "$rc_long" "$rep" \
    "$(awk -v a="$s0" -v b="$s1" 'BEGIN{printf "%.1f", b-a}')" \
    "$(awk -v a="$s0" -v b="$s1" -v x="$bs0" -v y="$bs1" 'BEGIN{printf "%.1f", (y-x)/((b-a)*1e9)*100}')"
  if [ "$rc_long" -eq 0 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
done
b1=$(busy); t1=$(date +%s.%N)
printf '\nSUMMARY pass=%s fail=%s skipped_loops=%s total_wall_s=%s ticket_duty_pct=%s\n' "$pass" "$fail" "$skipped" \
  "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')" \
  "$(awk -v a="$t0" -v b="$t1" -v x="$b0" -v y="$b1" 'BEGIN{printf "%.1f", (y-x)/((b-a)*1e9)*100}')"
printf '%s pass=%s fail=%s out=%s\n' "$(date -u +%FT%TZ)" "$pass" "$fail" "$OUT" >>"$ROOT/receipts.log"
[ "$fail" -eq 0 ]
