#!/usr/bin/env bash
# H14 non-square M=1 matvec cell on the M2 Max. Six programs built by mil-hwx-compiler, each checked with
# ane-run --check matvec after a short calibration (20 calls) and a long loop sized to about TARGET_S seconds.
# Duty = busy_ns delta over wall for the whole script, read from the driver's ane_stats (same definition as w73's cell-duty.py).
set -uo pipefail
ROOT=${ROOT:-/var/tmp/w7h-h14-nonsq}
RUNNER=${RUNNER:-/var/tmp/m2slot/tree/tools/ane-run}
STATS=${STATS:-/sys/class/accel/accel0/device/ane_stats}
TARGET_S=${TARGET_S:-110}
MAX_REPEAT=3000000
MIN_REPEAT=200
OUT="$ROOT/runs/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$ROOT/runs" && mkdir "$OUT" || exit 2
exec >"$OUT/results.txt" 2>&1
busy() { awk '$1=="busy_ns"{print $2}' "$STATS"; }
printf 'H14 non-square matvec cell start=%s kernel=%s target_s=%s\n' "$(date -u +%FT%TZ)" "$(uname -r)" "$TARGET_S"
printf 'chip=%s runner_sha256=%s\n' "$(tr -d '\0' </proc/device-tree/compatible 2>/dev/null | cut -c1-40)" "$(sha256sum "$RUNNER" | cut -c1-16)"
b0=$(busy); t0=$(date +%s.%N)
pass=0; fail=0
for shape in k256n512 k256n1024 k512n256 k512n1024 k1024n256 k1024n512; do
  pkg="$ROOT/$shape"
  printf '\n== %s program_sha256=%s ==\n' "$shape" "$(sha256sum "$pkg/out/program-0.anec" | cut -c1-16)"
  common=(--anec "$pkg/out/program-0.anec" --in "0=$pkg/x.f16" --weights "$pkg/w.raw.f16" --check matvec --time)
  timeout 90 "$RUNNER" "${common[@]}" --out "0=$OUT/$shape.cal.y.f16" --repeat 20 >"$OUT/$shape.cal.txt" 2>&1
  rc_cal=$?
  med=$(sed -n 's/.*median \([0-9.]*\).*/\1/p' "$OUT/$shape.cal.txt" | head -1)
  tail -n 3 "$OUT/$shape.cal.txt"
  printf 'rc_cal(%s)=%s median_ms=%s\n' "$shape" "$rc_cal" "${med:-none}"
  if [ "$rc_cal" -ne 0 ] || [ -z "${med:-}" ]; then fail=$((fail + 1)); printf 'SKIP long loop: calibration failed\n'; continue; fi
  rep=$(awk -v t="$TARGET_S" -v m="$med" -v lo="$MIN_REPEAT" -v hi="$MAX_REPEAT" 'BEGIN{r=int(t*1000/m); if(r<lo)r=lo; if(r>hi)r=hi; print r}')
  s0=$(date +%s.%N); bs0=$(busy)
  timeout $((TARGET_S * 3 + 90)) "$RUNNER" "${common[@]}" --out "0=$OUT/$shape.long.y.f16" --repeat "$rep" >"$OUT/$shape.long.txt" 2>&1
  rc_long=$?
  s1=$(date +%s.%N); bs1=$(busy)
  tail -n 3 "$OUT/$shape.long.txt"
  printf 'rc_long(%s)=%s repeat=%s wall_s=%s loop_duty_pct=%s\n' "$shape" "$rc_long" "$rep" \
    "$(awk -v a="$s0" -v b="$s1" 'BEGIN{printf "%.1f", b-a}')" \
    "$(awk -v a="$s0" -v b="$s1" -v x="$bs0" -v y="$bs1" 'BEGIN{printf "%.1f", (y-x)/((b-a)*1e9)*100}')"
  if [ "$rc_long" -eq 0 ]; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
done
b1=$(busy); t1=$(date +%s.%N)
printf '\nSUMMARY pass=%s fail=%s total_wall_s=%s ticket_duty_pct=%s\n' "$pass" "$fail" \
  "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')" \
  "$(awk -v a="$t0" -v b="$t1" -v x="$b0" -v y="$b1" 'BEGIN{printf "%.1f", (y-x)/((b-a)*1e9)*100}')"
printf '%s pass=%s fail=%s out=%s\n' "$(date -u +%FT%TZ)" "$pass" "$fail" "$OUT" >>"$ROOT/receipts.log"
[ "$fail" -eq 0 ]
