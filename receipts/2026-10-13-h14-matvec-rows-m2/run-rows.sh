#!/usr/bin/env bash
# H14 matvec with 2, 8 and 64 rows on the M2 Max. 27 programs built by mil-hwx-compiler (M in 2, 8, 64; K and N in 256, 512, 1024), each checked
# with ane-run --check matvec after a 20-call calibration, then a back-to-back loop sized to about TARGET_S seconds from the calibration median.
# A shape whose calibration check fails gets no long loop. Duty = busy_ns delta over wall time from the driver's ane_stats.
set -uo pipefail
ROOT=${ROOT:-/var/tmp/w7h-h14-rows}
RUNNER=${RUNNER:-/var/tmp/m2slot/tree/tools/ane-run}
STATS=${STATS:-/sys/class/accel/accel0/device/ane_stats}
TARGET_S=${TARGET_S:-40}
MAX_REPEAT=3000000
MIN_REPEAT=200
SHAPES=${SHAPES:-"m2k256n256 m2k256n512 m2k256n1024 m2k512n256 m2k512n512 m2k512n1024 m2k1024n256 m2k1024n512 m2k1024n1024 \
m8k256n256 m8k256n512 m8k256n1024 m8k512n256 m8k512n512 m8k512n1024 m8k1024n256 m8k1024n512 m8k1024n1024 \
m64k256n256 m64k256n512 m64k256n1024 m64k512n256 m64k512n512 m64k512n1024 m64k1024n256 m64k1024n512 m64k1024n1024"}
OUT="$ROOT/runs/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$ROOT/runs" && mkdir "$OUT" || exit 2
exec >"$OUT/results.txt" 2>&1
busy() { awk '$1=="busy_ns"{print $2}' "$STATS"; }
printf 'H14 row-variant matvec cell start=%s kernel=%s target_s=%s\n' "$(date -u +%FT%TZ)" "$(uname -r)" "$TARGET_S"
printf 'chip=%s runner_sha256=%s\n' "$(tr -d '\0' </proc/device-tree/compatible 2>/dev/null | cut -c1-40)" "$(sha256sum "$RUNNER" | cut -c1-16)"
b0=$(busy); t0=$(date +%s.%N)
pass=0; fail=0; skipped=0
for shape in $SHAPES; do
  pkg="$ROOT/$shape"
  printf '\n== %s program_sha256=%s ==\n' "$shape" "$(sha256sum "$pkg/out/program-0.anec" | cut -c1-16)"
  common=(--anec "$pkg/out/program-0.anec" --in "0=$pkg/x.f16" --weights "$pkg/w.raw.f16" --check matvec --time)
  timeout 90 "$RUNNER" "${common[@]}" --out "0=$OUT/$shape.cal.y.f16" --repeat 20 >"$OUT/$shape.cal.txt" 2>&1
  rc_cal=$?
  med=$(sed -n 's/.*median \([0-9.]*\).*/\1/p' "$OUT/$shape.cal.txt" | head -1)
  tail -n 3 "$OUT/$shape.cal.txt"
  printf 'rc_cal(%s)=%s median_ms=%s\n' "$shape" "$rc_cal" "${med:-none}"
  if [ "$rc_cal" -ne 0 ] || [ -z "${med:-}" ]; then
    fail=$((fail + 1)); skipped=$((skipped + 1)); printf 'SKIP long loop: calibration check failed\n'; continue
  fi
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
printf '\nSUMMARY pass=%s fail=%s skipped_loops=%s total_wall_s=%s ticket_duty_pct=%s\n' "$pass" "$fail" "$skipped" \
  "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')" \
  "$(awk -v a="$t0" -v b="$t1" -v x="$b0" -v y="$b1" 'BEGIN{printf "%.1f", (y-x)/((b-a)*1e9)*100}')"
printf '%s pass=%s fail=%s out=%s\n' "$(date -u +%FT%TZ)" "$pass" "$fail" "$OUT" >>"$ROOT/receipts.log"
[ "$fail" -eq 0 ]
