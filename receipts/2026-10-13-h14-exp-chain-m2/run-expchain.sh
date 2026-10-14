#!/usr/bin/env bash
# H14 exp with the scale correction, as four existing programs chained on the host (64 and 512 channels), on the M2 Max.
#   s1 e = exp(x)      unary exp program, input x
#   s2 c = x * K       elementwise mul program, inputs x and K (every lane holds the same fp16 constant)
#   s3 d = e * c       elementwise mul program, inputs e and c
#   s4 y = e + d       elementwise add program, inputs e and d
# The host only copies bytes: the output file of one stage is the input file of the next, unchanged. Each stage is one ane-run call.
# Per stage: a 20-call calibration with --time, then a back-to-back loop sized to about TARGET_S seconds from the calibration median.
# The whole run stops at DEADLINE_S (900 s): every timeout is capped to the time left. The next stage reads the calibration output of the earlier ones. A stage whose calibration or long loop fails stops the chain for that size.
# compare-expchain.py judges the saved output off the device, and checks the chip, the program hashes, the output hashes and that no other ane-run ran. Duty = busy_ns delta over wall time from the driver's ane_stats.
set -uo pipefail
ROOT=${ROOT:-/var/tmp/w7h-h14-expchain}
RUNNER=${RUNNER:-/var/tmp/m2slot/tree/tools/ane-run}
STATS=${STATS:-/sys/class/accel/accel0/device/ane_stats}
TARGET_S=${TARGET_S:-30}
DEADLINE_S=${DEADLINE_S:-900}  # the whole-run limit of the preregistration (15 minutes): no stage starts, and no ane-run call runs, past it
SIZES=${SIZES:-"64 512"}
MAX_REPEAT=3000000
MIN_REPEAT=200
OUT="$ROOT/runs/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$ROOT/runs" && mkdir "$OUT" || exit 2
exec >"$OUT/results.txt" 2>&1
busy() { awk '$1=="busy_ns"{print $2}' "$STATS"; }
printf 'H14 exp chain cell start=%s kernel=%s target_s=%s\n' "$(date -u +%FT%TZ)" "$(uname -r)" "$TARGET_S"
printf 'chip=%s runner_sha256=%s\n' "$(tr -d '\0' </proc/device-tree/compatible 2>/dev/null | cut -c1-40)" "$(sha256sum "$RUNNER" | cut -c1-16)"
# Other ane-run processes (the idle filler, another ticket) would add busy time to the duty figure: count them before and after.
others() { local n; n=$(${PGREP:-pgrep} -xc ane-run 2>/dev/null); echo "${n:-unknown}"; }
printf 'other_ane_run_before=%s\n' "$(others)"
b0=$(busy); t0=$(date +%s.%N)
pass=0; fail=0; skipped=0

# stage NAME PROGRAM INPUT_FILE... : calibration, then loop. Output files: $OUT/NAME.cal.y.f16 and NAME.long.y.f16.
left() { awk -v a="$t0" -v b="$(date +%s.%N)" -v d="$DEADLINE_S" 'BEGIN{r=int(d-(b-a)); print (r>0?r:0)}'; }
stage() {
  local name=$1 prog=$2; shift 2
  local args=(--anec "$prog") i=0 f
  for f in "$@"; do args+=(--in "$i=$f"); i=$((i + 1)); done
  args+=(--time)
  printf '\n== %s program_sha256=%s inputs=%s ==\n' "$name" "$(sha256sum "$prog" | cut -c1-16)" "$*"
  local rem; rem=$(left)
  if [ "$rem" -le 0 ]; then
    fail=$((fail + 1)); skipped=$((skipped + 1)); printf 'STOP whole-run deadline of %s s reached before this stage\n' "$DEADLINE_S"; return 1
  fi
  timeout $((rem < 90 ? rem : 90)) "$RUNNER" "${args[@]}" --out "0=$OUT/$name.cal.y.f16" --repeat 20 >"$OUT/$name.cal.txt" 2>&1
  local rc_cal=$?
  local med; med=$(sed -n 's/.*median \([0-9.]*\).*/\1/p' "$OUT/$name.cal.txt" | head -1)
  tail -n 2 "$OUT/$name.cal.txt"
  printf 'rc_cal(%s)=%s median_ms=%s\n' "$name" "$rc_cal" "${med:-none}"
  if [ "$rc_cal" -ne 0 ] || [ -z "${med:-}" ]; then
    fail=$((fail + 1)); skipped=$((skipped + 1)); printf 'SKIP long loop and later stages: calibration run failed\n'; return 1
  fi
  local rep; rep=$(awk -v t="$TARGET_S" -v m="$med" -v lo="$MIN_REPEAT" -v hi="$MAX_REPEAT" 'BEGIN{r=int(t*1000/m); if(r<lo)r=lo; if(r>hi)r=hi; print r}')
  local s0 s1 bs0 bs1
  rem=$(left)
  if [ "$rem" -le 0 ]; then
    fail=$((fail + 1)); printf 'STOP whole-run deadline of %s s reached before the long loop\n' "$DEADLINE_S"; return 1
  fi
  local limit=$((TARGET_S * 3 + 90))
  s0=$(date +%s.%N); bs0=$(busy)
  timeout $((rem < limit ? rem : limit)) "$RUNNER" "${args[@]}" --out "0=$OUT/$name.long.y.f16" --repeat "$rep" >"$OUT/$name.long.txt" 2>&1
  local rc_long=$?
  s1=$(date +%s.%N); bs1=$(busy)
  tail -n 2 "$OUT/$name.long.txt"
  printf 'rc_long(%s)=%s repeat=%s wall_s=%s loop_duty_pct=%s\n' "$name" "$rc_long" "$rep" \
    "$(awk -v a="$s0" -v b="$s1" 'BEGIN{printf "%.1f", b-a}')" \
    "$(awk -v a="$s0" -v b="$s1" -v x="$bs0" -v y="$bs1" 'BEGIN{printf "%.1f", (y-x)/((b-a)*1e9)*100}')"
  if [ "$rc_long" -ne 0 ]; then
    fail=$((fail + 1)); printf 'STOP later stages at this size: the long loop failed\n'; return 1
  fi
  pass=$((pass + 1))
  return 0
}

for c in $SIZES; do
  P="$ROOT/prog"; I="$ROOT/in"
  printf '\n######## %s channels ########\n' "$c"
  stage "exp_c$c.s1_e"  "$P/exp_c$c.anec" "$I/x_c$c.f16" || continue
  stage "exp_c$c.s2_c"  "$P/mul_c$c.anec" "$I/x_c$c.f16" "$I/k_c$c.f16" || continue
  stage "exp_c$c.s3_d"  "$P/mul_c$c.anec" "$OUT/exp_c$c.s1_e.cal.y.f16" "$OUT/exp_c$c.s2_c.cal.y.f16" || continue
  stage "exp_c$c.s4_y"  "$P/add_c$c.anec" "$OUT/exp_c$c.s1_e.cal.y.f16" "$OUT/exp_c$c.s3_d.cal.y.f16" || continue
done
b1=$(busy); t1=$(date +%s.%N)
printf '\nother_ane_run_after=%s\n' "$(others)"
# Tie results.txt to the output files it describes: compare-expchain.py checks these hashes against the files it judges.
for f in "$OUT"/*.y.f16; do printf 'OUTPUT_SHA256 %s %s\n' "$(basename "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"; done
printf '\nSUMMARY pass=%s fail=%s skipped_loops=%s total_wall_s=%s ticket_duty_pct=%s\n' "$pass" "$fail" "$skipped" \
  "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')" \
  "$(awk -v a="$t0" -v b="$t1" -v x="$b0" -v y="$b1" 'BEGIN{printf "%.1f", (y-x)/((b-a)*1e9)*100}')"
printf '%s pass=%s fail=%s out=%s\n' "$(date -u +%FT%TZ)" "$pass" "$fail" "$OUT" >>"$ROOT/receipts.log"
[ "$fail" -eq 0 ]
