#!/usr/bin/env bash
# Offline test of run-expchain.sh with a fake runner and fake ane_stats. Touches no device and no real directory.
# Part 1 checks the script. Part 2 breaks copies of the script and requires part 1's wiring checks to catch each copy.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/h14expchain-test.XXXXXX)
fails=0
ok()  { echo "PASS $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
mkdir -p "$T/root/prog" "$T/root/in"
for c in 64 512; do
  for p in exp mul add; do echo "prog-$p-$c" > "$T/root/prog/${p}_c$c.anec"; done
  for n in x k8 eight eighth; do echo "$n" > "$T/root/in/${n}_c$c.f16"; done
done
echo "busy_ns 1000" > "$T/stats"
printf '#!/usr/bin/env bash\necho "${FAKE_PGREP_COUNT:-0}"\n' > "$T/pgrep"; chmod +x "$T/pgrep"
cat > "$T/runner" <<'EOF'
#!/usr/bin/env bash
rep=0; prog=""; outfile=""; ins=()
while [ $# -gt 0 ]; do case "$1" in --repeat) rep=$2; shift;; --anec) prog=$2; shift;; --out) outfile=${2#0=}; shift;; --in) ins+=("${2#*=}"); shift;; esac; shift; done
echo "repeat=$rep prog=$(basename "$prog") ins=${ins[*]}" >> "$FAKE_LOG"
case "$(basename "$prog")" in "${FAIL_LONG_PROG:-none}") [ "$rep" != 20 ] && [ "${#ins[@]}" = "${FAIL_LONG_NINS:-9}" ] && { echo boom; exit 1; };; esac
case "$(basename "$prog")" in "$FAIL_PROG") [ "$rep" = 20 ] && [ "${#ins[@]}" = "${FAIL_NINS:-9}" ] && { echo boom; exit 1; };; esac
[ -n "${FAKE_IGNORE_TERM:-}" ] && trap "" TERM
[ -n "${FAKE_PID_FILE:-}" ] && echo $$ > "$FAKE_PID_FILE"
[ -n "${FAKE_SLEEP:-}" ] && sleep "$FAKE_SLEEP"
echo "$(basename "$outfile")" > "$outfile"
echo "exec ms over $rep calls: min 1.0 p10 1.0 p25 1.0 median 2.000 p75 2.0 p90 2.0 p99 2.0 max 3.0"
exit 0
EOF
chmod +x "$T/runner"
run() { local script=${SCRIPT:-$HERE/run-expchain.sh}; FAKE_LOG="$T/log" FAIL_PROG="${1:-none}" FAIL_NINS="${2:-9}" ROOT="$T/root" RUNNER="$T/runner" PGREP="$T/pgrep" STATS="$T/stats" TARGET_S="${3:-30}" timeout "${RUN_LIMIT:-600}" bash "$script"; }
latest() { find "$T/root/runs" -maxdepth 1 -name 'run-*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-; }

wiring() {  # checks of which file feeds which stage, from the fake runner's log
  local out=$1 c I="$T/root/in"
  for c in 64 512; do
    grep -q "repeat=20 prog=exp_c$c.anec ins=$I/x_c$c.f16\$" "$T/log" || { bad "s1 c$c does not read x only"; continue; }
    grep -q "repeat=20 prog=mul_c$c.anec ins=$I/x_c$c.f16 $I/k8_c$c.f16\$" "$T/log" || { bad "s2 c$c does not read x and k8"; continue; }
    grep -q "repeat=20 prog=add_c$c.anec ins=$out/exp_c$c.s2_c.cal.y.f16 $I/eight_c$c.f16\$" "$T/log" || { bad "s3 c$c does not add eight to the s2 output"; continue; }
    grep -q "repeat=20 prog=mul_c$c.anec ins=$out/exp_c$c.s1_e.cal.y.f16 $out/exp_c$c.s3_g.cal.y.f16\$" "$T/log" || { bad "s4 c$c does not multiply the s1 and s3 outputs"; continue; }
    grep -q "repeat=20 prog=mul_c$c.anec ins=$out/exp_c$c.s4_p.cal.y.f16 $I/eighth_c$c.f16\$" "$T/log" || { bad "s5 c$c does not scale the s4 output by eighth"; continue; }
    ok "c$c wiring: x -> s1; x,k8 -> s2; s2,eight -> s3; s1,s3 -> s4; s4,eighth -> s5"
  done
}

: > "$T/log"; run none 9 30; rc=$?
out1=$(latest)
[ "$rc" -eq 0 ] && ok "all ten stage runs pass, rc 0" || bad "rc=$rc"
grep -q "SUMMARY pass=10 fail=0 skipped_loops=0" "$out1/results.txt" && ok "summary pass=10 fail=0" || bad "summary line"
[ "$(grep -c 'repeat=15000 ' "$T/log")" -eq 10 ] && ok "long loop sized to 15000 calls for each of the 10 stages" || bad "repeat sizing"
[ "$(grep -c 'repeat=20 ' "$T/log")" -eq 10 ] && ok "ten twenty-call calibrations" || bad "calibration count"
[ "$(find "$out1" -name '*.cal.y.f16' | wc -l)" -eq 10 ] && [ "$(find "$out1" -name '*.long.y.f16' | wc -l)" -eq 10 ] && ok "both output files kept for every stage" || bad "output files"
grep -q "ticket_duty_pct=" "$out1/results.txt" && ok "duty line present" || bad "duty line"
wiring "$out1"
ident() {  # results.txt carries what compare-expchain.py checks: chip, other ane-run counts, program hashes, output hashes
  local out=$1 n
  grep -q "^chip=" "$out/results.txt" && grep -q "^other_ane_run_before=0$" "$out/results.txt" && grep -q "^other_ane_run_after=0$" "$out/results.txt" || { bad "chip or other_ane_run lines missing"; return; }
  [ "$(grep -c '^== exp_c[0-9]*\.s[1-5]_[a-z] program_sha256=' "$out/results.txt")" -eq 10 ] || { bad "program hash lines: $(grep -c 'program_sha256=' "$out/results.txt")"; return; }
  [ "$(grep -c '^OUTPUT_SHA256 ' "$out/results.txt")" -eq 20 ] || { bad "OUTPUT_SHA256 lines: $(grep -c '^OUTPUT_SHA256 ' "$out/results.txt")"; return; }
  n=$(sha256sum "$out/exp_c512.s3_g.cal.y.f16" | cut -d' ' -f1)
  grep -q "^OUTPUT_SHA256 exp_c512.s3_g.cal.y.f16 $n\$" "$out/results.txt" && ok "identity lines present; an output hash matches its file" || bad "output hash does not match the file"
}
ident "$out1"
# the idle check records other ane-run processes: a fake pgrep that finds one
: > "$T/log"; FAKE_PGREP_COUNT=1 run none 9 30 >/dev/null; out4=$(latest)
grep -q "^other_ane_run_before=1$" "$out4/results.txt" && grep -q "^other_ane_run_after=1$" "$out4/results.txt" && ok "other ane-run processes are recorded before and after" || bad "other ane-run count not recorded"
good_fails=$fails

# a failing s2 at 512 channels (a mul with two inputs, first mul run) must stop that size and leave 64 alone
: > "$T/log"; run mul_c512.anec 2 30; rc=$?
out2=$(latest)
[ "$rc" -ne 0 ] && ok "a failed calibration makes the cell fail (rc $rc)" || bad "failed calibration still rc 0"
grep -q "prog=mul_c512.anec ins=$out2" "$T/log" && bad "s3 ran after s2 failed" || ok "no later stage ran at the size whose stage failed"
grep -q "prog=add_c64.anec" "$T/log" && ok "the other size still ran to its last stage" || bad "the other size did not finish"
[ "$out2" != "$out1" ] && ok "a second run gets its own directory" || bad "directory reused"
grep -q "SUMMARY pass=10 fail=0" "$out1/results.txt" && ok "the first run's results were not overwritten" || bad "first results overwritten"

# a failing long loop of s2 at 512 channels (mul, two inputs) must stop that size too, and leave 64 alone
: > "$T/log"; FAIL_LONG_PROG=mul_c512.anec FAIL_LONG_NINS=2 run none 9 30; rc=$?
out3=$(latest)
[ "$rc" -ne 0 ] && ok "a failed long loop makes the cell fail (rc $rc)" || bad "failed long loop still rc 0"
grep -q "STOP later stages at this size" "$out3/results.txt" && ok "the stop is recorded" || bad "no stop line after a failed long loop"
grep -q "prog=mul_c512.anec ins=$out3" "$T/log" && bad "s3 ran after the s2 long loop failed" || ok "no later stage ran at the size whose long loop failed"
grep -q "prog=add_c64.anec" "$T/log" && ok "the other size still ran to its last stage" || bad "the other size did not finish"
grep -q "SUMMARY pass=6 fail=1" "$out3/results.txt" && ok "summary pass=6 fail=1" || bad "summary after a failed long loop: $(grep SUMMARY "$out3/results.txt")"

# the whole-run deadline: a deadline of 0 starts nothing; a runner that hangs is cut at the deadline, not at the per-call timeout (90 s)
: > "$T/log"; DEADLINE_S=0 run none 9 30; rc=$?
out5=$(latest)
[ "$rc" -ne 0 ] && ! grep -q 'repeat=' "$T/log" && grep -q "STOP whole-run deadline" "$out5/results.txt" && ok "a deadline of 0 starts no stage and fails the cell" || bad "deadline 0: rc=$rc"
: > "$T/log"; start=$(date +%s); RUN_LIMIT=20 FAKE_SLEEP=30 DEADLINE_S=3 run none 9 30; rc=$?; took=$(( $(date +%s) - start ))
[ "$rc" -ne 0 ] && [ "$took" -lt 15 ] && ok "a hanging runner is cut at the deadline (took ${took}s, not 90 s)" || bad "hanging runner: rc=$rc took=${took}s"
out6=$(latest)
grep -q "STOP whole-run deadline" "$out6/results.txt" && ok "the deadline stop is recorded" || bad "no deadline stop line"

# a runner that ignores TERM is still cut: timeout -k kills it 2 s after the limit
: > "$T/log"; start=$(date +%s); RUN_LIMIT=40 FAKE_IGNORE_TERM=1 FAKE_SLEEP=30 DEADLINE_S=4 run none 9 30 >/dev/null 2>&1; rc=$?; took=$(( $(date +%s) - start ))
[ "$rc" -ne 0 ] && [ "$took" -lt 15 ] && ok "a runner that ignores TERM is killed (took ${took}s, not 30 s)" || bad "TERM-ignoring runner: rc=$rc took=${took}s"
# a TERM sent to the script reaches the running call
term_check() {
  local script=${SCRIPT:-$HERE/run-expchain.sh} sup rpid
  : > "$T/log"; rm -f "$T/runner.pid"
  FAKE_LOG="$T/log" FAIL_PROG=none FAIL_NINS=9 ROOT="$T/root" RUNNER="$T/runner" PGREP="$T/pgrep" STATS="$T/stats" TARGET_S=30 \
    FAKE_PID_FILE="$T/runner.pid" FAKE_IGNORE_TERM=1 FAKE_SLEEP=60 bash "$script" >/dev/null 2>&1 &
  sup=$!
  for _ in $(seq 1 40); do [ -s "$T/runner.pid" ] && break; sleep 0.25; done
  rpid=$(cat "$T/runner.pid" 2>/dev/null || echo 0)
  kill -TERM "$sup" 2>/dev/null
  sleep 4
  if [ "$rpid" -gt 0 ] && kill -0 "$rpid" 2>/dev/null; then bad "the call was still running after TERM to the script (pid $rpid)"; kill -KILL "$rpid" 2>/dev/null; else ok "a TERM sent to the script ends the running call"; fi
  kill -KILL "$sup" 2>/dev/null; wait "$sup" 2>/dev/null
}
term_check
grep -q "ABORTED by a signal" "$(latest)/results.txt" && ok "the abort is recorded in results.txt" || bad "no ABORTED line"

# part 2: mutants of the script
mut() {  # label, sed expression
  local label=$1 expr=$2
  SCRIPT="$T/mutant.sh"; sed "$expr" "$HERE/run-expchain.sh" > "$SCRIPT"
  if cmp -s "$SCRIPT" "$HERE/run-expchain.sh"; then echo "FAIL mutant '$label': the text to break was not found"; fails=$((fails + 1)); return; fi
  : > "$T/log"; local before=$fails; run none 9 30 >/dev/null 2>&1; local o; o=$(latest); wiring "$o" >/dev/null 2>&1; ident "$o" >/dev/null 2>&1
  : > "$T/log"; local t1; t1=$(date +%s); RUN_LIMIT=20 FAKE_IGNORE_TERM=1 FAKE_SLEEP=30 DEADLINE_S=3 run none 9 30 >/dev/null 2>&1
  [ $(( $(date +%s) - t1 )) -lt 15 ] || bad "mutant hangs past the deadline" >/dev/null
  term_check >/dev/null 2>&1
  if [ "$fails" -gt "$before" ]; then
    echo "PASS mutant '$label' caught"; fails=$before
  else
    echo "FAIL mutant '$label' was NOT caught"; fails=$((fails + 1))
  fi
  unset SCRIPT
}
if [ -n "${HARMLESS_MUTANT:-}" ]; then  # self-test of this test: a mutant that changes nothing must make the test fail
  mut "harmless comment edit (self-test)" 's|^# The host only copies bytes|# The host copies only bytes|'
fi
mut "s4 reads x instead of the s1 output" 's|"\$OUT/exp_c\$c.s1_e.cal.y.f16" "\$OUT/exp_c\$c.s3_g.cal.y.f16"|"$I/x_c$c.f16" "$OUT/exp_c$c.s3_g.cal.y.f16"|'
mut "s5 scales with k8 instead of eighth" 's|"\$I/eighth_c\$c.f16"|"$I/k8_c$c.f16"|'
mut "s3 adds eighth instead of eight" 's|"\$I/eight_c\$c.f16"|"$I/eighth_c$c.f16"|'
mut "s3 uses the mul program" 's|"\$P/add_c\$c.anec"|"$P/mul_c$c.anec"|'
mut "output hashes not written" 's|OUTPUT_SHA256|NOHASH|'
mut "other ane-run count not recorded" 's|other_ane_run_after=%s|ignored_after=%s|'
mut "stage timeouts not capped by the deadline" 's|run_timed \$((rem < 90 ? rem : 90))|run_timed 90|'
mut "no kill-after on the timeout" 's|timeout -k 2 "\$limit"|timeout "$limit"|'
mut "no signal trap" 's|^trap on_signal TERM INT HUP$|:|'
mut "s2 uses the add program" 's|"\$P/mul_c\$c.anec" "\$I/x_c\$c.f16" "\$I/k8_c\$c.f16"|"$P/add_c$c.anec" "$I/x_c$c.f16" "$I/k8_c$c.f16"|'
if [ -z "${HARMLESS_MUTANT:-}" ]; then
  # run this test again with one surviving mutant added: it must exit nonzero
  HARMLESS_MUTANT=1 bash "$0" >"$T/selftest.out" 2>&1; rc=$?
  [ "$rc" -ne 0 ] && grep -q "NOT caught" "$T/selftest.out" && ok "a surviving mutant makes this test exit nonzero (rc $rc)" || bad "a surviving mutant did not make the test fail (rc $rc)"
fi
rm -rf "$T" 2>/dev/null
echo "failures: $fails"; exit $((fails > 0))
