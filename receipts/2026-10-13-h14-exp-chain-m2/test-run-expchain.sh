#!/usr/bin/env bash
# Offline test of run-expchain.sh with a fake runner and fake ane_stats. Touches no device and no real directory.
# Part 1 checks the script. Part 2 breaks a copy of the script in three ways and requires part 1's wiring checks to catch each copy.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/h14expchain-test.XXXXXX)
fails=0
ok()  { echo "PASS $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
mkdir -p "$T/root/prog" "$T/root/in"
for c in 64 512; do
  for p in exp mul add; do echo "prog-$p-$c" > "$T/root/prog/${p}_c$c.anec"; done
  echo x > "$T/root/in/x_c$c.f16"; echo k > "$T/root/in/k_c$c.f16"
done
echo "busy_ns 1000" > "$T/stats"
cat > "$T/runner" <<'EOF'
#!/usr/bin/env bash
rep=0; prog=""; outfile=""; ins=()
while [ $# -gt 0 ]; do case "$1" in --repeat) rep=$2; shift;; --anec) prog=$2; shift;; --out) outfile=${2#0=}; shift;; --in) ins+=("${2#*=}"); shift;; esac; shift; done
echo "repeat=$rep prog=$(basename "$prog") ins=${ins[*]}" >> "$FAKE_LOG"
case "$(basename "$prog")" in "$FAIL_PROG") [ "$rep" = 20 ] && [ "${#ins[@]}" = "${FAIL_NINS:-9}" ] && { echo boom; exit 1; };; esac
echo "$(basename "$outfile")" > "$outfile"
echo "exec ms over $rep calls: min 1.0 p10 1.0 p25 1.0 median 2.000 p75 2.0 p90 2.0 p99 2.0 max 3.0"
exit 0
EOF
chmod +x "$T/runner"
run() { local script=${SCRIPT:-$HERE/run-expchain.sh}; FAKE_LOG="$T/log" FAIL_PROG="${1:-none}" FAIL_NINS="${2:-9}" ROOT="$T/root" RUNNER="$T/runner" STATS="$T/stats" TARGET_S="${3:-30}" bash "$script"; }
latest() { find "$T/root/runs" -maxdepth 1 -name 'run-*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-; }

wiring() {  # checks of which file feeds which stage, from the fake runner's log
  local out=$1 c
  for c in 64 512; do
    grep -q "repeat=20 prog=exp_c$c.anec ins=$T/root/in/x_c$c.f16\$" "$T/log" || { bad "s1 c$c does not read x only"; continue; }
    grep -q "repeat=20 prog=mul_c$c.anec ins=$T/root/in/x_c$c.f16 $T/root/in/k_c$c.f16\$" "$T/log" || { bad "s2 c$c does not read x and K"; continue; }
    grep -q "repeat=20 prog=mul_c$c.anec ins=$out/exp_c$c.s1_e.cal.y.f16 $out/exp_c$c.s2_c.cal.y.f16\$" "$T/log" || { bad "s3 c$c does not read the s1 and s2 outputs"; continue; }
    grep -q "repeat=20 prog=add_c$c.anec ins=$out/exp_c$c.s1_e.cal.y.f16 $out/exp_c$c.s3_d.cal.y.f16\$" "$T/log" || { bad "s4 c$c does not read the s1 and s3 outputs"; continue; }
    ok "c$c wiring: x -> s1; x,K -> s2; s1,s2 -> s3; s1,s3 -> s4"
  done
}

: > "$T/log"; run none 9 30; rc=$?
out1=$(latest)
[ "$rc" -eq 0 ] && ok "all eight stage runs pass, rc 0" || bad "rc=$rc"
grep -q "SUMMARY pass=8 fail=0 skipped_loops=0" "$out1/results.txt" && ok "summary pass=8 fail=0" || bad "summary line"
[ "$(grep -c 'repeat=15000 ' "$T/log")" -eq 8 ] && ok "long loop sized to 15000 calls for each of the 8 stages" || bad "repeat sizing"
[ "$(grep -c 'repeat=20 ' "$T/log")" -eq 8 ] && ok "eight twenty-call calibrations" || bad "calibration count"
[ "$(find "$out1" -name '*.cal.y.f16' | wc -l)" -eq 8 ] && [ "$(find "$out1" -name '*.long.y.f16' | wc -l)" -eq 8 ] && ok "both output files kept for every stage" || bad "output files"
grep -q "ticket_duty_pct=" "$out1/results.txt" && ok "duty line present" || bad "duty line"
wiring "$out1"
good_fails=$fails

# a failing s2 at 512 channels (a mul with two inputs, first mul run) must stop that size and leave 64 alone
: > "$T/log"; run mul_c512.anec 2 30; rc=$?
out2=$(latest)
[ "$rc" -ne 0 ] && ok "a failed calibration makes the cell fail (rc $rc)" || bad "failed calibration still rc 0"
grep -q "prog=mul_c512.anec ins=$out2" "$T/log" && bad "s3 ran after s2 failed" || ok "no later stage ran at the size whose stage failed"
grep -q "prog=add_c64.anec" "$T/log" && ok "the other size still ran to its last stage" || bad "the other size did not finish"
[ "$out2" != "$out1" ] && ok "a second run gets its own directory" || bad "directory reused"
grep -q "SUMMARY pass=8 fail=0" "$out1/results.txt" && ok "the first run's results were not overwritten" || bad "first results overwritten"

# part 2: mutants of the script
mut() {  # label, sed expression
  local label=$1 expr=$2
  SCRIPT="$T/mutant.sh"; sed "$expr" "$HERE/run-expchain.sh" > "$SCRIPT"
  if cmp -s "$SCRIPT" "$HERE/run-expchain.sh"; then echo "FAIL mutant '$label': the text to break was not found"; fails=$((fails + 1)); return; fi
  : > "$T/log"; local before=$fails; run none 9 30 >/dev/null 2>&1; local o; o=$(latest); wiring "$o" >/dev/null 2>&1
  [ "$fails" -gt "$before" ] && { echo "PASS mutant '$label' caught"; fails=$before; } || echo "FAIL mutant '$label' was NOT caught"
  [ "$fails" -gt "$before" ] || true
  unset SCRIPT
}
base=$fails
mut "s3 reads x instead of s1 output" 's|"\$P/mul_c\$c.anec" "\$OUT/exp_c\$c.s1_e.cal.y.f16" "\$OUT/exp_c\$c.s2_c.cal.y.f16"|"$P/mul_c$c.anec" "$I/x_c$c.f16" "$OUT/exp_c$c.s2_c.cal.y.f16"|'
mut "s4 adds s2 instead of s3" 's|"\$OUT/exp_c\$c.s3_d.cal.y.f16" \|\| continue|"$OUT/exp_c$c.s2_c.cal.y.f16" \|\| continue|'
mut "s2 uses the add program" 's|"\$P/mul_c\$c.anec" "\$I/x_c\$c.f16" "\$I/k_c\$c.f16"|"$P/add_c$c.anec" "$I/x_c$c.f16" "$I/k_c$c.f16"|'
rm -rf "$T" 2>/dev/null
echo "failures: $fails"; exit $((fails > 0))
