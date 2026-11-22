#!/bin/sh
# Autonomous driver for the Nov 15-16 (UTC) scored runs. Written after the V44 waiter's notification arrived 7 hours
# late and a refunded Nov 14 run went unused: nothing here depends on a session being re-invoked.
#   A. Until the Nov 16 reset, watch J1 = V48b (V41 + all-reduce cap). If the judge fails it and refunds the run,
#      resubmit it once the same UTC day.
#   B. At 00:00:30 UTC Nov 16 submit the second attempt at the dense prefill path: V48 (V44 + cap), or V44 alone if
#      J1 showed the cap change is bad (failed its checks, crashed, or scored under V41).
#   C. Save every finished run's JSON and log. Exit when J1 and J2 are both finished, or at 23:30 UTC Nov 16.
# DRY=1 prints the decision for the judge's current state and exits without submitting.
cd /Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge || exit 1
RUNS=serving_v0523/runs
LOG=$RUNS/driver_nov16.log
J1=${J1:-8ebed2bced8942ff}
V48=serving_v0523/candidates/v48_ar_cap_128m/patch.diff
V48B=serving_v0523/candidates/v48b_ar_cap_on_v41/patch.diff
V44=serving_v0523/candidates/v44_dense_prefill_448/patch.diff
RESET=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-16 00:00:30' +%s)
END=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-16 23:30:00' +%s)
J2=""; RESUBMITTED=""
[ -f $RUNS/driver_nov16.j2 ] && J2=$(cat $RUNS/driver_nov16.j2)
[ -f $RUNS/driver_nov16.j1b ] && { RESUBMITTED=1; J1=$(cat $RUNS/driver_nov16.j1b); }
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a $LOG; }
# Prints "<class> <detail>": passed | capbad | inconclusive | refunded | pending | unknown
classify() {
  okc serve status $1 --json > $RUNS/$1.status.tmp 2>/dev/null && mv $RUNS/$1.status.tmp $RUNS/$1.status.json
  python3 - "$RUNS/$1.status.json" <<'P'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("unknown no status"); raise SystemExit
st, r = d.get("status"), d.get("result") or {}
score = (r.get("frontier") or {}).get("score")
if st == "done" and r.get("passed") and score:
    print(("passed" if score >= 1.090 else "capbad"), f"score={score}")
elif st == "done":
    print("capbad", f"done passed={r.get('passed')} score={score}")
elif st == "failed" and d.get("refunded"):
    print("refunded", f"{r.get('status')}: {r.get('error')}")
elif st == "failed" and r.get("status") == "timeout":
    print("inconclusive", f"timeout, refunded={d.get('refunded')}")
elif st == "failed":
    print("capbad", f"failed: {r.get('status')}: {r.get('error')}")
else:
    print("pending", f"status={st}")
P
}
save() {  # save a finished run's JSON and log once
  [ -s $RUNS/$1.json ] && grep -q '"finished_at"' $RUNS/$1.json && return
  okc serve status $1 --json > $RUNS/$1.json 2>&1
  okc serve status $1 --log > $RUNS/$1.log 2>&1
  say "saved $1: $(python3 -c "
import json,sys
d=json.load(open('$RUNS/$1.json')); r=d.get('result') or {}; f=r.get('frontier') or {}
print(d.get('status'), 'passed', r.get('passed'), 'score', f.get('score'), [(p['concurrency'], p['ratio']) for p in f.get('points', [])], 'refunded', d.get('refunded'), r.get('error'))" 2>&1)"
}
submit() {  # submit <patch> <label>; prints the job id or nothing
  OUT=$RUNS/$2
  okc serve submit $1 --flag swa-full-tokens-ratio=0.25 --no-wait > $OUT.submit.txt 2>&1
  JOB=$(grep -oE 'job [0-9a-f]{16}' $OUT.submit.txt | head -1 | cut -d' ' -f2)
  date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT.submitted_at.txt
  say "submit $2: $1 sha256 $(shasum -a 256 $1 | cut -d' ' -f1) -> job ${JOB:-NONE} ($(head -1 $OUT.submit.txt | cut -c1-120))" >&2
  echo "$JOB"
}
if [ -n "$DRY" ]; then
  C1=$(classify $J1); echo "J1 $J1: $C1"
  case "$C1" in capbad*) echo "at reset: would submit V44 ($V44)" ;; *) echo "at reset: would submit V48 ($V48)" ;; esac
  exit 0
fi
say "driver start: J1=$J1 J2=${J2:-none}"
while :; do
  NOW=$(date -u +%s)
  C1=$(classify $J1)
  case "$C1" in
    passed*|capbad*|inconclusive*|refunded*) save $J1 ;;
  esac
  # A. a refunded failure the same UTC day: resubmit once
  if [ -z "$RESUBMITTED" ] && [ "$NOW" -lt "$((RESET - 600))" ]; then
    case "$C1" in refunded*)
      say "J1 $J1 failed and was refunded ($C1); resubmitting V48b"
      NEW=$(submit $V48B v48b_resubmit)
      RESUBMITTED=1
      [ -n "$NEW" ] && { echo $NEW > $RUNS/driver_nov16.j1b; J1=$NEW; C1="pending resubmitted"; } ;;
    esac
  fi
  # B. the Nov 16 scored run
  if [ -z "$J2" ] && [ "$NOW" -ge "$RESET" ] && [ "$NOW" -lt "$END" ]; then
    case "$C1" in
      capbad*) say "J1: $C1 -> the cap change is not trusted; submitting V44 alone"; J2=$(submit $V44 v44_retry_scored)
               [ -z "$J2" ] && { say "V44 was not accepted; submitting V48"; J2=$(submit $V48 v48_scored_nov16); } ;;
      *) say "J1: $C1 -> submitting V48 (V44 + cap)"; J2=$(submit $V48 v48_scored_nov16) ;;
    esac
    if [ -n "$J2" ]; then echo $J2 > $RUNS/driver_nov16.j2; else say "no Nov 16 submission was accepted"; J2=NONE; fi
  fi
  # C. finished?
  if [ -n "$J2" ] && [ "$J2" != NONE ]; then
    C2=$(classify $J2)
    case "$C2" in pending*|unknown*) ;; *) save $J2 ;; esac
    case "$C1$C2" in *pending*|*unknown*) ;; *) say "both runs finished: J1 $C1 | J2 $C2"; exit 0 ;; esac
  fi
  [ "$J2" = NONE ] && exit 3
  [ "$NOW" -gt "$END" ] && { say "driver end time reached: J1 $C1 | J2 ${C2:-not submitted}"; exit 0; }
  sleep 300
done
