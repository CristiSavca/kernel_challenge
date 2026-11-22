#!/bin/sh
# Submit the all-reduce cap change as the Nov 15 (UTC) scored run, on the user's go-ahead of Nov 14.
# Base: V44 if the judge passed it at or above V41's 1.0934 (or passed the V48 check), otherwise V41.
# Holds if the V48 check came back failed. Prediction: serving_v0523/runs/predictions.md.
cd /Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge || exit 1
RUNS=serving_v0523/runs
OUT=$RUNS/v48_scored
V44=8ff48fb6618b4188
CHK=3a391700b7e94558
[ -s $OUT.submit.txt ] && grep -qE 'job [0-9a-f]{16}' $OUT.submit.txt && { echo "already submitted"; cat $OUT.submit.txt; exit 0; }
START=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-15 00:00:30' +%s)
LAST=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-15 23:30:00' +%s)
# DRY=1 prints the decision for the judge's current state and submits nothing.
[ -n "$DRY" ] && { OUT=$RUNS/v48_scored.dry; START=0; }
while [ "$(date -u +%s)" -lt "$START" ]; do sleep 30; done
[ "$(date -u +%s)" -gt "$LAST" ] && { echo "HOLD: woke after 23:30 UTC Nov 15, not submitting" | tee $OUT.gate.txt; exit 4; }
decide() {
  okc serve status $V44 --json > $RUNS/$V44.status.json 2>/dev/null
  okc serve status $CHK --json > $RUNS/$CHK.status.json 2>/dev/null
  python3 - "$RUNS" "$V44" "$CHK" <<'P'
import json, sys
runs, v44, chk = sys.argv[1:]
def load(j):
    try:
        return json.load(open(f"{runs}/{j}.status.json"))
    except Exception:
        return {}
a, c = load(v44), load(chk)
ra, rc = a.get("result") or {}, c.get("result") or {}
score = (ra.get("frontier") or {}).get("score")
if c.get("status") == "done" and rc.get("passed") is False:
    print("hold V48 check failed")
elif not a:
    print("retry no status for V44")
elif a.get("status") == "done" and ra.get("passed") and score and score >= 1.0934:
    print(f"v48 V44 passed at {score}")
elif a.get("status") == "done":
    print(f"v48b V44 done, passed={ra.get('passed')} score={score}")
elif c.get("status") == "done" and rc.get("passed"):
    print("v48 V48 check passed, V44 not finished")
elif a.get("status") == "queued":
    print("v48b V44 still queued")
else:
    print(f"wait V44 status={a.get('status')}")
P
}
WAIT_UNTIL=$(( $(date -u +%s) + 6000 ))
while :; do
  VERDICT=$(decide)
  case "$VERDICT" in
    v48\ *) PATCH=serving_v0523/candidates/v48_ar_cap_128m/patch.diff; break ;;
    v48b\ *) PATCH=serving_v0523/candidates/v48b_ar_cap_on_v41/patch.diff; break ;;
    hold*) echo "$(date -u +%H:%M) HOLD: $VERDICT" | tee $OUT.gate.txt; exit 2 ;;
  esac
  # V44 is running (or the status call failed): give it up to 100 minutes, then use the V41 base.
  if [ "$(date -u +%s)" -gt "$WAIT_UNTIL" ]; then
    VERDICT="v48b V44 unfinished after the wait ($VERDICT)"; PATCH=serving_v0523/candidates/v48b_ar_cap_on_v41/patch.diff; break
  fi
  sleep 120
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $VERDICT -> $PATCH sha256 $(shasum -a 256 $PATCH | cut -d' ' -f1)" | tee $OUT.gate.txt
[ -n "$DRY" ] && exit 0
N=0
while :; do
  okc serve submit $PATCH --flag swa-full-tokens-ratio=0.25 --no-wait > $OUT.submit.txt 2>&1
  JOB=$(grep -oE 'job [0-9a-f]{16}' $OUT.submit.txt | head -1 | cut -d' ' -f2)
  [ -n "$JOB" ] && break
  N=$((N + 1)); cp $OUT.submit.txt $OUT.submit.fail$N.txt
  [ $N -ge 5 ] && { echo "submit failed 5 times"; cat $OUT.submit.txt; exit 1; }
  sleep 120
done
date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT.submitted_at.txt
cat $OUT.submit.txt
echo "submitted job $JOB at $(cat $OUT.submitted_at.txt)"
