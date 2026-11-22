#!/bin/sh
# Submit V44 once as the Nov 16 (UTC) scored run, at 00:00:30 UTC. Detached so it survives the session.
cd /Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge || exit 1
OUT=serving_v0523/runs/v44_nov16
PATCH=serving_v0523/candidates/v44_dense_prefill_448/patch.diff
[ -s $OUT.submit.txt ] && grep -qE 'job [0-9a-f]{16}' $OUT.submit.txt && exit 0
START=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-16 00:00:30' +%s)
LAST=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-16 23:30:00' +%s)
while [ "$(date -u +%s)" -lt "$START" ]; do sleep 20; done
[ "$(date -u +%s)" -gt "$LAST" ] && { echo "woke after 23:30 UTC Nov 16, not submitting" > $OUT.gate.txt; exit 4; }
N=0
while :; do
  okc serve submit $PATCH --flag swa-full-tokens-ratio=0.25 --no-wait > $OUT.submit.txt 2>&1
  JOB=$(grep -oE 'job [0-9a-f]{16}' $OUT.submit.txt | head -1 | cut -d' ' -f2)
  [ -n "$JOB" ] && break
  N=$((N + 1)); cp $OUT.submit.txt $OUT.submit.fail$N.txt
  grep -qi "scored run" $OUT.submit.txt && grep -qi "left\|used\|limit" $OUT.submit.txt && exit 2
  [ $N -ge 5 ] && exit 1
  sleep 120
done
date -u +"%Y-%m-%dT%H:%M:%SZ" > $OUT.submitted_at.txt
okc serve status $JOB --wait > $OUT.wait.txt 2>&1
okc serve status $JOB --json > serving_v0523/runs/$JOB.json 2>&1
okc serve status $JOB --log > serving_v0523/runs/$JOB.log 2>&1
