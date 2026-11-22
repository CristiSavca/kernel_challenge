#!/bin/sh
# Submit V44 (clean dense prefill path, threshold 448) as the Nov 14 scored run, but only once the judge has passed a
# check of the same dense path (V42, threshold 384) for correctness and speed. Holds on a failed check or at the deadline.
cd /Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge
OUT=serving_v0523/runs/v44_scored
PATCH=serving_v0523/candidates/v44_dense_prefill_448/patch.diff
START=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-14 00:00:45' +%s)
DEADLINE=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-14 12:00:00' +%s)
while [ "$(date -u +%s)" -lt "$START" ]; do sleep 30; done
while :; do
  for J in ec7cad544a0843ef a0965d83c7b541ec; do
    okc serve status $J --json > serving_v0523/runs/$J.json 2>/dev/null
  done
  VERDICT=$(python3 - <<'P'
import json
ref = {16: 13380.4, 32: 14657.7}   # V41 on the judge at the same load: scored run (16) and check 8f0a7cc1 (32)
verdict = "wait"
for j in ("ec7cad544a0843ef", "a0965d83c7b541ec"):
    try:
        d = json.load(open(f"serving_v0523/runs/{j}.json"))
    except Exception:
        continue
    r = d.get("result")
    if not r:
        continue
    if r.get("passed") is False:
        verdict = f"fail {j} reasons={(r.get('correctness') or {}).get('reasons')}"
        break
    rungs = r.get("rungs") or []
    if r.get("passed") and rungs:
        g = rungs[0]
        rate, base = g["tok_s_per_gpu"], ref.get(g["concurrency"])
        if base and rate >= 0.99 * base:
            verdict = f"pass {j} users={g['concurrency']} tok_s_per_gpu={rate} vs V41 {base}"
        elif base and verdict == "wait":
            verdict = f"slow {j} users={g['concurrency']} tok_s_per_gpu={rate} vs V41 {base}"
print(verdict)
P
)
  case "$VERDICT" in
    pass*) echo "$(date -u +%H:%M) $VERDICT" | tee $OUT.gate.txt; break ;;
    fail*|slow*) echo "$(date -u +%H:%M) HOLD: $VERDICT" | tee $OUT.gate.txt; exit 2 ;;
  esac
  if [ "$(date -u +%s)" -gt "$DEADLINE" ]; then echo "HOLD: no V42 check finished by 12:00 UTC" | tee $OUT.gate.txt; exit 4; fi
  sleep 300
done
okc serve submit $PATCH --flag swa-full-tokens-ratio=0.25 --no-wait > $OUT.submit.txt 2>&1
cat $OUT.submit.txt
JOB=$(grep -oE '[0-9a-f]{16}' $OUT.submit.txt | head -1)
[ -z "$JOB" ] && { echo "no job id"; exit 1; }
okc serve status $JOB --wait > $OUT.wait.txt 2>&1
okc serve status $JOB --json > serving_v0523/runs/$JOB.json 2>&1
okc serve status $JOB --log > serving_v0523/runs/$JOB.log 2>&1
python3 - "$JOB" <<'P'
import json,sys
d=json.load(open(f'serving_v0523/runs/{sys.argv[1]}.json')); r=d.get('result') or {}; f=r.get('frontier') or {}
print(sys.argv[1], d.get('status'), 'passed', r.get('passed'), 'score', f.get('score'), [(p['concurrency'], p['ratio']) for p in f.get('points', [])])
P
