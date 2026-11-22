#!/bin/sh
# Submit V43 as the Nov 14 scored run once the daily quota resets, unless a queued V42 check has failed.
cd /Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge
OUT=serving_v0523/runs/v43_scored
while [ "$(date -u +%s)" -lt "$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-14 00:00:45' +%s)" ]; do sleep 30; done
for J in ec7cad544a0843ef a0965d83c7b541ec; do
  okc serve status $J --json > serving_v0523/runs/$J.json 2>/dev/null
  if python3 -c "
import json,sys
d=json.load(open('serving_v0523/runs/$J.json')); r=d.get('result')
sys.exit(0 if (r is not None and r.get('passed') is False) else 1)"; then
    echo "HOLD: V42 check $J failed; not submitting V43" | tee $OUT.submit.txt; exit 2
  fi
done
okc serve submit serving_v0523/candidates/v43_dense_prefill_profile/patch.diff --flag swa-full-tokens-ratio=0.25 --no-wait > $OUT.submit.txt 2>&1
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
t=(r.get('startup_log') or {}).get('text','')
for l in t.split('\n'):
    if 'Prefill stage profile' in l or 'Prefill expert check' in l: print(l[:600])
P
