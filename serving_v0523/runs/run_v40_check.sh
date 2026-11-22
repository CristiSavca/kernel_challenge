#!/bin/sh
# Submit the frozen V40 patch as a serving check once the track opens (17:00 UTC Nov 12), then save results.
cd /Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge
OUT=serving_v0523/runs/v40_check
while [ "$(date -u +%s)" -lt "$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-12 17:00:40' +%s)" ]; do sleep 30; done
for i in 1 2 3 4 5 6 7 8 9 10; do
  okc serve check serving_v0523/candidates/v40_ep_local_align/patch.diff --users 16 --no-wait > $OUT.submit.txt 2>&1
  if ! grep -qi "closed\|409" $OUT.submit.txt; then break; fi
  sleep 60
done
cat $OUT.submit.txt
JOB=$(grep -oE '[0-9a-f]{16}' $OUT.submit.txt | head -1)
[ -z "$JOB" ] && { echo "no job id"; exit 1; }
echo "JOB $JOB"
okc serve status $JOB --wait > $OUT.wait.txt 2>&1
okc serve status $JOB --json > $OUT.json 2>&1
okc serve status $JOB --log > $OUT.log 2>&1
tail -40 $OUT.wait.txt
