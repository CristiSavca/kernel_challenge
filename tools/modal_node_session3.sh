#!/bin/bash
# Session 3 on the private node: the profiler build alone, prefill graph disabled so real requests are profiled
# eagerly. Two conversations at 1 user (single-request chunks and later turns), then 16 users (enough chunks to
# fill the profiler's quota of 40 large forwards, alternating Marlin and dense experts).
cd "$(dirname "$0")/.." || exit 1
while pgrep -f 'modal_node.py bench' > /dev/null; do sleep 5; done
echo "starting session 3 request at $(date -u +%H:%M) UTC"
.venv-modal/bin/python tools/modal_node.py bench \
  v46p=serving_v0523/candidates/v46p_node_profile/patch.diff:swa-full-tokens-ratio=0.25,disable-prefill-cuda-graph \
  --loads 1,16 --reps 2,1 --minutes 25 --memory-gib 512 --hicache-size 24 --capacity-retries "${RETRIES:-30}" \
  > experiments/modal_node/session3.log 2>&1
CODE=$?
grep -v -E 'Traceback|^  File|^    ' experiments/modal_node/session3.log | tail -30 | cut -c1-400
exit $CODE
