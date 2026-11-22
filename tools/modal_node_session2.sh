#!/bin/bash
# Session 2 on the private node: V44 (clean dense prefill path, tonight's candidate) on all four loads, then the
# fixed profiler build with the prefill graph disabled so real requests are profiled eagerly.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f 'modal_node.py bench' > /dev/null; do sleep 5; done   # let session 1 release the machine and its reservation
echo "starting session 2 request at $(date -u +%H:%M) UTC"
.venv-modal/bin/python tools/modal_node.py bench \
  v44=serving_v0523/candidates/v44_dense_prefill_448/patch.diff:swa-full-tokens-ratio=0.25 \
  v45p=serving_v0523/candidates/v45p_node_profile/patch.diff:swa-full-tokens-ratio=0.25,disable-prefill-cuda-graph,loads=1+16,reps=1+1 \
  --loads 1,4,16,32 --reps 3,2,1,1 --minutes 60 --memory-gib 512 --hicache-size 24 --capacity-retries "${RETRIES:-20}" \
  > experiments/modal_node/session2.log 2>&1
CODE=$?
grep -v -E 'Traceback|^  File|^    ' experiments/modal_node/session2.log | tail -30 | cut -c1-400
exit $CODE
