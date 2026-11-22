#!/bin/bash
# Session 1 on the private node: V43 (dense prefill + stage profile) then V41 (the judged build, for calibration).
# Waits for the weights download to finish, then asks Modal for the machine until it gets one.
cd "$(dirname "$0")/.." || exit 1
DL=$(ls -d experiments/modal_node/*download | tail -1)
until grep -q 'DOWNLOAD DONE' "$DL/worker.log" 2>/dev/null; do
  if ! pgrep -f 'modal_node.py download' > /dev/null; then echo "download ended without finishing"; exit 1; fi
  sleep 30
done
echo "weights ready at $(date -u +%H:%M) UTC"
for SPEC in "512 24" "336 12"; do
  set -- $SPEC
  .venv-modal/bin/python tools/modal_node.py bench \
    v43=serving_v0523/candidates/v43_dense_prefill_profile/patch.diff:swa-full-tokens-ratio=0.25 \
    v41=serving_v0523/candidates/v41_swa_free/patch.diff:swa-full-tokens-ratio=0.25 \
    --loads 1,4,16,32 --reps 3,2,1,1 --minutes 90 --memory-gib "$1" --hicache-size "$2" --capacity-retries "${RETRIES:-12}" \
    >> experiments/modal_node/session1.$1.log 2>&1
  CODE=$?
  tail -25 experiments/modal_node/session1.$1.log | cut -c1-300
  if grep -q -E 'InvalidError|exceeds|maximum' experiments/modal_node/session1.$1.log && [ $CODE -ne 0 ]; then echo "retrying with less memory"; continue; fi
  exit $CODE
done
