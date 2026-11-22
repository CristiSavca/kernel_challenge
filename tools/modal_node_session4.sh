#!/bin/sh
# Node session 4: one paid 8xH200 session, three server starts.
#   v48p         profiler build with the all-reduce cap at 128 MB; prefill graph off. Large prefill forwards alternate
#                Marlin/dense experts and NCCL/custom sums, so one log gives the stage split and both comparisons.
#   v48          the scored candidate (V44 + all-reduce cap at 128 MB) with every graph on: does it start, how much
#                memory, TTFT and chunk rate at 1 and 16 users.
#   v48p_decode  profiler build with the decode graph off too: stage split of speculative verify steps.
# Estimated 39 minutes; the reservation is for MINUTES (default 45). Report with tools/okc_profile_report.py.
cd /Users/cristisavca/Documents/HASIDIC_PROJECTS/open-kernel-challenge || exit 1
P=serving_v0523/candidates
.venv-modal/bin/python tools/modal_node.py bench \
  "v48p=$P/v48p_node_profile/patch.diff:swa-full-tokens-ratio=0.25,disable-prefill-cuda-graph,loads=1+16,reps=2+1" \
  "v48=$P/v48_ar_cap_128m/patch.diff:swa-full-tokens-ratio=0.25,loads=1+16,reps=2+1" \
  "v48p_decode=$P/v48p_node_profile/patch.diff:swa-full-tokens-ratio=0.25,disable-prefill-cuda-graph,disable-decode-cuda-graph,loads=1+16,reps=1+1" \
  --minutes "${MINUTES:-45}" --min-variant-s 720 --memory-gib 512 --hicache-size 24 --capacity-retries "${RETRIES:-30}" \
  > experiments/modal_node/session4.log 2>&1
