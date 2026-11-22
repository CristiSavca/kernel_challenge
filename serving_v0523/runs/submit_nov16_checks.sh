#!/bin/sh
# Submit the frozen private checks after the November 16 UTC reset.
set -eu
cd "$(dirname "$0")/../.."
OKC=${OKC:-"$HOME/.okc/bin/okc"}
RUNS=serving_v0523/runs
CANDIDATES=serving_v0523/candidates
START=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-16 00:03:00' +%s)

submit_check() {
    name=$1
    patch=$2
    expected=$3
    shift 3
    receipt="$RUNS/$name.submit.txt"

    # Any prior attempt is reconciled against the judge before retrying.
    [ ! -e "$receipt" ] || return 0
    actual=$(shasum -a 256 "$patch")
    if [ "${actual%% *}" != "$expected" ]; then
        printf '%s patch hash mismatch\n' "$name" >&2
        return 1
    fi
    limits=$("$OKC" serve status --json)
    if ! printf '%s\n' "$limits" |
        jq -e '.resets_at == "2026-11-17T00:00:00Z" and .limits.checks.left > 0' > /dev/null; then
        printf '%s check quota unavailable\n' "$name" >&2
        return 1
    fi

    (set -C; : > "$receipt") 2>/dev/null || return 0
    "$OKC" serve check "$patch" --users 16 --flag swa-full-tokens-ratio=0.25 \
        "$@" --no-wait > "$receipt" 2>&1 || true
    cat "$receipt"
    rg -q 'job [0-9a-f]{16}' "$receipt"
}

while [ "$(date -u +%s)" -lt "$START" ]; do sleep 20; done
[ "$(date -u +%F)" = '2026-11-16' ] || exit 1

submit_check v55_nov16_check16 "$CANDIDATES/v55_tp_prefill_projections/patch.diff" \
    5ca9d26522ba84bc183537f819811962cac21de546244a48bf86a3a60dc5543e || true
submit_check v53_nov16_check16 "$CANDIDATES/v53_humming_indexed/patch.diff" \
    341c37c4d6e8cd7011218597de42870023fe2a33a046c6e2c7249d123107cb48 \
    --flag moe-runner-backend=humming --flag speculative-moe-runner-backend=marlin || true
submit_check v56_nov16_check16 "$CANDIDATES/v56_muse_marlin_sequence_parallel/patch.diff" \
    bf9f5590adf2117483ef0a4fcb86e434ff779de334c232dbc78fd2d19bce8126 \
    --flag enable-layernorm-sp || true
