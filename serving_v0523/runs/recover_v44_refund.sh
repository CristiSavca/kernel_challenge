#!/bin/sh
# Use a refunded Nov 16 scored slot once, if the exact V50 check earns it.
set -eu
cd "$(dirname "$0")/../.."
OKC=${OKC:-"$HOME/.okc/bin/okc"}
RUNS=serving_v0523/runs
PATCH=serving_v0523/candidates/v50_dense_compact_finalize/patch.diff
SHA=7ba1708e049dbfae89eac613efe0bdeb9b7c8ccbcf0c11859ccad49f7f4441ae
RECEIPT="$RUNS/v50_nov16_refund.submit.txt"
CHECK="$RUNS/fec66419d5154815.json"
END=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-16 23:30:00' +%s)
actual=$(shasum -a 256 "$PATCH")
[ "${actual%% *}" = "$SHA" ] || exit 1
[ ! -e "$RECEIPT" ] || exit 0

while [ "$(date -u +%s)" -lt "$END" ]; do
    run=
    if [ -s "$RUNS/v44_nov16.submit.txt" ]; then
        run=$(sed -nE 's/.*job ([a-f0-9]{16}).*/\1/p' "$RUNS/v44_nov16.submit.txt" | head -1)
    fi
    if [ -z "$run" ] || ! status=$("$OKC" serve status "$run" --json); then
        sleep 300
        continue
    fi

    if printf '%s\n' "$status" | jq -e '.status == "done" or .status == "withdrawn"' > /dev/null; then
        exit 0
    fi

    if [ "$(date -u +%F)" = "2026-11-16" ] &&
        printf '%s\n' "$status" |
            jq -e '.mode == "run" and .status == "failed" and .refunded == true' > /dev/null &&
        [ -s "$CHECK" ]; then
        if ! jq -e --arg sha "$SHA" '
            .status == "done" and .result.passed == true and
            .result.correctness.passed == true and .result.later_turns.passed == true and
            .result.patch.sha256 == $sha and .serving_flags["--swa-full-tokens-ratio"] == "0.25" and
            (.result.gpu_memory.over_stock_gb | type) == "number" and
            .result.gpu_memory.over_stock_gb <= 1.9 and .result.short_outputs == 0 and
            any(.result.vs_stock.rungs[]; .concurrency == 16 and .at_stock_point >= 1.17)
        ' "$CHECK" > /dev/null; then
            printf '%s V50 check did not meet the measured fallback criteria\n' "$(date -u +%FT%TZ)"
            exit 0
        fi
        if limits=$("$OKC" serve status --json) &&
            printf '%s\n' "$limits" |
                jq -e '.resets_at == "2026-11-17T00:00:00Z" and .limits.scored.left > 0' > /dev/null &&
            jobs=$("$OKC" serve jobs --json) &&
            printf '%s\n' "$jobs" |
                jq -e --arg id "$run" 'first(.[] | select(.mode == "run")) | .job_id == $id' > /dev/null; then
            (set -C; : > "$RECEIPT") 2>/dev/null || exit 0
            "$OKC" serve submit "$PATCH" --flag swa-full-tokens-ratio=0.25 --no-wait > "$RECEIPT" 2>&1 || true
            cat "$RECEIPT"
            job=$(sed -nE 's/.*job ([a-f0-9]{16}).*/\1/p' "$RECEIPT" | head -1)
            if [ -n "$job" ]; then
                "$OKC" serve status "$job" --wait > "$RUNS/v50_nov16_refund.wait.txt" 2>&1 || true
                "$OKC" serve status "$job" --json > "$RUNS/$job.json"
                "$OKC" serve status "$job" --log > "$RUNS/$job.log" 2>&1 || true
            fi
            exit 0
        fi
    fi
    sleep 300
done
