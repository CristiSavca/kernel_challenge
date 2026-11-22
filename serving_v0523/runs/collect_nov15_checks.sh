#!/bin/sh
# Save terminal receipts for this continuation, including the next UTC quota.
set -eu
cd "$(dirname "$0")"
OKC=${OKC:-"$HOME/.okc/bin/okc"}
END=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-20 10:00:00' +%s)
START=$(date -u -j -f '%Y-%m-%d %H:%M:%S' '2026-11-15 22:00:00' +%s)

while [ "$(date -u +%s)" -lt "$END" ]; do
    if ! jobs=$("$OKC" serve jobs --json); then
        sleep 300
        continue
    fi
    terminal_jobs=$(printf '%s' "$jobs" | jq -r --argjson start "$START" '
        .[] | select(.submitted_at >= $start) |
        select(.status == "done" or .status == "failed" or
               .status == "cancelled" or .status == "canceled" or .status == "withdrawn") |
        .job_id')
    for job in $terminal_jobs; do
        if [ -s "$job.json" ] && [ -s "$job.log" ]; then
            continue
        fi
        status=$(printf '%s' "$jobs" | jq -r --arg id "$job" '.[] | select(.job_id == $id) | .status')
        case "$status" in
            done|failed|cancelled|canceled|withdrawn)
                if "$OKC" serve status "$job" --json > "$job.json.pending" &&
                    jq -e --arg id "$job" '.id == $id' "$job.json.pending" > /dev/null; then
                    mv "$job.json.pending" "$job.json"
                    "$OKC" serve status "$job" --log > "$job.log" 2>&1 || true
                    printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$job" "$status"
                fi
                ;;
        esac
    done
    sleep 300
done
