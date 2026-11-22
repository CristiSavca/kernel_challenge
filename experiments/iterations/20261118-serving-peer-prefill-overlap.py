"""Aggregate client timing/length metadata from our on-node H200 run."""
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
TRACE = (ROOT / "experiments/modal_node/20261113T204343Z-bench/"
         "results__20261113T204343Z-bench__v41__load.json")


def union_duration(intervals, start):
    covered = 0.0
    cursor = start
    for lo, hi in sorted(intervals):
        covered += max(0.0, hi - max(cursor, lo))
        cursor = max(cursor, hi)
    return covered


def summarize(rows):
    rates = [[], [], [], []]
    durations = []
    overlapped = []
    by_tokens = []
    for row in rows:
        start = row["t0"] + row["ttft"]
        end = row["t0"] + row["dur"]
        generated = row["completion"] - 1
        if generated < 8 or end <= start:
            continue
        intervals = []
        for peer in rows:
            if peer["rep"] != row["rep"] or peer["user"] == row["user"]:
                continue
            lo = max(start, peer["t0"])
            hi = min(end, peer["t0"] + peer["ttft"])
            if lo < hi:
                intervals.append((lo, hi))
        seconds = end - start
        overlap = union_duration(intervals, start)
        fraction = overlap / seconds
        i = 0 if fraction <= 1e-6 else 1 if fraction <= 0.25 else 2 if fraction < 0.75 else 3
        ms = 1000 * seconds / generated
        rates[i].append(ms)
        durations.append(seconds)
        overlapped.append(overlap)
        by_tokens.append((row["prompt"], ms))
    by_tokens.sort()
    quarters = [by_tokens[:len(by_tokens)//4], by_tokens[-(len(by_tokens)//4):]]
    return {"requests": len(durations), "peer_wait_overlap_seconds_share": sum(overlapped)/sum(durations),
            "overlap_bins": ["zero", "(zero,0.25]", "(0.25,0.75)", "[0.75,1]"],
            "requests_by_bin": [len(b) for b in rates],
            "median_post_first_ms_per_token_by_bin": [median(b) if b else None for b in rates],
            "extreme_prompt_quartiles": [{"median_input_tokens": median(x[0] for x in q),
                                        "median_post_first_ms_per_token": median(x[1] for x in q)}
                                       for q in quarters]}


if __name__ == "__main__":
    rows = json.loads(TRACE.read_text())["requests"]
    print(json.dumps({"source": str(TRACE.relative_to(ROOT)), "version": "V41, eight H200, TP8",
                      "metric": "A peer is pending before first token. This can include scheduler waiting and GPU execution.",
                      "per_load": {n: summarize([x for x in rows if x["load"] == n]) for n in (1,4,16,32)}}, indent=2))
