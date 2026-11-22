"""Reconstruct client-visible phase overlap and cross-turn prefix lengths from archived H200 requests."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT / "experiments/modal_node/20261113T204343Z-bench/results__20261113T204343Z-bench__v41__load.json"


def analyze():
    source = json.loads(TRACE.read_text())
    assert source["gpus"] == 8
    grouped = defaultdict(list)
    for request in source["requests"]:
        grouped[request["load"], request["rep"]].append(request)
    by_load = defaultdict(Counter)
    for (load, rep), batch in grouped.items():
        events = defaultdict(lambda: [0, 0, 0])
        ids = {}
        counts = by_load[load]
        counts["repetitions"] += 1
        for r in batch:
            user, turn = r["user"], r["turn"]
            assert (user, turn) not in ids and r["asked"] == r["completion"]
            assert 0 <= r["cached"] <= r["prompt"] and 0 <= r["ttft"] <= r["dur"]
            ids[user, turn] = r
            start, first, end = r["t0"], r["t0"] + r["ttft"], r["t0"] + r["dur"]
            events[start][0] += 1
            events[first][0] -= 1
            events[first][1] += 1
            events[end][1] -= 1
            if turn == 0:
                events[start][2] += 1
                events[first][2] -= 1
            counts["new_prompt_rows"] += r["prompt"] - r["cached"]
            if turn:
                counts["later_new_prompt_rows"] += r["prompt"] - r["cached"]
        for (user, turn), r in ids.items():
            if turn:
                prior = ids[user, turn - 1]
                assert prior["t0"] + prior["dur"] <= r["t0"] + 1e-5
                counts["later_requests"] += 1
                counts["cached_prior_prompt_plus_one"] += r["cached"] == prior["prompt"] + 1
                # Upper bound on a preceding response's rows that might match the
                # next tokenized history. Exact token equality is not in the trace.
                bound = min(max(0, prior["prompt"] + prior["completion"] - r["cached"]),
                            prior["completion"], r["prompt"] - r["cached"])
                counts["remaining_previous_response_row_upper_bound"] += bound
        waiting = streaming = cold = 0
        prev = 0.
        for t, (wait_delta, stream_delta, cold_delta) in sorted(events.items()):
            elapsed = t - prev
            phase = ("mixed" if waiting and streaming else "waiting_only" if waiting
                     else "streaming_only" if streaming else "idle")
            counts[phase + "_s"] += elapsed
            if waiting and not streaming and cold:
                counts["waiting_only_with_first_turn_s"] += elapsed
                if waiting == cold:
                    counts["waiting_only_all_first_turn_s"] += elapsed
            waiting += wait_delta
            streaming += stream_delta
            cold += cold_delta
            assert 0 <= cold <= waiting
            assert 0 <= waiting + streaming <= load and min(waiting, streaming) >= 0
            prev = t
        assert waiting == streaming == 0
        counts["summed_wall_s"] += prev
    out = {}
    for load, row in sorted(by_load.items()):
        secs = [row[name + "_s"] for name in ("idle", "waiting_only", "streaming_only", "mixed")]
        assert abs(sum(secs) - row["summed_wall_s"]) < 1e-7
        out[load] = {**row, "percent_wall": {name: round(100 * row[name + "_s"] / sum(secs), 3)
                       for name in ("idle", "waiting_only", "streaming_only", "mixed")},
                     "max_preceding_response_fraction_of_all_new_rows":
                         row["remaining_previous_response_row_upper_bound"] / row["new_prompt_rows"]}
    return {"source": str(TRACE.relative_to(ROOT)), "source_sha256": hashlib.sha256(TRACE.read_bytes()).hexdigest(),
            "substrate": "V41 on eight H200 GPUs, historical benchmark with 13 turns. V64 and subsequent candidates have different model paths.",
            "scope": "Before/after first streamed token on outstanding client requests. These durations include scheduling, transfer and client overhead. Streaming and prompt compute can overlap or stall. The trace contains lengths, not token IDs or GPU kernel timings.",
            "loads": out}


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
