"""Read two official score receipts and the archived single-user H200 request traces."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
JOBS = {"V48b": "8ebed2bced8942ff", "V64": "abd217baf0ed49fd"}
TRACES = {
    "V41": "20261113T204343Z-bench/results__20261113T204343Z-bench__v41__load.json",
    "V44": "20261113T213641Z-bench/results__partial__v44__load.json",
}


def official(job):
    path = Path("serving_v0523/runs") / f"{job}.json"
    report = json.loads((ROOT / path).read_text())
    run = report["result"]
    assert report["status"] == "done" and report["mode"] == "run" and run["passed"]
    rung = next(r for r in run["rungs"] if r["concurrency"] == 1)
    point = next(p for p in run["frontier"]["points"] if p["concurrency"] == 1)
    assert run["short_outputs"] == 0 and not rung["truncated"]
    return {
        "source": str(path), "node": run["node"], "shape": run["shape"],
        "one_user": rung, "one_user_frontier": point, "board_score": run["frontier"]["score"],
    }


def archived(source):
    path = Path("experiments/modal_node") / source
    report = json.loads((ROOT / path).read_text())
    assert report["gpus"] == 8
    rows = sorted(
        (r for r in report["requests"] if r["load"] == 1),
        key=lambda r: (r["rep"], r["user"], r["turn"]),
    )
    assert len(rows) == 39 and all(r["asked"] == r["completion"] for r in rows)
    per_rep = []
    for rep in sorted({r["rep"] for r in rows}):
        batch = [r for r in rows if r["rep"] == rep]
        assert len(batch) == 13 and [r["turn"] for r in batch] == list(range(13))
        assert all(r["user"] == 0 and 0 <= r["ttft"] <= r["dur"] for r in batch)
        end = max(r["t0"] + r["dur"] for r in batch)
        fast = sum(r["dur"] - r["ttft"] for r in batch)
        per_rep.append({
            "rep": rep, "last_request_end_s": end,
            "summed_request_s": sum(r["dur"] for r in batch),
            "first_turn_ttft_s": batch[0]["ttft"],
            "later_turns_ttft_s": sum(r["ttft"] for r in batch[1:]),
            "after_first_token_s": fast, "after_first_fraction_of_wall": fast / end,
        })
    totals = {k: sum(r[k] for r in per_rep) for k in (
        "last_request_end_s", "summed_request_s", "first_turn_ttft_s",
        "later_turns_ttft_s", "after_first_token_s",
    )}
    wall = totals["last_request_end_s"]
    streaming = totals["after_first_token_s"]
    tokens = {k: sum(r[k] for r in rows) for k in ("prompt", "cached", "completion")}
    numerator = tokens["prompt"] + tokens["completion"]
    return rows, {
        "source": str(path), "tokens": tokens, "per_rep": per_rep, "totals": totals,
        "cached_share_of_all_tokens": tokens["cached"] / numerator,
        "output_share_of_all_tokens": tokens["completion"] / numerator,
        "one_user_speedup_if_postfirst_twice_as_fast": wall / (wall - streaming / 2),
        "one_user_speedup_if_postfirst_free": wall / (wall - streaming),
    }


def main():
    runs = {name: official(job) for name, job in JOBS.items()}
    base, new = runs.values()
    assert base["node"] == new["node"] and base["shape"] == new["shape"]
    assert base["one_user_frontier"]["tok_s_per_user"] == new["one_user_frontier"]["tok_s_per_user"]
    traces = {name: archived(path) for name, path in TRACES.items()}
    (old_rows, old), (new_rows, newer) = traces.values()
    match = ("rep", "user", "turn", "asked", "completion")
    assert all(tuple(a[k] for k in match) == tuple(b[k] for k in match) for a, b in zip(old_rows, new_rows))
    save = {k: old["totals"][k] - newer["totals"][k] for k in old["totals"]}
    official_ratios = {
        k: new["one_user"][k] / base["one_user"][k]
        for k in ("tok_s_per_gpu", "tok_s_per_user_e2e", "tok_s_per_user_p50",
                  "ttft_first_p90_s", "ttft_later_p90_s")
    }
    official_ratios["frontier_throughput_at_same_user_rate"] = (
        new["one_user_frontier"]["tok_s_per_gpu"] / base["one_user_frontier"]["tok_s_per_gpu"]
    )
    print(json.dumps({
        "official": runs, "official_v64_over_v48b": official_ratios,
        "archived_physical": {name: record for name, (_, record) in traces.items()},
        "matched_trace_rows": len(old_rows),
        "matched_but_prompt_lengths_differ": sum(a["prompt"] != b["prompt"] for a, b in zip(old_rows, new_rows)),
        "archived_v41_minus_v44_s": save,
        "fraction_of_net_observed_wall_gain_in_first_turn_ttft": save["first_turn_ttft_s"] / save["last_request_end_s"],
        "limits": [
            "Each trace measured three sequential, one-user conversations. Wall runs from client start to final response, including gaps.",
            "Client TTFT includes scheduling, network and first-token work. After-first duration includes transmission and client overhead.",
            "The archived speedups hold all TTFT and gaps fixed; the historical and official workloads have different lengths and generated content.",
            "Official percentiles and aggregate throughput cannot reconstruct the V64 time budget. Paired aggregates do not isolate a kernel effect.",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
