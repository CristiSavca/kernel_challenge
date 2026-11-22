"""Reconcile cold-wave arrivals with real client completions in the archived eight-H200 run."""
import bisect
import hashlib
import json
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments/modal_node/20261113T204343Z-bench"
EVENT = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+),"
    r".*?#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+),"
    r" cuda graph: (True|False), input throughput \(token/s\): ([\d.]+)"
)
WIDTH = 8


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid(text, prefix):
    return json.loads(re.search(re.escape(prefix) + r"(\[[\d ,]+\])", text).group(1))


def quantile(values, fraction):
    ordered = sorted(values)
    loc = fraction * (len(ordered) - 1)
    low = int(loc)
    return ordered[low] + (loc - low) * (ordered[min(low+1, len(ordered)-1)] - ordered[low])


def cold_wave(section, clients):
    events = []
    elapsed = 0
    for line in section.splitlines():
        if "Prefill batch," not in line:
            continue
        match = EVENT.search(line)
        assert match, line
        seqs, new, cached, running, queued, pending = map(int, match.groups()[:6])
        assert match.group(7) == "True" and "#replay-token" not in line
        # This reports the time between consecutive scheduler results. Drop the interval
        # leading to the first prompt, which includes unrelated work before this wave.
        if events:
            elapsed += new / float(match.group(8))
            assert running >= events[-1]["running"]
        events.append(dict(new=new, cached=cached, running=running, seqs=seqs, elapsed=elapsed))
        if pending == 0 and queued == 0 and running + seqs >= len(clients):
            break
    # The scheduler reports prefix hits once on page-one admission.
    assert sum(e["cached"]+e["new"] for e in events) == sum(c["prompt"] for c in clients)
    return events


def opportunities(events, users, prompt_grid, verify_grid, keys, rounds):
    cap = prompt_grid[-1]
    arrivals = [[] for _ in range(users)]
    overflow_calls = tail_pairs = marginal_padding = 0
    for event in events:
        running, rows = event["running"], event["new"]
        if not running:
            continue
        verify_rows = WIDTH * verify_grid[bisect.bisect_left(verify_grid, running)]
        separate = prompt_grid[bisect.bisect_left(prompt_grid, rows)] + verify_rows
        combined = rows + WIDTH * running
        if combined <= cap:
            if prompt_grid[bisect.bisect_left(prompt_grid, combined)] > separate:
                continue
        else:
            idx = bisect.bisect_left(keys, combined - cap)
            if idx == len(keys) or cap + keys[idx] > separate + verify_rows:
                continue
            overflow_calls += 1
            marginal_padding += max(0, cap + keys[idx] - separate)
        tail_pairs += running
        for rank_in_cohort in range(running):
            arrivals[rank_in_cohort].append(event["elapsed"])
    # Members are anonymous. Before the first wave ends the running count never drops;
    # assigning indices by join order needs no guesses about individual request IDs.
    end = events[-1]["elapsed"]
    out = {}
    for target in rounds:
        lead = [end - times[target-1] if len(times) >= target else 0.0 for times in arrivals]
        out[str(target)] = dict(
            clients_with_earlier_opportunity=sum(dt > 0 for dt in lead),
            median_seconds_before_last_prefill_report=round(statistics.median(lead), 4),
            lower_decile_seconds_before_last_prefill_report=round(quantile(lead, .1), 4),
        )
    return dict(overflow_calls=overflow_calls, all_prompt_and_running_pairs=tail_pairs,
                extra_padded_rows_on_overflow=marginal_padding, opportunities=out)


def main():
    server = next(ARCHIVE.glob("*v41__server.log"))
    load = next(ARCHIVE.glob("*v41__load.json"))
    text = server.read_text()
    data = json.loads(load.read_bytes())
    assert data["gpus"] == 8
    prompt = grid(text, "Capture target prefill CUDA graph begin. backend=breakable, num_tokens=")
    verify = grid(text, "Capture target verify CUDA graph end. bs=")
    geometric = [size * WIDTH for size in verify if size & (size - 1) == 0]
    epochs = text.split("Cache flushed successfully!")[1:]
    reps = [(r["concurrency"], rep) for r in data["rungs"] for rep in range(r["reps"])]
    assert len(reps) == len(epochs) == 7
    rows = []
    for section, (users, rep) in zip(epochs, reps):
        clients = [c for c in data["requests"] if (c["load"], c["rep"], c["turn"]) == (users, rep, 0)]
        assert len(clients) == users
        events = cold_wave(section, clients)
        visible = [c["t0"] + c["ttft"] for c in clients]
        rows.append(dict(users=users, rep=rep, prompt_calls=len(events), measured_prompt_rows=sum(e["new"] for e in events),
                         measured_first_visible_spread_ms=round(1000*(max(visible)-min(visible)), 5),
                         measured_first_turn_ttft_p90_s=round(quantile([c["ttft"] for c in clients], .9), 5),
                         between_first_and_last_prefill_reports_s=round(events[-1]["elapsed"], 5),
                         original_unshortened_chunks=opportunities(events, users, prompt, verify, [], (1,2,4)),
                         geometric_overflow=opportunities(events, users, prompt, verify, geometric, (1,2,4))))
    print(json.dumps(dict(
        source_server_sha256=digest(server), source_client_sha256=digest(load),
        source_server=str(server.relative_to(ROOT)), source_client=str(load.relative_to(ROOT)),
        prompt_cap=prompt[-1], overflow_rows=geometric, width=WIDTH, rows=rows,
        limits="Client times are measured first visible text. Opportunity times are inter-report intervals on the recorded V41 schedule assuming an admissible joint forward and no added time. They are not first-token timings for V232, a replay of changed request lifetimes, acceptance or an H200 speedup. Standard unshortened chunks omit V194's conditional reshaping."
    ), indent=2))


if __name__ == "__main__":
    main()
