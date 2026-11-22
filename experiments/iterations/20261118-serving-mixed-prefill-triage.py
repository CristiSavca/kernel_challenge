"""Summarize real eight-H200 request intervals and scheduler admissions by concurrency."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments/modal_node/20261113T204343Z-bench"
LOAD = next(ARCHIVE.glob("*v41__load.json"))
SERVER = next(ARCHIVE.glob("*v41__server.log"))
data = json.loads(LOAD.read_text())
rung_order = [r["concurrency"] for r in data["rungs"] for _ in range(r["reps"])]
sched = defaultdict(Counter)
rep = -1
for line in SERVER.read_text().splitlines():
    if "Cache flushed successfully!" in line:
        rep += 1
    match = re.search(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*?#running-req: (\d+)", line)
    if rep < 0 or not match:
        continue
    _, tokens, running = map(int, match.groups())
    bucket = sched[rung_order[rep]]
    bucket["prefills"] += 1
    bucket["with_existing_decode"] += int(running > 0)
    bucket["existing_request_prefill_pairs"] += running
assert rep == len(rung_order) - 1

rows = []
for b in sorted(sched):
    reqs = [r for r in data["requests"] if r["load"] == b]
    timed = Counter()
    for rep in sorted({r["rep"] for r in reqs}):
        events = []
        for r in reqs:
            if r["rep"] != rep:
                continue
            t = r["t0"]
            kind = "first" if r["turn"] == 0 else "later"
            events.extend(((t, kind, 1), (t+r["ttft"], kind, -1),
                           (t+r["ttft"], "gen", 1), (t+r["dur"], "gen", -1)))
        events.sort()
        state = Counter()
        prev = events[0][0]
        for t, k, delta in events:
            dt = t - prev
            timed["wall"] += dt
            any_waiter = state["first"] + state["later"] > 0
            if any_waiter and not state["gen"]:
                timed["all_before_first_token"] += dt
            if state["gen"] and not any_waiter:
                timed["all_after_first_token"] += dt
            if any_waiter and state["gen"]:
                timed["mixed"] += dt
            if state["first"] and not state["later"] and not state["gen"]:
                timed["all_cold"] += dt
            state[k] += delta
            prev = t
    answers = sum(r["completion"] for r in reqs)
    pairs = sched[b]["existing_request_prefill_pairs"]
    rows.append(dict(concurrency=b, **sched[b], output_tokens=answers,
                     percent_output_in_one_token_mixed=round(100*pairs/answers,2),
                     implied_percent_if_four_correct=round(400*pairs/answers,2),
                     time_fraction={k:round(v/timed["wall"],5) for k,v in timed.items() if k!="wall"}))
print(json.dumps(dict(source=str(ARCHIVE.relative_to(ROOT)), parent="V41 with 24 GB replica HiCache",
                      rows=rows, assumptions="Historical admissions fixed. Four correct is a scenario from V64 means. Client timing includes queue and service; physical overlap and speedup unmeasured."),indent=2))
