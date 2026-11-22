"""Size the candidate's peer exchanges on archived mixed-forward opportunities."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
with redirect_stdout(io.StringIO()):
    transport = runpy.run_path(str(next(HERE.parent.glob("v185_*/geometry.py"))))["geometry"]
schedule = runpy.run_path(str(next(HERE.parent.glob("v202_*/phase_budget.py"))))
analyze = schedule["summarize"]
batches = []


def capture(events):
    batches.append(events)
    return analyze(events)


schedule["main"].__globals__["summarize"] = capture
rungs = schedule["main"]()["loads"]
assert len(batches) == 3 * len(rungs)
results = {}
for i, (users, data) in enumerate(rungs.items()):
    all_events, ordinary, shifted = batches[3*i:3*i+3]
    assert sorted(e["live_rows"] for e in all_events) == sorted(e["live_rows"] for e in ordinary+shifted)
    owners = [e for e in all_events if e["projected_gate"]]
    sizes = [transport(e["live_rows"]) for e in owners]
    chosen = [v for v in sizes if v["passes_geometry"]]
    assert len(all_events) == data["paper_schedule"]["calls"]
    assert len(owners) == data["paper_schedule"]["normalized_context_calls"]
    # Captured graph padding can enlarge the producer shard. The current transport
    # also requires its rows to equal ceil(live_tokens / tp). Show both cases.
    chosen_exact = [v for e,v in zip(owners,sizes) if v["passes_geometry"] and
                    (e["captured_rows"]+7)//8 == v["aligned_rows"]//8]
    def totals(entries):
        return dict(calls=len(entries), sent_prior=sum(v["parent_wire_bytes"] for v in entries),
                    sent_candidate=sum(v["total_wire_bytes"] for v in entries),
                    carrier_prior=sum(v["parent_gathered_bytes"] for v in entries),
                    carrier_candidate=sum(v["live_projected_bytes"] for v in entries))
    results[users] = dict(mixed_opportunities=len(all_events), owner_eligible=len(owners),
                          live_shards=totals(chosen), padded_capture_shards=totals(chosen_exact))
print(json.dumps(dict(by_load=results, source="V41 H200 scheduling chronology; 7 repetitions; static TP8 geometry",
    assumptions="Compatible fused helper, unit model value multiplier, BF16 FC/RMSNorm/KV bank and PyNCCL transport. "
    "Data are prospective per-rank logical payloads at the prior trace geometry. The two columns assume that "
    "the model tail receives a tightly packed live shard or a capture-bucket-padded shard respectively.",
    physical_H200=False),indent=2))
