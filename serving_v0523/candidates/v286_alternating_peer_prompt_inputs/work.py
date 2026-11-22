"""Count device barrier epochs, storage, and conditional exposure in the earlier V41 trace."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
archive = json.loads((here.parent / "v258_compact_prompt_route_plan/work.json").read_bytes())
schedule = json.loads((here / "cold_schedule.json").read_bytes())
layers, world, width, max_rows = 62, 8, 4096, 16640
bank_bytes = (max_rows // world) * width * 2
observations = []
for sample in archive["archive_V41_conditional_exposure"]:
    calls = sample["eligible_size_prefill_passes"] * layers
    assert calls == sample["eligible_size_layer_calls"]
    observations.append(dict(**sample, parent_publication=calls, parent_retirement=calls,
                             candidate_publication=calls, candidate_retirement=0))
projection = []
for case in schedule["results"]:
    for grid in ("128","132"):
        ranks = [r[grid] for r in case["ranks"]]
        ratios = [r["windows"]["8"]["native_issue_time_ratio"] for r in ranks]
        projection.append(dict(tokens=case["rows"],resident_blocks=int(grid),
            full_native_over_perfect_balanced=max(r["native"] for r in ranks) / max(r["perfect_schedule_lower_bound"] for r in ranks),
            eight_expert_windows_issue_ratio_range=[min(ratios),max(ratios)],
            rank3_full_MiB=ranks[3]["windows"]["32"]["intermediate_MiB"],rank3_eight_expert_MiB=ranks[3]["windows"]["8"]["intermediate_MiB"]))
print(json.dumps(dict(sparse_layers=layers,TP=world,graph_rows=max_rows,hidden_width=width,
    parent_peer_payload_bytes=bank_bytes,double_peer_payload_bytes=2*bank_bytes,
    extra_peer_payload_bytes=bank_bytes,extra_logical_address_table_bytes=world*8,
    generic_per_bank_byte_cap=17*2**20,per_16K_prompt={"parent_publication":layers,"parent_retirement":layers,
        "double_publication":layers,"double_retirement":0,"single_fallback_publication":layers,"single_fallback_retirement":layers},
    per_load_conditional=observations,projection=projection,
    interpretation="V41 logged prompt sizes and durations. Counts assume all size-eligible calls dispatch V286 peer packing and acquire two slabs. A veto, incompatible topology, unsupported graph, or single-slot fallback reduces the exposure. Memory equations count owned data payload and pointer entries, excluding allocator rounding, signal pads, failure transients and other allocations. No physical V279 or V286 measurement.",
    candidate_timing=False),indent=2))
