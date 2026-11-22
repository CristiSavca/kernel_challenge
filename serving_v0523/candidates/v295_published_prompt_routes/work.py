"""Compare route publications on the same archived prefill calls and TP geometry."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
inputs = {}
def read(pattern):
    path, = here.parent.glob(pattern)
    raw = path.read_bytes()
    inputs[str(path.relative_to(here.parent))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

traffic = read("v242_*/work.json")
archive = read("v286_*/work.json")
timing = read("v288_*/check.json")
norm = read("v293_*/work.json")
replay = read("v295_*/independent_routes.json")
world, topk, tokens = (traffic[k] for k in ("world", "topk", "tokens"))
layers, width, cap = (archive[k] for k in ("sparse_layers", "hidden_width", "graph_rows"))
assert tokens == 16384 and world == topk == 8 and layers == 62
assert replay["totals"]["observed_calls"] == 276

def packet(rows):
    return 4 * (2 * rows * topk + 4)

row_bytes = width * 2
cap_rows = (cap + world - 1) // world
packet_rows = (packet(cap_rows) + row_bytes - 1) // row_bytes
extra = 2 * packet_rows * row_bytes
per_load = []
for row in archive["per_load_conditional"]:
    calls = row["eligible_size_layer_calls"]
    per_load.append(dict(users=row["users"], rep=row["rep"], size_eligible_layers=calls,
        route_collectives_removed_if_always_admitted=calls * 3,
        net_ms_per_layer_for_five_percent_raw=row["net_ms_per_call_for_5pct_raw"],
        net_ms_per_collective_if_sole_change=row["net_ms_per_call_for_5pct_raw"]/3))
remote = packet(tokens//world) * (world-1)
assert remote == traffic["packed_remote_bytes"]
print(json.dumps(dict(geometry=dict(layers=layers, width=width, topk=topk, tp=world, tokens=tokens),
    per_eligible_pass=dict(V291_separate_route_collectives=3*layers, V242_packed_route_collectives=layers,
        V295_and_V296_separate_route_collectives=0, V295_and_V296_activation_publication_barriers=layers,
        V296_logical_normalized_copy_bytes_removed_per_rank=norm["eligible_16k"]["whole_chunk_copy_bytes_removed_per_rank"],
        V296_copy_launches_removed=norm["eligible_16k"]["native_copy_launches_removed"]),
    modeled_received_route_bytes_per_rank_layer=dict(V291=traffic["native_remote_bytes"],V295_and_V296=remote),
    persistent_per_rank=dict(graph_row_cap=cap, bank_route_padding_rows=packet_rows,
        routing_slab_bytes_added_for_both_banks=extra,
        total_two_slabs_bytes=archive["double_peer_payload_bytes"]+extra,
        V295_eligibility_constant_bytes=16, V296_identity_map_bytes=norm["eligible_16k"]["identity_map_bytes_at_16640"]),
    archived_calls=per_load,
    coupled_uniform_32user_demand_ms_per_layer=timing["uniform_32user_demand_ms_per_layer"],
    scope="Count only size-eligible calls on the archived V41 schedule. Runtime admission, mapped peer-read cost, new packing/copy kernels, wire protocol bytes, cache effects, graph allocator peak and measured speed are unknown. A veto may add one publication before restoring the native collectives.",
    sources_sha256=inputs),indent=2))
