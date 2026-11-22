"""Keep conditional cold/stream work, press-to-test hardware thresholds, and baseline changes distinct."""
from hashlib import sha256
import json
from pathlib import Path

here = Path(__file__).resolve().parent
candidates = here.parent
project = candidates.parents[1]
inputs = {
    "attention": candidates/"v239_split_mixed_full_attention/work.json",
    "prompt": candidates/"v286_alternating_peer_prompt_inputs/work.json",
    "routing": candidates/"v290_mixed_ranked_peer_serving/routed_payload.json",
    "decode": candidates/"v289_overlapped_ranked_pool/work.json",
    "decoder_io": project/"experiments/iterations/20261119-serving-v281-v285.json",
    "expert_timing": candidates/"v288_paired_cold_and_verify_intervals/check.json",
    "score": candidates/"v265_native_serving_profile/audit.json",
}
j={label:json.loads(p.read_bytes()) for label,p in inputs.items()}
routed=next(r for r in j["routing"]["cases"] if r["tokens"]==16384)
transfer=routed["least_incoming_payload_saved_MiB"]
rungs=[]
for row in j["decode"]["by_rung"]:
    u=row["users"]
    geo=j["attention"]["rungs"][str(u)]
    spans=[r for r in j["prompt"]["per_load_conditional"] if r["users"]==u]
    arrivals=geo.get("full_joint_arrivals",0)
    fits=geo.get("prompt_pack_certain_fit",0)
    # This is a fixed-cost illustration. Real joins have variable cohort size, service time and acceptance.
    modeled_decode_seconds=arrivals*row["V64_proxy_verify_ms"]/1000
    old=next(r for r in j["decoder_io"]["combined_opportunities_vs_v229"] if r["users"]==u)
    seconds=sum(r["elapsed_seconds"] for r in spans)
    rungs.append(dict(users=u,logged_repetitions=len(spans),archived_wall_seconds=seconds,
        full_chunk_join_opportunities=arrivals,full_attention_prompt_packs_fitting_byte_cap=fits,
        full_layers=16,sliding_layers=46,fit_full_layer_calls=fits*16,
        size_eligible_prompt_layer_calls=sum(r["eligible_size_layer_calls"] for r in spans),
        ranked_nodes_removed_vs_V229_per_eligible_TP8_verify=old["avoided_graph_nodes_per_rank_pass"],
        uniform_logical_MiB_removed_vs_V229_per_eligible_TP8_verify=old["logical_payload_read_write_avoided_MiB_per_rank_pass"],
        V64_period_proxy_ms=row["V64_proxy_verify_ms"],at_one_fully_hidden_round_per_join_seconds=modeled_decode_seconds,
        illustrative_wall_ratio_if_only_this_work_were_free=seconds/(seconds-modeled_decode_seconds)))
print(json.dumps(dict(candidate="V291",per_load=rungs,
    peer_16k_synthetic_fixture=dict(dense_remote_per_rank_MiB=routed["baseline_incoming_MiB_per_rank"],
        highest_peer_input_per_rank_MiB=routed["peak_incoming_MiB"],highest_peer_output_per_rank_MiB=routed["peak_outgoing_MiB"],
        least_saved_per_rank_MiB=transfer),
    memory="TP8 peer slab is 32.5 MiB versus 16.25 MiB on the single-bank parent under expanded rows. Short-tail and packed prompt temporaries are independently bounded; peak allocations across both graph pools require a real node.",
    physical_cross_check=dict(matched_four_user_full_chunk_difference_ms=1000*j["expert_timing"]["matched_cold_summary"]["4"]["mean_difference_s"],
        cold_ms_per_16k_equivalent_layer_in_uniform_coupled_4x_scenario=j["expert_timing"]["uniform_32user_demand_ms_per_layer"]["4"],
        throughput_only_and_coupled_4x_crossings=j["score"]["score_scenarios"]["crossings"]["4"]),
    limitations="Each V41 run was a different historical candidate. The join illustration credits one full V64-period estimate per join and ignores new split attention costs, actual scheduling and changing cohort widths. The peer row counts use a synthetic routing fixture. The ranked I/O and launch counts assume eligible graph paths and uniform expert assignment. These scenarios have different baselines and do not add into a serving score.",
    input_sha256={key:sha256(p.read_bytes()).hexdigest() for key,p in inputs.items()}),indent=2))
