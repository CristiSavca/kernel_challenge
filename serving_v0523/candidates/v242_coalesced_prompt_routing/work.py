"""Count explicit eager metadata allocations and modeled prompt opportunities."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
T,K,P=16384,8,8
# n = rank-local bytes of [score FP32, id int32].
n=(T//P)*K*8
s=T*K*8
# Three original collectives: scores, ids and one 4-byte vote.
# New gather sends the same scores and ids plus 16 bytes of vote/padding.
# Largest new packet/transpose window includes both gather output and restored planes.
original=(P-1)*(n+4)
candidate=(P-1)*(n+16)
pack_peak=n+16 + (n+16)*P + s + 16
native_peak=s+4*(P+1)
archive=json.loads((HERE.parent/"v197_grouped_prompt_gate_up/independent_reach.json").read_text())["archive"]
print(json.dumps(dict(tokens=T,world=P,topk=K,native_collectives=3,packed_collectives=1,
    local_send_buffer_bytes=n+16,score_id_result_bytes=s,incremental_transport_bytes=candidate-original,
    native_remote_bytes=original,packed_remote_bytes=candidate,normal_packet_phase_peak_bytes=pack_peak,
    normal_parent_output_and_vote_bytes=native_peak,normal_modeled_extra_bytes=pack_peak-native_peak,
    fallback_retained_plane_and_two_clones_bytes=2*s, fallback_clone_upper_extra_bytes=s,
    local_repack_and_unpack_logical_bytes=2*n+2*s,
    excluded="Original per-rank routing tensors, CUDA graph allocator pool, framework temporaries and protocol scratch. Fallback clone upper bound excludes common hidden-state gather.",
    archive=[dict(users=x["users"],rep=x["rep"],eligible_prompt_passes=x["logical_grouped_passes"],
                  collectives_removed_if_eligible=124*x["logical_grouped_passes"],
                  net_us_per_eliminated_collective_for_5pct=x["net_ms_per_layer_call_for_5pct_raw"]*500)
             for x in archive]),indent=2))
