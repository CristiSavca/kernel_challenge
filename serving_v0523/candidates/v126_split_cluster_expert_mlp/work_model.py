"""Static native-width split geometry under a uniform top-eight routing model."""
import json
import math
MiB=1024**2

def row(tokens):
    e,g,k,h,sms=32,256,8,4096,132
    probabilities=[math.comb(tokens,j)*(k/g)**j*(1-k/g)**(tokens-j) for j in range(tokens+1)]
    active=e*(1-probabilities[0]); live=tokens*k*e/g
    splits=4 if active*16<2*sms else 2
    groups=min(e,max(1,math.ceil(active+math.sqrt(active*probabilities[0]))))
    jobs8=e*sum(math.ceil(j/8)*p for j,p in enumerate(probabilities))
    bank=3*h*h*9/16
    return dict(target_rows=tokens,P=splits,workgroups=groups,all_cluster_ctas_launched=groups*8*splits,
                all_cluster_ctas_with_full_static_grid=e*8*splits,
                expected_distinct_experts=active,expected_eightrow_subtiles=jobs8,
                expected_live_stripe_ctas_before_reuse=8*jobs8*splits,
                expected_live_unsplit_ctas=8*active,
                projected_bank_reads_mib=bank*jobs8/MiB,unique_packed_bank_mib=bank*active/MiB,
                cold_weight_floor_us=1e6*bank*active/4.8e12,
                live_partial_writes_plus_reads_mib=2*splits*live*h*4/MiB,
                logical_remote_shared_reads_mib=(8-1)*(h//(8*64))*live*h*2/MiB,
                stage_plus_partials_mib=tokens*k*h*(2+4*splits)/MiB,
                original_stage_plus_gate_arena_mib=tokens*k*h*6/MiB,
                additional_eager_tensor_bytes_vs_v68=tokens*k*h*(4*splits-4),
                w13_shared_stripe_write_read_mib=2*jobs8*8*splits*(8*2*4*32*16)/MiB)
print(json.dumps(dict(assumption="Independent uniform per-token routing, 256 global experts, EP8, 132 SM Hopper, native 8 proposal width; finite legal weights and inputs",
                      model=[row(8),row(32)],hardware_measurement=False),indent=2))
