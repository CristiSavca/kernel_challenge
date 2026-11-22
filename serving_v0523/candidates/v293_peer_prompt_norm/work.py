"""Bound expert ownership, normalized traffic and illustrative latency in the 16K trace."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1])
model_path="python/sglang/srt/models/muse_spark.py"
src=(stage/model_path).read_text()
for part in ('if config.num_shared_experts:', 'if parallel.moe_tp_size != 1:', 'if not int4_checkpoint:', 'The torch-native Muse MoE requires expert TP1',
             'self.num_local_experts = self.num_experts // self.ep_size','self.local_expert_start = self.ep_rank * self.num_local_experts',
             'num_local_experts=self.num_local_experts','local_expert_offset=self.local_expert_start'):
    assert part in src
before=here.parent/"v288_paired_cold_and_verify_intervals/check.json"
routes=here.parent/"v274_four_block_prompt_multicast/routes.json"
trace=json.loads(before.read_bytes())
rows=next(i["histograms"] for i in json.loads(routes.read_bytes())["cases"] if i["tokens"]==16384)
ranks=[sum(r) for r in rows]
resident=[set(range(32*i,32*(i+1))) for i in range(8)]
assert len(set.union(*resident))==256 and sum(map(len,resident))==256
layers,hidden,tokens,world=62,4096,16384,8
count=tokens//world
payload=count*hidden*2
traffic=2*payload
copy_ms=traffic*layers/4.8e12*1000
measured=trace["matched_cold_summary"]["4"]
print(json.dumps(dict(source_sha256={str(p.name):sha256(p.read_bytes()).hexdigest() for p in (before,routes)},
    geometry=dict(ep=8,tp=8,expert_tp=1,global_experts=256,local_experts=32,local_routes_16k=ranks,
        ideal_fully_replicated_route_balance_speedup=max(ranks)/(sum(ranks)/len(ranks)),
        packed_expert_bytes=(8192+4096)*4096*9//16,
        all_expert_weights_per_rank_assigned_once=True),
    eligible_16k=dict(layers=layers,shard_rows=count,bytes_of_native_norm_destination=payload,copy_bytes_removed_per_rank_layer=traffic,
        whole_chunk_copy_bytes_removed_per_rank=traffic*layers,native_copy_launches_removed=layers,
        native_rstd_output_bytes_also_elided_per_layer=count*4,
        identity_map_bytes_at_16640=16640//8*4,
        direct_ranked_writes_into_same_physical_slot_as_copy=True),
    per_16K_ms=dict(peak_bandwidth_equivalent_at_4_8_TB_s=copy_ms,
        time_at_one_microsecond_per_removed_launch=layers/1000,
        historical_four_user_V44_mean=measured["new_mean_s"]*1000,historical_four_user_V41_V44_delta=measured["mean_difference_s"]*1000,
        extra_saving_for_fourfold_historical_chunk=measured["required_seconds_beyond_v44_for_v41_chunk_factor"]["4"]*1000),
    conditional_by_load=[dict(users=r["users"],rep=r["rep"],publications=r["eligible_size_layer_calls"],
        minimum_saved_copy_bytes=(r["eligible_size_prompt_rows"]+world-1)//world*hidden*4*layers,
        upper_saved_copy_bytes=(r["eligible_size_prompt_rows"]+r["eligible_size_prefill_passes"]*(world-1))//world*hidden*4*layers)
        for r in json.loads((here.parent/"v258_compact_prompt_route_plan/work.json").read_bytes())["archive_V41_conditional_exposure"]],
    interpretation="Rank-balance ratio counts only live rows, assumes equal cost, ideal replication, free movement and no scheduling effects. 16K traffic is logical, throughput equivalent is not a physical saving. Per-repetition logical byte ranges apply the archived eligible token totals and bound TP8 padding per invocation. Actual branch admission, GPU memory service, scheduling and overlap, and per-token identity table loads are unmeasured."),indent=2))
