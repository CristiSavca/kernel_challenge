"""Verify the frozen route path and price its explicit shape-dependent allocations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parent
repo = Path(sys.argv[1]).resolve()
receipt = json.loads((HERE / "validation.json").read_bytes())
assert hashlib.sha256((HERE / "patch.diff").read_bytes()).hexdigest() == receipt["patch_sha256"]
parent, tree = receipt["parent_tree"], receipt["tree"]


def check(tree_id, path, fragments):
    path = "python/sglang/" + path
    content = subprocess.check_output(["git", "-C", repo, "show", f"{tree_id}:{path}"]).decode()
    assert all(part in content for part in fragments), path
    return {"git_tree": tree_id, "path": path,
            "sha256": hashlib.sha256(content.encode()).hexdigest()}


fp4 = "srt/layers/muse_spark/muse_spark_fp4_moe.py"
humming = "srt/layers/muse_spark/muse_spark_humming_moe.py"
peer = "srt/layers/muse_spark/peer_prompt_pack.py"
proof = [
    check(parent, "srt/models/muse_spark.py", ["replicate_expert_input = backend.is_marlin() or backend.is_humming()",
          "moe_layer._peer_prompt_inputs = peer_workspace", "self._routing_op()(",
          "return self._forward_sp_experts(", "return self._forward_sp_replicated_experts(",
          "num_tokens * self.top_k >= self.num_experts * _HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT",
          "if workspace is not None and not torch.compiler.is_compiling():", "peer_experts_into(workspace",
          "return layernorm_sp.reduce_scatter_token_rows(output, tp_group)"]),
    check(parent, "kernels/ops/moe/muse_spark_routing_jit.py", ["if topk_ids is None:",
          "(num_tokens, topk), dtype=torch.int32, device=gating_output.device",
          "return topk_weights_view, topk_ids_view, packed_view"]),
    check(parent, fp4, ["_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT = 256", "expert_map = torch.full((num_experts,), -1, dtype=torch.int32)",
          "expert_map[start : start + num_local_experts] = torch.arange("]),
    check(parent, peer, ["comm.has_multicast and comm.pull_enabled","torch.cuda.get_device_capability(comm.device)[0] == 9",
          "experts.use_w4a16 and experts.use_humming", "0 < shard.shape[1] <= experts.intermediate_size_per_partition",
          "_gather_prompt_routes(group, scores, ids, eligible)", "local_humming_expert_ids(experts, ids[:tokens])",
          "sorted_experts, slots, offsets = prompt_route_order(", "decision = torch.cat((votes, first_local)).tolist()"]),
    check(parent, "srt/layers/moe/token_dispatcher/standard.py", ["return torch.where(", "topk_ids < 0",
          "local_expert_mapping[topk_ids.clamp_min(0)]"]),
    check(parent, humming, ["local_ids = _map_topk_ids_to_local_experts(", "local_ids.flatten().sort(stable=True)",
          "offsets = torch.searchsorted(route_experts, experts, out_int32=True)", "_pack_prompt_routes[("]),
    check(tree, peer, ["counts = count_peer_routes(ids[:tokens], experts)", "if counts is None:",
          "count = counts[2][-1:]", "if all(decision[:-1]):", "grouped = place_peer_routes(local_ids, counts, decision[-1])"]),
    check(tree, "kernels/ops/moe/muse_spark_peer_pack.py", ["if ids.dtype != torch.int32", "32 <= tiles <= 2048",
          "expert = tl.load(ExpertMap + expert", "key = tl.sort(bucket * TILE + lane",
          "sorted_experts = ids.new_empty(routes)", "tl.store(RowMap + original",
          "tl.store(SortedExperts + row, bucket, bucket < EXPERTS)"]),
]
geometry = json.loads((CANDIDATES/"v179_peer_prompt_inputs/traffic.json").read_bytes())["geometry"]
world, experts, local, topk = [geometry[k] for k in ("ranks","experts","local_experts","distinct_routes")]
assert (world, experts, local, topk) == (8,256,32,8)
shapes = {}
for tokens in (8192, 16384, 16640):
    slots = tokens * topk
    tiles = (slots + 255) // 256
    live = slots // world
    old = {"full_sorted_int32_expert_ids": slots*4, "full_int64_permutation": slots*8}
    new = {"compact_int32_expert_ids": live*4, "int32_tile_prefix": (local+1)*tiles*4,
           "int32_totals": (local+1)*4}
    shapes[tokens] = dict(routed_slots=slots, average_live_routes=live, average_inactive_routes=slots-live,
                          tile_ctas_each_count_place=tiles, expert_scan_ctas=local+1,
                          parent_explicit_bytes=old, candidate_explicit_rank_mean_bytes=new,
                          rank_mean_reduction_bytes=sum(old.values())-sum(new.values()))
archive = json.loads((CANDIDATES/"v197_grouped_prompt_gate_up/independent_reach.json").read_bytes())["archive"]
wall = json.loads((CANDIDATES/"v193_scheduler_cache_poll/trace.json").read_bytes())["V41"]["rounds"]
score = json.loads((CANDIDATES/"v244_cold_prompt_scaling/check.json").read_bytes())["official_V64"]["score"]
observations = []
for sample, timed in zip(archive,wall,strict=True):
    assert (sample["users"],sample["rep"]) == (timed["concurrent_users"],timed["rep"])
    calls = sample["logical_grouped_passes"]*62
    elapsed = timed["wall_s"]
    def net_ms(factor):
        return elapsed * 1000 * (1-1/factor) / calls
    observations.append(dict(users=sample["users"], rep=sample["rep"], total_prefill_passes=sample["all_passes"],
                             eligible_size_prefill_passes=sample["logical_grouped_passes"],
                             eligible_size_prompt_rows=sample["eligible_logical_rows"],
                             eligible_size_layer_calls=calls, elapsed_seconds=elapsed,
                             net_ms_per_call_for_5pct_raw=net_ms(1.05),
                             net_ms_per_call_for_4x_from_V64_reference=net_ms(4/score)))
print(json.dumps(dict(candidate="V258", patch_sha256=receipt["patch_sha256"], native_source=proof,
    geometry=geometry, shapes=shapes, observed_V64_score=score, archive_V41_conditional_exposure=observations,
    allocation_model="Explicit sort-specific V256 tensors vs V258 histogram plan tensors. Assumes eight valid global routes and eight ranks. Every route has one owner, so rank-mean is exact regardless of skew. Common local IDs, inverse, offsets, activations, library sort scratch, allocator behavior and transpose/remap intermediates excluded. This is not a measured peak or a bandwidth estimate.",
    timing_model="Archived V41 wall and prompt-size distributions illustrate the required net reduction per eligible 62-layer prompt call at fixed work. Graph, topology, hooks or eligibility may reduce exposure; no candidate H200 latency or end-to-end score measured.",
    physical_candidate_timing=False),indent=2))
