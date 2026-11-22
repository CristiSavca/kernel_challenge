"""Uniform top-k expectation for the narrow ascending-K projection path."""
import json
import math

MIB = 1024 ** 2


def model(tokens):
    experts, local, topk, hidden, inner = 256, 32, 8, 4096, 4096
    slots = tokens * topk
    live = slots * local / experts
    chance = topk / experts
    pmf = [math.comb(tokens, n) * chance ** n * (1-chance) ** (tokens-n) for n in range(tokens+1)]
    unique = local * (1-pmf[0])
    tiles = local * sum(math.ceil(n/8) * p for n,p in enumerate(pmf))
    groups = max(1, min(slots, local, math.ceil(live)))
    used = min(slots, local)
    max_tiles = used + (slots-used)//8
    bank = 3 * inner * hidden * 9 / 16
    return dict(tokens=tokens, live_routes=live, unique_experts=unique, eight_row_tiles=tiles,
                groups=groups, launched_blocks_per_projection=groups*hidden//128,
                useful_block_iterations_per_projection=tiles*hidden/128, useful_warps_per_projection=tiles*hidden/64,
                v125_cluster_blocks_for_same_unique_experts=8*unique,
                logical_bank_mib=tiles*bank/MIB, unique_bank_mib=unique*bank/MIB,
                ideal_cold_weight_floor_us=unique*bank/4.8e12*1e6,
                avoided_indexed_gate_scale_activation_and_down_scale_mib=live*(16*inner+4*hidden)/MIB,
                logical_input_reads_per_projection_mib=live*hidden*2*(hidden//64)/MIB,
                new_eager_vector_slabs_mib=slots*(inner+hidden)*2/MIB,
                parent_eager_vector_slabs_mib=slots*(inner+max(2*inner,hidden))*2/MIB,
                plan_capacity_bytes=4*(1+9*max_tiles), planner_logical_id_reads_bytes=slots*local*4,
                planner_scratch_bytes=8*slots*4)


if __name__ == "__main__":
    print(json.dumps(dict(assumptions="Independent uniform top-eight per token, 256 global experts, EP8, hidden/intermediate 4096; 4.8 TB/s peak H200.",
                          cases=[model(8),model(32)],hardware_timing=False),indent=2))
