"""Model prompt expert communication, conditional on routing and a ring all-reduce."""
import json
import math
from pathlib import Path

here = Path(__file__).resolve().parent
parent = json.loads((here.parent / "v163_sparse_prompt_exchange" / "geometry.json").read_text())
hit = 1 - math.comb(256 - 32, 8) / math.comb(256, 8)
tp, tokens, hidden = 8, 16384, 4096
full = tokens * hidden * 2
gather = full * (tp - 1) / tp
sparse = gather * hit
assert abs(parent["uniform_model"]["full_chunk_expected_sparse_remote_B_per_rank_per_layer"] - sparse) < 1
print(json.dumps({
    "tokens": tokens, "hidden": hidden, "ranks": tp, "topk": 8,
    "assumptions": "Every token independently draws eight distinct global experts, 32 per rank. All ranks enter sparse exchange. Dense reference uses the logical byte volume of a bandwidth-optimal ring. The actual NCCL protocol on V64 is unknown.",
    "remote_MiB_sent_per_rank_per_layer": {
        "V64_dense_input_gather": gather / 2**20,
        "V64_or_V163_output_allreduce_if_ring": 2 * gather / 2**20,
        "V163_or_V167_sparse_input": sparse / 2**20,
        "V167_sparse_output": sparse / 2**20,
        "ring_V64_pair": 3 * gather / 2**20,
        "ring_V163_pair": (sparse + 2 * gather) / 2**20,
        "V167_pair": 2 * sparse / 2**20,
    },
    "finalizer_rows_per_expert_rank": {"V163": tokens, "expected_V167": tokens * hit},
    "finalizer_BF16_output_MiB_per_rank": {"V163": full / 2**20, "expected_V167": full * hit / 2**20},
    "fixed_graph_BF16_output_MiB_per_rank": {"V163_before_native_reduce": full / 2**20, "V167": full / tp / 2**20},
    "grouped_return_extra_FP32_accumulator_MiB_per_rank": full * 2 / tp / 2**20,
    "rank_zero_route_buffer_minimum_B": hidden * 2,
    "seeded_16384_uniform_parent_groups": parent["seeded_uniform_split_decisions"]["16384"],
    "return_collectives_per_sparse_pass": "one per group of expert ranks, with the same width as the outgoing input exchange",
    "physical_H200": False
}, indent=2))
