"""Compute activation traffic for real prompt shapes with modeled expert assignments."""

from collections import Counter
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
archive = json.loads((HERE.parent / "v163_sparse_prompt_exchange/geometry.json").read_bytes())
shape_trials = archive["seeded_uniform_split_decisions"]
experts, local_experts, topk, world, channels = 256, 32, 8, 8, 4096
hit = 1 - math.comb(experts - local_experts, topk) / math.comb(experts, topk)
mib = 2**20
rows = {}
for n_str, observations in shape_trials.items():
    n = int(n_str)
    shard = n // world * channels * 2 / mib
    packed = n * hit * channels * 2 / mib
    histogram = Counter(dict(observations["production_destination_width_histogram"]))
    assert all(0 < width < world for width in histogram)
    # Count explicit BF16 tensor reads and writes averaged over ranks. Transport
    # reads its send buffers and writes the receive buffer once in this model.
    packed_hbm = 4 * packed
    direct_hbm = 2 * shard + 2 * packed
    rows[n] = dict(local_publication_MiB=shard, expected_compact_input_MiB=packed,
        remote_payload_per_rank_MiB=packed * (world-1) / world,
        previous_nominal_tensor_traffic_MiB=packed_hbm, direct_nominal_tensor_traffic_MiB=direct_hbm,
        nominal_difference_MiB=packed_hbm-direct_hbm,
        conditional_difference_microseconds_at_4_8_TB_s=(packed_hbm-direct_hbm)*mib/4.8e6,
        observed_selector_simulations=observations["trials"],
        previous_input_alltoalls_histogram={world//width:count for width,count in histogram.items()},
        new_publication_copies=1, new_peer_reads=1, new_device_barriers=2)
result=dict(geometry=dict(experts=experts, local_experts=local_experts, distinct_routes=topk,
                         ranks=world, hidden=channels),
            assumptions="Uniform independent distinct top-k expert choices per token; 2-byte BF16. Each rank runs the sparse branch. Identical remote payload; HBM traffic and NVLink time are unmeasured.",
            shapes=rows, V41_archived_prompt_shapes=archive["rows"],
            resident_workspace_bound_B_at_16384=16384//world*channels*2,
            full_chunk_nominal_saved_GiB_over_62_upper_bound=rows[16384]["nominal_difference_MiB"]*62/1024,
            physical_serving_run=False, H200_peak_memory=None, measured_V64_excess_GB=1.9)
print(json.dumps(result, indent=2))
