"""Logical transport on eligible prompt calls; no assumptions about measured NVLink or cache bandwidth."""
from math import comb
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
size, experts, local, topk, hidden = 8, 256, 32, 8, 4096
p = 1 - comb(experts-local,topk)/comb(experts,topk)
shapes={}
for n in (8192,16384,16640):
    B = n*hidden*2/(1<<20)
    R = n*topk*local/experts
    normal_remote = B*(size-1)/size
    direct_remote = normal_remote*p
    # Both sub-16K input shapes run on the same configured 16K serving instance.
    resident=max(16384,n)*hidden*2/(size*(1<<20))
    shapes[n] = dict(rows=n, expected_rank_local_routes=R, expected_tokens_needed=n*p,
        replicated_write_MiB=B, symmetric_owner_copy_read_and_write_MiB=2*B/size,
        input_reads_before_W13_MiB=dict(V229=R*hidden*2/(1<<20),V230=B*p),
        equal_sorted_W13_input_writes_MiB=R*hidden*2/(1<<20),
        baseline_received_remote_MiB=normal_remote,peer_remote_MiB=direct_remote,
        remote_MiB_removed=normal_remote-direct_remote,
        approximate_live_workspace_net_MiB=dict(while_dense_input_would_be_live=resident-B,outside_prompt=resident))
physical = json.loads(next(ROOT.parent.glob('v197*/independent_reach.json')).read_bytes())['archive']
evidence = json.loads(next(ROOT.parent.glob('v179*/traffic.json')).read_bytes())['V41_archived_prompt_shapes']
exposure = [{**t,'need_ms_per_sparse_layer_for_five_percent':t['net_ms_per_layer_call_for_5pct_raw']}
            for t in physical if t['users'] in (16,32)]
print(json.dumps(dict(hardware_measurement=False,uniform_distinct_top8=True,needed_fraction=p,
    model_cuda_storage_units='MiB',shapes=shapes, archived_V41_captured_row_exposure=evidence,
    matching_trace_crossover_ms={r['users']:r['need_ms_per_sparse_layer_for_five_percent'] for r in exposure},
    bounds='Uniform routing; compare network bytes separately from touched arrays. Dense collective algorithm, SM occupancy, cache, write coalescing, new vote, map-read overhead, barriers and physical latencies unmeasured.'),indent=2))
