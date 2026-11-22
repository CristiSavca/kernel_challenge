"""Count the materializations on the V283 pooler and V284 routes; no timing assumptions."""
from hashlib import sha256
import json
from pathlib import Path

here=Path(__file__).resolve().parent
physical=here.parent/"v213_live_decode_expert_reduction/physical.json"
profile=json.loads(physical.read_text())
width, layers, topk = 4096, 62, 8

def work(rows, routes):
    moved=(4*rows*width if rows>1 else 0)
    former_norm_output=2*rows*width
    read_by_old_gather=2*routes*width
    eliminated=moved+former_norm_output+read_by_old_gather
    return dict(rows=rows,local_routes=routes,transpose_bytes=moved,normalized_output_bytes=former_norm_output,
                old_gather_payload_read_bytes=read_by_old_gather,logical_bytes_removed_per_layer=eliminated,
                MiB_across_62_layers=eliminated*layers/2**20,
                route_map_unique_bytes=4*rows*topk,rank_own_gather_send_bytes=2*rows*width//8)

print(json.dumps(dict(width=width,layers=layers,topk=topk,TP=8,
    eligible_materializations_before=["pool_gather_row_reorder","pre_expert_RMS","route_plan","sorted_gather"],
    eligible_materializations_after=["route_plan","rank_major_RMS_to_staging"],
    graph_nodes_removed_per_eligible_pass_per_rank=2*layers,
    uniform_topk_expected=[work(m,m) for m in [8,32,128,256]],
    archived_inputs=[work(c["rows"],c["local_routes"]) for c in profile["cases"]],
    archived_histogram_sha256=sha256(physical.read_bytes()).hexdigest(),
    critical_path="V283 max(P+T,R)+N+Q+G; V284 max(P,R)+Q+F assuming independent streams and ignoring shared resources. P includes projection+transport, T is reorder, N original norm, R router, Q plan, G old sorted gather and F fused routed norm. V272 places N and Q on separate branches, giving max(P+N,R+Q)+G on its separate parent. Moving N behind Q could lose this overlap in a future V272 integration.",
    limits="Logical accesses only. HBM and L2 traffic, fabric time, selective live writes, graph allocator reuse, instruction count, launch overlap and occupancy determine device performance. The native norm reads pooled rows twice and the donor executes both reads for every token. Both retain weighted output staging; the donor reads a small route map as it publishes."
),indent=2))
