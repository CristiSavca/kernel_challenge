"""Model useful warp count, packed requests, and resource ceilings for the target geometry."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
parent=json.loads((HERE.parent/"v129_routed_projection_pair/work_model.json").read_text())
base=parent[next(k for k,v in parent.items() if isinstance(v,list) and v and isinstance(v[0],dict))]
sm_count=132
cases=[]
for record in base:
    blocks=record["useful_block_iterations_per_projection"]
    phases={}
    for tag,planes,old_ahead in (("W13",2,1),("W2",1,2)):
        bank_b=planes*128*(16//2+1)
        shared=planes*18432+6144
        regs=57 if planes==2 else 58
        alloc_warp=32*((regs+7)//8)*8
        resident_blocks=min(blocks, sm_count*min(65536//(alloc_warp*8), (228*1024)//shared))
        phases[tag]=dict(packed_bytes_per_CTA_K16=bank_b, shared_bytes_per_CTA=shared,
            register_limited_blocks_per_SM=65536//(alloc_warp*8),
            shared_capacity_blocks_per_SM=(228*1024)//shared,
            V147_nominal_bank_lookahead_MiB=blocks*bank_b*old_ahead/2**20,
            V153_ideal_resident_blocks=resident_blocks,
            V153_one_chunk_bank_lookahead_MiB=resident_blocks*bank_b*8/2**20)
    cases.append(dict(target_rows=record["tokens"],mean_useful_blocks=blocks,
        V147_warps_per_SM=2*blocks/sm_count,V153_warps_per_SM=8*blocks/sm_count,
        V147_useful_warps=2*blocks,V153_useful_warps=8*blocks, phases=phases,
        cold_packed_bank_MiB=record["logical_bank_mib"]))
print(json.dumps(dict(h200_sms=sm_count, cases=cases, clusters=0, K=4096,
    stage_chunks=32,cta_barriers_per_route_tile=33, shape_H=4096, shape_I=4096,
    assumptions="Uniform distinct top-eight routing, all useful CTAs counted. Live issue and bandwidth depend on scheduling."),indent=2))
