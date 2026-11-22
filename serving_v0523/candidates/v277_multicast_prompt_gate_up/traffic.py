"""Enumerate original and clustered output owners on all reconstructed ranks."""
import hashlib
import json
from math import ceil
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
vendor = Path(sys.argv[1])
scheduler = (vendor/"include/humming/scheduler.cuh").read_text()
for expression in ("N_BLOCKS = ProblemShape::N / BlockShape::N / kMultiCastSizeA",
                   "m_id = mn_index / N_BLOCKS;", "n_id = mn_index % N_BLOCKS;",
                   "n_block_id = n_block_id * kMultiCastSizeA + cluster_rank", "dp_mn_next_index += gridDim.x / kMultiCastSize"):
    assert expression in scheduler


def count(hist, fanout, grid):
    tiles = [(expert, local, min(176,n-local)) for expert,n in enumerate(hist) for local in range(0,n,176)]
    bands = 32//fanout
    owners = set()
    blocks = [0]*grid
    issued_rows = [0]*grid
    for group in range(grid//fanout):
        for job in range(group,len(tiles)*bands,grid//fanout):
            tile,group_n = divmod(job,bands)
            for rank in range(fanout):
                n = group_n*fanout + rank
                task = (*tiles[tile],n)
                assert task not in owners
                owners.add(task)
                blocks[group*fanout+rank] += 1
                live=tiles[tile][-1]
                issued_rows[group*fanout+rank] += 32 if live<=32 else 96 if live<=96 else 176
    assert len(owners)==len(tiles)*32
    return owners, dict(M_tiles=len(tiles), CTA_jobs=len(owners), max_jobs_per_CTA=max(blocks), critical_issued_rows=max(issued_rows),
                        activation_full_box_MiB=len(tiles)*bands*176*4096*2/2**20,
                        packed_weight_and_group_scale_MiB=len(owners)*256*4096*(0.5+1/16)/2**20)


cases = []
routes = here.parent/"v274_four_block_prompt_multicast/routes.json"
for case in json.loads(routes.read_bytes())["cases"]:
    ranks=[]
    for rank,hist in enumerate(case["histograms"]):
        ref,x=count(hist,1,132)
        grids={}
        for grid in (116,120,124,128,132):
            work,y=count(hist,4,grid)
            assert ref==work and x["activation_full_box_MiB"]==4*y["activation_full_box_MiB"]
            grids[grid]=y
        ranks.append(dict(rank=rank,local_routes=sum(hist),ordinary=x,four_block=grids))
    cases.append(dict(prompt_tokens=case["tokens"],ranks=ranks))
for hist in ([0]*32,[0]*31+[1],[0]*8+[32,33,96,97,175,176,177]+[0]*17,[5000]+[0]*31):
    assert count(hist,4,128)[0]==count(hist,1,132)[0]
wave = json.loads((here.parent/"v273_prompt_decode_expert_serving/opportunity.json").read_bytes())
ordinary,wide = [cases[1]["ranks"][3][k] for k in ("ordinary","four_block")]
saved=(ordinary["activation_full_box_MiB"]-wide[128]["activation_full_box_MiB"])*2**20
seconds = saved / (4.8e12)*wave["equivalent_16k_cold_layer_calls"]
print(json.dumps(dict(scheduler_sha256=hashlib.sha256(scheduler.encode()).hexdigest(), route_replay_sha256=hashlib.sha256(routes.read_bytes()).hexdigest(),
    cases=cases, archived_32_first_wave_s=wave["archived_first_wave_s"], conditional_seconds=seconds,
    assumptions="Charge every saved 16K rank-3 source box as an HBM miss at 4.8 TB/s and scale by equivalent cold-layer calls. Hardware caching, padded rows and overlapping traffic can change the result. No speed measurement."),indent=2))
