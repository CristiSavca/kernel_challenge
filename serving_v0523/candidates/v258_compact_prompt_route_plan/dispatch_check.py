"""Apply the V242 multi-rank fixture to V258 and V256 with CPU stand-ins for the route kernels."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/"v242_coalesced_prompt_routing/distributed.py"
source=ORIGINAL.read_text()


def change(old,new):
    global source
    assert source.count(old)==1,old
    source=source.replace(old,new)


# Retain the inherited model, expert, peer and Gloo transport fixtures. Use the coalesced
# V256 parent, so both sides should make the same collectives on every branch.
change("v241_split_serving_decode","v256_window_backed_mixed_serving")
change("    parent_work=dict(work_env)",'''    definitions(MANIFEST['tree'],WORK,work_env,{'_gather_prompt_routes'})
    parent_work=dict(work_env)''')
change("e.use_w4a16=e.use_humming=True;e.num_local_experts=4","""e.use_w4a16=e.use_humming=True;e.num_local_experts=4
        e.global_num_experts=4*size
        mapping=torch.arange(4*size,dtype=torch.int32)
        e._expert_map=torch.where(mapping//4==rank,mapping%4,-1)""")
change("    def gate(layer,inputs,keys,offsets,tuning,prompt):",'''    kernel_path="python/sglang/kernels/ops/moe/muse_spark_peer_pack.py"
    class Kernel:
        def __init__(self,fn):self.fn=fn
        def __getitem__(self,grid):return self.fn
    def histogram(ids,mapping,local,prefix,slots,tiles,glob,map_ids,experts,bins,tile):
        if map_ids:
            local.copy_(torch.where(ids<0,-1,mapping[ids.clamp_min(0).long()]))
        labels=local.flatten(); pos=torch.arange(slots)
        valid=(labels>=0)&(labels<experts)
        keys=labels[valid].long()*tiles+pos[valid]//tile
        prefix[:experts].copy_(torch.bincount(keys,minlength=experts*tiles).view(experts,tiles))
        prefix[-1].copy_(torch.bincount(pos[valid]//tile,minlength=tiles))
        stats['counted']+=1
    def scan(prefix,totals,tiles,span):
        totals.copy_(prefix.sum(1));prefix.copy_(prefix.cumsum(1)-prefix)
    def place(ids,prefix,totals,ordered,mapped,offsets,slots,tiles,experts,bins,tile):
        idx=torch.arange(slots);v=ids.flatten();live=idx[v>=0]
        valid=live[v[live].argsort(stable=True)]
        ordered.copy_(v[valid]);mapped.fill_(-1)
        mapped.view(-1)[valid]=torch.arange(len(valid),dtype=torch.int32)
        offsets[0]=0;offsets[1:].copy_(totals[:-1].cumsum(0))
        stats['placed']+=1
    route=dict(triton=fake_triton,torch=torch,_count_route_tiles=Kernel(histogram),
               _scan_route_tiles=Kernel(scan),_place_routes=Kernel(place))
    definitions(MANIFEST['tree'],kernel_path,route,{'count_peer_routes','place_peer_routes'})
    work_env.update({k:route[k] for k in ('count_peer_routes','place_peer_routes')})
    def gate(layer,inputs,keys,offsets,tuning,prompt):''')
change("(8192,'all',False,False),(8201,'skew'", "(8192,'all',False,False),(8193,'wide_ids',False,False),(8201,'skew'")
change("assert counts == ([3,1] if fast else [4,2] if enters else [3,3])",
       "assert counts == ([1,1] if fast else [2,2] if enters else [3,3])")
change("if kind in ('all','mixed','hook'): ids=ids.int()","if kind in ('all','mixed','hook','remote','skew'): ids=ids.int()")
change("stats['transport_boundaries_removed'] += counts[0]-counts[1]","stats['matched_collective_counts']+=1")
change("    (Path(folder)/f'{rank}.json').write_text",
       "    assert stats['counted']>=4 and stats['placed']>=2,stats\n    (Path(folder)/f'{rank}.json').write_text")
change("host_substitutes=['peer mappings & stream barriers','Triton pack','Humming projections and weighted finalization']",
       "host_substitutes=['peer mappings & stream barriers','Triton pack, histogram, scan, placement','Humming projections and weighted finalization']")
exec(compile(source,str(ORIGINAL),'exec'),{"__name__":"__main__","__file__":str(ORIGINAL),"__builtins__":__builtins__})
