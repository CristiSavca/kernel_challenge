"""Replay the V293 norm destination under asymmetric eligibility and all-rank fallback."""
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import torch.multiprocessing as mp
from numeric import build,load,reference

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/"v290_mixed_ranked_peer_serving"))
import peer_grouped as donor
import transport

def normalized(shard,gain,eps):
    words=shard.view(torch.int16).numpy().view(np.uint16)
    gw=gain.view(torch.int16).numpy().view(np.uint16)
    return torch.from_numpy(reference(words,gw,eps).view(np.int16)).view(torch.bfloat16)

class HostNorm:
    def __init__(self,path):self.run_kernel=load(path);self.calls=0
    def run(self,x,gain,ids,dst,eps):
        assert x.shape==(1,*dst.shape) and ids.shape==(dst.shape[0],1) and x.is_contiguous()
        assert torch.equal(ids.flatten(),torch.arange(dst.shape[0],dtype=torch.int32))
        self.run_kernel(x.data_ptr(),gain.data_ptr(),ids.data_ptr(),dst.data_ptr(),1,*dst.shape,1,eps)
        self.calls+=1

def grouped(stage,binding,workspace,mode,rank,world,state,group,data,rows,width):
    # Reuse the V290 real eligibility/sort/group/fallback fixture. Replace its test
    # payload factory and expected values while leaving the stage source untouched.
    src=inspect.getsource(donor.grouped)
    def swap(before,after):
        nonlocal src
        assert src.count(before)==1
        src=src.replace(before,after)
    swap('        for epoch in range(15):', '''        for epoch in range(15):''')
    swap('            shard=data(epoch,rank,n).view(torch.bfloat16)\n            state.gold=torch.cat([data(epoch,r,n) for r in range(world)])','''            def raw(r):
                f=(data(epoch,r,n).float().remainder(127)-63.25)/11
                return f.bfloat16()
            def weight(r):
                return (torch.arange(width).remainder(13)/19+.25+r/64).bfloat16()
            eps=1e-6
            gain=weight(rank)
            state.gold=torch.cat([normalized(raw(r),weight(r),eps).view(torch.int16) for r in range(world)])
            use=(rank+epoch)%3!=0
            shard=raw(rank) if use else normalized(raw(rank),gain,eps)
            option={"source_norm":(gain,eps)} if use else {}
''')
    swap('ctx["_peer_experts_into"](workspace,experts,shard,state.scores,ids,group,out)',
         'ctx["_peer_experts_into"](workspace,experts,shard,state.scores,ids,group,out,**option)')
    swap('return dict(mode=mode,banks=banks,publications=pub,barriers=state.barriers-before,**obs)',
         'return dict(mode=mode,banks=banks,publications=pub,barriers=state.barriers-before,direct_norms=workspace.norm_module.calls,**obs)')
    swap('                ctx["_peer_experts_into"](workspace,experts,shard,state.scores,ids,group,out,**option)','''                if epoch == 6 and rank == world-1:
                    # Replace a previously captured helper locally. It must retain its old
                    # five-member plan signature and observe the normalized source shard.
                    old_impl=experts._grouped_humming_and_combine_impl
                    def observed(sh,local_ids,sc,out,*,peer_plan):
                        assert len(peer_plan)==5
                        expected=state.gold[rank*n:(rank+1)*n]
                        assert torch.equal(sh.view(torch.int16),expected)
                        old_impl(sh,local_ids,sc,out,peer_plan)
                    experts._grouped_humming_and_combine_impl=observed
                try:
                    ctx["_peer_experts_into"](workspace,experts,shard,state.scores,ids,group,out,**option)
                finally:
                    if "_grouped_humming_and_combine_impl" in vars(experts):
                        del experts._grouped_humming_and_combine_impl''')
    scope=dict(vars(donor));scope["normalized"]=normalized
    exec(compile(src,"V290 Gloo fixture with V293 norm payloads","exec"),scope)
    workspace.norm_module=HostNorm(libpath);workspace.norm_rows=torch.arange(rows,dtype=torch.int32).view(-1,1)
    def native(x,shape,gain,eps):
        assert shape==(width,);return normalized(x,gain,eps)
    with patch.object(torch.nn.functional,"rms_norm",native):
        return scope["grouped"](stage,binding,workspace,mode,rank,world,state,group,data,rows,width)

def worker(rank,world,stage,folder,dll):
    global libpath
    libpath=dll
    original=transport.exercise
    transport.exercise=lambda *args:original(*args,grouped_check=grouped)
    transport.run_world(rank,world,stage,folder)

def main(stage):
    totals={}
    with tempfile.TemporaryDirectory(prefix='v293-norm-gloo-') as root:
        lib=build(stage,Path(root))
        for world in (4,8):
            mp.spawn(worker,args=(world,str(stage),root,str(lib)),nprocs=world)
            for rank in range(world):
                for row in json.loads((Path(root)/f"{world}_{rank}.json").read_bytes()):
                    t=totals.setdefault(row["mode"],{})
                    for key in ("grouped","fallback","empty","stubs","publications","barriers","direct_norms","words","declined"):
                        t[key]=t.get(key,0)+row.get(key,0)
    for name in ("ordinary","large_allocation","large_probe","large_registration"):
        t=totals[name];assert t["grouped"]==120 and t["fallback"]==36 and t["empty"]>0 and 0<t["direct_norms"]<120
    print(json.dumps(dict(worlds=[4,8],counts=totals,words_compared=sum(x["words"] for x in totals.values()),
        direct_norms=sum(x["direct_norms"] for x in totals.values()),mismatches=0,
        scope="Real Gloo collective vote, production stable sort/Humming input dispatch, rendered CUDA norm with synchronous host stream/peer GPU substitutes. Natural pre-normalized and direct-to-slab ranks alternate. Fallback uses a CPU replay of the native CUDA reference; no GPU runtime."),indent=2))
if __name__=="__main__":main(Path(sys.argv[1]).resolve())
