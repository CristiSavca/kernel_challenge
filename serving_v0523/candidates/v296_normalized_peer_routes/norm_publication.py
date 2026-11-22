"""Replay published routes with the V293 host CUDA body and track every normalization."""
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

here=Path(__file__).resolve().parent
sys.path[:0]=[str(here.parent/"v293_peer_prompt_norm"),str(here.parent/"v295_published_prompt_routes")]
from numeric import build
from transport_norm import HostNorm, normalized
import independent_routes as original

BASE_FIXTURE=original.fixture

def run_fixture(rank,world,stage,root,dll):
    source=inspect.getsource(BASE_FIXTURE)
    def swap(a,b):
        nonlocal source
        assert source.count(a)==1,a
        source=source.replace(a,b)
    swap("    experts = Experts()", """    experts = Experts()
    machine=HostNorm(dll)
    norm_count=[0]
    eps=1e-6
    def weight(r):
        return (.4+torch.arange(WIDTH).remainder(9)/12+r/64).bfloat16()
    def native(x,shape,g,e):
        norm_count[0]+=1
        assert tuple(shape)==(WIDTH,)
        return normalized(x,g,e)
    def raw_norm(x):
        return native(x,(WIDTH,),weight(rank),eps)
    old_clone=torch.Tensor.clone
    def audit_clone(t,*a,**kw):
        if workspace is not None and t.data_ptr() in {b.data_ptr() for b,_ in workspace.banks}:
            state.counts["veto_slab_clones"]+=1
        return old_clone(t,*a,**kw)""")
    swap('        for mode in ("published","legacy","absent"):','''        for mode in ("published","legacy","absent"):''')
    swap('            model._peer_prompt_inputs = workspace','''            model._peer_prompt_inputs = workspace
            if workspace is not None:
                workspace.norm_module=machine
                workspace.norm_rows=torch.arange(CAPACITY,dtype=torch.int32).view(-1,1)
                workspace.source_norm=lambda shard,norm:(weight(rank),eps) if norm is not None and shard.shape[0]<=CAPACITY else None''')
    swap('                state.counts = dict(route_pairs=0,activation_words=0,empty_local=0)',
         '                state.counts = dict(route_pairs=0,activation_words=0,empty_local=0,veto_slab_clones=0)')
    swap('                state.ids,state.scores,state.gold = ids[:state.n],scores[:state.n],words[:state.n]','''                words=((words.float().remainder(101)-51.4)/10.3).bfloat16().view(torch.int16)
                reference=torch.cat([normalized(chunk.view(torch.bfloat16),weight(r),eps).view(torch.int16)
                                     for r,chunk in enumerate(words.chunk(world))])
                state.ids,state.scores,state.gold = ids[:state.n],scores[:state.n],reference[:state.n]''')
    swap('                men["_forward_sp_replicated_experts"](model,h,ls,li)','''                before_norm=norm_count[0],machine.calls
                # Alternate native-normalized and direct-capable ranks independently of route acceptance.
                use=workspace is not None and (epoch+rank)%3!=0
                with patch.object(torch.nn.functional,"rms_norm",native),patch.object(torch.Tensor,"clone",audit_clone):
                    if use:
                        men["_forward_sp_replicated_experts"](model,h,ls,li,pre_norm=raw_norm)
                    else:
                        men["_forward_sp_replicated_experts"](model,raw_norm(h),ls,li)
                normal=norm_count[0]-before_norm[0];direct=machine.calls-before_norm[1]
                assert normal+direct==1,(mode,case,rank,normal,direct)
                state.counts.update(native_norms=normal,into_slab=direct)''')
    scope={**vars(original),"HostNorm":HostNorm,"normalized":normalized,"dll":dll}
    exec(compile(source,"V295 integrated Gloo fixture with exact CUDA norm and veto witness","exec"),scope)
    return scope["fixture"](rank,world,stage,root)

def worker(rank,world,stage,folder,dll):
    original.fixture=lambda r,w,s,root:run_fixture(r,w,s,root,dll)
    original.worker(rank,world,stage,folder)

def main(stage):
    totals={}; outcomes={}
    with tempfile.TemporaryDirectory(prefix="v296-identity-") as root:
        lib=build(stage,Path(root))
        for world in (4,8):
            mp.spawn(worker,args=(world,str(stage),root,str(lib)),nprocs=world)
            for rank in range(world):
                for row in json.loads((Path(root)/f"{world}-{rank}.json").read_bytes()):
                    key=row["mode"]+"/"+row["case"];o=outcomes.setdefault(key,{})
                    for field in ("activation_words","route_pairs","empty_local","native_norms","into_slab","veto_slab_clones"):
                        totals[field]=totals.get(field,0)+row[field]
                        o[field]=o.get(field,0)+row[field]
    assert totals["into_slab"] and totals["veto_slab_clones"] and totals["native_norms"]
    print(json.dumps(dict(worlds=[4,8],counts=totals,per_case=outcomes,mismatches=0,
        scope="Actual V296 public SP replicated expert method, remote route publication and both fallbacks. Gloo plus shared CPU peer slabs, original CUDA body interpreted per lane, independent CUDA-vectorized CPU norm reference and synthetic Humming projections. Every rank normalizes each call exactly once; clone observer counts retained buffers acquired on a veto."),indent=2))
if __name__=="__main__":main(Path(sys.argv[1]).resolve())
