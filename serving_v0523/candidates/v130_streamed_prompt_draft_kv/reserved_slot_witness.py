"""Reproduce the frozen V130 full-prompt writer's physical slot counterexamples on CPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import torch

EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(EVIDENCE.parent/"v133_direct_prompt_draft_kv"))
import prompt_check as v133


def check(repo, root):
    patch = (EVIDENCE/"patch.diff").read_bytes()
    digest = hashlib.sha256(patch).hexdigest()
    assert digest == "4b4ab2f797c8580ed24a7eec28525f728fa1824d559b26c1020e9f22e26713f3"
    module = root/v133.MODULE
    module.parent.mkdir(parents=True)
    module.write_bytes(v133.pinned(repo,v133.MODULE))
    subprocess.run(["git","apply","--include="+v133.MODULE,"-"],cwd=root,input=patch,check=True)
    select=v133.method(module,"FusedKVMaterializeHelper","full_prompt_cache",dict(torch=torch))
    pid=[0,0,0]
    fn=v133.evaluator.source(module,{"_fused_norm_rope_kernel_stacked","_cache_round"},pid)._fused_norm_rope_kernel_stacked
    stock=v133.native_store(repo,root)
    rows, dim, logical_capacity=8,32,9
    input_kv=torch.ones((rows,1,2*dim),dtype=torch.bfloat16)
    for i in range(rows):input_kv[i,0,dim:]=i+3
    trig=torch.zeros((rows,dim),dtype=torch.bfloat16);trig[:,:dim//2]=1
    positions=torch.arange(rows,dtype=torch.int64)
    common=list(map(v133.PTR,(input_kv,torch.ones(dim),torch.full((1,),1e-6),trig,positions)))
    args=[2*dim,2*dim,dim,dim]
    geometry=[rows,1,1,dim,dim,dim,dim//2,dim]
    k=torch.empty((rows,dim),dtype=torch.bfloat16);value=torch.empty_like(k)
    for row in range(rows):
        pid[:]=row,0,0
        fn(*common,v133.PTR(k),v133.PTR(value),*args,rows*dim,dim,dim,rows*dim,dim,dim,*geometry)
    results=[]
    for first in [0,logical_capacity]:
        # The extra physical row is a guarded backing allocation. Both methods receive
        # the same logical bound; testing the second case never touches arbitrary memory.
        raw=torch.tensor([first,1,2,3,4,5,6,7],dtype=torch.int64)
        orig_k=torch.full((logical_capacity+1,dim),-99.,dtype=torch.bfloat16)
        orig_v=torch.full_like(orig_k,-99.)
        direct_k,direct_v=orig_k.clone(),orig_v.clone()
        failed=False
        try:stock(k,value,orig_k,orig_v,raw,row_bytes=2*dim,v_row_bytes=2*dim,size_limit=logical_capacity)
        except IndexError:failed=True
        assert failed == (first==logical_capacity)
        helper=NS(can_stream_prompt=lambda _:True,_full_prompt_lengths=None)
        _,loc,lens=select(helper,object(),raw)
        assert loc.shape==(1,8) and lens.tolist()==[8]
        for row in range(rows):
            pid[:]=row,0,0
            fn(*common,v133.PTR([direct_k]),v133.PTR([direct_v]),*args,0,0,0,0,0,0,*geometry,
               Loc=v133.PTR(loc),Lengths=v133.PTR(lens),WIDTH=8,CACHE_FP8=False)
        assert orig_v[first].eq(-99.).all() and direct_v[first].eq(3.).all()
        if first==0:
            assert torch.equal(orig_v[1:],direct_v[1:]) and torch.equal(orig_k[1:],direct_k[1:])
        results.append(dict(first_slot=first,native_asserted=failed,v130_asserted=False,
                            stock_sample=float(orig_v[first,0]),v130_sample=float(direct_v[first,0]),
                            overwritten_v_bytes=2*dim))
    return dict(patch_sha256=digest,logical_pool_capacity=logical_capacity,route_group_size=8,cases=results,cuda_execution=False)


if __name__=="__main__":
    checkout=EVIDENCE.parents[1]/"sglang"
    with tempfile.TemporaryDirectory(prefix="okc-v130-witness-") as tmp:
        print(json.dumps(check(checkout,Path(tmp)),indent=2))
