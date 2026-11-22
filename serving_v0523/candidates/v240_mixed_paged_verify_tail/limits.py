"""Verify the precise memory gate, native geometry guards and old FA3 out ABI."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument("stage",type=Path)
args=ap.parse_args()
path=args.stage/"python/sglang/kernels/ops/attention/paged_verify_attention.py"
tree=ast.parse(path.read_text())
nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
calls=[]
class Launch:
    def __getitem__(self,grid):
        return lambda *args,**kw:calls.append(tuple(grid))
env={"torch":torch,"math":math,"triton":NS(cdiv=lambda a,b:(a+b-1)//b,next_power_of_2=lambda n:1<<(n-1).bit_length()),
     "TensorDescriptor":None,"_partition_attention":Launch(),"_merge_partitions":Launch()}
exec(compile(ast.Module([nodes["_direct_tile_args"], nodes["paged_verify_attention"]],[]),str(path),"exec"),env)
fn=env["paged_verify_attention"]
counts={"cap_accept":0,"cap_decline":0,"native_geometry_decline":0,"default_unbounded":0}
max_workspace=0
for b in [1,3,7,15,31,64,65]:
    for w,heads,groups,dim in [(8,2,8,64),(4,4,8,64),(16,1,4,64),(8,2,1,64),(17,2,2,64),(8,2,8,128)]:
        q=torch.empty(b*w,heads*groups,dim,dtype=torch.float8_e4m3fn)
        cache=torch.empty(1,1,heads,dim,dtype=q.dtype)
        out=torch.empty_like(q,dtype=torch.bfloat16)
        p=torch.ones(b,4,dtype=torch.int32)
        cu=torch.arange(b+1,dtype=torch.int32)*w
        lens=torch.full((b,),w,dtype=torch.int32)
        sm=132
        native=b<=64 and w<=16 and 16<=w*groups<=64 and dim==64
        expected_bytes=((2*sm+heads-1)//heads+b-1)*heads*(1<<(w*groups-1).bit_length())*(dim+1)*4
        for cap in (expected_bytes-1,expected_bytes,expected_bytes+1,None):
            with patch.object(torch,"empty",wraps=torch.empty) as allocate:
                calls.clear()
                result=fn(q,cache,cache,p,lens,cu,max_q=w,window_left=-1,softmax_scale=.125,
                          k_descale=None,v_descale=None,out=out,num_sms=sm,workspace_limit_bytes=cap)
                expect=native and (cap is None or cap>=expected_bytes)
                assert (result is out)==expect and len(calls)==2*expect
                assert allocate.call_count==2*expect
                if expect:
                    sz=sum(math.prod(c.args[0])*4 for c in allocate.call_args_list)
                    assert sz==expected_bytes
                    max_workspace=max(max_workspace,sz)
                counts["native_geometry_decline" if not native else "default_unbounded" if cap is None
                       else "cap_accept" if expect else "cap_decline"]+=1

# The V239 provider probe replays the real production split with native FA3.
# A mixed-policy decline leads to precisely this fallback.
donor=HERE.parent/"v239_split_mixed_full_attention/independent_fa3.py"
src=donor.read_text()
anchor="state = dict(batched_prompt_kv=(layout,tail),"
assert src.count(anchor)==1
src=src.replace(anchor,"state = dict(verify_eligible=False, batched_prompt_kv=(layout,tail),")
ns={"__file__":str(donor),"__name__":"v240_fallback"}
with patch.object(sys,"argv",[str(donor),str(args.stage)]):
    exec(compile(src,str(donor),"exec"),ns)
legacy=words=copybacks=0
for rows,tails in [([4096],[8]),([3990,161],[8,8,8]),([4000,156,32],[8,7])]:
    for scales in ["none","ranked","expanded"]:
        for fresh in [True,False]:
            for old in [True,False]:
                for knew in [True,False]:
                    n,r,c=ns["run"](rows,tails,scales,fresh,old,knew)
                    words+=n;legacy+=r;copybacks+=c
print(json.dumps(dict(memory_geometry=counts,largest_accepted_workspace=max_workspace,native_abi_cases=72,
                      provider_legacy_retries=legacy,distinct_storage_copybacks=copybacks,
                      bf16_components_checked=words,paged_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
