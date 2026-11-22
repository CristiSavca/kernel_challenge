"""Use the production PyTorch wave planner to create small exact interpreter inputs."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

def plan(path):
    src=path.read_bytes()
    env=dict(torch=torch,triton=SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b))
    node=next(n for n in ast.parse(src).body if getattr(n,"name","")== "plan_waves")
    exec(ast.unparse(node),env)
    gen=torch.Generator().manual_seed(306)
    cases=[]
    for tokens,k,e,cap,width in ((27,4,7,11,37),(8,3,4,6,129),(36,8,32,37,4096),(73,8,16,512,33)):
        for layout in ("mixed","all local","all remote","skew","zero-route tail","boundary","sparse high"):
            ids=torch.randint(-3,e+1,(tokens,k),generator=gen,dtype=torch.int32)
            if layout=="all local":ids.remainder_(e)
            if layout=="all remote":ids.fill_(e+3)
            if layout=="skew":ids[:]=e-1
            if layout=="zero-route tail":ids[:tokens//2].remainder_(e);ids[tokens//2:]=-1
            if layout=="boundary":ids.flatten()[:cap]=0;ids.flatten()[cap:]=-1
            if layout=="sparse high":ids.fill_(-1);ids[::3,0]=e-1
            bins,slots,edges,bounds=env["plan_waves"](ids,e,cap)
            assert bins.dtype==edges.dtype==bounds.dtype==torch.int32 and slots.dtype==torch.int64
            cases.append(dict(layout=layout,tokens=tokens,k=k,e=e,cap=cap,width=width,ids=ids.tolist(),
                              bins=bins.tolist(),slots=slots.tolist(),edges=edges.tolist(),bounds=bounds.tolist()))
    return dict(source_sha256=hashlib.sha256(src).hexdigest(),cases=cases)

if __name__=="__main__":print(json.dumps(plan(Path(sys.argv[1]))))
