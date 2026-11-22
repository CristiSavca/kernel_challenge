"""Run both backend call expressions and real public attention dispatch with instrumented allocations and launches."""
import ast
from bisect import bisect_right
from collections import Counter
from hashlib import sha256
from itertools import accumulate
import json
import math
import os
from pathlib import Path
import random
import sys
import subprocess
import tempfile
from types import SimpleNamespace as NS


HERE=Path(__file__).resolve().parent
KERNEL="python/sglang/kernels/ops/attention/paged_verify_attention.py"
BACKEND="python/sglang/srt/layers/attention/flashattention_backend.py"
ceil=lambda a,b:(a+b-1)//b


class Tensor:
    def __init__(self,shape,device="cuda:0",dtype="int32"):
        self.shape=tuple(shape);self.device=device;self.dtype=dtype;self.is_cuda=device.startswith("cuda")
    def stride(self,n=None):
        return tuple(self.stride(i) for i in range(len(self.shape))) if n is None else math.prod(self.shape[n+1:])
    def is_contiguous(self):return True
    def data_ptr(self):return 4096


class Launch:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def call(*args,**kw):self.calls.append((grid,args,kw))
        return call


def load(source):
    astree=ast.parse(source)
    names={"paged_verify_attention","_direct_tile_args","_tile_geometry"}
    body=[k for k in astree.body if isinstance(k,ast.FunctionDef) and k.name in names]
    for k in body:k.decorator_list=[]
    triton=NS(cdiv=ceil,next_power_of_2=lambda n:1<<(n-1).bit_length())
    torch=NS(empty=Tensor,empty_like=lambda q,dtype:Tensor(q.shape,q.device,dtype),float32="f32",bfloat16="bf16",int32="int32",
             float8_e4m3fn="fp8",compiler=NS(is_compiling=lambda:False))
    env=dict(math=math,triton=triton,tl=NS(constexpr=None,maximum=max,full=lambda _,v,d:v,where=lambda p,a,b:a if p else b,cdiv=ceil,int32=int),
             torch=torch,_partition_attention=Launch(),_merge_partitions=Launch(),TensorDescriptor=lambda *a:a)
    exec(compile(ast.Module(body,type_ignores=[]),"<public attention>","exec"),env)
    return env


def caller(src):
    root=ast.parse(src)
    calls=sorted((n for n in ast.walk(root) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="paged_verify_attention"),key=lambda c:c.lineno)
    assert len(calls)==2
    codes=[compile(ast.Expression(n),"<actual backend call>","eval") for n in calls]
    assert "workspace_limit_bytes" not in [k.arg for k in calls[0].keywords]
    assert "workspace_limit_bytes" in [k.arg for k in calls[1].keywords]
    return codes


def run(env,calls,role,b,cols,page,m,w,limit=None,num_sms=132,cert=True,heads=2,groups=8):
    t=lambda s,dt:Tensor(s,dtype=dt)
    q=t((b*m,heads*groups,64),"fp8"); k=t((16384,page,heads,64),"fp8"); v=t(k.shape,"fp8")
    pages=t((b,cols),"int32"); lens=t((b,),"int32"); cu=t((b+1,),"int32"); dest=t(q.shape,"bf16")
    tile=64 if w>=0 else 128
    cert_span=cols if w<0 else min(cols,w+m+tile-1)
    origins=t((b,ceil(cert_span,tile)),"int32") if cert else None
    local=dict(paged_verify_attention=env["paged_verify_attention"],query=q,tail_query=q,key_cache=k,value_cache=v,page_table=pages,tail_pages=pages,
               cache_seqlens=lens,tail_lengths=lens,cu_seqlens_q=cu,cu_tail=cu,max_seqlen_q=m,max_tail=m,window_size=(w,0),layer=NS(scaling=.125),
               fa_k_descale=None,fa_v_descale=None,tail_scales={},_fa_out=dest,tail_out=dest,self=NS(_hopper_verify_sms=num_sms),origins=origins,available=limit)
    part=env["_partition_attention"];merge=env["_merge_partitions"]
    part.calls.clear();merge.calls.clear()
    out=eval(calls[role],local)
    if out is None:
        assert not part.calls and not merge.calls
        return None
    assert out is dest and len(part.calls)==len(merge.calls)==1
    grid,arg,kw=part.calls[0]; mg,ma,mk=merge.calls[0]
    assert kw.get("ONE_TILE",False)==mk.get("ONE_TILE",False) and mg==(b,heads,m*groups)
    # Partial workspace precedes LSE. See the actual call syntax bound above.
    workspace=arg[7]; lse=arg[8]
    assert workspace.shape==(grid[0]+b-1,heads,1<<(m*groups-1).bit_length(),64) and lse.shape==workspace.shape[:-1]
    return dict(partitions=grid[0],ctas=math.prod(grid),slots=workspace.shape[0],scratch=(math.prod(workspace.shape)+math.prod(lse.shape))*4,
                direct="Origins" in kw,segment_lanes=kw.get("SEGMENT_LANES",0),one_tile=kw.get("ONE_TILE",False))


def segments(prog,counts):
    ends=list(accumulate(counts)); start=[0]+ends[:-1]; total=ends[-1]
    width=1 if prog["one_tile"] else max(1,ceil(total,prog["partitions"]))
    result=[]
    for part in range(prog["partitions"]):
        left=part*width;right=min(total,left+width)
        while left<right:
            req=bisect_right(ends,left)
            finish=min(right,ends[req])
            result.append((part+req,req,left-start[req],finish-start[req]))
            left=finish
    return result


def parent_kernel():
    base=HERE.parents[1]/"sglang"
    parent=HERE.parent/"v297_wide_fused_decode"
    receipt=json.loads((parent/"validation.json").read_bytes())
    git_dir=subprocess.check_output(["git","-C",base,"rev-parse","--absolute-git-dir"],text=True).strip()
    with tempfile.TemporaryDirectory() as d:
        obj=Path(d)/"objects";obj.mkdir()
        env=dict(os.environ,GIT_INDEX_FILE=d+"/index",GIT_OBJECT_DIRECTORY=str(obj),GIT_ALTERNATE_OBJECT_DIRECTORIES=git_dir+"/objects")
        def git(*args,data=None):return subprocess.check_output(["git","-C",base,*args],env=env,input=data)
        git("read-tree",receipt["base"])
        git("apply","--cached","--whitespace=error-all","-",data=(parent/"patch.diff").read_bytes())
        assert git("write-tree").decode().strip()==receipt["tree"]
        return git("show",":"+KERNEL)


def main(stage):
    new=(stage/KERNEL).read_bytes(); backend=(stage/BACKEND).read_bytes(); old=parent_kernel()
    envs=[load(x) for x in (old,new)]; calls=caller(backend)
    rng=random.Random(301); count=Counter(); seen={}; wrong_cap=0
    for _ in range(3000):
        b=rng.choice([1,2,3,4,7,16,32,64]); cols=rng.choice([1,2,17,33,64,127,128,129,511,2051,8192,131072]); page=rng.choice([1,2,16,64]);m=rng.randint(2,16);w=rng.choice([-1,0,1,60,64,127,2047]);sms=rng.choice([20,80,132])
        groups=rng.choice([g for g in (1,2,4,8,16) if 16<=m*g<=64]);count["max_q_"+str(m)]+=1
        role=rng.choice([0,1]);cap=2**32;tile=128 if w<0 else 64
        a,c=[run(e,calls,role,b,cols,page,m,w,cap,sms,groups=groups) for e in envs]
        assert a and c and c["scratch"]<=a["scratch"] and c["ctas"]<=a["ctas"]
        assert a["direct"]==c["direct"]==(role==0 and page==1)
        lengths=[rng.randrange(cols*page+1) for _ in range(b)]
        qlens=[rng.randrange(min(l,m)+1) for l in lengths]
        geo=[[e["_tile_geometry"](l,q,w,tile) for l,q in zip(lengths,qlens)] for e in envs]
        assert geo[0]==geo[1]; counts=[n for _,n in geo[0]]
        left=segments(a,counts);right=segments(c,counts)
        assert left==right and sum(end-begin for _,_,begin,end in left)==sum(counts)
        assert all(0<=slot<c["slots"] for slot,*_ in right)
        cap_tiles=ceil(cols*page,tile)
        if w>=0:cap_tiles=min(cap_tiles,ceil(w+m+tile-1,tile))
        bound=max(1,b*cap_tiles); max_width=ceil(bound,c["partitions"])
        expected_lanes=1<<(max_width-1).bit_length() if w<0 and max_width<=32 else 0
        assert c["segment_lanes"]==expected_lanes
        if expected_lanes:
            assert all(0<end-begin<=expected_lanes for _,_,begin,end in right)
            count["bounded_certificate_lanes_"+str(expected_lanes)]+=1
        else:
            count["dynamic_certificate_lanes"]+=1
        if c["one_tile"]:assert max((end-begin for _,_,begin,end in right),default=0)<=1
        if w>=0 and any(n>ceil(w+m,tile) for n in counts):wrong_cap+=1
        count["layout_pairs"]+=1;count["one_tile"]+=c["one_tile"];count["empty_requests"]+=qlens.count(0)
        count["identical_intersections"]+=len(left)
    assert wrong_cap>0
    # Case near the cutoff using actual mixed-tail arguments. V297 uses 66 slots;
    # shortened workspace is 34 slots regardless of whether certificates exist.
    for w,cols in [(2047,131072),(-1,129)]:
        a,c=[run(e,calls,1,1,cols,1,8,w,2**32) for e in envs]
        for cap,mask in [(c["scratch"]-1,(False,False)),(c["scratch"],(False,True)),(a["scratch"]-1,(False,True)),(a["scratch"],(True,True))]:
            assert tuple(bool(run(e,calls,1,1,cols,1,8,w,cap)) for e in envs)==mask
        seen["window"+str(w)]=dict(parent=a,candidate=c)
    public=next(d for d in json.loads((HERE.parent/"v297_wide_fused_decode/server_config.json").read_bytes())["cases"] if d["name"]=="intact_chunk_both")
    budget=32*1024**2-2*public["target_mixed_capacity"]*2*64
    dense=run(envs[0],calls,1,public["max_running_requests"],131072,1,public["block_size"],-1,budget)
    assert dense and dense["scratch"]<budget
    for w in (-1,2047):
        seen["public"+str(w)]=[dict(batch=b,old=run(envs[0],calls,0,b,131072,1,8,w),new=run(envs[1],calls,0,b,131072,1,8,w)) for b in (1,2,4,16,32)]
    print(json.dumps(dict(source_sha256={k:sha256(v).hexdigest() for k,v in [("parent",old),("candidate",new),("backend",backend)]},
         counts=count,underestimated_if_mid_tile_allowance_removed=wrong_cap,public_max_mixed_rows=public["target_mixed_capacity"],public_min_mixed_budget=budget,
         public_max_old_full_scratch=dense["scratch"],cases=seen,scope="Both actual backend call expressions and public dispatch on modeled CUDA tensors; exact production scalar tile counts, in-order slots and intersection lengths; device arithmetic and hardware time excluded."),indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))
