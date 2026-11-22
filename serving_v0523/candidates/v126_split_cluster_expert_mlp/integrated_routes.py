"""Connect V68 route planning to V126's actual GPU bodies with CPU-emulated operations."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import numpy as np
import warp_check as w

def main(stage):
    fixture=Path(__file__).resolve().parents[1]/"v66_humming_grouped_decode/cpu_check.py"
    # Only the shared CPU tensor and bounds-checked Triton operators are required.
    helper={}
    exec(compile(fixture.read_text().split("if len(sys.argv)",1)[0],str(fixture),"exec"),helper)
    tensor,bf16,tl,launch,extract=[helper[n] for n in ("tensor","bf16","Language","Launch","extract")]
    tl.bfloat16="bf16"
    torch=NS(empty=lambda shape,dtype,device:tensor(np.full(shape,np.nan,dtype=dtype)),
             empty_like=lambda a,dtype=None:tensor(np.empty_like(a,dtype=dtype)),
             bfloat16=np.dtype(np.float32),uint8=np.uint8,int32=np.int32,float32=np.float32)
    sys.modules["humming.layer"]=NS(HummingMethod=NS())
    env=dict(math=math,torch=torch,triton=NS(next_power_of_2=lambda n:1<<(n-1).bit_length()),tl=tl,_MIN_TOKENS_FOR_FUSED=32)
    root=stage/"python/sglang"
    extract(root/"srt/layers/muse_spark/muse_spark_humming_moe.py",
            ("_plan_grouped_decode","_gather_grouped_decode","_sparse_cluster_grid","humming_grouped_decode_outputs"),env)
    extract(root/"kernels/ops/moe/muse_spark_post_expert_finalize_triton.py",
            ("_post_expert_norm_weighted_sum_kernel","fused_post_expert_norm_weighted_sum"),env)
    for name in ("_plan_grouped_decode","_gather_grouped_decode"):env[name]=launch(env[name])
    final=env["_post_expert_norm_weighted_sum_kernel"]
    rng=np.random.default_rng(1266);h,inner,e,n,k=512,2048,4,8,2
    gate=w.base.weight_bank(rng,e,2*inner,h);down=w.base.weight_bank(rng,e,h,inner)
    g13=tensor(np.array([.125,1.5,.5,1],np.float32));g2=tensor(np.array([-.5,1,.25,2],np.float32))
    x=tensor(bf16(rng.integers(-5,6,(n,h)).astype(np.float32)/8));scores=tensor(rng.uniform(.1,1,(n,k)).astype(np.float32))
    norm=tensor(bf16(rng.uniform(.75,1.25,h).astype(np.float32)));mapping=tensor(np.full(32,-1,np.int32));mapping[[1,7,12,28]]=[3,0,2,1]
    results=[]
    with tempfile.TemporaryDirectory() as d:
        dll,_=w.compile_actual(stage,Path(d))
        def split_call(rows,wg,sg,wd,sd,scale,ends,partials,groups):
            bits=w.v86.bf16(np.asarray(rows)); arr=[bits,wg,sg,wd,sd,w.v86.bf16(scale),ends]
            dll.run_split(*(a.ctypes.data for a in arr),h,inner,e,rows.shape[0],partials.shape[0],groups,partials.ctypes.data)
        env["_split_mlp_module"]=lambda:NS(run=split_call)
        for p in [2,4]:
            layer=NS(humming_cluster_mlp=False,humming_split_mlps=p,humming_device_sms=16,global_num_experts=32,num_local_experts=e,
                     intermediate_size_per_partition=inner,_expert_map=mapping,w13_weight=tensor(gate[2]),w13_weight_scale=tensor(gate[3]),
                     w2_weight=tensor(down[2]),w2_weight_scale=tensor(down[3]),w13_global_scale=g13)
            for pattern in ["remote","mixed","duplicate","skew"]:
                ids=tensor(np.full((n,k),30,np.int32))
                if pattern=="mixed":ids[:,1]=rng.choice([1,30,7,12,28],n)
                if pattern=="duplicate":ids[::2,:]=1
                if pattern=="skew":ids[:,:]=7
                out,rows,local=env["humming_grouped_decode_outputs"](layer,x,ids)
                wanted=mapping[ids];np.testing.assert_array_equal(wanted,local)
                assert out.shape==(p,n*k,h)
                dense=np.zeros((n,k,h),np.float32)
                for expert in np.unique(wanted[wanted>=0]):
                    tok,sl=np.where(wanted==expert)
                    gparts=w.dot(x[tok],gate,expert,p,True); gproj=np.zeros_like(gparts[0])
                    for piece in gparts:gproj+=piece
                    g,u=np.split(bf16(bf16(gproj)*g13[expert]),2,axis=-1)
                    with np.errstate(over="ignore"):act=bf16((g/(1+np.exp(-g)))*u)
                    dparts=w.dot(act,down,expert,p,False); np.testing.assert_array_equal(out[:,rows[tok,sl]],dparts)
                    raw=dparts[0].copy()
                    for piece in dparts[1:]:raw+=piece
                    dense[tok,sl]=bf16(bf16(raw)*g2[expert])
                env["_post_expert_norm_weighted_sum_kernel"]=launch(final,(3,))
                for first in [False,True]:
                    actual=env["fused_post_expert_norm_weighted_sum"](out,scores,norm,weight_first=first,
                            slot_expert_ids=local,slot_row_indices=rows,out=x.new_empty(x.shape),expert_output_scales=g2)
                    ans=np.zeros((n,h),np.float32)
                    for sl in range(k):
                        v=dense[:,sl,:]; pre=v*norm if first else v
                        scale=np.reciprocal(np.sqrt((pre*pre).sum(1,dtype=np.float32)/h+np.float32(1e-8)))
                        y=pre*scale[:,None] if first else v*scale[:,None]*norm
                        ans+=np.where((wanted[:,sl]>=0)[:,None],y*scores[:,sl,None],0)
                    np.testing.assert_array_equal(actual,bf16(ans))
                results.append(dict(P=p,pattern=pattern,live_slots=int(np.sum(wanted>=0)),compared_tokens=n,
                                    finalized_bf16=2*n*h))
    print(json.dumps(dict(cases=results,physical_gpu=False),indent=2))
if __name__=="__main__":main(Path(sys.argv[1]))
