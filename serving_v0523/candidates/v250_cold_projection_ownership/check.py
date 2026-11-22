"""Reproduce the native V68 row ownership and evaluate column-sharding crossovers."""
import ast
from collections import Counter
import contextlib
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ANCHOR=HERE.parent/'v68_grouped_decode_verify'
S=Path(sys.argv[1]); m=json.loads((ANCHOR/'validation.json').read_bytes())
TREE=m['trees']['v68']; MODEL='python/sglang/srt/models/muse_spark.py'; SP='python/sglang/srt/layers/layernorm_sp.py'
def source(tree,path):
    return subprocess.check_output(['git','-C',S,'show',tree+':'+path]).decode()
def load(code,container,names,env):
    body=ast.parse(code).body
    if container: body=next(n.body for n in body if getattr(n,'name',None)==container)
    nodes=[n for n in body if getattr(n,'name',None) in names]
    assert len(nodes)==len(names)
    for n in nodes:n.decorator_list=[]
    exec('from __future__ import annotations\n'+ast.unparse(ast.Module(nodes,[])),env)
    return {n:env[n] for n in names}

def worker(rank,world,tmp):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+tmp+'/init',rank=rank,world_size=world)
    stats=Counter(); state=NS(sp_active=True)
    class Group:
        ca_comm=None
        world_size=world
        rank_in_group=rank
        def is_symmetric_memory_enabled(self):return True
        def all_gather_into_tensor(self,out,shard):dist.all_gather_into_tensor(out,shard);stats['gathers']+=1
        def all_reduce(self,x):
            y=x.clone()
            # Widen for CPU Gloo support. The test's selected integer rows sum exactly.
            z=y.float();dist.all_reduce(z);stats['reductions']+=1
            return z.bfloat16()
    g=Group()
    par=NS(tp_group=g); c=NS()
    st=NS(torch=torch,ceil_align=lambda n,k:(n+k-1)//k*k,get_parallel=lambda:par,
          set_sp_num_tokens=lambda n:setattr(c,'rows',n))
    scope=vars(st)
    layer=NS(**load(source(TREE,SP),None,['sp_entry_scatter','sp_exit_gather','reduce_scatter_token_rows'],scope),
             sp_num_tokens=lambda:c.rows)
    w1=torch.arange(32).reshape(4,8).remainder(3).bfloat16()
    w2=torch.arange(32).reshape(8,4).remainder(3).bfloat16()
    shapes=[]
    def project(x,w,b):
        shapes.append(('pool' if w is w1 else 'unpool',x.shape[0]))
        return x@w.T
    env=dict(torch=torch,contextlib=contextlib,_is_cpu=False,get_forward=lambda:state,
             get_parallel=lambda:par,is_dp_attention_enabled=lambda:False,layernorm_sp=layer,
             _use_split_bf16_router=lambda rows:rows>256,bf16_gemm_dispatch=project,
             get_global_experts_capturer=lambda:None)
    names=['forward','_forward_tp','_forward_experts','_forward_sp_experts','_forward_sp_replicated_experts','_reduce_and_unpool']
    Model=type('Model',(),load(source(TREE,MODEL),'MuseSpark1xMoE',names,env))
    gateenv=dict(torch=torch,no_grad=torch.no_grad)
    class TorchMM:
        def __getattr__(self,name):return getattr(torch,name)
        def mm(self,x,w,out_dtype):
            shapes.append(('router',x.shape[0]));assert out_dtype==torch.float32
            return x.float()@w.float()
    gateenv['torch']=TorchMM()
    Gate=type('Gate',(),load(source(TREE,MODEL),'MuseSpark1xRouter',['_build_split2','forward_split2'],gateenv))
    class Experts:
        use_w4a16=True
        post_expert_norm=None
        def __call__(self,inputs,scores,ids,norm):
            assert inputs.shape[0]==scores.shape[0]==ids.shape[0]==c.rows
            return inputs if rank==0 else torch.zeros_like(inputs)
    op=Model();gate=Gate();gate.weight=torch.arange(32).reshape(4,8).float().div(143);gate._split2_key=None
    op.__dict__.update(finalize_ar_workspace=None,tp_size=world,moe_tp_size=1,
        sp_replicate_expert_input=True,pre_expert_norm=None,experts=Experts(),
        nvfp4_checkpoint=True,router_in_fp32=True,gate=gate,top_k=2,
        pre_expert_proj=NS(weight=w1),post_expert_proj=NS(weight=w2))
    def route():
        def call(unused,l,k,n,want_packed):
            vals,ids=l.topk(k)
            return vals,ids.int(),None
        return call
    op._routing_op=route
    fb=NS(forward_mode=NS(is_decode=lambda:False,is_target_verify=lambda:False))
    for n in (8193,16384,16639):
        full=(torch.arange(n*8).reshape(n,8)%3).bfloat16() # small exact integers
        shard=layer.sp_entry_scatter(full); padded=(n+world-1)//world
        shapes.clear()
        out=op.forward(shard,None,fb)
        assert shapes==[('pool',padded),('router',padded),('router',padded),('unpool',padded)],shapes
        expect=(shard@w1.T)@w2.T
        assert torch.equal(out,expect)
        stats['paired_rank_calls']+=1;stats['matched_BF16']+=out.numel()
        stats['owner_projection_rows_including_padding']+=4*padded
    (Path(tmp)/str(rank)).write_text(json.dumps(stats))
    dist.destroy_process_group()

def arithmetic():
    # Two indistinguishable BF16 high parts, opposite FP32 residuals. With
    # seven higher-scoring experts, the eighth position must depend on that
    # residual for at least one of the two orderings, independently of tie policy.
    cases=[]
    for pair in ((1.0009765625,0.9990234375),(0.9990234375,1.0009765625)):
        w=torch.tensor([3.]*7+list(pair)+[-3.]*247).reshape(256,1)
        hi=w.bfloat16();lo=(w-hi.float()).bfloat16()
        original=(hi.float()+lo.float()).flatten().sigmoid()
        high=hi.float().flatten().sigmoid()
        selected=lambda a:sorted(a.topk(8).indices.tolist())
        cases.append({'two_factors':selected(original),'high_only':selected(high)})
    assert sum(x['two_factors']!=x['high_only'] for x in cases)==1
    return cases

if __name__=='__main__':
    torch.set_num_threads(1); groups={}
    for world in (2,4,8):
        with tempfile.TemporaryDirectory() as temp:
            mp.start_processes(worker,(world,temp),nprocs=world,start_method='fork')
            groups[world]=sum((Counter(json.loads((Path(temp)/str(r)).read_text())) for r in range(world)),Counter())
    M,T,H,I,E=16384,8,8192,4096,256
    rows=M//T
    model={'owner_rows':rows,'pool_flops':2*rows*H*I,'unpool_flops':2*rows*H*I,'router_flops':4*rows*H*E}
    saving=2*(H*I+I*H+2*H*E)*(T-1)//T
    ratio=(T-1)/T
    rbytes={'pool_and_router_input_gather':int(M*H*2*ratio),'pool_owner_transpose':int(rows*I*2*ratio),
            'router_logits_owner_transpose':int(rows*E*4*ratio),'unpool_owner_transpose':int(rows*H*2*ratio),
            'conditional_unpool_input_gather':int(M*I*2*ratio)}
    base=sum(rbytes.values())-rbytes['conditional_unpool_input_gather'];extra=sum(rbytes.values())
    def read(path):return json.loads(path.read_text())
    arch=read(HERE.parent/'v197_grouped_prompt_gate_up/independent_reach.json')['archive']
    controls=read(ROOT/'experiments/runs/b5538a0dd25d46be.json')['response']['result']['records']
    profile_rows=sorted({row['case']['m'] for row in controls if row['event']=='benchmark'})
    manifest=read(HERE/'checkpoint_metadata.json')
    assert manifest['parameters']['F32']==62*(H*E+E)
    print(json.dumps(dict(base_tree=TREE,base_patch=m['sha256'],
        source_sha256={p:sha256(source(TREE,p).encode()).hexdigest() for p in (MODEL,SP)},
        public_F32_elements=manifest['parameters']['F32'],cpu_gloo=groups,full_shape_arithmetic=model,
        rank_layer_cold_weight_bytes_avoided=saving,rank_layer_receive_bytes=rbytes,
        nominal_HBM_saved_microseconds_at_4_8_TBps=saving/4.8e6,
        transfer_us_optimistic_900_GBs=[base/900e3,extra/900e3],
        route_top8_counterexample=arithmetic(),rank_compute_ratio_remove_low_only=sum(v for k,v in model.items() if k.endswith('_flops'))/(model['pool_flops']+model['unpool_flops']+model['router_flops']/2),
        projector_control_rows=profile_rows,archived_target_5pct_per_layer_ms={str(x['users']):x['net_ms_per_layer_call_for_5pct_raw'] for x in arch if x['users']>=16},
        scored_results={'V64':1.4119,'V68_matched_check_16':17512.9},physical_V250=False),indent=2))
