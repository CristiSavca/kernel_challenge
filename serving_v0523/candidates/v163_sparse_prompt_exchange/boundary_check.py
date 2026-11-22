"""Exercise the native V163 model, grouped experts, finalizer ABI and BCG replay on CPU."""
from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import json
import subprocess
import sys
from collections import Counter
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v149_overlapped_expert_route_planning'))
from cpu_check import install, Humming, Expert, finalize, HUM, MAP, MODEL, MOE, SRC

STAGE = Path(sys.argv[1]).resolve()
SP = SRC / 'layers/muse_spark/sparse_prompt_exchange.py'
FINAL = Path('python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py')
BCG = SRC / 'model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py'
V64 = 'b5a37c2923f8d1ab24c1c5cf7920a21e2c086774'


def source(path):
    file = STAGE / path
    if file.exists():
        return file.read_text()
    return subprocess.check_output(['git','-C',STAGE,'show',f'{V64}:{path}'],text=True)


stream = ContextVar('stream', default=None)
capture = ContextVar('capture', default=None)
phase = NS(raw=0, user_hook=False, records=[], masks=0, compiled=False, hw_capture=False, breaks=0)
observations = Counter()
device = NS(current_stream=lambda:'mock compute stream')
env = dict(torch=torch, functools=functools, _current_stream_var=stream, _current_capture_var=capture,
           _forked_streams_var=ContextVar('forks'), _install_wait_stream_hook=lambda:None,
           _uninstall_wait_stream_hook=lambda:None, get_device_module=lambda:device)
classes = {'_weak_ref_if_tensor','_copy_output','_EagerOnGraphCallable','eager_on_graph',
           'BreakableCUDAGraph','BreakableCUDAGraphCapture'}
unit = ast.parse(source(BCG))
exec(compile(ast.Module([n for n in unit.body if getattr(n,'name',None) in classes],[]),str(BCG),'exec'),env)


class Finalizer:
    """Stand in only below the native finalizer's launch and metadata checks."""
    def __getitem__(self, grid):
        def run(values, weight, scores, output, slots, **args):
            assert grid==(scores.shape[0],) and args['TOPK']==scores.shape[1]
            assert args['HAS_ROW_MAP'] and not args['HAS_IDS'] and args['H']==output.shape[1]
            assert args['APPLY_NORM'] and args['HAS_WEIGHT'] and not args['ROUND_PRODUCT']
            phase.records.append(('finalize',values.clone(),slots.clone(),scores.clone()))
            finalize(values,scores,norm_weight=weight,eps=args['eps'],weight_first=args['WEIGHT_FIRST'],
                     slot_expert_ids=None,slot_row_indices=slots,out=output)
        return run


env.update(nn=torch.nn, MuseSpark1xFp4Experts=Expert, _GROUPED_COMPUTE='grouped',
           HummingMethod=Humming, _post_expert_norm_weighted_sum_kernel=Finalizer(),
           triton=NS(next_power_of_2=lambda n:1<<(n-1).bit_length()), _MIN_TOKENS_FOR_FUSED=32,
           HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256,_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256,
           silu_and_mul=lambda g,o:o.copy_(torch.nn.functional.silu(g.float().chunk(2,1)[0])*g.float().chunk(2,1)[1]),
           is_in_breakable_cuda_graph=lambda:capture.get() is not None,
           get_moe_runner_backend=lambda:NS(is_humming=lambda:True))
for path, names in [(MAP,['_map_topk_ids_to_local_experts']),(HUM,['local_humming_expert_ids','humming_grouped_expert_outputs']),
                    (MOE, ['forward','_forward_w4a16','_grouped_humming_and_combine_impl']),
                    (FINAL,['fused_post_expert_norm_weighted_sum']), (MODEL,['_forward_sp_replicated_experts']),
                    (SP,['_group_width','_exchange_inputs','_local_grouped_eligible','_prompt_experts_into','_capture_stub'])]:
    install(source(path),names,env)

def wrap(fn,stub):
    return env['eager_on_graph'](True,capture_stub=stub)(fn)


def record_grouped(self,*args,**kw):
    phase.records.append(('grouped',kw.get('input_row_map') is not None,kw.get('local_route_count')))
    return env['_grouped_humming_and_combine_impl'](self,*args,**kw)


Expert._forward_w4a16=env['_forward_w4a16']
Expert._grouped_humming_and_combine_impl=record_grouped
Expert._grouped_humming_and_combine=wrap(record_grouped,lambda *a,**kw:None)
env['prompt_experts_into']=wrap(env['_prompt_experts_into'],env['_capture_stub'])


def reference(e,x,ids,scores):
    """Unsorted independent per-global-slot BF16 matmul and weighted RMSNorm."""
    local = env['local_humming_expert_ids'](e,ids)
    values=x.new_zeros((*ids.shape,x.shape[1]))
    for j in range(e.num_local_experts):
        row,col=(local==j).nonzero(as_tuple=True)
        if not len(row):continue
        mul=(x[row].float()@e.weights['w13'][j]).bfloat16().mul_(e.w13_global_scale[j])
        a,b=mul.float().chunk(2,dim=1)
        act=(torch.nn.functional.silu(a)*b).bfloat16()
        y=(act.float()@e.weights['w2'][j]).bfloat16().mul_(e.w2_global_scale[j]); values[row,col]=y
    out=torch.zeros_like(x,dtype=torch.float32)
    for k in range(ids.shape[1]):
        y=values[:,k].float()
        if e._post_expert_norm_weight_first:y=y*e._post_expert_norm_weight
        y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+e._post_expert_norm_eps)
        if not e._post_expert_norm_weight_first:y=y*e._post_expert_norm_weight
        out.add_(y*scores[:,k,None])
    return out.bfloat16(),values,local


def exercise(rank, bucket):
    size=2;rows=bucket;sh=(rows+size-1)//size;hidden=8;topk=8
    r=torch.Generator().manual_seed(14518+rank*3+rows)
    e=Expert(); e.env=env;e.num_local_experts=2;e.global_num_experts=4;e.use_humming=e.use_w4a16=True
    e.post_expert_norm=object();e._post_expert_norm=e.post_expert_norm;e._post_expert_norm_eps=1e-6
    e._post_expert_norm_weight=torch.arange(1,hidden+1).bfloat16()/4;e._post_expert_norm_weight_first=bool(rank)
    e._expert_map=torch.tensor(([1,-1,0,-1],[-1,0,-1,1])[rank],dtype=torch.int32)
    e.intermediate_size_per_partition=hidden;e.humming_grouped_tunings={'w13':None,'w2':None};e.gemms=[]
    e.weights={name:torch.randint(-4,5,(2,hidden,hidden*width),generator=r).float()/4 for name,width in [('w13',2),('w2',1)]}
    e.w13_global_scale=torch.tensor([.75,1.125]).bfloat16();e.w2_global_scale=torch.tensor([1.5,.875]).bfloat16()
    x=torch.zeros(size*sh,hidden,dtype=torch.bfloat16);ids=torch.zeros(size*sh,topk,dtype=torch.int32)
    scores=torch.zeros(size*sh,topk,dtype=torch.float32);shard=slice(rank*sh,(rank+1)*sh)
    a,b,c=x[shard],scores[shard],ids[shard];world=NS(rank_in_group=rank,world_size=size,device_group=None)
    def gather_mask(dest,local):
        assert capture.get() is None
        masks=[]
        for who in range(size):
            route=ids[:rows]>=0; owns=route & ((ids[:rows]%2)==who)
            vote = (not phase.user_hook if who==rank else not phase.peer_hook)
            mask=torch.zeros(size*sh+1,dtype=torch.uint8)
            if vote:mask[:rows]=owns.any(1);mask[-1]=1
            masks.append(mask)
        assert torch.equal(masks[rank],local.flatten())
        dest.copy_(torch.stack(masks)); phase.masks+=1
    world.all_gather_into_tensor=gather_mask
    def exchange(dest,packed,received,sent,*,group):
        assert capture.get() is None and group is None
        phase.records.append(('xchg',sum(sent),sum(received)))
        if sum(received):
            owns=(ids[:rows]>=0)&((ids[:rows]%2)==rank)
            dest.copy_(x[:rows][owns.any(1)])
    env['dist']=NS(all_to_all_single=exchange)
    model=NS(experts=e,num_experts=4,top_k=topk)
    env['get_parallel']=lambda:NS(tp_group=world,muse_sparse_prompt_exchange=True)
    # Captured inputs are stable address slots written by the preceding kernels.
    # The existing SP gather is represented by a view into those refreshed slots.
    def gather(t, n):
        assert n==rows
        if t is a:phase.records.append(('dense',));return x[:n]
        if t is b:return scores[:n]
        assert t is c;return ids[:n]
    finished=[]; env['layernorm_sp']=NS(sp_num_tokens=lambda:rows,sp_exit_gather=gather,
                          reduce_scatter_token_rows=lambda output,g:finished.append(output) or output)
    graph=env['BreakableCUDAGraph']();cap=env['BreakableCUDAGraphCapture'](graph)
    # Keep native enter/exit, segment order and replay. The hardware graph launch
    # is modeled by a callback that refreshes tensors before the eager operation.
    cap._begin_new_segment=lambda:None
    def end():
        num=len(graph._segments)
        graph._segments.append(NS(replay=lambda:phase.records.append(('segment',num,phase.raw))))
    cap._end_current_segment=end
    with cap:
        out=env['_forward_sp_replicated_experts'](model,a,b,c)
    assert len(graph._break_fns)==1 and len(graph._segments)==2 and phase.masks==0
    assert not out.any() and out is finished[-1]; stable=out.data_ptr()
    def pre(_mod,args):
        phase.records.append(('hook',));xx,ss,ii,norm=args
        return xx,ss,torch.where(ii>=0,(ii+2)%4,ii),norm
    def post(_mod,args,y):phase.records.append(('posthook',));return y+0.25
    for i in range(36):
        phase.records=[];phase.raw=rows-(i%13);phase.user_hook=(i%6 in (3,4));phase.peer_hook=(i%6==5)
        x.copy_((torch.randint(-8,9,x.shape,generator=r)/8).bfloat16());scores.copy_(torch.randn(scores.shape,generator=r).softmax(-1))
        ids.fill_(-1);prefix=torch.arange(rows,dtype=torch.int32)
        if i%6==0:ids[:rows:3]=((prefix[::3]%4)[:,None]+2*(torch.arange(topk)%2))%4
        elif i%6==1:ids.fill_(1-rank)
        elif i%6==2:ids[:rows]=torch.randint(-1,4,(rows,topk),generator=r)
        else:ids[:rows]=((prefix%3)[:,None]+2*(torch.arange(topk)%2))%4
        if phase.raw<rows:ids[phase.raw:rows]=torch.randint(0,4,(rows-phase.raw,topk),generator=r)
        if i%6==3:h=e.register_forward_pre_hook(pre)
        if i%6==4:h=e.register_forward_hook(post)
        target_ids=torch.where(ids[:rows]>=0,(ids[:rows]+2)%4,ids[:rows]) if i%6==3 else ids[:rows]
        y,values,local=reference(e,x[:rows],target_ids,scores[:rows]);out.fill_(float('nan'))
        graph.replay()
        if i%6==4:y+=.25
        assert torch.equal(out.view(torch.int16),y.view(torch.int16)),(rank,rows,i)
        assert out.data_ptr()==stable and len(graph._break_fns)==1
        assert phase.records[0]==('segment',0,phase.raw) and phase.records[-1]==('segment',1,phase.raw)
        grouped=[v for v in phase.records if v[0]=='grouped'];assert len(grouped)==1
        mapped,count=grouped[0][1:]
        if mapped:
            assert count==int((local>=0).sum())
            observations['sparse_count_crosschecked']+=1
        elif phase.user_hook:
            assert count is None and any(v[0] in ('hook','posthook') for v in phase.records)
            observations['nested_dense_hooked_replays']+=1
        else:
            assert count==int((local>=0).sum());observations['dense_eligible_replays']+=1
        if local.ge(0).any():
            f=next(v for v in phase.records if v[0]=='finalize');comp=f[1];slot=f[2]
            assert torch.equal(slot>=0, local>=0) and torch.equal(comp[slot[slot>=0].long()],values[local>=0])
            observations['original_slot_map_comparisons']+=1
        else:observations['zero_local_routes']+=1
        if phase.user_hook:h.remove()
        observations['replayed_rows']+=rows; observations['replays']+=1
    phase.masks=0



def admission():
    """Exercise thresholds and capture checks without a device collective."""
    from itertools import product
    cases=sparse=0
    for gpu,enable,humming,size,rows,compile_state,phase_name in product((False,True),(False,True),(False,True),(1,8),(8191,8192,16384),(False,True),('eager','full','breakable')):
        route=[]
        x=torch.empty(((rows+size-1)//size,1)); scores=torch.empty((len(x),8)); ids=torch.empty_like(scores,dtype=torch.int32)
        grp=NS(world_size=size)
        env['get_parallel']=lambda:NS(tp_group=grp,muse_sparse_prompt_exchange=enable)
        env['get_moe_runner_backend']=lambda:NS(is_humming=lambda:humming)
        env['is_in_breakable_cuda_graph']=lambda:phase_name=='breakable'
        env['prompt_experts_into']=lambda *a:route.append('sparse')
        env['layernorm_sp']=NS(sp_num_tokens=lambda:rows,sp_exit_gather=lambda t,n:t,
                              reduce_scatter_token_rows=lambda t,g:t)
        class Dense:
            post_expert_norm=None
            def __call__(self,*args): route.append('dense');return args[0]
        obj=NS(experts=Dense(),top_k=8,num_experts=256)
        with patch.object(torch.Tensor,'is_cuda',property(lambda _:gpu)), patch.object(torch.cuda,'is_current_stream_capturing',return_value=phase_name!='eager'), \
             patch.object(torch.compiler,'is_compiling',return_value=compile_state):
            env['_forward_sp_replicated_experts'](obj,x,scores,ids)
        expected=gpu and enable and humming and size>1 and rows>=8192 and not compile_state and phase_name!='full'
        assert route==['sparse' if expected else 'dense'],(enable,humming,size,rows,compile_state,phase_name,route);sparse+=expected;cases+=1
    return {'phase_and_threshold_combinations':cases,'eligible_combinations':sparse}


torch.set_num_threads(1)
with torch.inference_mode(), patch.object(torch.Tensor,'is_cuda',property(lambda _:True)), \
     patch.object(torch.cuda,'get_device_capability',return_value=(9,0)), \
     patch.object(torch.cuda,'is_current_stream_capturing',side_effect=lambda:capture.get() is not None):
    for n in (256,257,512,513):
        for rank in (0,1):exercise(rank,n)
print(json.dumps(dict(torch=torch.__version__,counts=dict(observations),admission=admission(),
    source_sha256={str(p):hashlib.sha256(source(p).encode()).hexdigest() for p in (SP,MODEL,HUM,MOE,FINAL,BCG)},
    hardware='CPU torch; Humming matmuls, NCCL exchanges, CUDA graph segments and Triton finalizer launch emulated'),indent=2))
