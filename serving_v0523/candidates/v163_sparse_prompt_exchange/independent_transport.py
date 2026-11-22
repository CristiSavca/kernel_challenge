"""Replay the exact variable-split sparse prompt exchange on real CPU Gloo ranks."""
import ast
import hashlib
import inspect
import json
import random
import sys
import tempfile
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT=Path(sys.argv[1])
SRC=ROOT/'python/sglang/srt/layers/muse_spark/sparse_prompt_exchange.py'
RAW=SRC.read_text()
HASH=hashlib.sha256(RAW.encode()).hexdigest()
TRACE={}
WIDTH=7
HIDDEN=12
TOP=8

def gather(tensor,group):
    target=tensor.new_empty((group.world_size*tensor.shape[0],*tensor.shape[1:]))
    dist.all_gather_into_tensor(target,tensor.contiguous(),group=group.device_group)
    return target

class Group:
    def __init__(self,rank,world):
        self.world_size=world;self.rank_in_group=rank;self.device_group=dist.group.WORLD
    def all_gather_into_tensor(self,out,x):
        dist.all_gather_into_tensor(out,x,group=self.device_group)

def local_humming_expert_ids(layer,ids):
    safe=ids.clamp(min=0,max=layer.global_num_experts-1)
    return torch.where((ids>=0)&(ids<layer.global_num_experts),layer.expert_map[safe.long()],-1).int().contiguous()

def oracle_result(layer,global_inputs,ids,scores):
    acc=torch.zeros((len(ids),HIDDEN),dtype=torch.float32)
    local=local_humming_expert_ids(layer,ids)
    for k in range(ids.shape[1]):
        mask=local[:,k]>=0
        val=global_inputs[mask].float()*(1+local[mask,k,None].float()/8)
        acc[mask]+=scores[mask,k,None]*val.to(torch.bfloat16).float()
    return acc.to(torch.bfloat16)

class Expert(nn.Module):
    def __init__(self,rank,world,remap):
        super().__init__()
        self._post_expert_norm_weight=object()
        self.post_expert_norm=object()
        self._post_expert_norm=self.post_expert_norm
        self.use_w4a16=self.use_humming=True
        self.intermediate_size_per_partition=HIDDEN
        self.global_num_experts=WIDTH*world
        self.expert_map=torch.full((WIDTH*world,),-1,dtype=torch.int32)
        if remap:
            self.expert_map[rank::world]=torch.arange(WIDTH,dtype=torch.int32)
        else:
            self.expert_map[rank*WIDTH:(rank+1)*WIDTH]=torch.arange(WIDTH,dtype=torch.int32)
        self.forwards=0;self.sparse=0;self.dense=0
    def forward(self,x,scores,ids,norm):
        self.forwards+=1
        return oracle_result(self,x,ids,scores)
    def _grouped_humming_and_combine_impl(self,x,local,scores,out,input_row_map=None,**rest):
        live=(local>=0).any(1)
        assert torch.equal(local,local_humming_expert_ids(self,self.global_ids))
        if input_row_map is not None:
            self.sparse+=1
            assert torch.equal(x.view(torch.int16),self.global_x[live].view(torch.int16))
            assert torch.equal(input_row_map[:len(local)][live],torch.arange(int(live.sum()),dtype=torch.int32))
            for key,val in rest.items():
                if key=='local_route_count':assert val==int((local>=0).sum())
            # A duplicate top-k route must continue to address one physical input row.
            rebuilt=torch.zeros_like(self.global_x)
            rebuilt[live]=x[input_row_map[:len(local)][live].long()]
        else:
            self.dense+=1
            assert torch.equal(x.view(torch.int16),self.global_x.view(torch.int16))
            rebuilt=x
        out.copy_(oracle_result(self,rebuilt,self.global_ids,scores))

def install():
    # The PyTorch code sees the real CUDA/Hopper admission predicates with CPU
    # data and a Gloo backend. Only the Humming numerical payload is simulated.
    global SP, ENV
    mod='sglang.srt.layers.muse_spark.muse_spark_fp4_moe'
    stub=ModuleType(mod);stub.MuseSpark1xFp4Experts=Expert;sys.modules[mod]=stub
    SP=NS()
    body=[n for n in ast.parse(RAW).body if isinstance(n,ast.FunctionDef)]
    ENV=dict(torch=torch,nn=nn,dist=dist,layernorm_sp=SP,local_humming_expert_ids=local_humming_expert_ids)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(SRC),'exec'),ENV)

def make_data(count,src_rows,size,pattern,seed,remap):
    gen=torch.Generator().manual_seed(seed)
    n=torch.arange(count)[:,None];k=torch.arange(TOP)[None,:]
    owner=(n+k)%size
    if pattern in ('all_same','zero_receiver'):owner[:]=0
    elif pattern=='self':owner=((n//src_rows)%size).expand(-1,TOP)
    elif pattern=='self_other':owner=torch.where(k%2==0,(n//src_rows)%size,(n//src_rows+1)%size)
    elif pattern.startswith('cover'):
        count_owners=max(1,min(size-1,int(pattern[5:])))
        owner=(n+(k%count_owners))%size
    elif pattern=='band':owner=((n+2*k)%max(1,size-1))
    elif pattern=='cluster':owner=torch.where(n%3==0,(n+1)%size,(n//src_rows)%size)
    elif pattern=='duplicate':owner=(n%size).expand(-1,TOP)
    elif pattern=='random':owner=torch.randint(0,size,(count,TOP),generator=gen)
    elif pattern=='hot':owner=torch.where(n%4!=0,torch.zeros_like(n),(n+k)%size)
    elif pattern in ('none','distributed_empty'):pass
    elif pattern!='dense':raise AssertionError(pattern)
    expert=(n*3+k*5)%WIDTH
    ids=expert*size+owner if remap else owner*WIDTH+expert
    if pattern=='none':ids.fill_(-1)
    elif pattern=='distributed_empty':ids[::2]=-1;ids[:,:,] = torch.where(k%3==0,ids,-1)
    if pattern=='duplicate':ids=ids[:,0:1].expand(-1,TOP).clone()
    scores=torch.rand(count,TOP,generator=gen)
    rows=torch.arange(src_rows*size)[:,None];cols=torch.arange(HIDDEN)[None,:]
    x=((rows*17+cols*5+seed)%97-46).to(torch.bfloat16)
    x[count:]=8192
    return ids.int(),scores,x

def worker(rank,world,uri,folder):
    torch.set_num_threads(1)
    install()
    dist.init_process_group('gloo',init_method=uri,rank=rank,world_size=world,timeout=timedelta(seconds=50))
    group=Group(rank,world);counts=Counter();modes=Counter();widths=Counter();remote_savings=0
    SP.sp_exit_gather=lambda x,n:gather(x,group)[:n]
    native=dist.all_to_all_single; transport_buffers=[]
    def tracked(out,inp,outs,ins,*a,**kw):
        transport_buffers.append((inp.untyped_storage().data_ptr(),inp.untyped_storage().nbytes(),out.untyped_storage().nbytes()))
        assert len(outs)==len(ins)==world and sum(outs)==out.shape[0] and sum(ins)==inp.shape[0]
        counts['nonempty_sparse_rounds']+=int(sum(ins)>0)
        counts['zero_input_rounds']+=int(sum(ins)==0)
        counts['zero_output_rounds']+=int(sum(outs)==0)
        counts['all_to_all_rounds']+=1
        return native(out,inp,outs,ins,*a,**kw)
    dist.all_to_all_single=tracked
    patterns=('none','dense','all_same','self','self_other','band','cluster','duplicate','random','hot','distributed_empty','cover3','cover5','cover6','cover7')
    cases=[(pattern,mode) for pattern in patterns for mode in ('plain','remap')]
    cases.extend((p,m) for p in ('dense','random','self') for m in ('one_rank_forward_hook','one_rank_pre_hook','one_rank_forward_override','one_rank_compiled','one_rank_unsupported','global_pre_hook'))
    with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)),patch.object(torch.cuda,'get_device_capability',lambda _:(9,0)):
        for i,(pattern,mode) in enumerate(cases):
            src_rows=7+i%3;count=src_rows*world - (i%3)   # Poisoned ragged tail on last source.
            ids,scores,x=make_data(count,src_rows,world,pattern,288+i,mode=='remap')
            expert=Expert(rank,world,mode=='remap');expert.global_ids=ids;expert.global_x=x[:count]
            local_x=x[rank*src_rows:(rank+1)*src_rows]
            specs=local_humming_expert_ids(expert,ids);real=(specs>=0).any(1)
            needs=gather(real.to(torch.int32)[None],group)
            needs=torch.nn.functional.pad(needs,(0,world*src_rows-count))
            table=needs.reshape(world,world,src_rows).sum(2).T.tolist()
            # Independently enumerate valid destination-group widths and the largest usable one.
            def fits(w):
                return all(sum(row[r] for row in table)+sum(table[r][lo:lo+w])<=world*src_rows for r in range(world) for lo in range(0,world,w))
            remote=sum(table[src][dst] for src in range(world) for dst in range(world) if dst!=src)
            allowed=remote<world*src_rows*(world-1)
            expected_width=next((w for w in (world,world//2,world//4,1) if w and allowed and fits(w)),0)
            width=ENV['_group_width'](table,world*src_rows)
            assert width==expected_width
            hook=None
            if rank==world-1 and mode=='one_rank_forward_hook':hook=expert.register_forward_hook(lambda m,args,out:out+1)
            if rank==world-1 and mode=='one_rank_pre_hook':hook=expert.register_forward_pre_hook(lambda m,args:(args[0],args[1]*2,args[2],args[3]))
            if rank==world-1 and mode=='one_rank_forward_override':expert.forward=lambda x,s,i,n:oracle_result(expert,x,i,s)+3
            if rank==world-1 and mode=='one_rank_compiled':expert._compiled_call_impl=lambda x,s,i,n:oracle_result(expert,x,i,s)+4
            if rank==world-1 and mode=='one_rank_unsupported':expert.use_humming=False
            if rank==world-1 and mode=='global_pre_hook':hook=nn.modules.module.register_module_forward_pre_hook(lambda m,args:(args[0],args[1]/2,args[2],args[3]))
            out=torch.empty_like(x[:count]);before=counts['all_to_all_rounds']
            transport_buffers.clear()
            ENV['_prompt_experts_into'](expert,local_x,scores,ids,group,out)
            assert len(set(address for address,_,_ in transport_buffers))<=1
            assert all(send+recv<=world*src_rows*HIDDEN*2 for _,send,recv in transport_buffers)
            reference=expert(x[:count],scores,ids,expert.post_expert_norm)
            if hook:hook.remove()
            assert torch.equal(out.view(torch.int16),reference.view(torch.int16)),(world,rank,i,pattern,mode)
            eligible_mode=mode in ('plain','remap')
            actual_width=width if eligible_mode else 0
            assert expert.sparse==int(bool(actual_width)),(pattern,mode,width)
            assert counts['all_to_all_rounds']-before==(world//actual_width if actual_width else 0)
            counts['bf16_output_values']+=out.numel()
            counts['rank_cases']+=1
            if actual_width:
                remote_savings+=world*src_rows*(world-1)-remote
            widths[actual_width]+=1;modes[mode]+=1
    dist.all_to_all_single=native
    Path(folder,f'{rank}.json').write_text(json.dumps(dict(counters=counts,widths=widths,modes=modes,remote_rows_saved_across_synthetic_cohorts=remote_savings)))
    dist.destroy_process_group()

if __name__=='__main__':
    result={}
    for size in (2,4,8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.start_processes(worker,(size,'file://'+str(Path(tmp)/'gloo'),tmp),nprocs=size,join=True,start_method='fork')
            items=[json.loads(Path(tmp,f'{r}.json').read_text()) for r in range(size)]
            result[str(size)]=dict(rank_cases=sum(v['counters']['rank_cases'] for v in items),bf16_values=sum(v['counters']['bf16_output_values'] for v in items),widths_per_rank=items[0]['widths'],modes_per_rank=items[0]['modes'],
                actual_transport_calls_per_rank=items[0]['counters']['all_to_all_rounds'],
                zero_input_rank_rounds=sum(v['counters']['zero_input_rounds'] for v in items),
                zero_output_rank_rounds=sum(v['counters']['zero_output_rounds'] for v in items))
    print(json.dumps(dict(source_sha256=HASH,torch=torch.__version__,gloo=result,
        scope='Production masked gather, split selection, variable-split Gloo, row maps and per-rank hooks. Synthetic BF16 expert function and CPU emulation of CUDA guard. No NCCL/BCG or H200 timing.'),indent=2))
