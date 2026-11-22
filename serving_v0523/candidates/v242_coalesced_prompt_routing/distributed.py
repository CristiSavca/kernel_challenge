"""Run the real SP entry, admission, grouped planner and symmetric workspace with CPU Gloo.
Physical CUDA streams, mapped peer memory, projections and finalization use explicit substitutes.
Pass an applied Git checkout and optionally a combined candidate directory for inherited replay.
"""
import ast
from collections import Counter
from contextlib import ExitStack, nullcontext
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS, MethodType
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
MANIFEST=json.loads(((Path(sys.argv[2]) if len(sys.argv)>2 else HERE)/'validation.json').read_bytes())
PARENT=json.loads((HERE.parent/'v241_split_serving_decode/validation.json').read_bytes())
ROOT='python/sglang/srt/'
HUM=ROOT+'layers/muse_spark/muse_spark_humming_moe.py'
MOE=ROOT+'layers/muse_spark/muse_spark_fp4_moe.py'
MODEL=ROOT+'models/muse_spark.py'
WORK=ROOT+'layers/muse_spark/peer_prompt_pack.py'

def definitions(tree,path,env,names,cls=None):
    code=subprocess.check_output(['git','-C',STAGE,'show',tree+':'+path])
    body=ast.parse(code).body
    if cls: body=next(n.body for n in body if getattr(n,'name',None)==cls)
    selected=[n for n in body if getattr(n,'name',None) in names]
    assert len(selected)==len(names)
    for n in selected: n.decorator_list=[]
    exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),env)

def module(**members):
    obj=ModuleType('fixture')
    vars(obj).update(members)
    return obj

def launch(rank, size, folder, slabs):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rdv'),rank=rank,world_size=size,timeout=timedelta(seconds=90))
    stats=Counter()
    tp=NS(world_size=size,rank_in_group=rank,device_group=dist.group.WORLD,cpu_group=dist.group.WORLD)
    device=torch.device('cpu')
    class Comm: disabled=False;pull_enabled=True;has_multicast=True;_is_graph_mode_supported=True
    tp.ca_comm=Comm(); tp.ca_comm.rank=rank;tp.ca_comm.world_size=size;tp.ca_comm.device=device
    state={'fault':'','graph':False,'breakable':False}

    def gather(out,x):
        stats['gathers'] += 1
        tensors=[torch.empty_like(x) for _ in range(size)]
        dist.all_gather(tensors,x.contiguous())
        out.copy_(torch.cat(tensors).view_as(out))
    tp.all_gather_into_tensor=gather
    def gather_rows(x,tokens):
        tmp=x.new_empty((size*x.shape[0],*x.shape[1:])); gather(tmp,x);return tmp[:tokens]
    def reduce(full,group):
        total=full.float();dist.all_reduce(total)
        rows=(len(full)+size-1)//size
        result=full.new_zeros((rows,full.shape[1])); chunk=total[rank*rows:(rank+1)*rows]
        result[:len(chunk)]=chunk;return result
    sp=NS(sp_exit_gather=gather_rows,reduce_scatter_token_rows=reduce,sp_num_tokens=lambda:state['tokens'])
    class Handle:
        def __init__(self):self.world_size=size;self.rank=rank
        def get_buffer(self,i,shape,dtype):
            if state['fault']=='alias' and rank==size-1 and i==rank:i=(i+1)%size
            return slabs[i][:shape[0],:shape[1]]
        def barrier(self,channel):
            assert channel==0
            dist.barrier();stats['barriers']+=1
    def alloc(shape,stride,dtype,dev,group):
        assert stride==(shape[1],1) and group==dist.group.WORLD.group_name
        if state['fault']=='allocation' and rank==size-1:raise RuntimeError('synthetic allocation failure')
        return slabs[rank][:shape[0],:shape[1]]
    def rendezvous(x):
        stats['rendezvous']+=1;return Handle()
    class Stream:
        def __init__(self,*_,**kw):pass
        def wait_stream(self,other):stats['waits']+=1
        def synchronize(self):pass
    def pull(peers,mapping,inputs,shard_rows):
        if state['fault']=='compile' and rank==size-1:raise ValueError('synthetic JIT failure')
        banks={b.data_ptr():b for b in slabs}
        if state.get('startup'):
            assert (mapping >= 0).sum() == 1 and mapping[-1, 0] == 0
            assert (len(mapping)-1)//shard_rows == rank and len(peers) == size
            inputs.copy_(banks[int(peers[rank])][:1])
            return
        live=torch.where(mapping>=0)
        for src in range(size):
            keep=live[0]//shard_rows==src
            inputs[mapping[live][keep].long()]=banks[int(peers[src])][live[0][keep]%shard_rows]
        stats['pulled_rows']+=len(live[0])
    work_env=dict(torch=torch,dist=dist,nn=nn,logger=NS(warning=lambda *a:None),TritonError=RuntimeError,
                  threading=__import__('threading'),pack_peer_rows=pull,layernorm_sp=sp)
    definitions(MANIFEST['tree'],WORK,work_env,{'PeerPromptInputs','peer_storage_rows','prepare_peer_prompt_inputs','_local_grouped_eligible','_peer_experts_into'})
    parent_work=dict(work_env)
    definitions(PARENT['tree'],WORK,parent_work,{'_peer_experts_into'})
    definitions(MANIFEST['tree'],WORK,work_env,{'_gather_prompt_routes'})
    class Pack:
        def __getitem__(self,grid):
            def run(x,slots,inputs,mapping,first,n,total,rs,cs,**kw):
                assert grid==(n+(total+255)//256,)
                if n:inputs.copy_(x[slots[first:]//kw['TOPK']])
                mapping.view(-1)[slots]=torch.arange(-first,total-first,dtype=mapping.dtype).clamp_min(-1)
            return run
    fake_triton=NS(next_power_of_2=lambda n:1<<(n-1).bit_length(),cdiv=lambda n,d:(n+d-1)//d)
    def gate(layer,inputs,keys,offsets,tuning,prompt):
        assert prompt and torch.equal(keys.long().bincount(minlength=layer.num_local_experts).int(),offsets.diff())
        return (inputs.float()*(1+keys[:,None].float()/10)).bfloat16()
    def down(layer,x,y,offsets,compute,tuning):
        y.copy_(x.flip([1]));return True
    def finalize(values,scores,norm_weight,eps,weight_first,slot_expert_ids,slot_row_indices,expert_output_scales,out):
        total=torch.zeros_like(out,dtype=torch.float)
        for k in range(scores.shape[1]):
            flag=slot_row_indices[:,k]>=0
            idx=slot_row_indices[flag,k].long()
            if len(idx):
                x=(values[idx].float()*expert_output_scales[slot_expert_ids[flag,k].long(),None].float()).bfloat16().float()
                if weight_first:x=x*norm_weight
                x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)
                if not weight_first:x=x*norm_weight
                total[flag]+=x*scores[flag,k,None]
        out.copy_(total)
    class Expert(nn.Module):
        def forward(self,x,scores,ids,norm):
            local=local_ids(self,ids)
            target=torch.empty_like(x);self._grouped_humming_and_combine_impl(x,local,scores,target);return target
    def local_ids(owner,idx):
        return torch.where(idx//4==rank,idx%4,-1).int().contiguous()
    def make_model(tree,workspace):
        e=Expert(); e.use_w4a16=e.use_humming=True;e.num_local_experts=4
        e.intermediate_size_per_partition=e.hidden_size=slabs[0].shape[1]
        e._post_expert_norm_weight=torch.linspace(.5,1.5,e.hidden_size)
        e._post_expert_norm=e.post_expert_norm=object();e._post_expert_norm_eps=.001
        e._post_expert_norm_weight_first=state.get('weight_first',True)
        e.w2_global_scale=torch.tensor([.375,.6875,1.0625,1.875]).bfloat16()
        e.humming_prompt_gate_up=e.humming_prompt_down=True;e.humming_grouped_tunings={'w13':'a','w2':'b'}
        env=dict(torch=torch,MuseSpark1xFp4Experts=Expert,triton=fake_triton,_GROUPED_COMPUTE='x',_pack_prompt_routes=Pack())
        names={'humming_grouped_expert_outputs'}
        names.add('prompt_route_order')
        definitions(tree,HUM,env,names)
        pkg=module(local_humming_expert_ids=local_ids,**{name:env[name] for name in names})
        sys.modules['sglang.srt.layers.muse_spark.muse_spark_humming_moe']=pkg
        comb=dict(torch=torch,fused_post_expert_norm_weighted_sum=finalize)
        definitions(tree,MOE,comb,{'_grouped_humming_and_combine_impl'},'MuseSpark1xFp4Experts')
        e._grouped_humming_and_combine_impl=MethodType(comb['_grouped_humming_and_combine_impl'],e)
        model=NS(top_k=8,num_experts=256,experts=e,_peer_prompt_inputs=workspace)
        env=dict(torch=torch,layernorm_sp=sp,get_parallel=lambda:NS(tp_group=tp))
        definitions(tree,MODEL,env,{'_forward_sp_replicated_experts'},'MuseSpark1xMoE')
        return model,env['_forward_sp_replicated_experts']
    mods={
        'humming.layer':module(HummingMethod=NS()),
        'sglang.kernels.ops.activation.activation':module(silu_and_mul=None),
        'sglang.srt.layers.muse_spark.decode_gate_up':module(run_gate_up=gate),
        'sglang.srt.layers.muse_spark.prompt_weight_pipeline':module(run_prompt_down=down),
        'sglang.srt.layers.muse_spark.muse_spark_fp4_moe':module(MuseSpark1xFp4Experts=Expert,_HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256),
        'sglang.srt.layers.muse_spark.peer_prompt_pack':module(peer_experts_into=work_env['_peer_experts_into']),
        'sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context':module(is_in_breakable_cuda_graph=lambda:state['breakable']),
        'sglang.srt.distributed.device_communicators.custom_all_reduce_v2':module(CustomAllReduceV2=Comm),
    }
    with ExitStack() as stack:
        add=stack.enter_context
        add(torch.no_grad());add(patch.dict(sys.modules,mods));add(patch.object(torch.Tensor,'record_stream',lambda *a:None))
        add(patch.object(torch.Tensor,'is_cuda',property(lambda _:True)))
        add(patch.object(torch.cuda,'current_stream',lambda *a:Stream()));add(patch.object(torch.cuda,'Stream',Stream))
        add(patch.object(torch.cuda,'stream',lambda s:nullcontext()))
        add(patch.object(torch.cuda,'get_device_capability',lambda d:(9,0)))
        add(patch.object(torch.cuda,'is_current_stream_capturing',lambda:state['graph']))
        add(patch.object(torch._C._distributed_c10d,'_SymmetricMemory',NS(empty_strided_p2p=alloc,rendezvous=rendezvous),create=True))
        # Transmit raw score bits, including signaling NaNs, infinities and
        # negative zero, over the actual Gloo byte collective. The receiver
        # has to reassemble rank order, both alternate dtypes and padding.
        bits=torch.tensor([0,-2147483648,0x3f800001,0x7f800000,0x7fa12000,-8388608,0x00000001,0x12345678],dtype=torch.int32)
        for score_type,id_type in ((torch.float32,torch.int32),(torch.float32,torch.int64),(torch.bfloat16,torch.int32)):
            for noncontiguous in (False,True):
                r,k=rank+5,3
                # Every peer must send identical shapes even when content differs.
                n=17
                raw=bits[(torch.arange(n*k*2)+r)%len(bits)].view(n,k*2)
                v=raw.view(score_type) if score_type==torch.float32 else raw.to(torch.int16).view(torch.bfloat16)
                idx=(torch.arange(n*k*2).reshape(n,k*2)*r-53).to(id_type)
                v,idx=(x[:,::2] if noncontiguous else x[:,:k].contiguous() for x in (v,idx))
                legal=rank%2==0
                reference=[gather_rows(x,n*size) for x in (v,idx)]
                before=stats['gathers']
                sv,iv,flags=work_env['_gather_prompt_routes'](tp,v,idx,legal)
                assert stats['gathers']==before+1
                assert flags.tolist()==[int(i%2==0) for i in range(size)]
                for expected,actual in zip(reference,(sv,iv)):
                    assert actual.is_contiguous() and actual.shape==expected.shape and actual.dtype==expected.dtype
                    assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
                    stats['route_bytes_exact']+=actual.nbytes
                stats['byte_cases']+=1
        make=work_env['prepare_peer_prompt_inputs'];state['startup']=True
        capacity=work_env['peer_storage_rows']
        for graph_max,scheduler_max,expected in ((16384,16384,2048),(16640,16640,2080),(16640,16384,2048),
                                                (16384,16640,2048),(None,16640,2048),(131072,131072,2176)):
            assert capacity(4096,8,16384,[8192,16384],graph_max,scheduler_max)==expected
            stats['capacity_checks']+=1
        for fault in ('allocation','alias','compile','topology'):
            state['fault']=fault;tp.ca_comm.has_multicast=not(fault=='topology' and rank==size-1)
            assert make(tp,slabs[0].shape[1],8,16384,[16384],16640,16640) is None;stats['setup_veto']+=1
        state['fault']='';tp.ca_comm.has_multicast=True
        workspace=make(tp,slabs[0].shape[1],8,16384,[16384],16640,16640);state['startup']=False
        assert workspace is not None and workspace.storage.shape[0]==(16640+size-1)//size
        for t,kind,graph,br in ((8191,'all',False,False),(8192,'all',False,False),(8201,'skew',True,True),
                               (16384,'remote',True,True),(16640,'mixed',True,True),(8193,'hook',True,True),
                               (8192,'all',True,False),(16700,'over',False,False)):
            state.update(tokens=t,graph=graph,breakable=br,weight_first=(t%2==0))
            rows=(t+size-1)//size;ids=torch.arange(rank*rows*8,(rank+1)*rows*8).reshape(rows,8)%(4*size)
            if kind=='skew':ids.fill_(0)
            if kind=='remote':ids.fill_(-1)
            if kind in ('mixed','hook'):ids[::3,:]=-1;ids[1::3,4:]=ids[1::3,:4]
            x=(torch.arange(rank*rows*slabs[0].shape[1],(rank+1)*rows*slabs[0].shape[1]).reshape(rows,-1)%317-158).bfloat16()/17
            scores=(torch.arange(rows*8).reshape(rows,8)%31+1).float()/37
            outs=[]
            barriers=stats['barriers']
            counts=[]
            if kind in ('all','mixed','hook'): ids=ids.int()
            for tree,owner in ((PARENT['tree'],parent_work),(MANIFEST['tree'],work_env)):
                mods['sglang.srt.layers.muse_spark.peer_prompt_pack'].peer_experts_into=owner['_peer_experts_into']
                before=stats['gathers']
                model,run=make_model(tree,workspace)
                if kind=='hook' and rank==size-1:
                    def verify_hook(m,a,o):
                        _,score_full,id_full,_=a
                        assert score_full.shape==id_full.shape==(t,8)
                        assert score_full.untyped_storage().data_ptr()!=id_full.untyped_storage().data_ptr()
                        for y in (score_full,id_full):
                            assert y.storage_offset()==0 and y.stride()==(8,1)
                            assert y.untyped_storage().nbytes()==rows*size*8*y.element_size()
                        full=((rows*size),8)
                        padded=[y.as_strided(full,(8,1)).clone() for y in (score_full,id_full)]
                        if tree==PARENT['tree']:
                            state['hook_buffers']=padded
                        else:
                            assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(padded,state['hook_buffers']))
                            stats['checked_hook_backings']+=2
                        return o+1
                    model.experts.register_forward_hook(verify_hook)
                outs.append(run(model,x,scores,ids))
                counts.append(stats['gathers']-before)
            assert torch.equal(outs[0].view(torch.int16),outs[1].view(torch.int16)),(rank,t,kind)
            fast=t>=8192 and t<=16640 and kind!='hook' and (not graph or br)
            assert stats['barriers']-barriers==4*fast,(rank,t,kind,stats)
            enters=t>=8192 and (not graph or br)
            assert counts == ([3,1] if fast else [4,2] if enters else [3,3]),(t,kind,counts,rank)
            stats['transport_boundaries_removed'] += counts[0]-counts[1]
            stats['cases']+=1;stats['joined_fast']+=int(fast);stats['bf16_words']+=outs[-1].numel()
    (Path(folder)/f'{rank}.json').write_text(json.dumps(stats))
    dist.destroy_process_group()

if __name__=='__main__':
    torch.set_num_threads(1);result={}
    for size in (2,4,8):
        with tempfile.TemporaryDirectory() as tmp:
            slabs=[torch.empty(((16640+size-1)//size,16),dtype=torch.bfloat16).share_memory_() for _ in range(size)]
            mp.start_processes(launch,(size,tmp,slabs),nprocs=size,start_method='fork')
            result[size]=sum((Counter(json.loads((Path(tmp)/f'{i}.json').read_text())) for i in range(size)),Counter())
    print(json.dumps(dict(physical_gpu=False,patch_sha256=MANIFEST['patch_sha256'],parent=PARENT['patch_sha256'],
                          host_substitutes=['peer mappings & stream barriers','Triton pack','Humming projections and weighted finalization'],gloo=result),indent=2))
