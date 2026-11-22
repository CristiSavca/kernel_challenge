"""Exercise V160 scheduler admission, native reservation and role transitions on CPU."""
import ast
import dataclasses
import hashlib
import json
import re
import struct
import subprocess
import sys
import tempfile
import types
import weakref
from collections import Counter
from contextlib import nullcontext
from copy import copy
from datetime import timedelta
from enum import IntEnum, auto
from functools import lru_cache, total_ordering
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGE = Path(sys.argv[1])
DIR = 'python/sglang/srt/'
PIN = '297e54fd105e974cdf401e5b4d20253a22c16f90'
HASHES = {}
F = 'speculative/dflash_worker_v2.py'
S = 'managers/schedule_batch.py'
I = 'speculative/dflash_info_v2.py'
A = 'managers/scheduler_components/mixed_dflash_admission.py'
Algo = NS(is_dflash=lambda:True, is_dflash_family=lambda:True, is_none=lambda:False)
DisaggregationMode = NS(NULL='none')
Backend = NS(FULL='full', BREAKABLE='breakable')
SIMULATE_ACC_LEN = 0
TOP_K_ALL = 1<<30
is_cuda = lambda:True
is_pin_memory_available = lambda _device:False
_get_overlap_plan_stream = lambda _device:(None,nullcontext())
get_exec = lambda:NS(deterministic=NS(enable_deterministic_inference=False),features=NS(enable_watermark=False,enable_custom_logit_processor=False),mamba=NS(enable_mamba_extra_buffer_lazy=False))
get_spec = lambda:NS(speculative_num_draft_tokens=8, speculative_dflash_mixed_prefill=ENABLED)
get_parallel = lambda:PARALLEL
get_alloc_reserve_per_decode = lambda:16
ceil_align = lambda x,a:(x+a-1)//a*a
dataclass = dataclasses.dataclass
class NoPenalties:
    is_required=False
    def __init__(self, *, batch, **kwargs):self.batch=weakref.ref(batch)
    def merge(self,their):
        assert not their.is_required

penaltylib=NS(BatchedPenalizerOrchestrator=NoPenalties,BatchedFrequencyPenalizer=object,BatchedMinNewTokensPenalizer=object,BatchedPresencePenalizer=object,BatchedRepetitionPenalizer=object)

@lru_cache
def source(rel):
    p=STAGE/(DIR+rel)
    if p.is_file():return p.read_text()
    return subprocess.check_output(['git','-C',STAGE,'show',':'+DIR+rel], text=True)

def publish(name, **attrs):
    module=types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name]=module

def grab(rel,name,owner=None,methods=None,fields=False):
    node=next(n for n in ast.parse(source(rel)).body if getattr(n,'name',None)==(owner or name))
    if owner:node=next(n for n in node.body if getattr(n,'name',None)==name)
    if methods is not None:
        node.bases=[]
        node.body=[n for n in node.body if getattr(n,'name',None) in methods or (fields and isinstance(n,ast.AnnAssign))]
    HASHES['/'.join(filter(None,(rel,owner,name)))]=hashlib.sha256(ast.unparse(node).encode()).hexdigest()
    exec('from __future__ import annotations\n'+ast.unparse(node),globals())
    return globals()[name]

class PrefillCudaGraphRunner(NS):pass
class DecodeCudaGraphRunner(NS):pass
class BaseTokenToKVPoolAllocator(NS):
    def check_decode_capacity(self,**kwargs):
        raise AssertionError('early check must never evict')
    def available_size(self):return self.free
class ExtraDemand(BaseTokenToKVPoolAllocator):
    def check_decode_capacity(self,**kwargs):raise AssertionError('unexpected')
class Reserved(NS):pass
def alloc_for_spec_decode(tree,pool,*,reqs,req_pool_indices,cur_kv_lens,cur_kv_lens_cpu,nxt_kv_lens,nxt_kv_lens_cpu,num_needed_tokens,batch):
    assert torch.equal(cur_kv_lens,cur_kv_lens_cpu) and torch.equal(nxt_kv_lens,nxt_kv_lens_cpu)
    bank=batch.token_to_kv_pool_allocator
    assert bank.free>=num_needed_tokens
    bank.free-=num_needed_tokens
    bank.reservations+=1
    for row,r,start,stop in zip(req_pool_indices,reqs,cur_kv_lens.tolist(),nxt_kv_lens.tolist()):
        assert start==r.kv.kv_allocated_len
        r.kv.kv_allocated_len=stop
        pool.req_to_token[row,start:stop]=torch.arange(start,stop)+1000*int(row)

def initialize():
    global ENABLED,PARALLEL,GRID,FLAGS
    ENABLED=True
    PARALLEL=NS(dp_size=1,pp_size=1,attn_cp_size=1,attn_dcp_size=1)
    GRID=json.loads((ROOT/'experiments/iterations/20261118-serving-mixed-graph-capacity.json').read_text())
    grab('model_executor/forward_batch_info.py','ForwardMode')
    grab('model_executor/forward_batch_info.py','CaptureHiddenMode')
    grab('beam_search/batch_tail.py','strip_beam_tail')
    grab('mem_cache/allocation_sizing.py','page_aligned_decode_alloc_lens')
    grab('sampling/sampling_batch_info.py','merge_bias_tensor')
    grab('sampling/sampling_batch_info.py','SamplingBatchInfo',methods={'__len__','from_schedule_batch','_make_sampling_mask_batch_indices','_make_sampling_support_logprobs_capture_indices','merge_batch','adjusted_from_schedule_batch','adjusted_merge_batch'},fields=True)
    grab(S,'ScheduleBatch',methods={'batch_size','is_empty','prepare_for_decode','new_tokens_required_next_decode','_new_tokens_required_next_decode_spec_v2','merge_batch'},fields=True)
    ScheduleBatch.maybe_evict_swa=lambda s:setattr(s,'evictions',s.evictions+1)
    publish('sglang.srt.managers.schedule_batch',ScheduleBatch=ScheduleBatch)
    prepare=grab('speculative/spec_utils.py','spec_prepare_for_decode')
    publish('sglang.srt.speculative.spec_utils',spec_prepare_for_decode=prepare)
    for n in ['_ensure_prepare_length_buffers','prepare_for_decode']:
        setattr(Reserved,n,grab(I,n,'DFlashDraftInputV2'))
    grab(F,'DFlashWorkerV2',methods={'can_settle_mixed_verify','_can_skip_verify_logits'})
    import bisect
    globals()['bisect']=bisect
    pad=grab('model_executor/runner/base_cuda_graph_runner.py','_pad_to_bucket','BaseCudaGraphRunner')
    PrefillCudaGraphRunner._pad_to_bucket=pad
    DecodeCudaGraphRunner._pad_to_bucket=pad
    PrefillCudaGraphRunner._uses_eager_prefill_tail=grab('model_executor/runner/prefill_cuda_graph_runner.py','_uses_eager_prefill_tail','PrefillCudaGraphRunner')
    for n in ['_candidate_geometry','admit_mixed_dflash','build_mixed_dflash_batch']:grab(A,n)
    publish('sglang.srt.managers.scheduler_components.mixed_dflash_admission',admit_mixed_dflash=admit_mixed_dflash,build_mixed_dflash_batch=build_mixed_dflash_batch)
    s='managers/scheduler.py'; cls=next(n for n in ast.parse(source(s)).body if getattr(n,'name',None)=='Scheduler')
    fn=next(n for n in cls.body if getattr(n,'name',None)=='_get_new_batch_prefill_raw')
    branch=next(n for n in fn.body if isinstance(n,ast.If) and 'get_spec().speculative_dflash_mixed_prefill' in ast.unparse(n.test))
    text=ast.unparse(branch)
    HASHES['scheduler.admission_branch']=hashlib.sha256(text.encode()).hexdigest()
    exec('def schedule(self,new_batch,running_batch,chunked_prefill_size):\n'+'\n'.join('    '+l for l in text.splitlines())+'\n    return new_batch,running_batch',globals())
    arg='arg_groups/field_order.py';name='POSITIONAL_FIELD_ORDER'
    def positional(s):
        scope={};exec(s,scope);return scope[name]
    old=subprocess.check_output(['git','-C',ROOT/'serving_v0523/sglang','show',PIN+':'+DIR+arg],text=True)
    assert positional(source(arg))==positional(old)+('speculative_dflash_mixed_prefill',)
    field=next(n for n in ast.walk(ast.parse(source('arg_groups/fields/spec.py'))) if isinstance(n,ast.AnnAssign) and n.target.id=='speculative_dflash_mixed_prefill')
    assert ast.literal_eval(field.value) is False
    # Exercise the challenge's actual msgspec namespace and argparse adapters.
    import argparse
    import msgspec
    def module(name, rel):
        loaded=types.ModuleType(name)
        sys.modules[name]=loaded
        exec(compile(source(rel),rel,'exec'),loaded.__dict__)
        return loaded
    args=module('sglang.srt.arg_groups.arg_utils','arg_groups/arg_utils.py')
    module('sglang.srt.arg_groups.choices','arg_groups/choices.py')
    specs=module('sglang.srt.arg_groups.fields.spec','arg_groups/fields/spec.py')
    flag='speculative_dflash_mixed_prefill'
    assert args.namespace_of(specs.Spec)[flag]=='spec' and getattr(specs.Spec(),flag) is False
    parser=argparse.ArgumentParser()
    args.add_cli_args_from_dataclass(parser,specs.Spec)
    assert not getattr(parser.parse_args([]),flag)
    assert getattr(parser.parse_args(['--speculative-dflash-mixed-prefill']),flag)
    assert getattr(parser.parse_args(['--speculative-dflash-mixed-prefill','--speculative-dflash-block-size','8']), 'speculative_dflash_block_size')==8

def create(p=1,t=3,q=582,allocate=True):
    grids=GRID['target_prefill_graph_rows'],GRID['target_verify_graph_requests']
    kv=NS(req_to_token=torch.zeros(66,1024,dtype=torch.int64) if allocate else NS(shape=(66,2048)))
    alloc=BaseTokenToKVPoolAllocator(free=32000,reservations=0,page_size=1)
    pre=PrefillCudaGraphRunner(capture_num_tokens=grids[0],max_bs=64,max_num_tokens=grids[0][-1],prefill_backend_name=Backend.BREAKABLE,enable_cp_bcg_capture=False)
    ver=DecodeCudaGraphRunner(capture_bs=grids[1],max_bs=grids[1][-1],captured_req_width=8,enable_two_batch_overlap=False,require_mlp_tp_gather=False,ragged_verify_mode=False,enable_pdmux=False,disable_padding=False)
    runner=NS(req_to_token_pool=kv,prefill_cuda_graph_runner=pre,decode_cuda_graph_runner=ver,sampling_observer=None,ngram_embedding_manager=NS(enabled=False))
    worker=DFlashWorkerV2();worker.target_worker=NS(model_runner=runner);worker.block_size=8
    for key in ('lilicorr','_selector_sample'):setattr(worker,key,None)
    for key in ('_is_domino','_need_mamba_verify_commit','use_compact_draft_cache'):setattr(worker,key,False)
    sched=NS(draft_worker=worker,model_worker=worker,tp_worker=worker.target_worker,max_prefill_tokens=16384,tp_group=NS(world_size=1),spec_algorithm=Algo,page_size=1,dllm_config=None,require_mlp_sync=False,enable_unified_memory=False,enable_hisparse=False,enable_overlap=True,is_mixed_chunk=False,disaggregation_mode=DisaggregationMode.NULL,_reqs_finishing_inflight=lambda _:[])
    info=Reserved(**{k:None for k in ('_prepare_batch_seq_lens_cpu_buf','_prepare_cur_kv_lens_cpu_buf','_prepare_nxt_kv_lens_cpu_buf','_prepare_cur_kv_lens_gpu_buf','_prepare_nxt_kv_lens_gpu_buf')})
    def role(start,count,decode):
        reqs=[]
        for i in range(start,start+count):
            length=24+i
            reqs.append(NS(kv=NS(req_pool_idx=i,kv_committed_len=length,kv_allocated_len=length+2),is_retracted=False,finished=lambda:False,beam_group=None,return_sampling_mask=False,return_indexer_topk=False,return_routed_experts=False,stream=True,multimodal_inputs=None,decode_batch_idx=0,extend_end=length,output_ids=[1],sampling_params=NS(top_k=1,top_p=1.,temperature=0.75+i/100,min_p=0.,logit_bias=None)))
        indices=torch.arange(start,start+count)
        batch=ScheduleBatch(reqs=reqs,req_to_token_pool=kv,token_to_kv_pool_allocator=alloc,
            spec_algorithm=Algo,device='cpu',spec_info=info if decode else None,enable_overlap=True,
            req_pool_indices=indices,req_pool_indices_cpu=indices.clone(),seq_lens=indices+24,
            orig_seq_lens=indices+24,seq_lens_cpu=None if decode else indices+24,
            model_config=NS(vocab_size=32,is_encoder_decoder=False),forward_mode=ForwardMode.EXTEND)
        batch.return_hidden_states_mode=CaptureHiddenMode.NULL;batch.evictions=0
        batch.sampling_info=SamplingBatchInfo.from_schedule_batch(batch,32)
        batch.extend_lens=[1]*(count-1)+[max(1,q-count+1)];batch.prefix_lens=[0]*count
        batch.extend_num_tokens=q;batch.extend_logprob_start_lens=[x-1 for x in batch.extend_lens]
        batch.input_ids=None;batch.out_cache_loc=indices
        return batch
    return sched,role(1,p,False),role(1+p,t,True)

def denied_cases():
    def set_path(path,value):
        def apply(s,p,d):
            refs={'s':s,'p':p,'d':d,'r':d.reqs[0],'w':s.draft_worker,'g':PARALLEL,'f':s.tp_worker.model_runner.prefill_cuda_graph_runner,'v':s.tp_worker.model_runner.decode_cuda_graph_runner}
            parts=path.split('.');obj=refs[parts[0]]
            for attr in parts[1:-1]:obj=getattr(obj,attr)
            setattr(obj,parts[-1],value)
        return apply
    settings={'s.enable_overlap':False,'s.enable_unified_memory':True,'s.disaggregation_mode':'PREFILL','s.require_mlp_sync':True,'s.is_mixed_chunk':True,'s.page_size':160,'s.enable_hisparse':True,'s.dllm_config':1,'p.input_embeds':1,'p.replace_embeds':1,'s.max_prefill_tokens':605,'s._reqs_finishing_inflight':lambda _:[1],'v.ragged_verify_mode':True,'v.enable_pdmux':True,'v.require_mlp_tp_gather':True,'v.enable_two_batch_overlap':True,'v.captured_req_width':7,'v.max_bs':2,'f.max_bs':3,'f.enable_cp_bcg_capture':True,'f.prefill_backend_name':'eager','d.is_extend_in_batch':True,'d.spec_info':None,'p.spec_info':1,'p.return_logprob':True,'d.return_hidden_states':True,'d.sampling_info.is_all_greedy':False,'p.sampling_info.grammars':[1],'w.lilicorr':1,'w._selector_sample':1,'r.multimodal_inputs':1,'r.beam_group':1,'r.return_sampling_mask':True,'r.is_retracted':True,'r.finished':lambda:True,'r.kv.req_pool_idx':0,'r.kv.kv_committed_len':2000,'r.kv.kv_allocated_len':4098,'d.token_to_kv_pool_allocator.free':41,'p.prefix_lens':[],'p.extend_lens':[334],'p.extend_num_tokens':16383,'d.req_to_token_pool':object(),'g.pp_size':2,'g.attn_dcp_size':2}
    cases={k:set_path(k,v) for k,v in settings.items()}
    cases['duplicate_physical_row']=lambda s,p,d:setattr(d.reqs[0].kv,'req_pool_idx',p.reqs[0].kv.req_pool_idx)
    def override(s,p,d):
        alloc=ExtraDemand(free=999,reservations=0,page_size=1);p.token_to_kv_pool_allocator=d.token_to_kv_pool_allocator=alloc
    cases['allocator_demand_override']=override
    for key,setup in cases.items():
        sched,p,d=create();setup(sched,p,d)
        result=schedule(sched,p,d,16384)
        assert result[0] is p and result[1] is d and d.evictions==0 and d.reqs[0].decode_batch_idx==0,key
        PARALLEL.pp_size=PARALLEL.attn_dcp_size=1
    return list(cases)

def acceptance():
    global ENABLED
    for tails in [1,2,3,4]:
        sched,p,d=create(t=tails)
        # Same 640-token capture bucket; native double-width allocation requests exactly 14/row.
        sched.max_prefill_tokens=582+8*tails
        d.token_to_kv_pool_allocator.free=14*tails
        prior=[(r.extend_end,list(r.output_ids)) for r in d.reqs]
        p_sample,d_sample=p.sampling_info,d.sampling_info
        prompt_temps=p_sample.temperatures.clone();tail_temps=d_sample.temperatures.clone()
        for cap,enabled in [(sched.max_prefill_tokens-1,True),(sched.max_prefill_tokens,False)]:
            ENABLED=enabled;before=schedule(sched,p,d,cap)
            assert before[0] is p and before[1] is d and d.evictions==0
        ENABLED=True
        outer,carry=schedule(sched,p,d,sched.max_prefill_tokens)
        assert carry.is_empty() and outer.mixed_dflash_roles[0] is p and outer.mixed_dflash_roles[1] is d
        assert outer.req_pool_indices.tolist()==list(range(1,2+tails)) and outer.forward_mode==ForwardMode.MIXED
        assert outer.extend_lens==[582]+[8]*tails and outer.extend_num_tokens==582+8*tails
        assert [n-k for n,k in zip(outer.extend_lens,outer.extend_logprob_start_lens)]==[1]*(1+tails)
        assert p.sampling_info is p_sample and d.sampling_info is d_sample
        assert outer.sampling_info.penalizer_orchestrator.batch() is outer
        assert p_sample.penalizer_orchestrator.batch() is p and d_sample.penalizer_orchestrator.batch() is d
        assert torch.equal(outer.sampling_info.temperatures,torch.cat([prompt_temps,tail_temps]))
        assert p.forward_mode==ForwardMode.EXTEND and d.forward_mode==ForwardMode.DECODE
        assert p.req_pool_indices.tolist()==[1] and len(d.reqs)==tails and d.evictions==1
        assert d.token_to_kv_pool_allocator.free==0 and d.token_to_kv_pool_allocator.reservations==1
        for r,before in zip(d.reqs,prior):
            assert r.stream and r.decode_batch_idx==1 and (r.extend_end,r.output_ids)==before
            row=r.kv.req_pool_idx; start=r.kv.kv_committed_len+2
            assert r.kv.kv_allocated_len==start+14
            assert torch.equal(d.req_to_token_pool.req_to_token[row,start:start+14],torch.arange(start,start+14)+1000*row)
    return 4

def historic():
    archive=ROOT/'experiments/modal_node/20261113T204343Z-bench'
    loads=json.loads(next(archive.glob('*load.json')).read_text())
    order=[r['concurrency'] for r in loads['rungs'] for _ in range(r['reps'])]
    log=next(archive.glob('*v41__server.log')).read_text()
    pos=-1;counts={n:Counter() for n in order}
    for line in log.splitlines():
        if 'Cache flushed successfully!' in line:pos+=1
        m=re.search(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+),.*?#running-req: (\d+)',line)
        if not m or pos<0:continue
        p,q,t=map(int,m.groups())
        if t==0:continue
        entry=counts[order[pos]];entry['opportunities']+=1
        s,pr,de=create(p,t,q,allocate=False)
        if admit_mixed_dflash(s,pr,de,16384):entry['admitted']+=1;entry['tails']+=t
    assert [counts[n]['admitted'] for n in (1,4,16,32)]==[0,66,99,55],counts
    return counts

def vote_worker(rank, world, uri):
    dist.init_process_group('gloo',init_method=uri,rank=rank,world_size=world,timeout=timedelta(seconds=45))
    for fault in ('match','capacity','order','real_tokens','same_buckets_other_prefix','finishing'):
        s,p,d=create();s.tp_group=NS(world_size=world,cpu_group=dist.group.WORLD)
        if rank==world-1:
            if fault=='capacity':d.token_to_kv_pool_allocator.free=0
            if fault=='order':d.reqs.reverse()
            if fault=='real_tokens':p.extend_num_tokens-=1;p.extend_lens[0]-=1
            if fault=='same_buckets_other_prefix':p.prefix_lens[0]+=1
            if fault=='finishing':s._reqs_finishing_inflight=lambda _:[1]
        assert admit_mixed_dflash(s,p,d,16384)==(fault=='match')
        assert d.evictions==0 and d.token_to_kv_pool_allocator.reservations==0
    dist.destroy_process_group()

if __name__=='__main__':
    torch.set_num_threads(1);initialize()
    faults=denied_cases();accepted=acceptance(); admitted=historic()
    for world in (2,4,8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.start_processes(vote_worker,(world,'file://'+str(Path(tmp)/'gloo')),nprocs=world,join=True,start_method='fork')
    print(json.dumps(dict(positive_native_reservations=accepted,admission_declines=faults,archived_geometry=admitted,
       gloo_world_sizes=[2,4,8],gloo_patterns_per_world=6,
       existing_positional_prefix_preserved=True,cli_default_false=True,cli_bare_flag_true=True,native_runtime_namespace="spec",source_hashes=HASHES,
       coverage='Native CPU scheduler branch, DFLASH reservation, sampler reconstruction/merge and actual Gloo. GPU allocation ABI emulated; no H200 run.'),indent=2))
