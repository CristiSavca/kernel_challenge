"""Run the native chunk adder and mixed scheduler transition using real PyTorch/Gloo votes."""
import ast
from collections import Counter
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location("parent_admission",next(HERE.parent.glob("v160_*/scheduler_admission_cpu.py")))
f=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=f
spec.loader.exec_module(f)
f.initialize()
for name in ("_continuation_budget","reserve_mixed_continuation"):
    f.grab(f.A,name)
sys.modules["sglang.srt.managers.scheduler_components.mixed_dflash_admission"].reserve_mixed_continuation=f.reserve_mixed_continuation
policy="managers/schedule_policy.py"
native_add=f.grab(policy,"add_chunked_req","PrefillAdder")
native_charge=f.grab(policy,"_update_prefill_budget","PrefillAdder")
native_floor=f.grab(policy,"floor_chunked_tokens","PrefillAdder")
f.CLIP_MAX_NEW_TOKENS=4096
# Execute the entire native scheduler block between constructing the adder and visiting the queue.
scheduler=next(n for n in ast.parse(f.source("managers/scheduler.py")).body if getattr(n,"name",None)=="Scheduler")
forward=next(n for n in scheduler.body if getattr(n,"name",None)=="_get_new_batch_prefill_raw")
part=next(n for n in forward.body if isinstance(n,ast.If) and "reserve_mixed_continuation" in ast.unparse(n))
exec("def select(self,running_batch,adder):\n"+'\n'.join('    '+line for line in ast.unparse(part).splitlines()),vars(f))


class Request(N):
    @property
    def extend_len(self):
        return self.extend_end-self.prefix_len


class Adder(N):
    add_chunked_req=native_add
    _update_prefill_budget=native_charge
    floor_chunked_tokens=native_floor
    _mamba_gap_budget_for_req=staticmethod(lambda req:0)
    _request_slot_reserve_for_req=staticmethod(lambda *a,**kw:0)
    _kv_shard_reserve_scratch=staticmethod(lambda **kw:True)
    _swa_new_tokens=staticmethod(lambda req: req.sampling_params.max_new_tokens)
    ceil_paged_tokens=staticmethod(int)


def setup(tails,remaining,cap=128,prefix=4):
    s,p,d=f.create(p=1,t=tails,q=cap)
    graph=s.tp_worker.model_runner.prefill_cuda_graph_runner
    graph.capture_num_tokens=[1,2,4,8,16,32,64,128]
    graph.max_num_tokens=cap
    s.chunked_prefill_size=s.max_prefill_tokens=cap
    s.enable_lora=False
    s.dynamic_chunk_sizer=None
    s.waiting_queue=[]
    s.policy=N(shortest_prefill_chunk_limit=lambda *a:None)
    s.tp_group=N(world_size=1)
    req=Request(**vars(p.reqs[0]))
    p.reqs=[req]
    req.prefix_len=prefix
    req.full_untruncated_fill_ids=range(prefix+remaining)
    req.retracted_stain=req.return_logprob=False
    req.grammar=None
    req.sampling_params.max_new_tokens=64
    req.init_next_round_input=lambda:None
    s.chunked_req=req
    calls=[]
    memory=N(available_chunk_tokens=lambda cap:cap,fit_chunk=lambda **kw:kw["chunk_limit"],
             reserve=lambda *a,**kw:calls.append((a,kw)))
    a=Adder(memory_budget=memory,rem_chunk_tokens=cap,rem_input_tokens=cap,chunked_req_limit=None,
            kv_shard_granule=0,mamba_chunk_grid=1,dllm_config=None,prefill_delayer_single_pass=None,
            exact_chunk_fill=True,page_size=1,per_req_token_overhead=1,log_hit_tokens=0,log_input_tokens=0,
            log_replay_tokens=0,can_run_list=[],remaining_charges=calls)
    return s,p,d,req,a


def local():
    result=Counter()
    for count in (1,2,3,4):
        for remaining in (1,64,127,128,129,144,155,168,191,192,207,224,255,256,257,311,401,499):
            for enabled in (True,False):
                s,p,d,req,a=setup(count,remaining)
                f.ENABLED=enabled
                f.select(s,d,a)
                m=min(remaining,128)
                row=req.extend_len
                assert a.can_run_list==[req] and len(a.remaining_charges)==1
                assert a.remaining_charges[0][0][0]==row and a.rem_input_tokens==128-row
                assert (s.chunked_req is req)==(row<remaining)
                result["native_chunk_calls"]+=1
                if row<m:
                    assert enabled and row+count*8==128 and a.rem_chunk_tokens==0
                    old_chunks=[min(128,remaining-i) for i in range(0,remaining,128)]
                    new_chunks=[row]+[min(128,remaining-row-i) for i in range(0,remaining-row,128)]
                    pad=s.tp_worker.model_runner.prefill_cuda_graph_runner._pad_to_bucket
                    grid=s.tp_worker.model_runner.prefill_cuda_graph_runner.capture_num_tokens
                    assert [pad(i,grid) for i in old_chunks]==[128]+[pad(i,grid) for i in new_chunks[1:]]
                    assert sum(new_chunks)==remaining
                    p.extend_lens=[row];p.prefix_lens=[req.prefix_len];p.extend_num_tokens=row
                    outer,_=f.schedule(s,p,d,128)
                    assert outer is not p and outer.mixed_dflash_roles==(p,d) and outer.extend_num_tokens==128
                    result["admitted_real_mixed_batches"]+=1
                    result["native_decoder_allocations"]+=d.token_to_kv_pool_allocator.reservations
                else:
                    assert row==m
    f.ENABLED=True
    s,p,d,r,a=setup(2,290)
    assert f._continuation_budget(s,d,r,a)[0]==112
    graph=s.tp_worker.model_runner.prefill_cuda_graph_runner
    verify=s.tp_worker.model_runner.decode_cuda_graph_runner
    for owner,name,value in [
        (s,"dynamic_chunk_sizer",object()),(a,"chunked_req_limit",12),(s,"enable_unified_memory",True),
        (d,"spec_info",None),(d,"sampling_info",None),(r,"grammar",object()),(r,"return_logprob",True),
        (r.kv,"req_pool_idx",None),(r.sampling_params,"top_k",5),(s,"enable_lora",True),
        (a,"mamba_chunk_grid",16),(a,"kv_shard_granule",4),(graph,"capture_num_tokens",[1,64]),
        (verify,"captured_req_width",4),(verify,"enable_two_batch_overlap",True),(a,"rem_input_tokens",64),
        (a.memory_budget,"available_chunk_tokens",lambda cap:cap-1),(a.memory_budget,"fit_chunk",lambda **kw:None),
        (s,"_reqs_finishing_inflight",lambda _:True),(graph,"max_bs",1)]:
        with patch.object(owner,name,value): f.reserve_mixed_continuation(s,d,r,a)
        assert a.rem_chunk_tokens==128
        result["unsupported_declines"]+=1
    return result


def worker(rank,world,path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",rank=rank,world_size=world,init_method="file://"+path,timeout=timedelta(seconds=45))
    counts=Counter()
    for fault in ("none","different_remaining","one_low_budget","different_identity","none"):
        s,p,d,req,a=setup(2,290)
        s.tp_group=N(world_size=world,cpu_group=dist.group.WORLD)
        if rank==world-1:
            if fault=="different_remaining": req.full_untruncated_fill_ids=range(295)
            if fault=="one_low_budget":
                # Recover between the optional vote and the ordinary budget check.
                values=iter((127,128))
                a.memory_budget.available_chunk_tokens=lambda cap:next(values)
            if fault=="different_identity": req.kv.req_pool_idx+=1
        with torch.inference_mode(): f.select(s,d,a)
        assert req.extend_len==(112 if fault=="none" else 128), (rank,world,fault,req.extend_len)
        counts["consensus" if fault=="none" else "rank_veto"]+=1
    if rank==0: Path(path+".json").write_text(json.dumps({k:v*world for k,v in counts.items()}))
    dist.destroy_process_group()


if __name__=="__main__":
    t=local(); out={}
    for size in (2,4,8):
        with tempfile.TemporaryDirectory() as temp:
            path=str(Path(temp)/"gloo")
            mp.spawn(worker,(size,path),nprocs=size,join=True)
            out[size]=json.loads(Path(path+".json").read_bytes())
    print(json.dumps(dict(single_rank=t,torch=torch.__version__,multi_rank=out,
        physical_h200=False,scope="Native scheduler continuation selection, native PrefillAdder.add_chunked_req and accounting, native V160 mixed admission and decode reservation. Fixture supplies the memory backend and request metadata; Gloo consensus is real."),indent=2))
