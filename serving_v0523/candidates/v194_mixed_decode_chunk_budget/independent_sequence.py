"""Exercise repeated native continuation decisions and arbitrary later admissions or vetoes."""
import ast
import bisect
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Sequence

ROOT=Path(__file__).resolve().parents[3]
stage=Path(sys.argv[1])
mod=stage/"python/sglang/srt/managers/scheduler_components/mixed_dflash_admission.py"
source=mod.read_text()
original=ROOT/"serving_v0523/sglang/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py"
log=next((ROOT/"experiments/modal_node").rglob("*v41__server.log")).read_text()
rng=random.Random(7432)


def method(text,name,env):
    n=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)
    n.decorator_list=[]
    exec(compile(ast.Module(body=[n],type_ignores=[]),name,"exec"),env)
    return env[name]


class Prefill(NS): pass
class Verify(NS): pass
pad=method(original.read_text(),"_pad_to_bucket",dict(bisect=bisect,Sequence=Sequence))
choose=method(source,"_continuation_budget",dict(PrefillCudaGraphRunner=Prefill,DecodeCudaGraphRunner=Verify,
                                                DisaggregationMode=NS(NULL="ordinary")))
prefill_grid=json.loads(re.search(r"Capture target prefill CUDA graph begin\. backend=breakable, num_tokens=(\[[^\]]+\])",log)[1])
verify_grid=json.loads(re.search(r"Capture target verify CUDA graph end\. bs=(\[[^\]]+\])",log)[1])
decided=mixed=late_no=rounds=0
for case in range(14000):
    cap=rng.choice((1024,4096,16384))
    grids=[x for x in prefill_grid if x<=cap]
    assert grids[-1]==cap
    total=rng.randint(1,8)*cap + rng.choice([1,cap-1,cap,rng.randint(1,cap),rng.choice(grids)])
    req=NS(kv=NS(req_pool_idx=5),sampling_params=NS(top_k=1),return_logprob=False, grammar=None,
           is_retracted=False,finished=lambda:False,multimodal_inputs=None,beam_group=None,
           full_untruncated_fill_ids=range(total),prefix_len=0)
    p=Prefill(_uses_eager_prefill_tail=lambda:True,enable_cp_bcg_capture=False,max_num_tokens=cap,
              max_bs=32,capture_num_tokens=grids,_pad_to_bucket=pad)
    v=Verify(max_bs=32,capture_bs=verify_grid,disable_padding=False,
             ragged_verify_mode=False,enable_pdmux=False,require_mlp_tp_gather=False,enable_two_batch_overlap=False,
             _pad_to_bucket=pad)
    worker=NS(_can_skip_verify_logits=lambda *a:True)
    s=NS(tp_worker=NS(model_runner=NS(prefill_cuda_graph_runner=p,decode_cuda_graph_runner=v)),draft_worker=worker,
         model_worker=worker,chunked_prefill_size=cap,dynamic_chunk_sizer=None,enable_overlap=True,
         is_mixed_chunk=False,require_mlp_sync=False,page_size=1,enable_unified_memory=False,enable_hisparse=False,
         disaggregation_mode="ordinary",dllm_config=None,enable_lora=False,spec_algorithm=NS(is_dflash=lambda:True),
         _reqs_finishing_inflight=lambda b:False,max_prefill_tokens=cap)
    a=NS(rem_chunk_tokens=cap,chunked_req_limit=None,kv_shard_granule=0,mamba_chunk_grid=1,
         rem_input_tokens=cap,memory_budget=NS(available_chunk_tokens=lambda c:c,fit_chunk=lambda **kw:kw["chunk_limit"]),
         _swa_new_tokens=lambda _:0)
    actual=prompt_cost=0
    last=(total-1)%cap+1
    target=(total-1)//cap*cap+pad(last,grids)
    while req.prefix_len<total:
        left=total-req.prefix_len
        count=rng.randint(0,32);w=rng.choice((2,4,8,12,16))
        v.captured_req_width=w;worker.block_size=w
        d=NS(spec_info=True,is_extend_in_batch=False,sampling_info=True,reqs=[req]*count)
        proposal=choose(s,d,req,a); rounds+=1
        if proposal:
            new,old,remaining,row,prefix,n,width=proposal
            assert (old,remaining,row,prefix,n,width)==(cap,left,5,req.prefix_len,count,w)
            assert new==cap-count*w and 0<new<cap
            assert (left-1)//cap == 1+(left-new-1)//cap
            assert pad((left-new-1)%cap+1,grids)==pad((left-1)%cap+1,grids)
            assert pad(new,grids)+pad(count,verify_grid)*w>=cap
            decided+=1
            # Native late eligibility may decline after the prompt has been shortened.
            if rng.getrandbits(1):
                prompt_cost+=pad(new+count*w,grids);mixed+=1
            else:
                prompt_cost+=pad(new,grids);late_no+=1
            req.prefix_len+=new
        else:
            chunk=min(left,cap);prompt_cost+=pad(chunk,grids);req.prefix_len+=chunk
        actual+=1
    assert actual==(total+cap-1)//cap and req.prefix_len==total
    assert prompt_cost<=target
print(json.dumps(dict(module_sha256=hashlib.sha256(mod.read_bytes()).hexdigest(),seed=7432,
                      sequences=case+1,continuation_decisions=rounds,eligible=decided,materialized_mixed=mixed,
                      post_reservation_vetoes=late_no,real_graph_buckets=prefill_grid,
                      scope="Actual CPU budget function across synthetic constant-capacity sequences and arbitrarily changing live cohorts; compare all covered prompt slots, number of forwards and padded prompt/mixed rows with separate decode outside this count."),indent=2))
