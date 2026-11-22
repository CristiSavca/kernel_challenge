"""Execute the submitted native host methods with CPU devices and a deterministic target body."""
import ast
import copy
import dataclasses
import enum
from functools import total_ordering
import hashlib
import json
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as N
import torch

SRC=Path(sys.argv[1])/"python/sglang/srt"
torch.set_num_threads(1)
env=dict(total_ordering=total_ordering,__name__=__name__,torch=torch,IntEnum=enum.IntEnum,auto=enum.auto,dataclass=dataclasses.dataclass,
         dataclasses=dataclasses,copy=copy,_is_cpu=True,_is_npu=False,PPProxyTensors=type("Proxy",(),{}),
         _autotune_run_lm_head=None,DpPaddingMode=N(SUM_LEN=1),LoRABatchLayout=N(DP_LOCAL=1),
         _trace_e2e_logits=lambda *a,**kw:None,get_forward=lambda:N(scoped=lambda **kw:nullcontext()),
         enable_num_token_non_padded=lambda:False,is_pin_memory_available=lambda dev:False,
         support_triton=lambda backend:False,ForwardBatchDeepSeekMHAMixin=object)
hashes={}
def build(file,name,methods=None):
    path=SRC/file
    raw=path.read_text()
    node=next(n for n in ast.parse(raw).body if getattr(n,"name",None)==name)
    if methods is not None:
        node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name in methods]
        if name not in ("ForwardBatch","LogitsMetadata","LogitsProcessorOutput"):
            node.bases=[]
            node.decorator_list=[]
            node.body=[n for n in node.body if isinstance(n,ast.FunctionDef)]
    hashes[file]=hashlib.sha256(path.read_bytes()).hexdigest()
    exec(compile("from __future__ import annotations\n"+ast.unparse(node),file,"exec"),env)
    return env[name]

fb="model_executor/forward_batch_info.py"; lp="layers/logits_processor.py"
pcg="model_executor/runner/prefill_cuda_graph_runner.py"
Mode=build(fb,"ForwardMode");Capture=build(fb,"CaptureHiddenMode")
build(fb,"compute_position_torch");build(fb,"compute_position")
FB=build(fb,"ForwardBatch",{"init_new","set_mixed_verify_rows","apply_deprecated_skip_attn_backend_init",
    "_maybe_init_non_generation_fields","_init_num_token_non_padded","init_mlp_sync_metadata",
    "contains_mm_inputs","contains_audio_inputs","contains_video_inputs","contains_image_inputs"})
LM=build(lp,"LogitsMetadata",{"from_forward_batch"})
LO=build(lp,"LogitsProcessorOutput",set())
LP=build(lp,"LogitsProcessor",{"forward","_get_pruned_states","_get_hidden_states_to_store",
             "_get_last_hidden_states_to_store","_get_logits","_gather_dp_attn_hidden_states",
             "_scatter_dp_attn_logits","_copy_logits_to_buffer","_can_gather_softcap_fused"})
Muse=build("models/muse_spark.py","MuseSparkTextForCausalLM",{"forward"})
Worker=build("managers/tp_worker.py","TpModelWorker",{"forward_batch_generation"})
build(pcg,"_slice_output_rows")
Graph=build(pcg,"PrefillCudaGraphRunner",{"_execute_body_capture","_trim_logits_output","execute",
            "_finalize_execute_output","_prefill_logits_buffer_rows","_next_token_logits_buffer","can_run_graph"})
PAR=N(dp_size=1,pp_size=1,attn_cp_size=1,attn_dcp_size=1)
def disabled():return False
class Flags:
    def __getattr__(self,name):return N(get=disabled)
env.update(envs=Flags(),get_parallel=lambda:PAR,get_exec=lambda:N(features=N(enable_encoder_swa_bounded_replay=False)),
           get_spec=lambda:N(speculative_num_draft_tokens=8),capture_pre_sample_logits=lambda *args:None,
           GenerationBatchResult=N,EmbeddingPoolerOutput=type("Pool",(),{}),
           _maybe_build_forward_token_modalities=lambda *a:None,
           maybe_publish_prefill_shared_read_done=lambda *a:None,
           layernorm_sp=N(runs_sp=lambda mode:mode==Mode.EXTEND,
                    sp_exit_gather=lambda t,num_tokens:t[:num_tokens],
                    maybe_exit_gather=lambda **kw:(kw["hidden_states"],kw["hidden_states_before_norm"])),
           pack_aux_hidden_states=lambda t:t if isinstance(t,torch.Tensor) else torch.cat(t,dim=-1))
mod=types.ModuleType("sglang.srt.model_executor.model_runner")
mod.ModelRunner=type("ModelRunner",(),{})
sys.modules[mod.__name__]=mod

class Batch(N):
    # Every optional member read by the real constructor is explicitly initialized.
    pass

def schedule(lengths,offset=0):
    n=len(lengths);total=sum(lengths)
    keys="orig_seq_lens out_cache_loc_dsv4 engram_history engram_staged_rows mamba_track_indices mamba_track_seqlens mamba_track_mask mamba_track_seqlens_cpu mamba_prefill_track_mask_cpu mamba_cow_src_indices mamba_cow_dst_indices mamba_clear_indices encoder_lens encoder_out_cache_loc input_embeds replace_embeds replace_positions tbo_split_seq_index top_logprobs_nums token_ids_logprobs multimodal_inputs encoder_cached encoder_lens_cpu sampling_info spec_info extend_input_logprob_token_ids dllm_config beam_tail global_num_tokens global_num_tokens_for_logprob decoding_reqs"
    b=Batch(**dict.fromkeys(keys.split()))
    b.reqs=[N(beam_group=None,extend_end=1,full_untruncated_fill_ids=[0],lora_id=None,rid=str(i),token_type_ids=None) for i in range(n)]
    b.extend_lens=lengths;b.prefix_lens=[3+i for i in range(n)];b.extend_num_tokens=total
    b.extend_logprob_start_lens=[0]*n;b.seq_lens=torch.tensor(lengths)+torch.tensor(b.prefix_lens)
    b.seq_lens_cpu=b.seq_lens.clone();b.seq_lens_sum=int(b.seq_lens.sum())
    b.req_pool_indices=torch.arange(n);b.out_cache_loc=torch.arange(total-1,-1,-1)+23
    b.input_ids=(torch.arange(total)+offset)%33
    b.is_extend_in_batch=False;b.is_prefill_only=False;b.can_run_decode_cuda_graph=False
    b.can_run_dp_draft_cuda_graph=False;b.can_run_dp_prefill_cuda_graph=False
    b.dp_prefill_cuda_graph_max_prefix_len=None;b.forward_mode=b.global_forward_mode=Mode.EXTEND
    b.spec_algorithm=N(is_dflash=lambda:True);b.return_logprob=False;b.dp_spec_prefill_coordination_applied=False
    b.hicache_consumer_index=-1
    return b

class Plan:
    def __init__(self,fb):self.slots=fb.out_cache_loc.clone()
    def bind(self,fb,reader,*,cols):
        fb.kv_loc_plan=self
        fb.out_cache_loc=self.slots

H,V=8,43
torch.manual_seed(902)
WEIGHT=torch.randn(V,H)
def body_tokens(ids):
    h=((ids.float()[:,None]*torch.arange(1,H+1)[None,:])%41).to(torch.bfloat16)
    return h,[h,h*2,h*3,h*4,h*5]

def test_worker(backend,shared_cap):
    proc=LP();proc.expected_aux_hidden_states=None;proc.keep_last_hidden_states=False
    proc.do_tensor_parallel_all_gather=False;proc.do_tensor_parallel_all_gather_dp_attn=False
    proc.logit_scale=0.75;proc.final_logit_softcapping=25;proc.vocab_size=V
    proc._compute_lm_head=lambda hidden,head,bias,out:torch.mm(hidden.float(),head.T,out=out)
    body=N(forward=lambda ids,pos,fb,input_embeds=None:body_tokens(ids))
    model=Muse();model.capture_aux_hidden_states=True;model.model=lambda *a,**kw:body.forward(*a,**kw)
    model.logits_processor=proc.forward;model.lm_head=WEIGHT
    model_runner=N(device="cpu",is_draft_worker=False,spec_algorithm=N(is_speculative=lambda:True),model_config=N(model_is_mrope=False,vocab_size=V),
            lora_manager=None,ngram_embedding_manager=N(enabled=False),model=model,pp_group=N(is_last_rank=True,is_first_rank=True),
            prefill_attention_backend_str="torch_native",kv_index_translator=N(own_plan=Plan),
            graph_shared_output=N(max_rows=shared_cap,get_logits_buffer=lambda vocab,rows:torch.full((rows,vocab),float("nan"))))
    w=Worker();w.pp_group=model_runner.pp_group;w.model_runner=model_runner;w.is_dllm=lambda:False
    w.set_hicache_consumer=lambda _:None
    g=Graph();g._is_full_backend=backend=="FULL";g.layer_model=body
    g._qwen_bcg_hc_sidechannel=False;g.is_dsv41_model=False;g._use_draft_input_embeds=False
    g.model_runner=model_runner;g.buffer_registry=N(has_slot=lambda name:False)
    g._prefill_forward_context=lambda *a,**kw:nullcontext();g.model_context=nullcontext
    g.backend=N(replay_session=nullcontext,replay=lambda key,fb,**kw:body_tokens(fb.input_ids))
    g._restore_qwen_bcg_hc_output=lambda x:x
    g._uses_eager_prefill_tail=lambda:backend in ("FULL","BREAKABLE")
    g._validate_capture_hidden_mode=lambda _:None;g._shape_key=lambda *a:N(variant_label=None)
    g.device_module=None;g._wait_for_layer_loads=lambda:None;g.enable_cp_bcg_capture=False
    last=[]
    def load(fb,**kw):
        n=fb.input_ids.numel();cap=(n+15)//16*16
        g.raw_num_tokens=n;g.raw_bs=fb.batch_size
        static=dataclasses.replace(fb,input_ids=torch.cat((fb.input_ids,torch.full((cap-n,),32))),
                positions=torch.arange(cap),next_token_logits_buffer=g._next_token_logits_buffer(g._prefill_logits_buffer_rows(fb)))
        if g._is_full_backend:fb.next_token_logits_buffer=static.next_token_logits_buffer
        return static,None
    g.load_batch=load
    def forward(fb,pp_proxy_tensors):
        last.append(fb)
        output=g.execute(fb) if backend!="eager" else model.forward(fb.input_ids,fb.positions,fb)
        return N(logits_output=output,can_run_graph=backend!="eager",expert_distribution_metrics=None,
                 routed_experts_output=None,indexer_topk_output=None)
    model_runner.forward=forward
    return w,last,g

def suite():
    successes=ordinary=0
    for backend in ("eager","FULL","BREAKABLE"):
        for cap in (0,16,4096):
            w,entries,g=test_worker(backend,cap)
            for i,(prompts,tails) in enumerate((([1],1),([3,5],2),([8,1,7],1),([100],4),([2,4,2],3),([5],1))):
                lens=prompts+[8]*tails;b=schedule(lens,offset=3*i)
                expected_idx=(torch.tensor(prompts).cumsum(0)-1).tolist()+list(range(sum(prompts),sum(lens)))
                for count in (len(prompts),None):
                    res=w.forward_batch_generation(b,is_verify=True,capture_hidden_mode=Capture.FULL,
                              return_kv_loc_plan=True,mixed_verify_prompt_count=count)
                    fb=entries[-1]
                    ix=expected_idx if count else (torch.tensor(lens).cumsum(0)-1).tolist()
                    h,t=body_tokens(b.input_ids);v=25*torch.tanh((h[ix].float()@WEIGHT.T)*.75/25)
                    assert torch.equal(res.logits_output.next_token_logits,v)
                    assert torch.equal(res.logits_output.hidden_states,torch.cat(t,dim=1))
                    assert res.kv_loc_plan is fb.kv_loc_plan and torch.equal(res.kv_loc_plan.slots,b.out_cache_loc)
                    assert not hasattr(res,"next_token_ids") and fb.forward_mode==Mode.EXTEND
                    assert torch.equal(fb.positions,torch.cat([torch.arange(p,p+q) for p,q in zip(b.prefix_lens,lens)]))
                    if count:successes+=1
                    else:ordinary+=1
    w,entries,g=test_worker("eager",0)
    failures=0
    for mutate in (
        lambda b,k:setattr(b,"extend_num_tokens",1),lambda b,k:b.prefix_lens.__setitem__(1,-1),
        lambda b,k:setattr(b,"seq_lens_cpu",b.seq_lens_cpu+1),lambda b,k:setattr(b,"seq_lens_cpu",None),
        lambda b,k:setattr(b,"input_embeds",torch.zeros(2,2)),lambda b,k:setattr(b,"replace_embeds",torch.zeros(1)),
        lambda b,k:setattr(b,"req_pool_indices",b.req_pool_indices[:-1]),
        lambda b,k:k.update(is_verify=False),lambda b,k:k.update(return_kv_loc_plan=False),
        lambda b,k:k.update(capture_hidden_mode=Capture.LAST),lambda b,k:k.update(mixed_verify_prompt_count=0),
        lambda b,k:k.update(mixed_verify_prompt_count=True),lambda b,k:k.update(mixed_verify_prompt_count=3),
        lambda b,k:setattr(b,"return_logprob",True),lambda b,k:setattr(b,"spec_info",N()),
        lambda b,k:setattr(b,"forward_mode",Mode.MIXED),lambda b,k:setattr(b,"beam_tail",N()),
        lambda b,k:setattr(b.reqs[0],"beam_group",N()),lambda b,k:setattr(b,"out_cache_loc",b.out_cache_loc[:-1]),
        lambda b,k:setattr(b,"tbo_split_seq_index",1),lambda b,k:setattr(b,"spec_algorithm",N(is_dflash=lambda:False)),
        lambda b,k:setattr(b,"input_ids",b.input_ids[:-1]),lambda b,k:b.extend_lens.__setitem__(1,7),
    ):
        b=schedule([4,8,8]);k=dict(is_verify=True,return_kv_loc_plan=True,capture_hidden_mode=Capture.FULL,mixed_verify_prompt_count=1)
        mutate(b,k)
        try:w.forward_batch_generation(b,**k)
        except ValueError:failures+=1
        else:raise AssertionError(mutate)
        assert not entries
    post_guard=0
    for k in ("logit_rows","vocab","tap_rows","no_taps"):
        batch=schedule([4,8])
        def corrupted(fb,**kw):
            out=LO(next_token_logits=torch.empty(8 if k=="logit_rows" else 9,V-1 if k=="vocab" else V),
                   hidden_states=None if k=="no_taps" else torch.empty(11 if k=="tap_rows" else 12,5*H))
            return N(logits_output=out,can_run_graph=False,expert_distribution_metrics=None,
                     routed_experts_output=None,indexer_topk_output=None)
        original=w.model_runner.forward
        try:
            w.model_runner.forward=corrupted
            w.forward_batch_generation(batch,is_verify=True,return_kv_loc_plan=True,
                    capture_hidden_mode=Capture.FULL,mixed_verify_prompt_count=1)
        except RuntimeError as ex:
            assert "incomplete logits" in str(ex)
            post_guard+=1
        finally:w.model_runner.forward=original
    assert post_guard==4
    raw=schedule([4,8]);r=w.forward_batch_generation(raw,is_verify=True,return_kv_loc_plan=True,
                    capture_hidden_mode=Capture.FULL,mixed_verify_prompt_count=1)
    assert not g.can_run_graph(entries[-1])
    return dict(mixed_worker_forward_cases=successes,ordinary_forward_cases=ordinary,prelaunch_declines=failures,incomplete_output_rejections=post_guard,
                tc_piecewise_falls_back_to_eager=True,kernels="synthetic CPU body, dense full-vocabulary head and zero GPU allocations")
if __name__=="__main__":
    with torch.inference_mode():results=suite()
    print(json.dumps(dict(torch=torch.__version__,sources=hashes,**results),indent=2))
