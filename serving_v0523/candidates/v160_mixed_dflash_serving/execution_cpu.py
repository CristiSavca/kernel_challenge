"""Exercise the overlap scheduler with real role snapshots, publication and host result dispatch."""
import ast
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import timezone
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import torch

CAND = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("hostcheck",CAND/"v156_mixed_verify_settlement/host_check.py")
h = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = h
spec.loader.exec_module(h)
src = Path(sys.argv[1])/"python/sglang/srt"
hashes = {}
def method(rel, name, owner=None, *, fields_only=False):
    raw=(src/rel).read_text()
    node=next(x for x in ast.parse(raw).body if getattr(x,"name",None)==(owner or name))
    if owner:
        node=next(x for x in node.body if getattr(x,"name",None)==name)
        if name=="run_batch": node.decorator_list=[]
    if fields_only:
        node.bases=[]
        node.body=[f for f in node.body if isinstance(f,ast.AnnAssign) or getattr(f,"name",None) in ("copy","batch_size")]
    hashes["/".join(x for x in (rel,owner,name) if x)] = hashlib.sha256(ast.unparse(node).encode()).hexdigest()
    ns={}
    exec(compile("from __future__ import annotations\n"+ast.unparse(node),rel,"exec"),globals(),ns)
    return ns[name]

ForwardMode=h.ForwardMode
CaptureHiddenMode=NS(NULL=0)
dataclass=dataclasses.dataclass
_debug=False
_DEBUG_ASSERT=False
_is_hip=False
n=h.noop
GenerationBatchResult=h.GenerationBatchResult
MixedDFlashResult=method("managers/utils.py","MixedDFlashResult")
h.MixedDFlashResult=MixedDFlashResult
ScheduleBatch=method("managers/schedule_batch.py","ScheduleBatch",fields_only=True)
if "mixed_dflash_roles" not in {f.name for f in dataclasses.fields(ScheduleBatch)}:
    raise RuntimeError("assemble the admission delta before running this test")
ov="managers/overlap_utils.py"
resolve_forward_inputs=method(ov,"resolve_forward_inputs")
RelayPayload=method(ov,"RelayPayload")
DisaggregationMode=NS(PREFILL="prefill")
r="managers/scheduler.py"


class Algo(h.Algo):
    is_some=lambda self:True
    is_ngram=lambda self:False
    is_dspark=lambda self:False
    supports_grammar_overlap=lambda self:True


class Ready:
    def __init__(self): self.records=0
    def record(self): self.records+=1
    def wait(self): pass


class Future:
    resolve_seq_lens_cpu=method(ov,"resolve_seq_lens_cpu","FutureMap")
    _resolve_spec_extras=method(ov,"_resolve_spec_extras","FutureMap")
    publish=method(ov,"publish","FutureMap")
    stash=method(ov,"stash","FutureMap")
    _maybe_init_forward_bufs=staticmethod(n)
    _maybe_init_dsa_topk_indices_buf=staticmethod(n)
    def __init__(self,mirror):
        self.spec_algo=Algo()
        self.needs_cpu_seq_lens=mirror
        self.new_seq_lens_buf=torch.arange(64)+40
        self.output_tokens_buf=torch.arange(64)+100
        self.need_topk=self.need_hidden_states=self.needs_confidence_relay=False
        self.draft_probs_buf=self.dsa_topk_indices_buf=self._relay=None
        self.fwd_prepare_d2h_stream=None
        self.publish_ready=Ready()


class Info:
    def copy_for_forward(self):
        return Info()


class Scheduler(h.Scheduler):
    _run_mixed_dflash_roles=method(r,"_run_mixed_dflash_roles","Scheduler")
    run_batch=method(r,"run_batch","Scheduler")
    _forward_isolation=method(r,"_forward_isolation","Scheduler")
    record_batch_in_overlap=method(r,"record_batch_in_overlap","Scheduler")
    _relay_forward_payload=method(r,"_relay_forward_payload","Scheduler")
    _copy_result_to_cpu=method(r,"_copy_result_to_cpu","Scheduler")

    def __init__(self,worker,mirror):
        super().__init__(h.Processor())
        self.is_generation=self.enable_overlap=True
        self.enable_unified_memory=self.enable_unified_cache_external_linker=False
        self._confidence_budget_prepare=self.scripted_scheduler_hook=self.forward_sleep_time=None
        self.enable_async_d2h_copy=False
        self.forward_ct=self.processed_tokens_counter=self.batch_record_ct=0
        self.batch_record_buf=[[],[]]
        self.disaggregation_mode="none"
        self._sched_idled=True
        self.forward_stream_ctx=self.copy_stream_ctx=nullcontext()
        self.forward_stream=self.copy_stream=self.schedule_stream=h.Sink()
        self.profiler_manager=self.beam_coordinator=h.Sink()
        self.spec_algorithm=Algo()
        self.future_map=Future(mirror)
        self.model_worker=worker
        self.device_module=NS(Event=h.Event)


def relay(*,bonus_tokens,new_seq_lens):
    return NS(bonus_tokens=bonus_tokens,new_seq_lens=new_seq_lens,topk_p=None,topk_index=None,hidden_states=None,
              dsa_topk_indices=None,future_dsa_topk_indices_available=False,future_indices=None)


def module(path, **bindings):
    m=ModuleType(path);m.__dict__.update(bindings);sys.modules[path]=m


class Worker:
    def __init__(self,allow,pg,dg,width):
        self.allow=allow; self.pg=pg; self.dg=dg; self.width=width
        self.log=[]; self.current=None
    def propose_decode_block(self,d):
        assert d.forward_mode==ForwardMode.DECODE and d.spec_info.bonus_tokens.numel()==len(d.reqs)
        self.log.append("draft")
        self.current=NS(prefix=d.seq_lens.clone(),anchor=d.spec_info.bonus_tokens.clone())
        return self.current
    def decode_output(self,d,proposal):
        assert self.current is proposal
        assert torch.equal(proposal.anchor,d.spec_info.bonus_tokens)
        ids=proposal.anchor[:,None]+torch.arange(self.width)
        lens=1+(torch.arange(len(d.reqs))+self.width//2)%self.width
        next_input=relay(bonus_tokens=ids[torch.arange(len(d.reqs)),lens-1],new_seq_lens=proposal.prefix+lens)
        self.current=None
        return GenerationBatchResult(next_token_ids=ids.flatten(),accept_lens=lens,logits_output=h.LogitsProcessorOutput(None),
            next_draft_input=next_input,new_seq_lens=next_input.new_seq_lens,can_run_cuda_graph=self.dg,
            speculative_num_draft_tokens=self.width)
    def prompt_output(self,p):
        assert p.input_ids.tolist()==[51+i for i in range(len(p.reqs))]
        ids=p.input_ids+10
        ri=relay(bonus_tokens=ids,new_seq_lens=p.seq_lens)
        return GenerationBatchResult(next_token_ids=ids,logits_output=h.LogitsProcessorOutput(None),can_run_cuda_graph=self.pg,
                                     next_draft_input=ri,new_seq_lens=p.seq_lens)
    def verify_proposed_decode_block(self,d,proposal,**kw):
        self.log.append("separate verification")
        result=self.decode_output(d,proposal)
        self.fallback_logits=result.logits_output
        self.fallback_logits.next_token_logits=torch.ones(23)
        d.forward_mode=ForwardMode.TARGET_VERIFY
        d.spec_info=NS(overwritten=True)
        d.out_cache_loc=torch.tensor([777])
        return result
    def forward_batch_generation(self,p,**kw):
        self.log.append("separate prompt" if p.forward_mode==ForwardMode.EXTEND else "ordinary decode")
        if p.forward_mode==ForwardMode.DECODE:
            out=self.decode_output(p,self.propose_decode_block(p))
            kw["on_publish"](out.new_seq_lens)
            return out
        assert self.fallback_logits.next_token_logits is None
        return self.prompt_output(p)


def try_mixed(worker,p,d,proposal,*,on_publish,prefix_readback):
    assert prefix_readback is d.seq_lens
    worker.log.append("voted")
    if not worker.allow:
        return None
    a=worker.prompt_output(p);b=worker.decode_output(d,proposal)
    join=relay(bonus_tokens=torch.cat((a.next_token_ids,b.next_draft_input.bonus_tokens)),
               new_seq_lens=torch.cat((a.new_seq_lens,b.new_seq_lens)))
    on_publish(join.new_seq_lens)
    worker.log.append("combined target")
    return MixedDFlashResult.from_role_results(p,d,a,b,join)


module("sglang.srt.speculative.dflash_mixed_forward",try_forward_proposed_mixed=try_mixed,
       start_mixed_prefix_readback=lambda d:d.seq_lens)
module("sglang.srt.speculative.draft_worker_common",make_draft_input_v2=relay)


def setup(seed, allow, mirror, pg, dg):
    p=1+seed%3;t=1+(seed//3)%5;width=(2,4,8)[seed%3]
    worker=Worker(allow,pg,dg,width);s=Scheduler(worker,mirror)
    ix=torch.randperm(36,generator=torch.Generator().manual_seed(seed))[:p+t]
    prompt_reqs=[h.request(i,"prompt","chunk" if i==1 else "normal",1) for i in range(p)]
    tail_reqs=[h.request(i,"decode","cap" if i%2 else "normal",width) for i in range(t)]
    def role(reqs,indices,mode):
        return ScheduleBatch(reqs=reqs,model_config=NS(is_encoder_decoder=False),forward_mode=mode,
               spec_algorithm=Algo(),sampling_info=Info(),req_pool_indices=indices,req_pool_indices_cpu=indices,
               seq_lens=torch.full((len(reqs),),13,dtype=torch.int64),device="cpu",enable_overlap=True)
    prompt=role(prompt_reqs,ix[:p],ForwardMode.EXTEND)
    prompt.prefill_input_ids_cpu=torch.arange(p)+51
    prompt.prefix_lens=[12]*p;prompt.extend_lens=[1]*p;prompt.extend_num_tokens=p
    decode=role(tail_reqs,ix[p:],ForwardMode.DECODE)
    decode.spec_info=relay(bonus_tokens=None,new_seq_lens=decode.seq_lens)
    decode.spec_info.future_indices=ix[p:]
    outer=role(prompt_reqs+tail_reqs,ix,ForwardMode.MIXED)
    outer.mixed_dflash_roles=(prompt,decode)
    outer.prefill_input_ids_cpu=prompt.prefill_input_ids_cpu
    outer.mix_running_indices=outer.mix_running_indices_cpu=ix[p:]
    outer.decoding_reqs=tail_reqs[:]
    outer.extend_num_tokens=p+t*width
    original_lens=s.future_map.new_seq_lens_buf.clone()
    original_bonus=s.future_map.output_tokens_buf.clone()
    result=s.run_batch(outer)
    assert result.prompt_batch.forward_iter==result.decode_batch.forward_iter==outer.forward_iter==1
    assert result.prompt_batch.launch_ts==result.decode_batch.launch_ts==outer.launch_ts
    assert outer.mixed_dflash_roles is None and outer.prefill_input_ids_cpu is None
    assert outer.mix_running_indices is None and outer.mix_running_indices_cpu is None and outer.input_ids is None
    assert decode.forward_mode==ForwardMode.DECODE and not hasattr(decode.spec_info,"overwritten")
    assert result.copy_done.readies==1 and result.can_run_cuda_graph==(pg and dg)
    assert worker.log==(["draft","voted","combined target"] if allow else ["draft","voted","separate verification","separate prompt"])
    assert s.batch_record_ct==1 and len(s.batch_record_buf[s.batch_record_ct])>=4
    expected_lens=torch.cat((prompt.seq_lens,original_lens[ix[p:]]+result.decode_result.accept_lens))
    expected_ids=torch.cat((prompt.input_ids+10,original_bonus[ix[p:]]+result.decode_result.accept_lens-1))
    assert torch.equal(result.new_seq_lens,expected_lens)
    assert torch.equal(result.next_draft_input.bonus_tokens,expected_ids)
    original_lens[ix]=expected_lens;original_bonus[ix]=expected_ids
    assert torch.equal(s.future_map.new_seq_lens_buf,original_lens) and torch.equal(s.future_map.output_tokens_buf,original_bonus)
    assert s.future_map.publish_ready.records==1
    # Drain a queued host snapshot after the real role inputs have been rebound.
    s.process_batch_result(outer.copy(),result)
    assert all(r.kv.kv_committed_len==30+i+int(result.decode_result.accept_lens[i]) for i,r in enumerate(tail_reqs))
    # Reusing the same outer object for the next ordinary decode tick must
    # gather the just-published values and must not replay the previous prompt.
    outer.forward_mode=ForwardMode.DECODE
    ordinary=s.run_batch(outer)
    assert worker.log[-2:]==["ordinary decode","draft"] and s.future_map.publish_ready.records==2
    assert ordinary.has_sampled_token_ids
    return int(ix.numel())


if __name__=="__main__":
    rows=0;total=0
    for seed in range(18):
        for mirror in (True,False):
            for pg,dg in ((True,True),(True,False),(False,True),(False,False)):
                for allow in ((False,True) if pg==dg else (False,)):
                    rows+=setup(seed,allow,mirror,pg,dg);total+=1
    print(json.dumps(dict(rounds=total,role_requests=rows,following_ordinary_decode_ticks=total,
        prior_length_paths=["CPU mirror","GPU-only"],hardware=False,source_hashes=hashes,
        boundaries="Native Scheduler.run_batch, forward isolation, host processors and FutureMap. CPU tensors and instrumented model and CUDA streams."),indent=2))

