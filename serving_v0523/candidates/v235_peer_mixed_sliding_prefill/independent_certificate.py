"""Exercise production certificate minting at the target entry and refusal before consuming a proposal."""
from contextlib import nullcontext
import hashlib
import logging
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn.modules import module as module_hooks

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
manifest=json.loads((HERE/"validation.json").read_bytes())
assert hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest()==manifest["patch_sha256"]
SRT="python/sglang/srt/"
PATHS=["speculative/dflash_mixed_forward.py","speculative/dflash_mixed_graph_admission.py",
       "speculative/dflash_worker_v2.py","model_executor/runner/prefill_cuda_graph_runner.py",
       "model_executor/forward_batch_info.py","layers/attention/direct_prompt_kv.py"]
hashes={}
for path in PATHS:
    source=subprocess.check_output(["git","-C",str(STAGE),"show",manifest["tree"]+":"+SRT+path])
    assert (STAGE/SRT/path).read_bytes()==source
    hashes[path]=hashlib.sha256(source).hexdigest()


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,file); obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj;spec.loader.exec_module(obj);return obj


bridge=module("v235_bridge",HERE.parent/"v159_mixed_verify_forward/bridge_cpu.py")
limits=module("v235_limits",STAGE/SRT/"arg_groups/prefill_buffer_ceiling.py")
bridge.pl.mixed_prefill_graph_fits=limits.mixed_prefill_graph_fits
bridge.env.update(module_hooks=module_hooks,dflash_sp_prefill_scope=lambda *args:nullcontext())
num_tails=0
scenario=""
active=None


class Finished(Exception): pass


class TargetReached(Exception): pass


def inspect_plan(runner,prompt,verify,mixed,requests):
    global active
    assert not getattr(prompt,"mixed_verify_is_causal",False)
    assert not getattr(mixed,"mixed_verify_is_causal",False)
    active=mixed
    if scenario=="masked":verify.spec_info.custom_mask=torch.ones(1,dtype=torch.bool)
    if scenario=="top2":verify.spec_info.topk=2
    if scenario=="ragged":verify.spec_info.ragged_verify_layout=object()
    if scenario=="mixed_spec":mixed.spec_info=verify.spec_info
    result=bridge.pl.plan_mixed_dflash_target(runner,prompt,verify,mixed,requests)
    if scenario in ("masked","top2","ragged","mixed_spec"):assert result is None
    return result


real_try=bridge.env["try_forward_proposed_mixed"]
bridge.env["plan_mixed_dflash_target"]=inspect_plan
cases=[]


def inspect(worker,prompt,tail,proposal,**kwargs):
    global active
    active=None
    q,t=prompt.extend_num_tokens,len(tail.reqs)
    bridge.pl.SETTINGS.chunked_prefill_size=q
    bridge.pl.SETTINGS.max_prefill_tokens=q+t*8
    pre=worker.target_worker.model_runner.prefill_cuda_graph_runner
    pre.mixed_dflash_token_capacity=pre.max_num_tokens
    worker._target_logits_processor=None
    worker._sp_prefill_context=None
    worker._prefill_kv_helper=lambda:None
    worker._wait_for_fused_hicache_load=lambda batch:None
    worker.target_worker.set_hicache_consumer=lambda index:None
    was=worker._pending_proposal_token
    called=[]
    def enter_target(**kw):
        fb=kw["forward_batch"]
        assert kw["batch"] is None and kw["is_verify"] and kw["return_kv_loc_plan"]
        assert fb is active and fb.mixed_verify_is_causal is True
        assert not getattr(prompt,"mixed_verify_is_causal",False) and not getattr(tail,"mixed_verify_is_causal",False)
        assert worker._pending_proposal_token is None and fb.spec_info is None
        assert torch.equal(fb.input_ids[q:],proposal.tokens.flatten()) and torch.equal(fb.positions[q:],proposal.positions.flatten())
        assert torch.equal(fb.out_cache_loc[q:],proposal.virtual_slots.flatten())
        called.append(fb)
        raise TargetReached
    worker.target_worker.forward_batch_generation=enter_target
    saved=bridge.env["agree_mixed_dflash_target"]
    if scenario=="remote_veto":
        bridge.env["agree_mixed_dflash_target"]=lambda *args,**kwargs:None
    try:
        try:result=real_try(worker,prompt,tail,proposal,**kwargs)
        except TargetReached:result=None
    finally:bridge.env["agree_mixed_dflash_target"]=saved
    positive=scenario=="causal"
    assert bool(called)==positive and result is None
    if not positive:
        assert worker._pending_proposal_token is was and not getattr(active,"mixed_verify_is_causal",False)
    cases.append(dict(tails=t,prompt=q,case=scenario,target_entries=len(called),certificate=positive,
                      live_tail_rows=t*8,proposal_consumed=worker._pending_proposal_token is None))
    raise Finished


bridge.env["try_forward_proposed_mixed"]=inspect
for num_tails in (1,2,3,4):
    for scenario in ("causal","masked","top2","ragged","mixed_spec","remote_veto"):
        try:bridge.run("breakable",num_tails,1,True)
        except Finished:pass
assert len(cases)==24
prefix=module("v235_prefix",HERE.parent/"v212_mixed_paired_draft_kv/worker_prefix_check.py")
logging.disable(logging.WARNING)
settled=[]
for remap in (False, True):
    for padded in (False, True):
        for mode in ("per_layer", "partial_tail", "nonunit"):
            lengths=list(range(1,9))
            env=prefix.build(mode,remap,lengths,8,517)
            ref=prefix.build(mode,remap,lengths,8,517)
            data,dead,_=prefix.settle(env,mode!="nonunit",padded)
            original,_,_=prefix.settle(ref,False,padded)
            assert torch.equal(data,original)
            settled.append(dict(mode=mode,physical_remap=remap,padded=padded,committed_lengths=lengths,
                rejected_tail_rows=8*8-sum(lengths),pool_words=data.numel(),poisoned_or_unused_slots=dead))
for path,digest in prefix.native.digests.items():
    src=subprocess.check_output(["git","-C",str(STAGE),"show",manifest["tree"]+":"+path])
    assert hashlib.sha256(src).hexdigest()==digest,path
    hashes[path.removeprefix(SRT)]=digest
print(json.dumps(dict(candidate=manifest["candidate"],patch_sha256=manifest["patch_sha256"],tree=manifest["tree"],
    source_sha256=hashes,cases=cases,live_accepted=4,declines=20,
    accepted_prefix_replays=settled,physical_h200=False,
    scope="Real proposal ownership, view formation, plan validation, certificate minting and target entry; separately exact native settlement and five physical draft KV layers. CPU boundary for CUDA writer."),indent=2))
