"""Run the real proposal-to-mixed ForwardBatch bridge with linear draft tokens and physical slots."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

bridge=load('v232_bridge_fixture',HERE.parent/'v159_mixed_verify_forward/bridge_cpu.py')
helpers=load('v232_limits',Path(sys.argv[1])/'python/sglang/srt/arg_groups/prefill_buffer_ceiling.py')
bridge.pl.mixed_prefill_graph_fits=helpers.mixed_prefill_graph_fits
class Boundary(Exception): pass
current=()
rows=[]
def inspect(worker,prompt,tail,proposal,**kwargs):
    q=prompt.extend_num_tokens
    t=len(tail.reqs)
    # The fixture holds small real tensors. Scaling the policy limit to the live prompt
    # makes the same production 'q <= chunk < q + width*t' branch activate.
    policy=bridge.pl.SETTINGS
    policy.chunked_prefill_size=q
    if current[-1]!='cost':policy.max_prefill_tokens=q+t*8
    pre=worker.target_worker.model_runner.prefill_cuda_graph_runner
    pre.mixed_dflash_token_capacity=pre.max_num_tokens
    worker._target_logits_processor=None
    value=bridge.env['_prepare_views'](worker,prompt,tail,proposal)
    assert (value is None)==bool(current[-1]),(current,value)
    if value is not None:
        plan, mixed_signature, prompt_fb, mixed_fb=value
        assert mixed_fb.forward_mode==bridge.v155.Mode.EXTEND
        assert mixed_fb.spec_info is None
        assert torch.equal(mixed_fb.input_ids[:q],prompt.input_ids)
        assert torch.equal(mixed_fb.input_ids[q:],proposal.tokens.flatten())
        assert torch.equal(mixed_fb.positions[q:],proposal.positions.flatten())
        assert torch.equal(mixed_fb.out_cache_loc,torch.cat((prompt.out_cache_loc,proposal.virtual_slots.flatten())))
        assert mixed_fb.extend_seq_lens_cpu==[*prompt.extend_lens,*([8]*t)]
        assert mixed_fb.extend_prefix_lens_cpu==[*prompt.prefix_lens,*tail.seq_lens.tolist()]
        assert plan.mixed_rows<=plan.prompt_rows+2*plan.verify_rows
    rows.append(dict(tails=t,live_prompt=q,veto=current[-1],prepared=value is not None))
    raise Boundary
bridge.env['try_forward_proposed_mixed']=inspect
for tail in (1,2,3,4):
    for veto in (None,'no_bucket','no_headroom','sampling','cost','stale'):
        current=(tail,veto)
        try:bridge.run('breakable',tail,1,True,veto)
        except Boundary:pass
assert len(rows)==24
print(json.dumps(dict(actual_prepare_views_cases=len(rows),positive=sum(v['prepared'] for v in rows),checks=rows),indent=2))
