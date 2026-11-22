"""Drive the real proposal, consensus and target call. Check marker lifetime across decisions."""
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

HERE=Path(__file__).resolve().parent
S=Path(sys.argv[1]).resolve()/'python/sglang/srt'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

b=load('v234_immutable_bridge_fixture',HERE.parent/'v159_mixed_verify_forward/bridge_cpu.py')
b.pl.mixed_prefill_graph_fits=load('v234_physical_caps',S/'arg_groups/prefill_buffer_ceiling.py').mixed_prefill_graph_fits
native_prepare=b.env['_prepare_views']; native_try=b.env['try_forward_proposed_mixed']
fb=ast.parse((S/'model_executor/forward_batch_info.py').read_text())
fields=next(x.body for x in fb.body if getattr(x,'name',None)=='ForwardBatch')
default=ast.literal_eval(next(x.value for x in fields if isinstance(x,ast.AnnAssign) and x.target.id=='mixed_verify_is_causal'))
assert default is False
# The old torch CPU fixture uses a stand-in for the GPU ForwardBatch. Use the exact
# declared field default and the submitted runner's actual replay constructor expression.
b.v155.FB.mixed_verify_is_causal=default
runner=ast.parse((S/'model_executor/runner/prefill_cuda_graph_runner.py').read_text())
calls=[n for n in ast.walk(runner) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='ForwardBatch']
field='mixed_verify_is_causal'
replayed=[kw.value for c in calls for kw in c.keywords if kw.arg==field]
assert len(replayed)==1 and ast.unparse(replayed[0])=='forward_batch.mixed_verify_is_causal'
copy_field=compile(ast.Expression(replayed[0]),'<static ForwardBatch constructor>','eval')
b.env.update(dflash_sp_prefill_scope=lambda *_:nullcontext())
class Reached(Exception):pass
observed=[]

def check(worker,prompt,tail,proposal,**kw):
    cases=[];target=[]
    q=prompt.extend_num_tokens;t=len(tail.reqs)
    b.pl.SETTINGS.chunked_prefill_size=q
    if veto!='cost':b.pl.SETTINGS.max_prefill_tokens=q+t*8
    pre=worker.target_worker.model_runner.prefill_cuda_graph_runner
    pre.mixed_dflash_token_capacity=pre.max_num_tokens
    worker._target_logits_processor=None
    worker._sp_prefill_context=None
    worker._prefill_kv_helper=lambda:None
    worker._wait_for_fused_hicache_load=lambda *_:None
    worker.target_worker.set_hicache_consumer=lambda *_:None
    def prepare(*args,**kwargs):
        value=native_prepare(*args,**kwargs)
        if value is not None:
            p,m=value[-2:];assert not p.mixed_verify_is_causal and not m.mixed_verify_is_causal
            cases.append((p,m))
        return value
    def execute(*,batch,forward_batch,is_verify,return_kv_loc_plan):
        assert batch is None and is_verify and return_kv_loc_plan and cases[0][1] is forward_batch
        assert not cases[0][0].mixed_verify_is_causal
        assert eval(copy_field,{'forward_batch':forward_batch})
        assert torch.equal(forward_batch.out_cache_loc[q:],proposal.virtual_slots.flatten())
        assert torch.equal(forward_batch.input_ids[q:],proposal.tokens.flatten())
        target.append(True)
        raise Reached
    worker.target_worker.forward_batch_generation=execute
    b.env['_prepare_views']=prepare
    try:
        try: result=native_try(worker,prompt,tail,proposal,**kw)
        except Reached:result=True
    finally:b.env['_prepare_views']=native_prepare
    if veto is None:
        assert result is True and len(target)==1 and not worker.is_current_decode_proposal(tail,proposal)
    else:
        assert result is None and not target
        if veto=='rank_disagrees':
            assert len(cases)==1 and not cases[0][1].mixed_verify_is_causal
    # A new synthetic/ordinary batch has the default after each attempted replay.
    assert not eval(copy_field,{'forward_batch':NS(mixed_verify_is_causal=default)})
    observed.append(dict(tails=t,veto=veto,target_calls=len(target),prepared=len(cases)))
    raise Reached

b.env['try_forward_proposed_mixed']=check
for tails in (1,2,3,4):
    for veto in (None,'rank_disagrees','no_bucket','no_headroom','sampling','cost','stale',None):
        try:b.run('breakable',tails,1,True,veto)
        except Reached:pass
assert len(observed)==32
print(json.dumps(dict(native_proposal_rounds=len(observed),joint_target_calls=sum(x['target_calls'] for x in observed),
    equal_tail_write_ids=True,zero_marks_on_rank_disagreement=True,new_forward_default=False,
    production_replay_constructor=ast.unparse(replayed[0]),cases=observed), indent=2))
