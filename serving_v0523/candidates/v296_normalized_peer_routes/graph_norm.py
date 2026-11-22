"""Exercise deferred pre-expert norm in V296's actual Breakable callback, including gain identity."""
import ast
import inspect
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"v295_published_prompt_routes"))
import graph_replay as prior

code = inspect.getsource(prior.test)
def change(before,after):
    global code
    assert code.count(before)==1, before
    code=code.replace(before,after)

change('    layer_sources = []','''    layer_sources = []
    norm_env = dict(torch=torch,nn=torch.nn)
    bind(P+"layers/muse_spark/muse_spark_layers.py", ("MuseSpark1xRMSNorm",), norm_env)
    Norm=norm_env["MuseSpark1xRMSNorm"]
    functional = torch.nn.functional.rms_norm
    class Machine:
        def run(self, raw, gain, dest_rows, dest, epsilon):
            assert ctx.get() is None and raw.shape==(1,rows,width)
            assert gain.data_ptr() == state.current.gain.data_ptr()
            assert epsilon==state.current.eps
            same(dest_rows,torch.arange(rows,dtype=torch.int32).view(-1,1))
            same(raw[0],state.current.raw[rank])
            dest.copy_(functional(raw[0],(width,),gain,epsilon))
            state.events.append("direct_norm")
    def native(x,shape,gain,epsilon):
        assert ctx.get() is None and gain.data_ptr()==state.current.gain.data_ptr()
        assert epsilon==state.current.eps
        same(x,state.current.raw[rank])
        state.events.append("native_norm")
        return functional(x,shape,gain,epsilon)
    ''')
change('    men = dict(torch=torch,layernorm_sp=sp,get_parallel=lambda:NS(tp_group=group))','''    modules.extend([module(PKG+"muse_spark_layers", MuseSpark1xRMSNorm=Norm),
                    module("sglang.srt.batch_invariant_ops", is_batch_invariant_mode_enabled=lambda:False)])
    class Model:
        pass
    men = dict(torch=torch,layernorm_sp=sp,get_parallel=lambda:NS(tp_group=group),
               MuseSpark1xMoE=Model, MuseSpark1xFp4Experts=Experts)''')
change('bind(MODEL, ("_forward_sp_replicated_experts",),men,owner="MuseSpark1xMoE")','''bind(MODEL, ("_forward_sp_experts","_forward_sp_replicated_experts"),men,owner="MuseSpark1xMoE")
    Model._forward_sp_replicated_experts=men["_forward_sp_replicated_experts"]''')
change('    with (torch.no_grad(),','''    ws.norm_module=Machine()
    ws.norm_rows=torch.arange(rows,dtype=torch.int32).view(-1,1)
    norms=[]
    with (torch.no_grad(),patch.object(torch.nn.functional,"rms_norm",native),''')
change('            model=NS(top_k=k,num_experts=4,experts=exp,_peer_prompt_inputs=ws)','''            model=Model()
            model.top_k=k;model.num_experts=4;model.experts=exp;model._peer_prompt_inputs=ws
            model.sp_replicate_expert_input=True
            model.pre_expert_norm=Norm(width,eps=1e-6*(i+1),zero_centered_gamma=i==1).to(dtype=torch.bfloat16).eval()
            model.pre_expert_norm.weight.fill_(1.25)
            model.pre_expert_norm.precompute_effective_weight()
            norms.append(model.pre_expert_norm)''')
change('out=men["_forward_sp_replicated_experts"](model,sh,scores,ids)','out=men["_forward_sp_experts"](model,sh,scores,ids,None)')
change('                golden=table_words(n,width,epoch*29)','''                gain=norms[i]._get_effective_weight()
                gain.copy_((torch.arange(width)*.03125+epoch/16+.25).to(torch.bfloat16))
                normalized=torch.stack([functional(x,(width,),gain,norms[i].eps) for x in shards])
                golden=table_words(n,width,epoch*29)''')
change('state.current=NS(shards=shards,','state.current=NS(raw=shards,shards=normalized,gain=gain,eps=norms[i].eps,')
change('same(ws.banks[b][0][:rows],shard)','same(ws.banks[b][0][:rows],state.current.shards[rank])')
change('                same(out,golden)','''                same(out,golden)
                assert state.events.count("direct_norm")+state.events.count("native_norm")==1
                stats["direct_norm"]+=state.events.count("direct_norm")
                stats["native_norm"]+=state.events.count("native_norm")
                stats["replayed_frozen_gain" if norms[i].zero_centered_gamma else "replayed_parameter_gain"]+=1''')
ctx = dict(vars(prior))
exec(compile(code,__file__,"exec"),ctx)
torch.set_num_threads(1)
for size in (2,4,8):
    for start in (0,1):
        ctx["test"](size,start)
print(json.dumps(dict(tree=prior.TREE,source_sha256=prior.sources,results=dict(sorted(prior.stats.items())),
    coverage="Real V296 model -> captured norm-admission -> production route/legacy graph callback, raw and normalized inputs checked with host RMSNorm; device-op and remote transport simulated. Gain buffers are mutated in place between replays. Changing the captured Python graph structure requires recapture.",physical_gpu=False),indent=2))
