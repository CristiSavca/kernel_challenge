"""Replay the pinned eager-on-graph wrapper over refreshed CPU tensors and sort buffers."""
from __future__ import annotations
import ast
from contextvars import ContextVar
import functools
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import torch
from cpu_check import source, install, Humming, finalize, HUM, MOE, SRC

stage=Path(sys.argv[1]).resolve()
torch.manual_seed(149);torch.set_num_threads(1)
ctx=ContextVar("capture",default=None)
runtime=dict(torch=torch,functools=functools,_current_capture_var=ctx)
bcg=SRC/"model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py"
unit=ast.parse(source(stage,bcg,False))
names={"eager_on_graph","_EagerOnGraphCallable","_weak_ref_if_tensor","_copy_output"}
exec(compile(ast.Module([n for n in unit.body if getattr(n,"name",None) in names],type_ignores=[]),str(bcg),"exec"),runtime)
env=dict(torch=torch,HummingMethod=Humming,_GROUPED_COMPUTE="grouped",fused_post_expert_norm_weighted_sum=finalize,
         silu_and_mul=lambda g,o:o.copy_(torch.nn.functional.silu(g.float().chunk(2,1)[0])*g.float().chunk(2,1)[1]))
install(source(stage,HUM,True),["humming_grouped_route_order","humming_grouped_expert_outputs"],env)
install(source(stage,MOE,True),["_grouped_humming_and_combine_impl"],env)
layer=NS(num_local_experts=4,intermediate_size_per_partition=8,_post_expert_norm_weight=torch.randn(8).bfloat16(),
         _post_expert_norm_eps=1e-6,_post_expert_norm_weight_first=True,gemms=[],humming_grouped_tunings={"w13":None,"w2":None},
         weights={"w13":torch.randn(4,8,16),"w2":torch.randn(4,8,8)},w13_global_scale=torch.ones(4).bfloat16(),w2_global_scale=torch.ones(4).bfloat16())
inner=env["_grouped_humming_and_combine_impl"]
count=0;elements=0
for n in (32,257,4096):
    ids=torch.full((n,8),-1,dtype=torch.int32)
    order=list(env["humming_grouped_route_order"](ids,4))
    inputs=torch.empty(n,8,dtype=torch.bfloat16);scores=torch.empty(n,8)
    out=torch.empty_like(inputs)
    graph=NS(_break_fns=[])
    cap=NS(_barrier_fn=None,_end_current_segment=lambda:None,_begin_new_segment=lambda:None,cuda_graph=graph)
    token=ctx.set(cap)
    wrapped=runtime["eager_on_graph"](True,capture_stub=lambda *a,**kw:None)(inner)
    wrapped(layer,inputs,ids,scores,out,order)
    ctx.reset(token)
    assert len(graph._break_fns)==1 and not layer.gemms
    storage=tuple(x.data_ptr() for x in order)
    for i in range(32):
        ids.random_(-1,4);inputs.copy_(torch.randn(n,8));scores.copy_(torch.randn(n,8).softmax(-1))
        if i%5==0: ids.fill_(-1)
        if i%5==1: ids.fill_(3)
        if i%5==2: ids[:,1::2]=ids[:,::2]
        for dst,src in zip(order,env["humming_grouped_route_order"](ids,4)):dst.copy_(src)
        graph._break_fns[0]()
        expected=torch.empty_like(out);inner(layer,inputs,ids,scores,expected)
        assert torch.equal(out.view(torch.int16),expected.view(torch.int16))
        assert storage==tuple(x.data_ptr() for x in order)
        count+=1;elements+=out.numel()
    layer.gemms=[]
print(json.dumps(dict(torch=torch.__version__,wrapper="pinned eager_on_graph/captured_args/replay_fn",replays=count,
                      verified_BF16_elements=elements,changing_rank_routes=True,static_buffer_addresses=True,
                      GPU_capture=False,limitation="CPU tensors in pinned recursive argument helper; the graph kernels and weak accelerator-storage views need H200."),indent=2))
