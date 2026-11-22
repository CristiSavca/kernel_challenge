"""Run the real wave, W2 bindings, Humming launch ABI and finalizer with CPU device-operation models."""
import ast
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
import torch

stage = Path(sys.argv[1]).resolve()
candidates = Path(__file__).resolve().parent.parent
src = stage/"python/sglang/srt/layers/muse_spark"
model = runpy.run_path(str(next(candidates.glob("v306_*/replay.py"))))
base, graph = model["base"], model["Graph"]()
wf = src/"humming_prefill_waves.py"
ops, _, _ = base["make_ops"](stage)
pid,(loader,) = base["program"](wf, ("_load_wave",))
ops["_load_wave"] = graph.kernel(base["Launch"](loader,pid,(2,3,4,5,6)))
ops["_pack_wave"] = graph.kernel(ops["_pack_wave"])
ops["fused_post_expert_norm_weighted_sum"] = graph.call(ops["fused_post_expert_norm_weighted_sum"])
disjoint = model["disjoint"]
totals,bases,modes = Counter(),Counter(),Counter()
calls = []
counts = Counter()

def module(name, **fields):
    m = ModuleType(name)
    vars(m).update(fields)
    return m

pair = module("sglang.srt.layers.muse_spark.decode_mma_pair")
prompt = module("sglang.srt.layers.muse_spark.prompt_weight_pipeline")
for m,n in [(pair,"decode_mma_pair.py"),(prompt,"prompt_weight_pipeline.py")]:
    exec(compile((src/n).read_bytes(),str(src/n),"exec"),vars(m))
f = next(n for n in ast.parse((src/"muse_spark_humming_moe.py").read_bytes()).body
         if getattr(n,"name",None)=="run_grouped_prompt_projection")
ops["run_grouped_prompt_projection"] = None
scope = {"_GROUPED_COMPUTE":"grouped"}
exec(compile(ast.Module([f],[]),str(src/"muse_spark_humming_moe.py"),"exec"),scope)
ops["run_grouped_prompt_projection"] = scope[f.name]

def projection(path,layer,x,y,bounds,name):
    assert disjoint(x,y)
    layout = bounds.tolist()  # Device-operation model only.
    assert layout == sorted(layout) and layout[-1] == len(x)
    rows = len(x)-layout[0]
    bases[path, 'empty' if rows==0 else 'positive' if layout[0] else 'zero'] += 1
    calls.append((path,rows))
    y.fill_(float("nan"))
    for e,(lo,hi) in enumerate(zip(layout,layout[1:])):
        if lo<hi:
            y[lo:hi] = x[lo:hi].float() @ getattr(layer,name)[e].T

def ordinary(layer, *, inputs, outputs, expert_layout, compute_config, tuning_config, sublayer_name):
    assert compute_config=="grouped" and tuning_config==sublayer_name
    projection(sublayer_name+" vendor",layer,inputs,outputs,expert_layout,sublayer_name)

def custom(*,configs,inputs,outputs,expert_layout,weight,weight_scale,locks):
    assert configs is config and weight is layer.w2_weight and weight_scale is layer.w2_weight_scale and locks is layer.locks
    projection("w2 custom",layer,inputs,outputs,expert_layout,"w2")

def scaled(raw, activation, keys, offsets, scales, span):
    assert disjoint(raw,activation)
    lo = int(offsets[0])
    k = keys[lo:]//span-1
    left,right=(raw[lo:].float()*scales[k,None].float()).bfloat16().float().chunk(2,dim=1)
    activation[lo:] = torch.nn.functional.silu(left)*right
ops["_expert_scaled_silu_module"] = lambda:NS(run=graph.call(scaled))

def bind(layer,packed,act):
    if w13=="declined":
        return None
    def project(keys,offsets):
        assert disjoint(packed,act)
        edges=offsets.tolist()
        calls.append(("w13 fused",len(packed)-edges[0]))
        for e,(lo,hi) in enumerate(zip(edges,edges[1:])):
            if lo<hi:
                assert (keys[lo:hi]==e).all()
                raw=(packed[lo:hi].float()@layer.w13[e].T).bfloat16()
                gate,up=(raw.float()*layer.w13_global_scale[e].float()).bfloat16().float().chunk(2,dim=1)
                act[lo:hi] = torch.nn.functional.silu(gate)*up
    return graph.call(project)
mods = [pair,prompt,module("humming",ops=NS(launch_kernel=graph.call(custom))),
        module("humming.layer",HummingMethod=NS(forward_layer=graph.call(ordinary))),
        module("sglang.srt.layers.muse_spark.decode_gate_up",bind_prompt_wave=bind)]
config = torch.tensor([0,1<<30,197,0],dtype=torch.int64)
randg = torch.Generator().manual_seed(308)
def rand(*shape):return torch.randint(-4,5,shape,generator=randg).float()/8
t,k,e,h,i,cap=37,8,7,8,16,25
layer=NS(num_local_experts=e,hidden_size=h,intermediate_size_per_partition=i,w13=rand(e,2*i,h),w2=rand(e,h,i),
    w13_global_scale=torch.linspace(.5,1.25,e).bfloat16(),w2_global_scale=torch.linspace(.625,1.5,e).bfloat16(),
    _post_expert_norm_weight=rand(h).bfloat16(),_post_expert_norm_eps=1e-5,_post_expert_norm_weight_first=False,
    humming_grouped_tunings={"w13":"w13","w2":"w2"},humming_metas={"w2":NS(to_str=lambda:"meta")},
    w2_weight=torch.empty((e,1,2*h),dtype=torch.int32),w2_weight_scale=torch.empty((e,1,h),dtype=torch.float8_e4m3fn),
    locks=torch.zeros(1,dtype=torch.int32))
weight_binding=(prompt._binding(layer.w2_weight),prompt._binding(layer.w2_weight_scale))
hidden,scores=rand(t,h).bfloat16(),rand(t,k)
with patch.dict(sys.modules,{m.__name__:m for m in mods}):
    for layout in ("ragged","local","remote","trailing","exact"):
        ids=torch.randint(-3*e,e,(t,k),dtype=torch.int32,generator=randg)
        if layout=="local": ids.remainder_(e)
        elif layout=="remote": ids.fill_(-1)
        elif layout=="trailing": ids[:t//2].remainder_(e);ids[t//2:]=-1
        elif layout=="exact": ids.fill_(-1);ids.flatten()[:cap]=e-1
        reference=base["reference"](layer,hidden,scores,ids)
        live=int(((ids>=0)&(ids<e)).sum())
        for w13,w2,capture,cond in product(("fused","declined","missing"),("bound","stale","missing"),(True,False),(True,False)):
            layer.humming_prompt_gate_up = None if w13=="missing" else True
            layer.humming_conditional_waves=cond
            layer.humming_prompt_down = None if w2=="missing" else (config,"meta","grouped","w2",weight_binding if w2=="bound" else ())
            calls.clear();graph.iterations=None
            with patch.object(torch.cuda,"is_current_stream_capturing",return_value=capture),patch.object(torch.cuda,"CUDAGraph",graph):
                output=ops["grouped_prefill_waves"](layer,hidden,scores,ids,cap)
            assert torch.equal(reference.view(torch.int16),output.view(torch.int16))
            assert all(x[0]==("w13 fused" if w13=="fused" else "w13 vendor") for x in calls[::2])
            assert all(x[0]==("w2 custom" if w2=="bound" else "w2 vendor") for x in calls[1::2])
            assert sum(r for _,r in calls[::2])==sum(r for _,r in calls[1::2])==live
            if graph.iterations is not None:counts["conditional_runs"]+=1;counts["max_overflow"]=max(counts["max_overflow"],graph.iterations)
            counts["cases"]+=1;counts["bf16_words"]+=output.numel();counts["live_routes"]+=live
            counts["W2_launches"]+=len(calls[1::2]);modes[w13+"/"+w2]+=1
assert all(bases[k,p]>0 for k in ("w2 custom","w2 vendor") for p in ("empty","positive","zero")),bases
hot=[ast.parse(wf.read_bytes()),f]
for implementation in hot:
    assert not any(isinstance(node,ast.Attribute) and node.attr in {"item","tolist","cpu","synchronize"} for node in ast.walk(implementation))
print(json.dumps(dict(counts=counts,W13_W2_cases=modes,bases={"/".join(key):value for key,value in bases.items()},
    sources={n:sha256((src/n).read_bytes()).hexdigest() for n in ["muse_spark_humming_moe.py","humming_prefill_waves.py","prompt_weight_pipeline.py","decode_mma_pair.py"]},
    scope="Actual planner, packer, shared dispatch, custom W2 binding and launch adapter, finalizer. Modeled CUDA graph and numerical GEMMs on host PyTorch.",physical_H200=False),indent=2))
