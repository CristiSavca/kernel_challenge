"""The frozen V129 preparation and dispatch suite exercised with both V139 JITs."""
import importlib.util
import json
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

stage = Path(sys.argv[1]).resolve()
candidates = Path(__file__).resolve().parents[1]
fname = "python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
source = stage / fname
spec = importlib.util.spec_from_file_location("v129_dispatch", candidates / "v129_routed_projection_pair/dispatch_check.py")
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
original = owner.function
record = []
dispatch = owner.dispatch(source)

# Run the unchanged V129 assertions against combined initialization. Instrument
# both JIT lookup functions and the requested input shape before running it.
for enabled, available, width_ok in product((False, True), repeat=3):
    def combined(path, name, env):
        env["envs"].SGLANG_ENABLE_MUSE_SPARK_COMPACT_PREFILL = NS(get=lambda:enabled)
        marker = object()
        count = []
        def compile_activation():
            count.append("activation")
            return marker if available else None
        env["_prefill_scaled_silu_module"] = compile_activation
        f = original(path, name, env)
        def invoke(layer):
            layer.hidden_size = 4096
            layer.intermediate_size_per_partition = 4096 if width_ok else 2048
            can_load = (enabled and width_ok and layer.gate_up_proj.is_cuda
                        and env["torch"].cuda.get_device_capability("cuda:0") == (9,0)
                        and not env["envs"].SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
                        and not env["get_exec"]().deterministic.enable_deterministic_inference)
            f(layer)
            assert len(count)==int(can_load)
            assert (layer.humming_prefill_silu is marker) == (can_load and available)
            assert all(hasattr(layer,f"{name}_{k}") for name in ("w13","w2") for k in ("weight","weight_scale","global_scale"))
            assert all(not hasattr(layer,name) for name in ("gate_up_proj","down_proj"))
            record.append((int(layer.humming_route_pair),int(layer.humming_prefill_silu is not None)))
        return invoke
    with patch.object(owner,"function",combined):
        owner.startup(source)

# The normal W4A16 frontend selects exactly one of these three implementations.
# Install observable stops at the three external kernel boundaries. Both Python
# planners, including the native route-pair allocation and V135 front door, run.
fp4 = source.with_name("muse_spark_fp4_moe.py")
class Selected(Exception):
    def __init__(self, name): self.name = name
def stop(name):
    def caller(*args, **kwargs): raise Selected(name)
    return caller

summary = {}
norm = object()
for tiny_on, large_on, graph in product((False,True), repeat=3):
    for m in (0,8,32,33,128,8191,8192,16384):
        dim,topk,inner = 128,8,128
        h=torch.zeros((m,dim),dtype=torch.bfloat16)
        ids=torch.arange(m*topk,dtype=torch.int32).reshape(m,topk)%32
        scores=torch.empty((m,topk),dtype=torch.float)
        raw=dict(torch=torch, _HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT=256,
                 is_in_breakable_cuda_graph=lambda:False,local_humming_expert_ids=lambda model,x:x,
                 _route_pair_module=lambda:NS(run=stop("tiny")), moe_align_block_size=stop("indexed"),
                 fused_post_expert_norm_weighted_sum=stop("empty"),
                 compact_prefill_and_combine=stop("compact"),
                 humming_grouped_expert_outputs=stop("legacy-grouped"))
        en=original(source,"humming_expert_outputs",raw)
        raw["humming_expert_outputs"]=en
        grouped=original(fp4,"_grouped_humming_and_combine_impl",raw)
        raw["_grouped_humming_and_combine_impl"]=grouped
        full=original(fp4,"_forward_w4a16",raw)
        layer=NS(use_humming=True, humming_route_pair=tiny_on,
                 humming_prefill_silu=object() if large_on else None,
                 _post_expert_norm=norm,_post_expert_norm_weight=norm,
                 _post_expert_norm_eps=1e-5,_post_expert_norm_weight_first=False,
                 intermediate_size_per_partition=inner,num_local_experts=32, global_num_experts=256,
                 humming_indexed_tunings={k:[(10**8,{"block_shape":(8,64,128)})] for k in ("w13","w2")})
        for k,n in (("w13",2*inner),("w2",dim)):
            setattr(layer,k+"_weight",torch.empty((32,8,2*n),dtype=torch.int32,device="meta"))
            setattr(layer,k+"_weight_scale",torch.empty((32,8,n),dtype=torch.float8_e4m3fn,device="meta"))
            setattr(layer,k+"_global_scale",object())
        layer._grouped_humming_and_combine=lambda *a:grouped(layer,*a)
        with patch.object(torch.cuda,"is_current_stream_capturing",return_value=graph):
            try: full(layer,h,scores,ids,norm)
            except Selected as got:
                path=got.name
            else: raise AssertionError("frontend failed to reach a kernel")
        expected=("empty" if not m else "tiny" if tiny_on and m*topk<=256 else
                  ("compact" if large_on else "legacy-grouped") if m*topk>=65536 and not graph else "indexed")
        assert expected==path,(m,tiny_on,large_on,graph,path,expected)
        summary[path]=summary.get(path,0)+1

print(json.dumps(dict(startup_combinations=len(record),
  independent_readiness={f"tiny={i},large={j}":record.count((i,j)) for i,j in product((0,1),repeat=2)},
  ordinary_forward_cases=sum(summary.values()),ordinary_forward_kernel_dispatches=summary,
  inherited_low_load_dispatch=dispatch,physical_hardware=False),indent=2))
