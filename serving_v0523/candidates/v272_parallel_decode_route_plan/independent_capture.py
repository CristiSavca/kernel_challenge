"""Exercise native model types and PyTorch dispatch states at V272's capture guard."""
import ast
import contextlib
import hashlib
import json
from enum import Enum
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

root = Path(__file__).resolve().parents[3]
stage = Path(sys.argv[1])/"python/sglang"
hashes = {}


def src(path):
    raw = (stage/path).read_bytes()
    hashes[path] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def load_type(path, name, bases, names):
    native = next(n for n in ast.parse(src(path)).body if getattr(n,"name",None) == name)
    native.bases = [ast.parse(base, mode="eval").body for base in bases]
    methods = [m for m in native.body if getattr(m,"name",None) in names]
    assert {m.name for m in methods} == set(names)
    fields = [m for m in native.body if name=="FusedMuseSparkRouting" and isinstance(m,ast.Assign)
              and getattr(m.targets[0],"id",None) in ("priority","capabilities")]
    native.body = fields + methods
    exec("from __future__ import annotations\n" + ast.unparse(native), env)
    return env[name]


env = dict(debug_kernel_api=lambda fn:fn, torch=torch, nn=nn, F=F, _is_cpu=False, get_global_experts_capturer=lambda:None,
           _trace_enabled=False, _forced_backend=None, _UNRESOLVED=object(), _oot_dispatch_key=lambda:"test")
backend=Enum("KernelBackend",["TORCH","TORCH_COMPILE","JIT","TRITON"])
env.update(KernelBackend=backend,CapabilityRequirement=Enum("CapabilityRequirement",["CUDA"]),
           BACKEND_METHODS={backend.JIT:"forward_jit",backend.TRITON:"forward_triton"})
base = load_type("kernels/fused_op.py","BaseFusedOp",["nn.Module"],[
    "__init__","forward","register_oot_forward","_resolve_forward_method","_forward_backend_dynamic",
    "_auto_backend_candidates","_defined_method"])
base._oot_forward_registry = {}
mfile = "srt/models/muse_spark.py"
routing_type = load_type("kernels/ops/moe/muse_spark_routing.py","FusedMuseSparkRouting",["BaseFusedOp"],
    ["__init__","forward_jit","forward_triton","forward_native","backend_eligible"])
norm = load_type("srt/layers/muse_spark/muse_spark_layers.py","MuseSpark1xRMSNorm",["nn.Module"],
    ["__init__","forward","_get_effective_weight"])
router = load_type(mfile,"MuseSpark1xRouter",["nn.Module"],["forward","forward_split2"])
expert = load_type("srt/layers/muse_spark/muse_spark_fp4_moe.py","MuseSpark1xFp4Experts",["nn.Module"],["forward"])
moe = load_type(mfile,"MuseSpark1xMoE",["nn.Module"],["_capture_decode_plan","_routing_op"])
hpath = "srt/layers/muse_spark/muse_spark_humming_moe.py"
har = ast.parse(src(hpath))
native = [n for n in har.body if getattr(n,"name",None) in ("allocate_decode_plan","humming_decode_eligible")]
assert len(native)==2
exec("from __future__ import annotations\n"+ast.unparse(ast.Module(native,[])),env)
hook_module = ModuleType("sglang.srt.layers.muse_spark.muse_spark_humming_moe")
hook_module._MAX_DECODE_SLOTS = env["_MAX_DECODE_SLOTS"] = 2048
hook_module.allocate_decode_plan=env["allocate_decode_plan"]
dispatch_module = ModuleType("sglang.kernels.fused_op")
dispatch_module.BaseFusedOp=base
dispatch_module.get_fused_op_backend=lambda:None
kernels=ModuleType("sglang.kernels"); kernels.fused_op=dispatch_module
sys.modules["sglang.kernels"]=kernels
sys.modules[hook_module.__name__]=hook_module
sys.modules[dispatch_module.__name__]=dispatch_module
m=moe()
m.gate=router(); m.gate.e_score_correction_bias=torch.zeros(256)
m.route_scale=1.; m.router_multiplier=None; m.route_eps=None; m.top_k=8
m.pre_expert_norm=norm(4)
m.pre_expert_norm.weight.data.fill_(1.)
m.experts=expert()
m.experts.use_humming=True; m.experts.use_w4a16=True
m.experts.num_local_experts=32; m.experts.intermediate_size_per_partition=4096
m.experts.humming_decode_tunings={"w13":True, "w2":True}
m.experts.post_expert_norm=norm(4)
object.__setattr__(m.experts,"_post_expert_norm",m.experts.post_expert_norm)
m.experts._post_expert_norm_weight=torch.ones(4)
x=torch.ones((8,4),dtype=torch.bfloat16)

def accepted():
    return m._capture_decode_plan(x) is not None

class FunctionPass(TorchFunctionMode):
    def __torch_function__(self,func,types,args=(),kwargs=None):
        return func(*args,**(kwargs or {}))

class DispatchPass(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        return func(*args,**(kwargs or {}))

with torch.no_grad():
    assert not accepted() # Unprimed lazy routing must run stock.
    r=m._routing_op()
    assert type(r) is routing_type and r.expert_bias is m.gate.e_score_correction_bias
    # Normal routing's post-warmup resolved method is the in-tree dynamic dispatch.
    env["_oot_dispatch_key"]=lambda:None
    r._forward_method=r._resolve_forward_method()
    assert r._forward_method.__func__ is base._forward_backend_dynamic
    assert r._dynamic_backend_candidates == (backend.JIT,backend.TRITON)
    assert accepted()
    cases={}
    for users in (1,4,16,32):
        xs=torch.empty((users*8,8192),device="meta",dtype=torch.bfloat16)
        plan=m._capture_decode_plan(xs)
        assert tuple(p.device.type for p in plan) == ("meta",)*4
        assert all(p.is_contiguous() and p.dtype == torch.int32 for p in plan)
        sizes=[p.numel()*p.element_size() for p in plan]
        assert sizes == [users*8*8*4,132,users*8*8*4,users*8*8*4]
        cases[users]={"verification_rows":users*8, "plan_bytes_per_rank_layer":sum(sizes)}
    assert m._capture_decode_plan(torch.empty((257,4))) is None
    blocked=[]
    for name,ctx in (("global_forward_pre",nn.modules.module.register_module_forward_pre_hook(lambda *_:None)),
                     ("function_mode",FunctionPass()),("dispatch_mode",DispatchPass()),
                     ("grad_enabled",torch.enable_grad()),("router_capture",patch.dict(env,get_global_experts_capturer=lambda:object()))):
        with ctx:
            assert not accepted(), name
            blocked.append(name)
    for name in ("experts","gate","pre_expert_norm","_fused_routing_op"):
        mod=getattr(m,name)
        with mod.register_forward_hook(lambda *_:None), mod.register_forward_pre_hook(lambda *_:None):
            assert not accepted(), name
        blocked.append(name)
    for mod, key, value in ((r,"forward_jit",lambda *_:None),(r,"backend_eligible",lambda *_:True),
            (m.gate,"forward_split2",lambda *_:None),(r,"sol_topk_out",torch.zeros(1)),(r,"is_torch_compile",True),
            (dispatch_module,"get_fused_op_backend",lambda:backend.TORCH)):
        with patch.object(mod,key,value):
            assert not accepted(), key
        blocked.append(key)
    # Public BaseFusedOp registration, actual resolver, default nn.Module.__call__
    # and unmodified forward on the routing instance.
    def registered(self,*args,**kwargs):
        m.pre_expert_norm.weight.fill_(2.)
        return None
    base.register_oot_forward(routing_type,registered,"test")
    env["_oot_dispatch_key"]=lambda:"test"
    r._forward_method=r._resolve_forward_method()
    assert r.forward.__func__ is base.forward
    plugin_selected=accepted()
    assert not plugin_selected
    before=m.pre_expert_norm(x)
    r()
    ordered=m.pre_expert_norm(x)
    assert torch.ne(before,ordered).all()

full=src("srt/model_executor/runner_backend/full_cuda_graph_backend.py")
assert full.index("for warmup_step in range(2)")<full.index("graph = torch.cuda.CUDAGraph()")<full.index("out = forward_fn()")
assert "return layernorm_sp_enabled() and forward_mode == ForwardMode.EXTEND" in src("srt/layers/layernorm_sp.py")
assert "@torch.no_grad()\n    def init_all_cuda_graphs" in src("srt/managers/scheduler.py")
assert "self.capture_forward_mode = ForwardMode.TARGET_VERIFY" in src("srt/model_executor/runner/decode_cuda_graph_runner.py")
assert "if not get_exec().features.enable_return_routed_experts:" in src("srt/state_capturer/routed_experts.py")
# Per-call overlap is limited by the later of the two arms.
for p in range(5):
    for r_ in range(5):
        for n in range(5):
            for q in range(5):
                saving=max(p,r_)+n+q-max(p+n,r_+q)
                assert saving==(min(q,p-r_+n) if p>=r_ else min(n,r_-p+q))
j=json.loads((root/"serving_v0523/runs/04b01f958aa5487c.json").read_bytes())["result"]
rung=j["rungs"][0]; period=1000*rung["acceptance"]/rung["tok_s_per_user_p50"]
score=json.loads(next((root/"serving_v0523/candidates").glob("v265*/audit.json")).read_bytes())["score_scenarios"]["crossings"]
print(json.dumps(dict(torch=torch.__version__,eligible_after_native_warmup=True,ordinary_shapes=cases,
    verified_fallbacks=blocked,registered_router_parallel_plan_accepted=plugin_selected,
    possible_norm_values_in_custom_router_order=[before[0,0].item(),ordered[0,0].item()],
    old_and_new_per_layer_time="max(P,R)+N+Q versus max(P+N,R+Q) for pool P, routing R, norm N and plan Q; 625 nonnegative cases checked",
    v68_16_user_conditional_decode_period_ms=period,
    v68_16_user_saving_per_layer_us_to_scale_decode_period_by_uniform_model={
        k:1000*period*(1-1/v["both_axes"])/62 for k,v in score.items()},
    scope="True extracted model constructors, native bound methods, real PyTorch modes and hooks, native in-tree candidate enumeration and dispatch resolution under modeled in-tree platform identity. Sizing uses meta tensors. No CUDA capture or latency measurement.",
    source_sha256=hashes),indent=2))
