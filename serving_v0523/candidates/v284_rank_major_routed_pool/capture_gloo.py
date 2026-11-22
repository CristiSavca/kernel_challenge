"""Run V284's actual capture vote on 4/8 Gloo ranks, with one rank declining at a time."""
import ast
import contextlib
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

MODEL = "python/sglang/srt/models/muse_spark.py"
HELPER = "python/sglang/srt/layers/communication/muse_spark_gemm_ag.py"
PIN = "5c4886908584029761b579af026dcfb627c84070"

def read(stage,path,cls,name,ns):
    tree = ast.parse((stage/path).read_bytes()).body
    if cls:
        tree = next(n.body for n in tree if getattr(n,"name","") == cls)
    node = next(n for n in tree if getattr(n,"name","") == name)
    exec("from __future__ import annotations\n"+ast.unparse(node), ns)
    return ns[name]

def install(name,**attrs):
    mod = ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod

class Experts(nn.Module): pass
class Norm(nn.Module): pass
class Router(nn.Module): pass
class Base(nn.Module):
    def _forward_backend_dynamic(self): pass
class Routing(Base): pass
class Full: pass

def worker(rank,world,folder,source):
    torch.set_num_threads(1)
    group_dir = Path(folder)
    dist.init_process_group("gloo",init_method="file://"+str(group_dir/f"g{world}"),rank=rank,world_size=world,timeout=timedelta(seconds=60))
    try:
        stage = Path(source)
        group = NS(cpu_group=dist.group.WORLD,world_size=world,ranks=list(range(world)))
        parallel = NS(tp_size=world,moe_ep_size=world,attn_tp_size=world,moe_tp_size=1,attn_cp_size=1,
                      attn_dcp_size=1,attn_dp_enabled=False,tp_group=group,attn_tp_group=group)
        envs = NS(SGLANG_ENABLE_MUSE_SPARK_DECODE_COLUMN_GATHER=NS(get=lambda:True),
                  SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False))
        install("sglang.srt.environ",envs=envs)
        install("sglang.srt.batch_invariant_ops",is_batch_invariant_mode_enabled=lambda:False)
        install("sglang.srt.runtime_context",get_parallel=lambda:parallel,get_exec=lambda:NS(deterministic=NS(enable_deterministic_inference=False)))
        install("sglang.srt.model_executor.runner_backend.full_cuda_graph_backend",FullCudaGraphBackend=Full)
        install("sglang.kernels",fused_op=NS(BaseFusedOp=Base,get_fused_op_backend=lambda:None,_trace_enabled=False))
        control=NS(jit=True,capturer=None)
        install("sglang.srt.layers.muse_spark.muse_spark_humming_moe",_ranked_pool_module=lambda:object() if control.jit else None)
        ns=dict(torch=torch,nn=nn,MuseSpark1xFp4Experts=Experts,MuseSpark1xRMSNorm=Norm,MuseSpark1xRouter=Router,
                FusedMuseSparkRouting=Routing,get_global_experts_capturer=lambda:control.capturer)
        rank_cap = read(stage,MODEL,"MuseSpark1xMoE","can_capture_ranked_pool",ns)
        guard = read(stage,HELPER,None,"can_capture_decode_columns",ns)
        model=nn.Module()
        body=nn.Module()
        model.model=nn.Module()
        model.model.language_model=body
        layer=nn.Module()
        body.layers=nn.ModuleList([layer])
        layer.is_moe_layer=True
        native_moe=type("MoE",(nn.Module,),{"can_capture_ranked_pool":rank_cap})
        install("sglang.srt.models.muse_spark",MuseSpark1xMoE=native_moe)
        layer.mlp=native_moe()
        moe=layer.mlp
        moe.nvfp4_checkpoint=True
        for name,shape in [("pre_expert_proj",(8192,4096)),("post_expert_proj",(4096,8192))]:
            obj=nn.Linear(*shape,bias=False,dtype=torch.bfloat16,device="meta").requires_grad_(False)
            setattr(moe,name,obj)
        moe.experts=exp=Experts()
        exp.use_w4a16=exp.use_humming=True
        exp.humming_decode_tunings={"w13":1,"w2":2}
        exp._post_expert_norm=exp.post_expert_norm=nn.Module()
        exp._post_expert_norm_weight=torch.ones(1)
        moe.pre_expert_norm=norm=Norm()
        norm.hidden_size=4096
        norm.zero_centered_gamma=True
        gain=torch.ones(4096,dtype=torch.bfloat16)
        norm._frozen_weight=gain
        moe.gate=Router()
        moe._fused_routing_op=route=Routing()
        route.is_torch_compile=False
        route.sol_topk_out=None
        route.expert_bias=moe.gate.e_score_correction_bias=torch.zeros(8)
        route._forward_method=route._forward_backend_dynamic
        model.eval()
        runner=NS(backend=Full(),capture_forward_mode=NS(is_decode=lambda:False,is_target_verify=lambda:True),
            enable_pdmux=False,enable_two_batch_overlap=False,ragged_verify_mode=False,enable_torch_compile=False,
            model_runner=NS(is_draft_worker=False,lora_manager=None,capture_tail_hooks=[],capture_head_hooks=[]))
        out={}
        cases=("enabled","fp32_gain","unaligned_gain","uncached_gain","wrong_pytorch","jit_declined","router_capture",
               "custom_norm_method","custom_expert_helper","custom_router","forward_hook",
               "original_tp_signature","original_reduce_signature","moe_subclass",
               "incompatible_arch","other_backend","enabled_again")
        for case in cases:
            last=rank==world-1
            with contextlib.ExitStack() as stack:
                active=lambda name: last and case==name
                stack.enter_context(torch.inference_mode())
                stack.enter_context(patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True))
                stack.enter_context(patch.object(torch.cuda,"is_available",return_value=True))
                stack.enter_context(patch.object(torch.cuda,"get_device_capability",return_value=(8,0) if active("incompatible_arch") else (9,0)))
                stack.enter_context(patch.object(torch.version,"git_version","other" if active("wrong_pytorch") else PIN))
                if active("fp32_gain"): stack.enter_context(patch.object(norm,"_frozen_weight",gain.float()))
                if active("unaligned_gain"): stack.enter_context(patch.object(norm,"_frozen_weight",torch.empty(4097,dtype=torch.bfloat16)[1:]))
                if active("uncached_gain"): stack.enter_context(patch.object(norm,"_frozen_weight",None))
                if active("custom_norm_method"): stack.enter_context(patch.object(norm,"_get_effective_weight",lambda:gain,create=True))
                if active("custom_expert_helper"): stack.enter_context(patch.object(exp,"_forward_w4a16",lambda *_:None,create=True))
                if active("custom_router"): stack.enter_context(patch.object(moe,"_routing_op",lambda:route,create=True))
                if active("original_tp_signature"): stack.enter_context(patch.object(moe,"_forward_tp",lambda x,r,b:x,create=True))
                if active("original_reduce_signature"): stack.enter_context(patch.object(moe,"_reduce_and_unpool",lambda x,*,already_reduced,column_group:x,create=True))
                if active("moe_subclass"):
                    moe.__class__=type("Derived",(native_moe,),{})
                    stack.callback(setattr,moe,"__class__",native_moe)
                if active("forward_hook"): stack.callback(norm.register_forward_hook(lambda *_:None).remove)
                if active("other_backend"): stack.enter_context(patch.object(runner,"backend",object()))
                control.jit=not active("jit_declined")
                control.capturer=object() if active("router_capture") else None
                columns=guard(model,runner)
                observed=(columns,runner._capture_muse_ranked_pool)
                expected=(case not in ("forward_hook","incompatible_arch","other_backend","original_tp_signature",
                                       "original_reduce_signature","moe_subclass"),case in ("enabled","enabled_again"))
                assert observed==expected,(case,rank,observed)
                out[case]=list(observed)
        (group_dir/f"rank{world}-{rank}").write_text(json.dumps(out))
    finally:
        dist.destroy_process_group()

if __name__=="__main__":
    stage=Path(sys.argv[1])
    per_world={}
    with tempfile.TemporaryDirectory() as folder:
        for world in (4,8):
            mp.spawn(worker,args=(world,folder,str(stage)),nprocs=world)
            rows=[json.loads((Path(folder)/f"rank{world}-{r}").read_text()) for r in range(world)]
            assert all(row==rows[0] for row in rows)
            per_world[str(world)]=rows[0]
    print(json.dumps(dict(worlds=per_world,rank_conditions=sum(len(v)*int(w) for w,v in per_world.items()),
        sources={p:sha256((stage/p).read_bytes()).hexdigest() for p in (MODEL,HELPER)},
        scope="Actual admission methods with real CPU Gloo MIN, mock CUDA properties, projection parameters on meta device, modeled routing/module capability and release identity. Both flags were freshly decided on each pass."), indent=2))
