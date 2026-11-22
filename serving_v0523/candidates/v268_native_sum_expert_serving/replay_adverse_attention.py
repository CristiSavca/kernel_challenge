"""Repeat every V267 precision case on the frozen V268 native output path."""
import ast
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent/"v267_pipelined_expert_serving"
TEST = runpy.run_path(str(CONTROL/"attention_pipeline_check.py"))
base, SRT = TEST["base"], TEST["SRT"]
QKV = SRT/"layers/communication/sp_half_qkv.py"
SP = SRT/"layers/layernorm_sp.py"
frozen = json.loads((CONTROL/"attention_pipeline.json").read_bytes())


def worker(rank, size, tmp):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(tmp)/f"tp{size}"),rank=rank,
                            world_size=size,timeout=timedelta(minutes=4))
    sched=TEST["prior"]["Schedule"]()
    group=TEST["prior"]["Group"](size,rank,sched)
    group.width=256
    group.mode="ar"
    os.environ["NCCL_NVLS_ENABLE"]="1"
    load,install,Method=base["load"],base["install"],base["LinearMethod"]
    par=lambda:NS(tp_group=group,attn_tp_group=group)
    def register(fake_impl):
        def decorate(fn):
            op=torch.library.custom_op("v268_replay::"+fn.__name__,mutates_args=(),device_types="cpu")(fn)
            op.register_fake(fake_impl)
            return op
        return decorate
    qenv=dict(torch=torch,get_parallel=par,get_lora=lambda:NS(enable_lora=False),
              get_exec=lambda:NS(deterministic=NS(enable_deterministic_inference=False)),register_custom_op=register)
    load(QKV,[n.name for n in ast.parse(QKV.read_bytes()).body if isinstance(n,ast.FunctionDef)],qenv)
    install("sglang.srt.layers.communication.sp_half_qkv",qenv)
    backend=NS(is_cutedsl=lambda:False,is_gemv=lambda:False)
    install("sglang.srt.layers.quantization.unquant",dict(UnquantizedLinearMethod=Method,get_bf16_gemm_backend=lambda:backend))
    install("sglang.srt.batch_invariant_ops",dict(is_batch_invariant_mode_enabled=lambda:False))
    install("sglang.srt.environ",dict(envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False))))
    tokens=512*size-(size-1)
    sp=dict(torch=torch,os=os,get_parallel=par,sp_num_tokens=lambda:tokens,
            sp_fused_matmul_eligible=lambda _:False,ceil_align=lambda n,s:((n+s-1)//s)*s)
    load(SP,("column_parallel_g_matmul","row_parallel_gbar_matmul","sp_exit_gather","reduce_scatter_token_rows"),sp)
    install("sglang.srt.layers.layernorm_sp",sp)
    oldsp=dict(sp)
    load(Path(tmp)/"layernorm_sp.py",("row_parallel_gbar_matmul","reduce_scatter_token_rows"),oldsp)
    oldop=dict(torch=torch)
    load(Path(tmp)/"sp_attention_return.py",("striped_return_eligible","striped_attention_return"),oldop)
    oldmod=NS(**oldop)
    cases=sorted([c for c in frozen["precision_cases"] if (c["tp"],c["rank"])==(size,rank)],
                 key=lambda c:(c["mode"], c["q_fused"]))
    counts=dict(cases=0,positive_control_mismatches=0,checked_qkv=0,checked_owner=0,QKV_different=0,
                owner_different=0,vocabulary_rows=0,fixed_head_changed=0,midpoint_cases=0,
                midpoint_argmax_changes=0,full_reductions=0,QKV_half_calls=0)
    samples=[]
    __import__("torch._dynamo")
    with ExitStack() as mocks,torch.no_grad():
        for obj,name,fn in ((torch.cuda,"get_device_capability",lambda dev:(9,0)),(torch.cuda,"Stream",lambda device:sched.transfer),
                            (torch.cuda,"current_stream",lambda device:sched.current),(torch.cuda,"stream",sched.use)):
            mocks.enter_context(patch.object(obj,name,fn))
        for c in cases:
            g=torch.Generator().manual_seed(26700+rank)
            x=torch.randint(-2,3,(512,32),generator=g).to(torch.bfloat16)/8
            q=NS(weight=torch.randint(-2,3,(32,32),generator=g).to(torch.bfloat16)/8,
                 quant_method=Method(),gather_output=False,bias=None)
            o=NS(weight=torch.randint(-2,3,(256,16),generator=g).to(torch.bfloat16)/8,
                 tp_group=group,quant_method=Method(),bias=None,_sp_unfused_collective=True)
            if c["mode"]=="ar" and not c["q_fused"] and rank==0: q.gather_output=True
            qenv["enable_half_qkv"]([q],group)
            q.gather_output=False
            if c["mode"]=="q_reload": q.quant_method=Method()
            q_native=NS(**{k:v for k,v in vars(q).items() if not k.startswith("_sp_")})
            group.pynccl_comm.group_shapes.clear()
            refq=sp["column_parallel_g_matmul"](q_native,x,None)
            qout=sp["column_parallel_g_matmul"](q,x,None)
            assert len(group.pynccl_comm.group_shapes)==2*int(c["q_fused"])
            def bridge(y):
                return (y[:,:16]*torch.sigmoid(y[:,16:])).contiguous()
            native=oldsp["row_parallel_gbar_matmul"](o,bridge(refq),None).clone()
            before=NS(**vars(o),_sp_return_stream=sched.transfer,_sp_return_method=o.quant_method)
            with patch.dict(sys.modules,{"sglang.srt.layers.layernorm_sp":NS(**oldsp),
                            "sglang.srt.layers.communication.sp_attention_return":oldmod}), \
                 patch.object(oldmod,"striped_return_eligible",lambda n,w:w>=256):
                striped=oldsp["row_parallel_gbar_matmul"](before,bridge(refq),None)
            old_count=int((striped.view(torch.int16)!=native.view(torch.int16)).sum())
            assert old_count==c["owner_differences"],(size,rank,c["mode"],old_count,c["owner_differences"])
            sched.trace.clear()
            got=sp["row_parallel_gbar_matmul"](o,bridge(qout),None)
            assert sched.trace==[("ar","compute",256)]
            readout=TEST["precision_readout"](native,got)
            counts["cases"]+=1
            counts["full_reductions"]+=1
            counts["positive_control_mismatches"]+=old_count
            counts["checked_qkv"]+=qout.numel()
            counts["checked_owner"]+=got.numel()
            counts["QKV_different"]+=int((qout.view(torch.int16)!=refq.view(torch.int16)).sum())
            counts["owner_different"]+=int((got.view(torch.int16)!=native.view(torch.int16)).sum())
            counts["vocabulary_rows"]+=readout["fixed_head_compared_rows"]
            counts["fixed_head_changed"]+=readout["fixed_head_argmax_different"]
            counts["QKV_half_calls"]+=int(c["q_fused"])
            if "constructed_two_logit_example" in c:
                t=c["constructed_two_logit_example"]
                row,feature=t["owner_row"],t["feature"]
                assert float(native[row,feature])==t["old_value"] and float(striped[row,feature])==t["new_value"]
                previous=int(t["constant_competitor"]>float(striped[row,feature]))
                current=int(t["constant_competitor"]>float(got[row,feature]))
                assert previous!=t["old_argmax"] and previous==t["new_argmax"]
                counts["midpoint_cases"]+=1
                counts["midpoint_argmax_changes"]+=int(current!=t["old_argmax"])
                if not samples: samples.append(dict(tp=size,rank=rank,**t,corrected_value=float(got[row,feature]),corrected_argmax=current))
        assert counts["QKV_different"]==counts["owner_different"]==counts["fixed_head_changed"]==counts["midpoint_argmax_changes"]==0
    (Path(tmp)/f"r-{size}-{rank}.json").write_text(json.dumps(dict(counts=counts,samples=samples)))
    dist.destroy_process_group()


if __name__=="__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo0" if sys.platform=="darwin" else "lo")
    with tempfile.TemporaryDirectory() as tmp:
        standalone=TEST["reconstruct_return_control"](tmp)
        for size in (4,8): mp.spawn(worker,args=(size,tmp),nprocs=size,join=True)
        result=[json.loads(p.read_bytes()) for p in sorted(Path(tmp).glob("r-*.json"))]
    aggregate={k:sum(r["counts"][k] for r in result) for k in result[0]["counts"]}
    assert aggregate["cases"]==len(frozen["precision_cases"])
    assert aggregate["positive_control_mismatches"]==frozen["owner_different"]
    print(json.dumps(dict(patch_sha256=hashlib.sha256((HERE/"patch.diff").read_bytes()).hexdigest(),
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (QKV,SP)},
        frozen_V267_receipt_sha256=hashlib.sha256((CONTROL/"attention_pipeline.json").read_bytes()).hexdigest(),
        reconstructed_V264_patch_sha256=standalone,**aggregate,
        representative_midpoints=[s for r in result for s in r["samples"]][:2],physical_H200=False,
        scope="Exact V267 adverse seeds, nonlinear bridge, TP4/8 ranks, full vs striped Gloo controls, fixed 64-logit readout and original constructed competitors."),indent=2))
