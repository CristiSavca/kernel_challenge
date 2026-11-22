"""Compare complete Muse decoder-to-MoE calls and rank-major projection gathers on CPU Gloo."""
import argparse
import ast
import contextlib
from datetime import timedelta
from enum import IntEnum, auto
from functools import lru_cache
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch, PropertyMock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
SRT = "python/sglang/srt/"
MODEL = SRT + "models/muse_spark.py"
GROUP = SRT + "distributed/parallel_state.py"
FB = SRT + "model_executor/forward_batch_info.py"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
SHAPES = {"pool": (4096,8192), "unpool": (8192,4096)}


def sources(candidate,stage):
    template=HERE.parent/"v217_captured_verify_context/independent_model_entry.py"
    spec=importlib.util.spec_from_file_location("v228_reader",template)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    old=sys.argv
    sys.argv=[str(template)]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv=old
    mod.SRT=""
    mod.PARENT=(HERE.parent/"v224_packed_prefill_serving/patch.diff").read_text()
    mod.CHILD=(Path(candidate)/"patch.diff").read_text()
    @lru_cache(None)
    def source(path, child):
        if child and stage is not None and (Path(stage)/path).exists():
            raw=(Path(stage)/path).read_text()
            if mod.CHILD:
                assert raw==mod.read_source(path,True),path
            return raw
        return mod.read_source(path,child and bool(mod.CHILD))
    return source


def select(source,path,parent,names,env,child):
    tree=ast.parse(source(path,child))
    body=tree.body if parent is None else next(n.body for n in tree.body if getattr(n,"name",None)==parent)
    for name in names:
        node=next(n for n in body if getattr(n,"name",None)==name)
        exec("from __future__ import annotations\n"+ast.unparse(node),env)
    return {name:env[name] for name in names}


def init_weights(folder):
    for key,(out,inp) in SHAPES.items():
        p=Path(folder)/key
        with p.open("wb") as f:
            f.truncate(out*inp*2)
        w=torch.from_file(str(p),shared=True,size=out*inp,dtype=torch.bfloat16).view(out,inp)
        idx=torch.arange(out)
        # Two powers-of-two support columns make a mathematically exact full-K
        # dot possible for every output feature, while also identifying shards.
        w[:,0]=((idx*3)%31-15).bfloat16()
        w[:,1]=((idx*5)%29-14).bfloat16()
        del w


class Experts:
    use_humming=True
    use_w4a16=True
    post_expert_norm=None

    def __init__(self, state):
        self.st=state

    def __call__(self, pooled, scores, ids, *_args, ar_workspace=None, **kwargs):
        self.st["pooled"]=pooled.clone()
        self.st["route"]=ids.clone()
        self.st["grouped"]=kwargs.get("grouped_decode")
        y=torch.zeros_like(pooled)
        idx = torch.arange(pooled.shape[0])
        y[:,0]=(pooled[idx,(self.st["rank"]*127+idx)%pooled.shape[1]].to(torch.int32)%11).bfloat16()
        y[:,1]=(pooled[idx,(self.st["rank"]*257+2*idx)%pooled.shape[1]].to(torch.int32)%7).bfloat16()
        return self.st["reduce"](y,"workspace") if ar_workspace is not None else y


class Router(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.st=st
        self.weight=torch.nn.Parameter(torch.empty(256,8192),requires_grad=False)

    def forward(self,x):
        self.st["router"]="full"
        offsets=torch.arange(256).view(1,256)
        return (x[:,0:1].float()+offsets).fmod(67)

    def forward_split2(self,x):
        self.st["router"]="split2"
        return self.forward_logits(x)

    def forward_logits(self,x):
        offsets=torch.arange(256).view(1,256)
        return (x[:,0:1].float()+offsets).fmod(67)


def install(path, **bindings):
    mod=ModuleType(path)
    mod.__dict__.update(bindings)
    sys.modules[path]=mod
    return mod


def capture(source,group,rank,world,mode,weights):
    flag=NS(get=lambda:True)
    invariant=NS(value=False)
    environ=NS(SGLANG_ENABLE_MUSE_SPARK_DECODE_COLUMN_GATHER=flag,
               SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False))
    parallel=NS(tp_group=group,tp_size=world,moe_ep_size=world,attn_tp_size=world,
                moe_tp_size=1,attn_cp_size=1,attn_dcp_size=1,attn_dp_enabled=False,attn_tp_group=group)
    exec_cfg=NS(deterministic=NS(enable_deterministic_inference=False),graph=NS(enable_cudagraph_gc=False,debug_cuda_graph=False))
    install("sglang.srt.batch_invariant_ops",is_batch_invariant_mode_enabled=lambda:invariant.value)
    install("sglang.srt.environ",envs=environ)
    install("sglang.srt.runtime_context",get_exec=lambda:exec_cfg,get_parallel=lambda:parallel)
    class Full:
        def capture_session(self,stream): return contextlib.nullcontext()
    install("sglang.srt.model_executor.runner_backend.full_cuda_graph_backend", FullCudaGraphBackend=Full)
    ns=dict(torch=torch)
    guard=select(source,SRT+"layers/communication/muse_spark_gemm_ag.py",None,["can_capture_decode_columns"],ns,True)
    install("sglang.srt.layers.communication.muse_spark_gemm_ag",**guard)
    topmethod=select(source,MODEL,"MuseSparkTextForCausalLM",["can_capture_decode_columns"],ns,True)
    target=type("Target",(torch.nn.Module,),topmethod)()
    target.model=torch.nn.Module()
    target.model.language_model=body=torch.nn.Module()
    body.layers=torch.nn.ModuleList([torch.nn.Module()])
    layer=body.layers[0]
    layer.is_moe_layer=True
    layer.mlp=moe=torch.nn.Module()
    moe.nvfp4_checkpoint=True
    for key,name in [("pool","pre_expert_proj"),("unpool","post_expert_proj")]:
        w=weights[key]
        linear=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,device="meta")
        linear.weight=torch.nn.Parameter(w,requires_grad=False)
        setattr(moe,name,linear)
    target.eval()
    # Muse lazily registers a per-layer router after loader .eval() during warmup.
    class Late(torch.nn.Module): pass
    # Ordinary Breakable prefill restores the bound class method by assignment.
    body.forward=body.forward
    null=contextlib.nullcontext
    rns=dict(empty_context=null,freeze_gc=lambda *_:null(),graph_capture=lambda **_:null(NS(stream=object())),
             get_or_create_global_graph_capture_stream=object,get_exec=lambda:exec_cfg,
             forward_context=lambda *a,**k:null(),ForwardContext=lambda **k:NS(**k))
    path=SRT+"model_executor/runner/decode_cuda_graph_runner.py"
    methods=select(source,path,"DecodeCudaGraphRunner",["capture","_capture","capture_one_shape"],rns,True)
    runner=type("Runner",(),methods)()
    runner.backend=Full()
    runner.capture_forward_mode=mode.TARGET_VERIFY
    runner.enable_pdmux=runner.enable_two_batch_overlap=runner.ragged_verify_mode=runner.enable_torch_compile=False
    runner.enable_profile_cuda_graph=False
    runner.model_runner=NS(model=target,is_draft_worker=False,lora_manager=None,capture_head_hooks=[],capture_tail_hooks=[])
    runner.model_context=null
    runner.seq_len_fill_value=1
    runner.buffers=NS(seq_lens=torch.empty(2),seq_lens_cpu=torch.empty(2),reset_index_buffers=lambda:None)
    runner.captured_req_width=8
    runner.warmup=lambda:setattr(moe,"_fused_routing_op",Late())
    runner.capture_prepare=lambda *a,**k:(NS(muse_decode_column_gather=False),None,None)
    captures={}
    class CaptureBoundary(Exception): pass
    def sample(fb,*,num_tokens):
        captures[num_tokens]=fb.muse_decode_column_gather
        raise CaptureBoundary
    runner.tbo_plugin=NS(capture_one_batch_size=sample)
    def stream():
        for size in (32,1):
            try: runner.capture_one_shape(size,lambda *a:None)
            except CaptureBoundary: pass
    runner._capture_one_stream=stream
    conditions=[]
    for condition in ("yes","rank_optout","rank_dtype","rank_topology","rank_order","rank_attn_dp","rank_invariant",
                      "rank_old_gpu","rank_hook","rank_override","rank_tail_hook","rank_training","draft","yes"):
        last=rank==world-1
        flag.get=lambda:not (condition=="rank_optout" and last)
        invariant.value=condition=="rank_invariant" and last
        parallel.attn_cp_size=2 if condition=="rank_topology" and last else 1
        parallel.attn_tp_group=NS(ranks=group.ranks[::-1]) if condition=="rank_order" and last else group
        parallel.attn_dp_enabled=condition=="rank_attn_dp" and last
        cc=(8,0) if condition=="rank_old_gpu" and last else (9,0)
        runner.model_runner.capture_tail_hooks=[object()] if condition=="rank_tail_hook" and last else []
        runner.model_runner.is_draft_worker=condition=="draft"
        target.training=condition=="rank_training" and last
        captures.clear()
        with contextlib.ExitStack() as stack:
            if condition=="rank_dtype" and last:
                w=moe.pre_expert_proj.weight
                q=torch.nn.Parameter(torch.empty(w.shape,device="meta",dtype=torch.float16),requires_grad=False)
                stack.enter_context(patch.object(moe.pre_expert_proj,"weight",q))
            if condition=="rank_override" and last:
                stack.enter_context(patch.object(layer,"forward",lambda x:2*x))
            if condition=="rank_hook" and last:
                h=layer.register_forward_pre_hook(lambda mod,args:args)
                stack.callback(h.remove)
            stack.enter_context(patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True))
            stack.enter_context(patch.object(torch.cuda,"is_available",return_value=True))
            stack.enter_context(patch.object(torch.cuda,"get_device_capability",return_value=cc))
            runner.capture()
        expected=condition=="yes"
        assert captures=={256:expected,8:expected},(rank,condition,captures)
        conditions.append(condition)
    return dict(captures),conditions


def worker(rank,world,folder,candidate,stage):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(folder)/f"gloo{world}"),
                            rank=rank,world_size=world,timeout=timedelta(seconds=100))
    try:
        run(rank,world,folder,candidate,stage)
    finally:
        dist.destroy_process_group()


def run(rank,world,folder,candidate,stage):
    source=sources(candidate,stage)
    class State: pass
    record=[]
    def gather_single(output,x,group=None):
        with torch.inference_mode(False):
            parts=[torch.empty(tuple(x.shape),dtype=x.dtype) for _ in range(world)]
            dist.all_gather(parts,x.clone().contiguous())
        output.view(world,*x.shape).copy_(torch.stack(parts))
    groupns=dict(torch=torch,is_shm_available=lambda *_:False,all_gather_single=gather_single)
    gdef=select(source,GROUP,"GroupCoordinator",["all_gather"],groupns,False)
    class Group:
        local_size=world
        world_size=world
        rank_in_group=rank
        ranks=list(range(world))
        cpu_group=dist.group.WORLD
        ca_comm=NS(rank=rank,world_size=world)
        device_group=dist.group.WORLD
        hpu_communicator=None
        def is_allocation_symmetric(self): return False
        def use_symmetric_memory(self,*a,**kw): return contextlib.nullcontext()
        def all_gather(self,x,dim=-1):
            state["gathers"].append((tuple(x.shape),dim))
            return gdef["all_gather"](self,x,dim)
    group=Group()
    fbenv=dict(IntEnum=IntEnum,auto=auto)
    mode=select(source,FB,None,["ForwardMode"],fbenv,False)["ForwardMode"]
    weights={}
    for key,(out,inp) in SHAPES.items():
        weights[key]=torch.from_file(str(Path(folder)/key),shared=False,size=out*inp,dtype=torch.bfloat16).view(out,inp)
    captured,votes=capture(source,group,rank,world,mode,weights)
    for call,m,condition in [
        ("DECODE",1,""),("TARGET_VERIFY",8,""),("TARGET_VERIFY",8,"native_mc"),
        ("TARGET_VERIFY",32,""),("TARGET_VERIFY",128,""),("TARGET_VERIFY",256,""),
        ("TARGET_VERIFY",32,"workspace"),("TARGET_VERIFY",32,"refused_workspace"),
        ("EXTEND",32,""),("MIXED",32,""),("DECODE",8,"hook"),("TARGET_VERIFY",32,"updated"),("TARGET_VERIFY",32,"eager"),
        ("TARGET_VERIFY",257,"cost_bound"),
    ]:
        stream=torch.zeros(m,8192,dtype=torch.float32)
        activ=torch.zeros(m,8192,dtype=torch.bfloat16)
        activ[:,0]=((torch.arange(m)%5)-2).bfloat16()
        activ[:,1]=((torch.arange(m)%7)-3).bfloat16()
        rows=[]
        if condition=="updated":
            # All ranks reload the live replicated banks in place.
            for w in weights.values():
                w[:,0].mul_(-1)
        for child in (False,True):
            state=dict(gathers=[],reductions=[],gemms=[],rank=rank,router=None,hook=0)
            def reduce(y,which="normal"):
                state["reductions"].append(which)
                z=y.clone()
                dist.all_reduce(z)
                return z
            state["reduce"]=reduce
            def gemm(x,w,bias):
                assert bias is None and x.dtype==w.dtype==torch.bfloat16
                # All omitted columns are zero in the underlying file-backed
                # actual production-size weight banks. Every output is checked.
                out=(x[:,0:1].float()*w[:,0].float()+x[:,1:2].float()*w[:,1].float()).bfloat16()
                state["gemms"].append([x.shape[1],w.shape[0],w.storage_offset()])
                return out
            def mc(_comm,x,w):
                cols=w.shape[0]//world
                state["native_mc"]=True
                local=gemm(x,w[rank*cols:(rank+1)*cols],None)
                return group.all_gather(local,dim=-1)
            getpar=lambda:NS(tp_group=group,tp_size=world,moe_ep_size=world,moe_tp_size=1,
                             attn_dp_size=1,attn_cp_size=1,attn_dcp_size=1,pp_size=1,
                             enable_dp_attention=False,enable_prefill_cp=False,attn_tp_size=world)
            ctx=NS(sp_active=False)
            runtime=dict(torch=torch,contextlib=contextlib,nn=torch.nn,get_forward=lambda:ctx,get_parallel=getpar,
                is_dp_attention_enabled=lambda:False,_is_cpu=False,get_global_experts_capturer=lambda:None,
                _use_split_bf16_router=lambda n:n>8,bf16_gemm_dispatch=gemm,gemm_ag_supported=lambda c,x,w:(
                    condition=="native_mc" and w.shape[0]==8192),gemm_ag=mc,PrefillStageProfile=NS(lap=lambda *_:None),
                tensor_model_parallel_all_reduce=reduce,is_cuda=lambda:True,is_in_breakable_cuda_graph=lambda:False)
            if child:
                install("sglang.srt.layers.quantization.unquant",bf16_gemm_dispatch=gemm)
                runtime.update(select(source,SRT+"layers/communication/muse_spark_gemm_ag.py",None,
                                      ["try_decode_column_gather"],runtime,True))
            # Cuda/fused expert math stays at the boundaries of this fixture.
            methods=["forward","_forward_tp","_forward_experts","_reduce_and_unpool","_defers_unpool_gather"]
            mo_methods=select(source,MODEL,"MuseSpark1xMoE",methods,runtime,child)
            moe=type("MoE",(),dict(mo_methods,__call__=mo_methods["forward"]))()
            moe.tp_size=world;moe.moe_tp_size=1;moe.top_k=8
            moe.nvfp4_checkpoint=moe.router_in_fp32=True
            moe.sp_replicate_expert_input=False
            moe.pre_expert_norm=None
            moe.unpool_gather_boundary=None
            moe.experts=Experts(state)
            moe.pre_expert_proj=NS(weight=weights["pool"])
            moe.post_expert_proj=NS(weight=weights["unpool"])
            moe.gate=Router(state)
            if condition=="hook" and rank==world-1:
                def hook(module, args, result):
                    state["hook"]+=1
                    return result.neg()
                moe.gate.register_forward_hook(hook)
            moe._routing_op=lambda:lambda _,scores,k,flag,*,want_packed:(scores.topk(k)[0],scores.topk(k)[1],None)
            moe.finalize_ar_workspace=(NS(max_tokens=512,can_fuse=lambda *_:condition=="workspace",fallback_logged=True)
                                       if condition in ("workspace","refused_workspace") else None)
            decoder=type("Decoder",(),select(source,MODEL,"MuseSpark1xDecoderLayer",["forward"],runtime,child))()
            decoder.self_attn=lambda pos,hidden,batch:hidden
            decoder.is_moe_layer=True;decoder.mlp=moe;decoder.operator_dtype=torch.bfloat16;decoder._router_dtype=torch.float32
            decoder._fused_attn_boundary=lambda s,x,**kwargs:(x,s,x.float())
            def after(s,y,**kw):
                assert y.shape==s.shape,(y.shape,s.shape)
                return y,s+y.float(),None
            decoder._fused_ffn_boundary=after
            enable=child and condition not in ("hook","eager") and captured[8]
            fb=NS(forward_mode=mode[call],attn_tp_sequence_sharded=False,muse_decode_column_gather=enable)
            # Fake architecture and the CUDA GEMM implementations; Gloo is real.
            with torch.inference_mode(),patch.object(torch.Tensor,"is_cuda",new_callable=PropertyMock,return_value=True), \
                 patch.object(torch.cuda,"get_device_capability",return_value=(9,0)):
                s,n=decoder.forward(None,stream,activ,fb)
            assert len(state["reductions"])==1,state
            rows.append((s,n,state))
        (s0,n0,a),(s1,n1,b)=rows
        assert torch.equal(s0,s1) and torch.equal(n0.view(torch.int16),n1.view(torch.int16)),(call,m,condition)
        assert torch.equal(a["pooled"].view(torch.int16),b["pooled"].view(torch.int16)) and torch.equal(a["route"],b["route"])
        if call in ("DECODE","TARGET_VERIFY") and condition not in ("hook","eager"):
            assert [g[1] for g in b["gemms"]]==[4096//world,8192 if m>256 else 8192//world],(m,condition,b["gemms"])
        else:
            assert [g[1] for g in b["gemms"]]==[4096,8192] and not b["gathers"]
        assert a["router"]==b["router"] and a["hook"]==b["hook"] and a["reductions"]==b["reductions"]
        record.append(dict(capture_epochs=votes if not record else [],mode=call,m=m,guard=condition or None,parent_gemms=a["gemms"],candidate_gemms=b["gemms"],
                           parent_collectives=a["gathers"],candidate_collectives=b["gathers"],reductions=b["reductions"],
                           router=b["router"],hooks=b["hook"],checked_words=s0.numel()+n0.numel()+a["pooled"].numel()))
    # Independent dense random GEMMs exercise the full-K numerical implementation
    # without the sparse production-size basis used to exhaust all output lanes.
    select(source,SRT+"layers/communication/muse_spark_gemm_ag.py",None,["try_decode_column_gather"],runtime,True)
    install("sglang.srt.layers.quantization.unquant",bf16_gemm_dispatch=torch.nn.functional.linear)
    gen=torch.Generator().manual_seed(228)
    dense=[]
    for m in (1,8,16,32):
        dims=512
        x=torch.randn(m,2*dims,generator=gen).bfloat16()[:,::2]
        w=torch.randn(1024,dims,generator=gen).bfloat16()
        original=torch.nn.functional.linear(x,w)
        sharded=runtime["try_decode_column_gather"](group,x,w)
        assert sharded.shape==original.shape
        err=(sharded.float()-original.float()).abs()
        torch.testing.assert_close(sharded,original,rtol=.01,atol=.03125)
        dense.append(dict(m=m,outputs=sharded.numel(),bf16_unequal=torch.count_nonzero(
            sharded.view(torch.int16)!=original.view(torch.int16)).item(),max_abs=err.max().item()))
    words=torch.tensor([0,32768,32641,65473,rank],dtype=torch.int32).to(torch.int16).view(torch.bfloat16)[None,:]
    received=group.all_gather(words,dim=-1).view(torch.int16)
    expected=torch.cat([torch.tensor([0,32768,32641,65473,r],dtype=torch.int32).to(torch.int16) for r in range(world)])
    assert torch.equal(received[0],expected)
    (Path(folder)/f"rank{world}-{rank}.json").write_text(json.dumps({"paths":record,"dense":dense,"transport_words":received.numel()}))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate",type=Path,nargs="?",default=HERE,help="full-patch candidate directory, including an integrated version")
    ap.add_argument("--stage",type=Path,help="optionally verify staged source bytes against that candidate")
    ap.add_argument("--world",type=int,nargs="+",default=[4,8])
    args=ap.parse_args()
    torch.set_num_threads(1)
    out={}
    with tempfile.TemporaryDirectory(prefix="v228-gloo-") as folder:
        init_weights(folder)
        for size in args.world:
            mp.spawn(worker,args=(size,folder,str(args.candidate.resolve()),str(args.stage) if args.stage else None),nprocs=size,join=True)
            out[size]=[json.loads((Path(folder)/f"rank{size}-{r}.json").read_text()) for r in range(size)]
    counts=dict(paired_rank_calls=sum(len(c["paths"]) for world in out.values() for c in world),
                checked_words=sum(i["checked_words"] for world in out.values() for r in world for i in r["paths"]),
                dense_outputs=sum(i["outputs"] for w in out.values() for r in w for i in r["dense"]),
                unequal_dense_outputs=sum(i["bf16_unequal"] for w in out.values() for r in w for i in r["dense"]))
    print(json.dumps(dict(torch=torch.__version__,physical_gpu=False,counts=counts,
        candidate=args.candidate.name,patch_sha256=sha256((args.candidate/"patch.diff").read_bytes()).hexdigest(),
        parent_patch_sha256=sha256((HERE.parent/"v224_packed_prefill_serving/patch.diff").read_bytes()).hexdigest(),
        rank_zero_per_tp={g:v[0]["paths"] for g,v in out.items()},
        dense_gemm_rank_zero={g:v[0]["dense"] for g,v in out.items()},signed_zero_nan_transport_words=sum(r["transport_words"] for w in out.values() for r in w)),indent=2))
