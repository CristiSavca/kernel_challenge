"""Run the V150 projection through the real SP path with four and eight Gloo ranks."""

import ast
from contextlib import ExitStack, nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import weakref
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

CANDIDATES = Path(__file__).resolve().parent.parent
STAGE = Path(sys.argv[1]) / "python/sglang/srt"
PINNED = CANDIDATES.parent / "sglang/python/sglang/srt"
PRODUCER = STAGE / "layers/communication/sp_owner_projection.py"
ring = runpy.run_path(str(CANDIDATES / "v138_pipelined_qkv_ring/cpu_check.py"))
reuse = ring["reuse"]
install, load = reuse["install"], reuse["load"]


def reduce_fp32(parts):
    total = torch.zeros_like(parts[0], dtype=torch.float32)
    for source in parts:
        total.add_(source.float())
    return total.to(parts.dtype)


def worker(rank, size, tmp):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(tmp) / str(size)), rank=rank,
                            world_size=size, timeout=timedelta(seconds=90))
    group = reuse["Group"](size, rank)
    group.use_symmetric_memory = lambda *args: nullcontext()
    group.is_symmetric_memory_enabled = lambda: False
    group._can_use_flashinfer_allreduce = lambda x: False
    group._resolve_outplace_all_reduce_method = lambda x: "pynccl"
    comm = group.pynccl_comm
    trace = ring["Accesses"]()
    paths = {"p2p": 0, "allreduce": 0, "dispatch": 0, "linear": 0}
    samples = []
    send_times = {}
    premature_releases = []
    releases = []
    flags = NS(sp_active=True)
    parallel = lambda: NS(tp_group=group, attn_tp_group=group)
    runtime = NS(deterministic=NS(enable_deterministic_inference=False))
    backend = NS(is_gemv=lambda: False, is_cutedsl=lambda: False)

    def all_ranks(x):
        received = [torch.empty_like(x) for _ in range(size)]
        dist.all_gather(received, x.contiguous())
        return torch.stack(received)

    def allreduce(x):
        paths["allreduce"] += 1
        return reduce_fp32(all_ranks(x))

    group.all_reduce = allreduce
    def release(key):
        sent = send_times.pop(key, None)
        if sent is None:
            return
        releases.append(key)
        if any(a>b for a,b in zip(sent, trace.compute.time)):
            premature_releases.append((sent,tuple(trace.compute.time)))

    def project(x, w, bias=None):
        assert trace.active is trace.compute
        result = F.linear(x, w, bias)
        trace.visit(result, True)
        weakref.finalize(result,release,id(result))
        paths["linear"] += 1
        return result

    def dispatch(x, w, bias):
        paths["dispatch"] += 1
        return project(x, w, bias)

    def finish(partials, output):
        assert trace.active is trace.compute
        for source in range(size):
            trace.visit(partials[source], False)
        output.copy_(reduce_fp32(partials))

    def register(fake_impl):
        def wrap(fn):
            op = torch.library.custom_op("owner_fixture::" + fn.__name__, mutates_args=(),
                                         device_types="cpu")(fn)
            op.register_fake(fake_impl)
            return op
        return wrap

    install("sglang.srt.batch_invariant_ops", {"is_batch_invariant_mode_enabled":lambda: False})
    install("sglang.srt.environ", {"envs": NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda: False))})
    install("sglang.srt.layers.quantization.unquant", {
        "UnquantizedLinearMethod": reuse["LinearMethod"], "get_bf16_gemm_backend": lambda: backend,
        "bf16_gemm_dispatch": dispatch})
    ns = dict(torch=torch, F=NS(linear=project), get_parallel=parallel, get_exec=lambda:runtime,
              register_custom_op=register, _finish_partials=finish)
    load(PRODUCER, ("enable_owner_projection", "owner_projection_eligible", "_exchange",
                    "_owner_shape", "_project_stripe", "projection_stripe_width", "owner_projection"), ns)
    owner = install("sglang.srt.layers.communication.sp_owner_projection", ns)
    row = dict(torch=torch, os=os, get_parallel=parallel, ceil_align=lambda x,n: (x+n-1)//n*n,
               sp_fused_matmul_eligible=lambda fc:False)
    load(STAGE / "layers/layernorm_sp.py", ("row_parallel_gbar_matmul","reduce_scatter_token_rows"), row)
    linear_src = (PINNED / "layers/linear.py").read_text()
    tree = ast.parse(linear_src)
    cls = next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=="RowParallelLinear")
    fwd = next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=="forward")
    env = dict(get_group_rank_size=lambda group:(rank,size),get_forward=lambda:flags,
               layernorm_sp=NS(**row),use_symmetric_memory=lambda *a,**kw:nullcontext(),
               is_allocation_symmetric=lambda:False,get_parallel=parallel,
               PrefillStageProfile=NS(lap_scoped=lambda _:None),
               should_skip_mlp_all_reduce=lambda:False)
    exec("from __future__ import annotations\n"+ast.unparse(fwd),env)
    layer = NS(quant_method=reuse["LinearMethod"](), weight=torch.empty((16,16),dtype=torch.bfloat16),
               tp_group=group,bias=None,skip_bias_add=False,input_is_parallel=True,
               _sp_unfused_collective=True,use_dp_attention_reduce=False,reduce_results=False,
               use_decode_attn_tp=False)
    send, recv = comm.send, comm.recv
    def send_checked(x, peer):
        assert trace.active is trace.transfer
        trace.visit(x, False)
        send_times[id(x)] = tuple(trace.transfer.time)
        return send(x, peer)
    def recv_checked(x, peer):
        assert trace.active is trace.transfer
        trace.visit(x, True)
        return recv(x, peer)

    import torch._dynamo as _dynamo
    with ExitStack() as st:
        for obj,key,val in ((torch.cuda,"get_device_capability",lambda dev:(9,0)),
                            (torch.cuda,"Stream",lambda device:trace.transfer),
                            (torch.cuda,"current_stream",lambda device:trace.active),
                            (torch.cuda,"stream",trace.exchange_scope),(comm,"send",send_checked),
                            (comm,"recv",recv_checked)):
            st.enter_context(patch.object(obj,key,val))
        for problem in ("method","autocast","deterministic","unavailable","ranks","weight_dtype","params"):
            bad=NS(**vars(layer))
            with ExitStack() as one:
                if rank==size-1:
                    if problem=="method": bad.quant_method=object()
                    elif problem=="autocast": one.enter_context(patch.object(torch,"is_autocast_enabled",return_value=True))
                    elif problem=="deterministic": one.enter_context(patch.object(torch,"are_deterministic_algorithms_enabled",return_value=True))
                    elif problem=="unavailable": one.enter_context(patch.object(comm,"available",False))
                    elif problem=="ranks": bad.tp_group=NS(ranks=group.ranks[::-1])
                    elif problem=="weight_dtype": bad.weight=bad.weight.float()
                    elif problem=="params": bad.weight=bad.weight.clone().requires_grad_()
                ns["enable_owner_projection"]([bad],group)
            assert not getattr(bad,"_sp_owner_projection",False) and not comm.group_shapes
        ns["enable_owner_projection"]([layer],group)
        send_times.clear()  # scalar connection warmup does not use the GEMM allocation path
        assert layer._sp_owner_projection and len(comm.group_shapes)==size-1
        eligible=owner.owner_projection_eligible
        assert eligible(512*size,8192,size) and not eligible(512*size-1,8192,size)
        assert not eligible(512*size,1,size)
        assert ns["projection_stripe_width"](16384,8192)==4096
        for kind,rows,k,n in (("normal",64,48,64),("tail",67,45,37),("strided",65,32,48),
                             ("dispatch",76,16,31),("stock-small",64,16,13),("float",66,32,17),
                             ("full-width",17,16,8192),("full-k",34,1024,16),("non-sp",62,16,13),
                             ("later-autocast",56,16,12),("later-deterministic",61,16,12)):
            trace.clear()
            releases.clear()
            comm.group_shapes.clear()
            paths.update(dict.fromkeys(paths,0))
            flags.sp_active = kind!="non-sp"
            dtype=torch.float32 if kind=="float" else torch.bfloat16
            gen=torch.Generator().manual_seed(8000+rank*19+rows)
            def rand(shape):
                return torch.randint(-4,5,shape,dtype=torch.int8,generator=gen).to(dtype)/8
            x=rand((rows,k));layer.weight=rand((n,k))
            if kind=="strided":
                buf=torch.zeros((rows,k*2),dtype=dtype);buf[:,::2]=x;x=buf[:,::2]
            direct=kind not in ("stock-small","float","non-sp","later-autocast","later-deterministic")
            backend.is_gemv=lambda:kind=="dispatch"
            full=F.linear(x,layer.weight)
            if flags.sp_active:
                merged=reduce_fp32(all_ranks(full))
                merged=F.pad(merged,(0,0,0,(-rows)%size))
                expected=merged.chunk(size)[rank]
            else:
                expected=full
            trace.clear()
            run_gate=eligible if kind=="stock-small" else lambda *args:True
            with ExitStack() as ctx:
                ctx.enter_context(patch.object(owner,"owner_projection_eligible",run_gate))
                if kind=="later-autocast":ctx.enter_context(patch.object(torch,"is_autocast_enabled",return_value=True))
                if kind=="later-deterministic":ctx.enter_context(patch.object(torch,"are_deterministic_algorithms_enabled",return_value=True))
                stripe = 12 if kind in ("tail","dispatch") else n
                ctx.enter_context(patch.dict(ns,projection_stripe_width=lambda *args: stripe))
                actual,bias=env["forward"](layer,x)
            assert bias is None and torch.equal(expected,actual),(rank,kind,actual.shape,expected.shape)
            nstripes=(n+stripe-1)//stripe
            assert len(comm.group_shapes)==((size-1)*nstripes if direct else 0),(rank,kind,comm.group_shapes)
            assert paths["allreduce"]==int(not direct and flags.sp_active)
            assert paths["dispatch"]==(size*nstripes if kind=="dispatch" else 0)
            for idx,pair in enumerate(comm.group_shapes):
                step=1+idx%(size-1)
                assert pair[0][:2]==("send",(rank+step)%size)
                assert pair[1][:2]==("recv",(rank-step)%size)
            assert comm.disabled and comm.ops is None
            assert not premature_releases and len(releases)==len(comm.group_shapes),(rank,kind,releases,premature_releases)
            assert trace.compute.time[1]>=trace.transfer.time[1]
            samples.append({"case":kind,"values":actual.numel(),"groups":len(comm.group_shapes)})
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            x=torch.empty((16384,1024),dtype=torch.bfloat16)
            w=torch.empty((8192,1024),dtype=x.dtype)
            out=ns["owner_projection"](x,w,size,False)
            assert out.shape==(16384//size,8192) and out.dtype==torch.bfloat16
    (Path(tmp)/f"rank-{size}-{rank}.json").write_text(json.dumps(samples))
    dist.destroy_process_group()


if __name__=="__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo0" if sys.platform=="darwin" else "lo")
    with tempfile.TemporaryDirectory() as tmp:
        for size in (4,8):
            mp.spawn(worker,args=(size,tmp),nprocs=size,join=True)
        rows=[v for f in Path(tmp).glob("rank-*.json") for v in json.loads(f.read_text())]
    print(json.dumps({"torch":torch.__version__,"source_sha256":hashlib.sha256(PRODUCER.read_bytes()).hexdigest(),
                      "tp":[4,8],"rank_cases":len(rows),"startup_asymmetric_cases":7*12,
                      "floating_values":sum(r["values"] for r in rows),
                      "p2p_exchange_groups":sum(r["groups"] for r in rows),
                      "ordered_send_buffer_releases":sum(r["groups"] for r in rows),
                      "rows":["threshold validated separately","small CPU rows forced through actual owner operation"],
                      "collective":"real Gloo grouped sends/receives, emulated PyNCCL adapter",
                      "finalizer":"ordered FP32 sum of BF16 rank partials on CPU",
                      "accelerator":"CUDA streams replaced with vector-clock conflict checks; GPU not executed"},indent=2))
