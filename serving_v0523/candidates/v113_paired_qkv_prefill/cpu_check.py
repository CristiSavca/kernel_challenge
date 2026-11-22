"""Exercise the actual SP column entry, shape gate, P2P choreography and both GEMMs."""

import ast
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F


SOURCE = Path(sys.argv[1]) / "python/sglang/srt"
NS = types.SimpleNamespace


def install(name, attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


def load(path, names, env):
    owner = ast.parse(path.read_text())
    methods = [n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(methods)==len(names)
    exec("from __future__ import annotations\n" + ast.unparse(ast.Module(methods,[])), env)


class Comm:
    def __init__(self, rank):
        self.available = True
        self.device = torch.device("cpu")
        self.rank = rank
        self.disabled = True
        self.ops = None
        self.sent = 0
        self.group_shapes = []

    @contextmanager
    def change_state(self, enable):
        old, self.disabled = self.disabled, not enable
        try:
            yield
        finally:
            self.disabled = old

    def group_start(self):
        assert self.ops is None and not self.disabled
        self.ops = []
        self.group_shapes.append([])

    def send(self, x, peer):
        assert x.is_contiguous()
        self.sent += x.numel() * x.element_size()
        self.group_shapes[-1].append(("send",peer,tuple(x.shape)))
        self.ops.append(dist.P2POp(dist.isend,x,peer))

    def recv(self, x, peer):
        assert x.is_contiguous()
        self.group_shapes[-1].append(("recv",peer,tuple(x.shape)))
        self.ops.append(dist.P2POp(dist.irecv,x,peer))

    def group_end(self):
        assert not self.disabled
        for op in dist.batch_isend_irecv(self.ops):
            op.wait()
        self.ops = None


class Group:
    def __init__(self,size,rank):
        self.world_size=size
        self.rank_in_group=rank
        self.cpu_group=dist.group.WORLD
        self.ranks=list(range(size))
        self.pynccl_comm=Comm(rank)
        self.allgathers=0

    def all_gather_into_tensor(self, result, x):
        self.allgathers += 1
        dist.all_gather_into_tensor(result,x)


class LinearMethod:
    def apply(self,fc,x,bias):
        return F.linear(x,fc.weight,bias)


def worker(rank,size,rendezvous):
    dist.init_process_group("gloo",init_method="file://"+rendezvous,rank=rank,world_size=size,
                            timeout=timedelta(seconds=120))
    torch.set_num_threads(1)
    group=Group(size,rank)
    parallel=lambda:NS(tp_group=group,attn_tp_group=group)
    model_paths=[]
    backend=NS(is_cutedsl=lambda:False,is_gemv=lambda:False)
    def bf16_gemm_dispatch(x,w,b):
        model_paths.append((tuple(x.shape),tuple(w.shape)))
        return F.linear(x,w,b)
    unquant=install("sglang.srt.layers.quantization.unquant",{
        "bf16_gemm_dispatch":bf16_gemm_dispatch,
        "UnquantizedLinearMethod":LinearMethod,"get_bf16_gemm_backend":lambda:backend})
    install("sglang.srt.batch_invariant_ops",{"is_batch_invariant_mode_enabled":lambda:False})
    install("sglang.srt.environ",{"envs":NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False))})
    def register_custom_op(fake_impl):
        # Exercise the same input/output schema and real torch.library dispatcher on CPU.
        def decorate(fn):
            fn.__annotations__={"local":torch.Tensor,"weight":torch.Tensor,"size":int,
                                "dispatch":bool,"return":torch.Tensor}
            op=torch.library.custom_op("paired_fixture::"+fn.__name__,mutates_args=(),device_types="cpu")(fn)
            op.register_fake(fake_impl)
            return op
        return decorate
    env=dict(torch=torch,F=F,get_parallel=parallel,get_exec=lambda:NS(deterministic=NS(enable_deterministic_inference=False)),
             register_custom_op=register_custom_op)
    load(SOURCE/"layers/communication/sp_paired_qkv.py",
         ("enable_paired_qkv","paired_qkv_wins","_paired_shape","paired_qkv"),env)
    install("sglang.srt.layers.communication.sp_paired_qkv",env)
    meta={"num_tokens":0}
    routes=dict(torch=torch,get_parallel=parallel,sp_num_tokens=lambda:meta["num_tokens"],
                sp_fused_matmul_eligible=lambda _:False)
    load(SOURCE/"layers/layernorm_sp.py",("sp_exit_gather","column_parallel_g_matmul"),routes)
    linear=NS(quant_method=LinearMethod(),bias=None,gather_output=False,
              weight=torch.ones((8,64),dtype=torch.bfloat16))
    with patch.object(torch.cuda,"get_device_capability",return_value=(9,0)):
        bad=NS(quant_method=object(),bias=None,gather_output=False,weight=linear.weight)
        env["enable_paired_qkv"]([bad if rank==size-1 else linear],group)
        assert not hasattr(linear,"_sp_paired_qkv") and not group.pynccl_comm.group_shapes
        if rank==1: group.pynccl_comm.available=False
        env["enable_paired_qkv"]([linear],group)
        assert not hasattr(linear,"_sp_paired_qkv") and not group.pynccl_comm.group_shapes
        group.pynccl_comm.available=True
        if rank==0: linear.weight.requires_grad_(True)
        env["enable_paired_qkv"]([linear],group)
        assert not hasattr(linear,"_sp_paired_qkv")
        linear.weight.requires_grad_(False)
        with patch.object(torch.cuda,"get_device_capability",return_value=(8 if rank==0 else 9,0)):
            env["enable_paired_qkv"]([linear],group)
            assert not hasattr(linear,"_sp_paired_qkv")
        env["enable_paired_qkv"]([linear],group)
    assert linear._sp_paired_qkv and len(group.pynccl_comm.group_shapes)==1
    assert env["paired_qkv_wins"](2048,8192,2304,8)
    assert not env["paired_qkv_wins"](128,8192,2304,8)
    for rows,h,n,drop,kind in [(1,64,48,0,"short"),(65,64,8,1,"odd"),
                               (34,96,6,size-1,"strided"),(73,128,20,0,"kernels"),
                               (128,64,80,1,"wide"),(70,8192,7,1,"full-k")]:
        g=torch.Generator().manual_seed(30+h+n)
        x=(torch.randint(-4,5,(rows*size,h),generator=g,dtype=torch.int8).bfloat16()/4)
        g.manual_seed(441+rank)
        w=torch.randint(-4,5,(n,h),generator=g,dtype=torch.int8).bfloat16()/4
        linear.weight=w
        local=x[rank*rows:(rank+1)*rows]
        if kind=="strided":
            temp=torch.empty((rows,h*2),dtype=x.dtype)
            temp[:,::2].copy_(local)
            local=temp[:,::2]
        meta["num_tokens"]=rows*size-drop
        backend.is_cutedsl=lambda:kind=="kernels"
        comm=group.pynccl_comm
        comm.sent=0
        comm.group_shapes.clear()
        group.allgathers=0
        model_paths.clear()
        got=routes["column_parallel_g_matmul"](linear,local,None)
        ref=F.linear(x[:meta["num_tokens"]],w)
        assert torch.equal(got,ref),(rank,kind)
        # Every source row / destination head rank participates in the dense original product.
        assert got.shape==(meta["num_tokens"],n)
        use=env["paired_qkv_wins"](rows,h,n,size)
        assert use == (kind not in ("short","wide"))
        assert group.allgathers == int(not use)
        assert comm.disabled and comm.ops is None
        if use:
            half=size//2
            assert comm.sent==((half-1)*rows*h+n*h+half*rows*n)*2
            assert len(comm.group_shapes)==2
            assert len(comm.group_shapes[0])==size and len(comm.group_shapes[1])==2
            assert len(model_paths)==(2 if kind=="kernels" else 0)
    # FakeTensor shape inference for the integrated custom op at actual TP8 and QKV geometry.
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        x=torch.empty((2048,8192),dtype=torch.bfloat16)
        w=torch.empty((2304,8192),dtype=torch.bfloat16)
        got=env["paired_qkv"](x,w,8,False)
        assert got.shape==(16384,2304) and got.dtype==x.dtype
    dist.destroy_process_group()


def native_channels():
    torch.set_num_threads(2)
    g=torch.Generator().manual_seed(437)
    # The low-bit dyadic multiplicands make the FP32 full-K dot sums exact.
    x=torch.randint(-2,3,(13,8192),dtype=torch.int8,generator=g).bfloat16()/4
    w=torch.randint(-2,3,(2304,8192),dtype=torch.int8,generator=g).bfloat16()/4
    baseline=F.linear(x,w)
    blocked=torch.cat([F.linear(t,w) for t in x.split((7,6))])
    ref=F.linear(x.float(),w.float()).bfloat16()
    assert torch.equal(blocked,baseline) and torch.equal(ref,baseline)
    return {"tokens":13,"in_channels":8192,"out_channels_per_rank":2304,
            "checked_bf16_values":baseline.numel()}


if __name__ == "__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo0")
    with tempfile.TemporaryDirectory() as d:
        for size in (4,8):
            mp.spawn(worker,args=(size,str(Path(d)/str(size))),nprocs=size,join=True)
    print(json.dumps({"torch":torch.__version__,"real_gloo_tp":[4,8],"projection_cases_per_tp":6,
        "native_channel_fixture":native_channels(),"rank_local_startup_fallbacks_per_tp":4,"full_k":8192,"check":"production SP column entry and registered custom op; paired send/receive byte counts, rank-major outputs, padded/strided inputs, BF16 full-K reference, native dense/dispatched GEMM selection, warmup, and registered fake output at full geometry","h200_executed":False},indent=2))
