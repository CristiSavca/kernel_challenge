"""Run the candidate's SP entry with real Gloo ranks and a CUDA stream-order oracle."""

from contextlib import ExitStack, nullcontext
from datetime import timedelta
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
from torch.nn import functional as F


CANDIDATES = Path(__file__).resolve().parent.parent
reuse = runpy.run_path(str(next(CANDIDATES.glob("v113*/cpu_check.py"))))
SOURCE = Path(sys.argv[1]) / "python/sglang/srt"
load, install = reuse["load"], reuse["install"]


class Stream:
    def __init__(self, index):
        self.index = index
        self.time = [0, 0]

    def wait_stream(self, other):
        self.time = list(map(max, self.time, other.time))

    def tick(self):
        self.time[self.index] += 1
        return tuple(self.time)


class Accesses:
    """Check that the two logical CUDA streams order every conflicting use."""
    def __init__(self):
        self.compute = self.active = Stream(0)
        self.transfer = Stream(1)
        self.clear()

    def clear(self):
        self.writes = {}
        self.reads = {}
        self.exchanges = 0
        self.projected = []

    def visit(self, tensor, write):
        key = tensor.data_ptr()
        now = self.active.tick()
        conflicts = [self.writes.get(key, (0, 0))]
        if write:
            conflicts += self.reads.get(key, [])
            self.writes[key] = now
            self.reads[key] = []
        else:
            self.reads.setdefault(key, []).append(now)
        assert all(all(a <= b for a, b in zip(prior, now)) for prior in conflicts), (
            "stream data race", key, conflicts, now
        )

    def exchange_scope(self, stream):
        return patch.object(self, "active", stream)


def worker(rank, size, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=size, timeout=timedelta(seconds=90))
    torch.set_num_threads(1)
    group = reuse["Group"](size, rank)
    comm = group.pynccl_comm
    trace = Accesses()
    parallel = lambda: NS(tp_group=group, attn_tp_group=group)
    backend = NS(is_cutedsl=lambda: False, is_gemv=lambda: False)
    operators = []
    project = F.linear

    def linear(x, w, b=None):
        assert trace.active is trace.compute
        trace.visit(x, False)
        trace.projected.append(x.data_ptr())
        return project(x, w, b)

    def dispatch(x, w, b):
        operators.append(True)
        return linear(x, w, b)

    install("sglang.srt.layers.quantization.unquant", {
        "bf16_gemm_dispatch": dispatch, "UnquantizedLinearMethod": reuse["LinearMethod"],
        "get_bf16_gemm_backend": lambda: backend,
    })
    install("sglang.srt.batch_invariant_ops", {"is_batch_invariant_mode_enabled": lambda: False})
    install("sglang.srt.environ", {"envs": NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda: False))})

    def register(fake_impl):
        def decorate(fn):
            op = torch.library.custom_op("ring_fixture::" + fn.__name__, mutates_args=(), device_types="cpu")(fn)
            op.register_fake(fake_impl)
            return op
        return decorate

    env = dict(torch=torch, F=NS(linear=linear), get_parallel=parallel, register_custom_op=register,
               get_exec=lambda: NS(deterministic=NS(enable_deterministic_inference=False)))
    load(SOURCE / "layers/communication/sp_ring_qkv.py",
         ("enable_ring_qkv", "_exchange", "ring_qkv_eligible", "_ring_shape", "ring_qkv"), env)
    install("sglang.srt.layers.communication.sp_ring_qkv", env)
    meta = {"n": 0}
    routes = dict(torch=torch, get_parallel=parallel, sp_num_tokens=lambda:meta["n"],
                  sp_fused_matmul_eligible=lambda _: False)
    load(SOURCE / "layers/layernorm_sp.py", ("sp_exit_gather", "column_parallel_g_matmul"), routes)
    layer = NS(quant_method=reuse["LinearMethod"](), bias=None, gather_output=False,
               weight=torch.ones((8,32), dtype=torch.bfloat16))
    send, recv = comm.send, comm.recv

    def send_checked(x, peer):
        assert peer == (rank + 1) % size
        trace.visit(x, False)
        send(x, peer)

    def recv_checked(x, peer):
        assert peer == (rank - 1) % size
        trace.visit(x, True)
        recv(x, peer)
        trace.exchanges += 1

    __import__("torch._dynamo")  # Initialize device interfaces before swapping in CPU stream stand-ins.
    with ExitStack() as stack:
        for obj, name, value in (
            (torch.cuda, "get_device_capability", lambda dev:(9, 0)),
            (torch.cuda, "Stream", lambda device:trace.transfer),
            (torch.cuda, "current_stream", lambda dev:trace.active),
            (torch.cuda, "stream", trace.exchange_scope),
            (comm, "send", send_checked), (comm, "recv", recv_checked),
        ):
            stack.enter_context(patch.object(obj, name, value))
        for why in ("different_method", "peer_unavailable", "requires_grad", "wrong_gpu"):
            bad = NS(**vars(layer))
            if rank == 0:
                if why == "different_method":
                    bad.quant_method = object()
                if why == "peer_unavailable":
                    comm.available = False
                if why == "requires_grad":
                    bad.weight = layer.weight.clone().requires_grad_(True)
            context = patch.object(torch.cuda, "get_device_capability", lambda dev:(8, 0)) if why == "wrong_gpu" and rank == 0 else nullcontext()
            with context:
                env["enable_ring_qkv"]([bad], group)
            assert trace.exchanges == 0 and not hasattr(bad, "_sp_ring_qkv")
            comm.available = True
        env["enable_ring_qkv"]([layer], group)
        assert trace.exchanges == 1 and layer._sp_ring_qkv
        for rows, h, n, drop, kind in ((511,64,8,0,"small"), (512,32,16,size-1,"strided"),
                                       (523,96,13,1,"dispatch"), (513,48,28,0,"normal"),
                                       (512,16,2304,0,"wide"), (513,32,8,1,"bias")):
            trace.clear()
            comm.sent = group.allgathers = 0
            operators.clear()
            backend.is_gemv = lambda: kind == "dispatch"
            gen = torch.Generator().manual_seed(490 + n)
            x = torch.randint(-8,9,(size*rows,h),generator=gen).bfloat16()/8
            gen.manual_seed(85 + rank)
            layer.weight = torch.randint(-8,9,(n,h),generator=gen).bfloat16()/8
            local = x.narrow(0,rank*rows,rows)
            if kind == "strided":
                tmp = local.new_zeros((rows,h*2))
                tmp[:,::2] = local
                local = tmp[:,::2]
            bias = torch.ones(n,dtype=x.dtype) if kind == "bias" else None
            meta["n"] = rows*size - drop
            got = routes["column_parallel_g_matmul"](layer,local,bias)
            assert torch.equal(got, project(x[:meta["n"]],layer.weight,bias)), (rank,kind)
            use = kind in ("strided", "dispatch", "normal")
            assert group.allgathers == int(not use)
            assert trace.exchanges == ((size-1) if use else 0)
            assert comm.sent == trace.exchanges * rows * h * 2
            assert len(operators) == (size if kind == "dispatch" else 0)
            if use:
                assert len(trace.projected) == size and len(set(trace.projected)) == 3
                assert trace.compute.time[1] >= trace.transfer.time[1]
            assert comm.disabled and comm.ops is None
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            x = torch.empty((2048,8192),dtype=torch.bfloat16)
            w = torch.empty((2304,8192),dtype=x.dtype)
            out = env["ring_qkv"](x,w,8,False)
            assert out.shape == (16384,2304) and out.dtype == x.dtype
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo0" if sys.platform == "darwin" else "lo")
    with tempfile.TemporaryDirectory() as tmp:
        for size in (4,8):
            mp.spawn(worker,args=(size,str(Path(tmp)/str(size))),nprocs=size,join=True)
    print(json.dumps(dict(torch=torch.__version__,real_gloo_tp=[4,8],rank_cases=72,
                          eligible_rank_cases=36,rank_startup_rejections=48,
                          native_channels=reuse["native_channels"](),graph_shape=[16384,2304],
                          proof="CPU BF16 reference, rank and byte routing, vector-clock stream hazard checks",
                          cuda_execution=False,nccl_profile=False),indent=2))
