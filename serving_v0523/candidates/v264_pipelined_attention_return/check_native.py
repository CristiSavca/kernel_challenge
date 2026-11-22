"""Exercise each attention return through the production row-parallel entry on Gloo."""
from contextlib import ExitStack, contextmanager, nullcontext
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
STAGE = Path(sys.argv[1]).resolve()
SRT = STAGE / "python/sglang/srt"
OP = SRT / "layers/communication/sp_attention_return.py"
base = runpy.run_path(str(HERE.parent / "v113_paired_qkv_prefill/cpu_check.py"))


class Schedule:
    def __init__(self):
        self.compute, self.transfer = Stream(self, "compute"), Stream(self, "transfer")
        self.current = self.compute
        self.trace = []

    @contextmanager
    def use(self, stream):
        before, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = before


class Stream:
    def __init__(self, owner, lane):
        self.owner, self.lane = owner, lane

    def wait_stream(self, other):
        self.owner.trace.append(("wait", self.lane, other.lane))


class Group(base["Group"]):
    def __init__(self, size, rank, schedule):
        super().__init__(size, rank)
        self.mode, self.width, self.schedule = "", 0, schedule

    def _resolve_outplace_all_reduce_method(self, tensor):
        if self.mode == "switch" and tensor.shape[-1] < self.width:
            return "ca"
        return "pynccl" if self.mode == "pynccl" else None

    def _can_use_flashinfer_allreduce(self, tensor):
        return self.mode == "fi" and tensor.shape[-1] < self.width

    def is_symmetric_memory_enabled(self):
        return self.mode == "symmetric"

    def use_symmetric_memory(self, _):
        return nullcontext()

    def all_reduce(self, tensor):
        self.schedule.trace.append(("ar", self.schedule.current.lane, tensor.shape[-1]))
        if self.mode == "pynccl":
            tensor = tensor.clone()
        dist.all_reduce(tensor)
        return tensor

    def reduce_scatter_along_dim(self, tensor, dim):
        assert dim == 0
        self.schedule.trace.append(("rs", self.schedule.current.lane, tensor.shape[-1]))
        result = tensor.new_empty(tensor.shape[0] // self.world_size, tensor.shape[1])
        dist.reduce_scatter_tensor(result, tensor)
        return result


def worker(rank, size, loc):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(loc) / str(size)),
                            rank=rank, world_size=size, timeout=timedelta(minutes=4))
    schedule = Schedule()
    group = Group(size, rank, schedule)
    load, install = base["load"], base["install"]
    method = base["LinearMethod"]
    backend = NS(is_cutedsl=lambda: group.mode == "special", is_gemv=lambda: False)
    install("sglang.srt.layers.quantization.unquant", dict(UnquantizedLinearMethod=method,
                                                             get_bf16_gemm_backend=lambda: backend))
    install("sglang.srt.environ", dict(envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda: False))))
    install("sglang.srt.batch_invariant_ops", dict(is_batch_invariant_mode_enabled=lambda: False))
    install("sglang.srt.runtime_context", dict(get_lora=lambda: NS(enable_lora=False),
            get_exec=lambda: NS(deterministic=NS(enable_deterministic_inference=False))))
    layernorm = dict(torch=torch, os=os, ceil_align=lambda n, s: ((n+s-1)//s)*s)
    load(SRT / "layers/layernorm_sp.py", ("row_parallel_gbar_matmul", "reduce_scatter_token_rows"), layernorm)
    install("sglang.srt.layers.layernorm_sp", layernorm)
    env = {"torch": torch}
    load(OP, ("enable_striped_return", "striped_return_eligible", "striped_attention_return"), env)
    opmod = install("sglang.srt.layers.communication.sp_attention_return", env)
    eligible = opmod.striped_return_eligible
    real_mm = torch.mm
    layer = NS(weight=torch.empty(256, 16, dtype=torch.bfloat16), tp_group=group,
               quant_method=method(), bias=None, _sp_unfused_collective=True)
    cases = [("ar", 67, 256, 16), ("rs", 66, 256, 16), ("pynccl", 64, 256, 16),
             ("strided", 65, 256, 16), ("bias", 66, 256, 16), ("grad", 64, 256, 16),
             ("small", 31, 256, 16), ("special", 64, 256, 16), ("det", 64, 256, 16),
             ("switch", 64, 256, 16), ("fi", 64, 256, 16), ("symmetric", 64, 256, 16),
             ("reloaded_method", 65, 256, 16), ("weight_grad", 65, 256, 16), ("weight_stride", 65, 256, 16)]
    if size == 2:
        # The production byte threshold and real input/output widths without a scaled gate.
        cases.append(("real", 8192, 8192, 1024))
    checked, traced, vetoes = 0, 0, 0
    with ExitStack() as mocks, torch.no_grad():
        for obj, key, value in (
            (torch.cuda, "get_device_capability", lambda dev: (9, 0)),
            (torch.cuda, "Stream", lambda device: schedule.transfer),
            (torch.cuda, "stream", schedule.use),
            (torch.cuda, "current_stream", lambda device: schedule.current),
        ):
            mocks.enter_context(patch.object(obj, key, value))
        for veto in ("method", "topology", "no_comm"):
            bad = NS(**vars(layer))
            if rank == 0:
                if veto == "method":
                    bad.quant_method = object()
                elif veto == "topology":
                    bad.tp_group = NS(ranks=group.ranks[::-1])
                else:
                    group.pynccl_comm.available = False
            env["enable_striped_return"]([bad], group)
            assert not hasattr(bad, "_sp_return_stream")
            group.pynccl_comm.available = True
            vetoes += 1
        env["enable_striped_return"]([layer], group)
        assert layer._sp_return_stream is schedule.transfer
        for mode, rows, width, inner in cases:
            group.mode, group.width = mode, width
            layer.quant_method = method() if mode == "reloaded_method" else layer._sp_return_method
            os.environ["NCCL_NVLS_ENABLE"] = "0" if mode == "rs" else "1"
            g = torch.Generator().manual_seed(7523 + rows + rank)
            layer.weight = torch.randint(-3, 4, (width,inner), generator=g).to(torch.bfloat16) / 8
            if mode == "weight_grad":
                layer.weight.requires_grad_()
            elif mode == "weight_stride":
                layer.weight = layer.weight.t().contiguous().t()
            stride = 2 if mode == "strided" else 1
            x = torch.randint(-4, 5, (rows,inner*stride), generator=g).to(torch.bfloat16)[:, ::stride] / 4
            if stride > 1:
                tmp = torch.empty((rows, inner*stride), dtype=x.dtype)
                tmp[:,::stride] = x
                x = tmp[:,::stride]
            bias = torch.ones(width, dtype=x.dtype) if mode == "bias" else None
            x.requires_grad_(mode == "grad")
            mark = layer._sp_return_stream
            layer._sp_return_stream = None
            old = layernorm["row_parallel_gbar_matmul"](layer, x, bias)
            # Make a compact independent reference before the candidate allocates.
            expected = old.clone()
            del old
            layer._sp_return_stream = mark
            def mm(a, b, out):
                schedule.trace.append(("gemm", schedule.current.lane, out.shape[-1]))
                return real_mm(a, b, out=out)
            schedule.trace.clear()
            guard = eligible if mode == "real" else lambda n, c: n >= 32
            with patch.object(opmod, "striped_return_eligible", guard), patch.object(torch, "mm", mm), \
                 patch.object(torch, "are_deterministic_algorithms_enabled", return_value=mode == "det"):
                actual = layernorm["row_parallel_gbar_matmul"](layer, x, bias)
            assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), (rank,size,mode)
            checked += expected.numel()
            if mode in ("ar","rs","pynccl","strided","real"):
                collective = "rs" if mode == "rs" else "ar"
                assert schedule.trace == [
                    ("gemm","compute",width//2), ("wait","transfer","compute"), (collective,"transfer",width//2),
                    ("gemm","compute",width//2), ("wait","compute","transfer"), (collective,"compute",width//2),
                ], schedule.trace
                traced += 1
            else:
                assert schedule.trace == [("ar","compute",width)], schedule.trace
            del actual, expected
        # Every rank takes the same injected failure after enqueuing its first reduction.
        group.mode, group.width = "ar", 256
        layer.weight = torch.ones(256,16,dtype=torch.bfloat16)
        x = torch.ones(64, 16, dtype=torch.bfloat16)
        hit = []
        def fail(a, b, out):
            hit.append(1)
            if len(hit) == 2:
                raise ValueError("second GEMM failed")
            return real_mm(a,b,out=out)
        schedule.trace.clear()
        with patch.object(torch, "mm", fail):
            try:
                env["striped_attention_return"](x,layer.weight,group,schedule.transfer)
            except ValueError as exc:
                assert str(exc) == "second GEMM failed"
            else:
                raise AssertionError("missing exception")
        assert schedule.trace[-1] == ("wait","compute","transfer")
    (Path(loc)/f"r-{size}-{rank}.json").write_text(json.dumps(dict(cases=len(cases), fast=traced,
            words=checked, vetoes=vetoes, failure_join=1, full_native_width=bool(size==2))))
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    with tempfile.TemporaryDirectory() as tmp:
        for size in (2,4,8):
            mp.spawn(worker,args=(size,tmp),nprocs=size,join=True)
        values=[json.loads(p.read_bytes()) for p in Path(tmp).glob("r-*.json")]
    print(json.dumps(dict(source_sha256=hashlib.sha256(OP.read_bytes()).hexdigest(),
               patch_sha256=json.loads((HERE/"validation.json").read_bytes())["patch_sha256"],
               tp=[2,4,8], rank_cases=sum(v["cases"] for v in values), fast=sum(v["fast"] for v in values),
               checked_BF16=sum(v["words"] for v in values), failure_joins=sum(v["failure_join"] for v in values),
               asymmetric_vetoes=sum(v["vetoes"] for v in values), full_width_rank_cases=sum(v["full_native_width"] for v in values),
               scope="Real CPU dense BF16 and Gloo all-reduce/reduce-scatter; stream dependencies modeled. All small fixtures scale only the geometry guard.",
               physical_H200=False),indent=2))
