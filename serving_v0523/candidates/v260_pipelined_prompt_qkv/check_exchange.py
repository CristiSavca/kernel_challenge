"""Run the public SP entry with Gloo transports and checked CUDA stream dependencies."""
import ast
from contextlib import ExitStack, contextmanager
from datetime import timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
SRT = STAGE / "python/sglang/srt"
OP = SRT / "layers/communication/sp_half_qkv.py"
COLUMN = SRT / "layers/layernorm_sp.py"
base = runpy.run_path(str(HERE.parent / "v113_paired_qkv_prefill/cpu_check.py"))
NS = SimpleNamespace


class Stream:
    def __init__(self, lane):
        self.lane = lane
        self.clock = [0, 0]

    def wait_stream(self, other):
        self.clock = list(map(max, self.clock, other.clock))

    def event(self):
        self.clock[self.lane] += 1
        return tuple(self.clock)


class Schedule:
    def __init__(self):
        self.compute = self.current = Stream(0)
        self.transfer = Stream(1)
        self.accesses = []
        self.outputs = []

    @contextmanager
    def using(self, stream):
        last = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = last

    def visit(self, tensor, write):
        # Compare byte ranges so a half-batch read overlaps EVERY received shard.
        assert tensor.is_contiguous()
        first = tensor.data_ptr()
        last = first + tensor.numel() * tensor.element_size()
        now = self.current.event()
        for lo, hi, was_write, when in self.accesses:
            if max(first, lo) < min(last, hi) and (write or was_write):
                assert all(a <= b for a, b in zip(when, now)), (first, last, when, now)
        self.accesses.append((first, last, write, now))


def worker(rank, size, rendezvous, receipts):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=size,
                            timeout=timedelta(seconds=120))
    torch.set_num_threads(1)
    group = base["Group"](size, rank)
    comm = group.pynccl_comm
    events = Schedule()
    active = NS(dispatched=False, lora=False, injected=False, projections=0, dispatches=0)
    backend = NS(is_cutedsl=lambda: active.dispatched, is_gemv=lambda: False)
    parallel = lambda: NS(tp_group=group, attn_tp_group=group)

    def dispatch(x, weight, bias):
        active.dispatches += 1
        return F.linear(x, weight, bias)

    install = base["install"]
    install("sglang.srt.layers.quantization.unquant", dict(get_bf16_gemm_backend=lambda: backend,
            bf16_gemm_dispatch=dispatch, UnquantizedLinearMethod=base["LinearMethod"]))
    install("sglang.srt.batch_invariant_ops", dict(is_batch_invariant_mode_enabled=lambda: False))
    install("sglang.srt.environ", dict(envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda: False))))

    def register(fake_impl):
        def decorate(fn):
            op = torch.library.custom_op("cohort_fixture::" + fn.__name__, mutates_args=(), device_types="cpu")(fn)
            op.register_fake(fake_impl)
            return op
        return decorate

    env = dict(torch=torch, get_parallel=parallel, get_lora=lambda: NS(enable_lora=active.lora),
               get_exec=lambda: NS(deterministic=NS(enable_deterministic_inference=False)),
               register_custom_op=register)
    names = [n.name for n in ast.parse(OP.read_bytes()).body if isinstance(n, ast.FunctionDef)]
    base["load"](OP, names, env)
    install("sglang.srt.layers.communication.sp_half_qkv", env)
    meta = {"tokens": 0}
    column = dict(torch=torch, get_parallel=parallel, sp_num_tokens=lambda: meta["tokens"],
                  sp_fused_matmul_eligible=lambda _: False)
    base["load"](COLUMN, ("column_parallel_g_matmul", "sp_exit_gather"), column)
    linear = NS(quant_method=base["LinearMethod"](), bias=None, gather_output=False,
                weight=torch.ones(16, 32, dtype=torch.bfloat16))
    send, recv = comm.send, comm.recv
    project = env["_project"]

    def checked_send(x, peer):
        events.visit(x, False)
        send(x, peer)

    def checked_recv(x, peer):
        events.visit(x, True)
        recv(x, peer)

    def checked_project(x, w, dest, specialized):
        assert events.current is events.compute
        events.visit(x, False)
        events.visit(dest, True)
        active.projections += 1
        if active.injected:
            raise RuntimeError("injected projection failure")
        project(x, w, dest, specialized)

    env["_project"] = checked_project
    __import__("torch._dynamo")
    count = 0
    words = 0
    cases = []
    with ExitStack() as mocks:
        for obj, name, fn in (
            (torch.cuda, "get_device_capability", lambda dev: (9, 0)),
            (torch.cuda, "Stream", lambda device: events.transfer),
            (torch.cuda, "current_stream", lambda device: events.current),
            (torch.cuda, "stream", events.using), (comm, "send", checked_send), (comm, "recv", checked_recv)
        ):
            mocks.enter_context(patch.object(obj, name, fn))
        for veto in ("wrong_method", "ineligible_weight", "communicator", "lora", "architecture"):
            candidate = NS(**vars(linear))
            if rank == 0:
                if veto == "wrong_method":
                    candidate.quant_method = object()
                if veto == "ineligible_weight":
                    candidate.weight = linear.weight.clone().requires_grad_(True)
                if veto == "communicator":
                    comm.available = False
                active.lora = veto == "lora"
            with patch.object(torch.cuda, "get_device_capability",
                              lambda dev: (8 if rank == 0 and veto == "architecture" else 9, 0)):
                env["enable_half_qkv"]([candidate], group)
            assert not hasattr(candidate, "_sp_half_qkv") and not comm.group_shapes
            active.lora, comm.available = False, True
        env["enable_half_qkv"]([linear], group)
        assert linear._sp_half_qkv and len(comm.group_shapes) == 2
        for rows, hidden, n, drop, mode, fast in (
            (511, 32, 16, 0, "small", False), (512, 32, 16, size-1, "strided", True),
            (523, 64, 24, 1, "specialized", True), (513, 64, 28, 0, "plain", True),
            (512, 16, 2304, 0, "wide", False), (513, 32, 8, 1, "bias", False),
            (513, 32, 8, 0, "requires_grad", False), (512, 32, 16, 1, "deterministic", False),
            (2048, 64, 32, 0, "full_rows", True)
        ):
            active.dispatched = mode == "specialized"
            events.accesses.clear()
            active.projections = active.dispatches = comm.sent = group.allgathers = 0
            comm.group_shapes.clear()
            g = torch.Generator().manual_seed(rows + n)
            x = torch.randint(-8, 9, (size*rows, hidden), generator=g).bfloat16() / 8
            g.manual_seed(rank + 732)
            linear.weight = torch.randint(-8, 9, (n, hidden), generator=g).bfloat16() / 8
            local = x.narrow(0, rank * rows, rows)
            if mode == "strided":
                tmp = x.new_empty(rows, hidden * 2)
                tmp[:, ::2] = local
                local = tmp[:, ::2]
            local.requires_grad_(mode == "requires_grad")
            meta["tokens"] = size * rows - drop
            bias = torch.ones(n, dtype=x.dtype) if mode == "bias" else None
            with torch.no_grad(), patch.object(torch, "are_deterministic_algorithms_enabled", return_value=mode=="deterministic"):
                got = column["column_parallel_g_matmul"](linear, local, bias)
            wanted = F.linear(x[:meta["tokens"]], linear.weight, bias)
            assert torch.equal(got.view(torch.int16), wanted.view(torch.int16))
            assert group.allgathers == int(not fast) and len(comm.group_shapes) == (2 if fast else 0)
            assert active.projections == (2 if fast else 0) and active.dispatches == (2 if active.dispatched else 0)
            if fast:
                assert comm.sent == (size-1)*rows*hidden*local.element_size()
                half = size // 2
                base_rank = rank // half * half
                peers = [p for p in range(base_rank, base_rank+half) if p != rank]
                assert [p for direction,p,_ in comm.group_shapes[0] if direction=="send"] == peers
                assert comm.group_shapes[1] == [(k,(rank+half)%size,(half,rows,hidden)) for k in ("send","recv")]
                assert events.compute.clock[1] >= events.transfer.clock[1]
            cases.append(mode)
            count += int(fast)
            words += got.numel()
        # All ranks inject after the bidirectional transfer, and must rejoin before returning.
        active.injected = True
        events.accesses.clear()
        try:
            column["column_parallel_g_matmul"](linear, local, None)
        except RuntimeError as error:
            assert "injected projection failure" in str(error)
        else:
            raise AssertionError("expected first projection failure")
        assert comm.disabled and comm.ops is None and events.compute.clock[1] >= events.transfer.clock[1]
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            x = torch.empty(2048, 8192, dtype=torch.bfloat16)
            w = torch.empty(2304, 8192, dtype=x.dtype)
            assert env["half_qkv"](x,w,8,False).shape == (16384,2304)
    (Path(receipts)/f"{size}-{rank}.json").write_text(json.dumps(dict(size=size,rank=rank,fast=count,
                     cases=cases,checked_bf16=words,startup_vetoes=5,joined_after_failure=True)))
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform=="darwin" else "lo")
    with tempfile.TemporaryDirectory() as tmp:
        for size in (4, 8):
            mp.spawn(worker, args=(size, str(Path(tmp)/str(size)), tmp), nprocs=size, join=True)
        records = [json.loads(p.read_bytes()) for p in sorted(Path(tmp).glob("*.json"))]
    assert len(records)==12
    manifest = json.loads((HERE/"validation.json").read_bytes())
    print(json.dumps(dict(patch_sha256=manifest["patch_sha256"], source_sha256=sha256(OP.read_bytes()).hexdigest(),
              tp=[4,8], rank_cases=sum(len(r["cases"]) for r in records),
              eligible_rank_cases=sum(r["fast"] for r in records), bf16_outputs=sum(r["checked_bf16"] for r in records),
              injected_stream_joins=len(records), startup_vetoes=sum(r["startup_vetoes"] for r in records),
              records=records, physical_H200=False,
              scope="Real Gloo P2P and PyTorch custom-op dispatcher, CPU BF16 products, checked two-stream dependency model."), indent=2))
