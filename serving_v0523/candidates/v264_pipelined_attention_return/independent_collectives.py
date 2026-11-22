"""Replay the striped return's native collective selection and breakable-graph dependencies."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
COMMON = runpy.run_path(str(HERE.parent / "v260_pipelined_prompt_qkv/independent_transport.py"))
FILES = dict(op="layers/communication/sp_attention_return.py", sp="layers/layernorm_sp.py",
             parallel="distributed/parallel_state.py", ca="distributed/device_communicators/custom_all_reduce.py",
             weak="distributed/device_communicators/custom_all_reduce_utils.py",
             graph="model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py",
             common="utils/common.py", pynccl="distributed/device_communicators/pynccl.py")
native = COMMON["native"]


def run():
    loader = COMMON["read_export"]
    loader.__globals__.update(HERE=HERE, FILES=FILES)
    cfg, src = loader()
    get_active = ContextVar("active")
    cap = ContextVar("capture", default=None)
    forks = ContextVar("forks", default=None)
    edge_log = []
    wait_hook = native(src["graph"], "_hooked_wait_stream", dict(
        _forked_streams_var=forks, _current_stream_var=cap, _is_stream_capturing=lambda s: s in forks.get(),
        _original_wait_stream=lambda this, other: edge_log.append((this.stream_id, other.stream_id))))

    class Stream:
        wait_stream = wait_hook
        def __init__(self, name):
            self.stream_id = name

    main, side = Stream("main"), Stream("transfer")
    get_active.set(main)
    device = NS(current_device=lambda: 2, current_stream=lambda device: get_active.get())
    resolver = native(src["common"], "get_current_device_stream_fast",
                      dict(cached_device_index=-1, torch=NS(get_device_module=lambda: device)))

    @contextmanager
    def select(stream):
        token = get_active.set(stream)
        try:
            yield
        finally:
            get_active.reset(token)

    cuda = NS(current_stream=lambda _:get_active.get(), stream=select)
    captured = [False]
    method = native(src["parallel"], "_resolve_outplace_all_reduce_method", dict(is_in_tc_piecewise_cuda_graph=lambda: captured[0]), "GroupCoordinator")
    weak = native(src["weak"], "is_weak_contiguous", {})
    custom = native(src["ca"], "should_custom_ar", dict(_is_hip=False, is_weak_contiguous=weak), "CustomAllreduce")
    reduce_rows = native(src["sp"], "reduce_scatter_token_rows", dict(
        torch=torch, os=os, ceil_align=lambda n, align: (n + align-1)//align*align))
    default = NS(is_cutedsl=lambda: False, is_gemv=lambda: False)
    calls = []
    fail = [None]

    def mm(left, right, *, out):
        calls.append(("project", resolver().stream_id, tuple(out.shape), out.storage_offset(), out.untyped_storage()._cdata))
        if len([x for x in calls if x[0]=="project"])==2 and fail[0]=="second":
            raise RuntimeError("injected second")
        return torch.mm(left, right, out=out)

    striped = native(src["op"], "striped_attention_return", dict(torch=NS(cuda=cuda, mm=mm)))
    class Group:
        world_size, rank_in_group = 8, 1
        pcie_ipc_comm = qr_comm = pymscclpp_comm = torch_symm_mem_comm = None
        pynccl_comm = object()
        _resolve_outplace_all_reduce_method = method
        def __init__(self, ca_limit, flash_limit):
            self.ca_comm = NS(disabled=False, max_size=ca_limit, full_nvlink=True, world_size=8)
            self.ca_comm.should_custom_ar = lambda t: custom(self.ca_comm, t)
            self.flash_limit = flash_limit

        def _can_use_flashinfer_allreduce(self, t):
            return t.numel() * t.element_size() <= self.flash_limit
        def is_symmetric_memory_enabled(self):
            return False
        def outcome(self, name, t):
            calls.append((name, resolver().stream_id, tuple(t.shape), t.storage_offset(), t.untyped_storage()._cdata))
            if fail[0]=="reduce" and get_active.get() is side:
                raise RuntimeError("injected reduce")
        def all_reduce(self, t):
            self.outcome("all_reduce", t)
            return t
        def reduce_scatter_along_dim(self, t, dim):
            self.outcome("reduce_scatter", t)
            return t.narrow(dim, t.shape[0]//8, t.shape[0]//8)

    modules={"sglang.srt.layers.layernorm_sp": NS(reduce_scatter_token_rows=reduce_rows),
             "sglang.srt.layers.quantization.unquant": NS(get_bf16_gemm_backend=lambda: default)}
    samples, rejected, failures = [], [], []
    with patch.dict(sys.modules, modules):
        for rows in (8192, 8193, 16384):
            x = torch.empty(rows, 1024, dtype=torch.bfloat16, device="meta")
            weight = torch.empty(8192, 1024, dtype=x.dtype, device=x.device)
            half_size = rows * 4096 * 2
            for capture, tc in (("eager",False), ("full",False), ("breakable",False), ("piecewise",True)):
                captured[0] = tc
                cap.set(main if capture=="breakable" else None)
                forks.set(set() if capture=="breakable" else None)
                for nvls in ("0", "1"):
                    for ca_limit, flash_limit, reason in ((0,0,None),(half_size,0,"custom change"),
                                                          (2*half_size,0,"custom full"), (0,half_size,"flash change")):
                        group = Group(ca_limit, flash_limit)
                        calls.clear()
                        edge_log.clear()
                        with patch.dict(os.environ, NCCL_NVLS_ENABLE=nvls):
                            output = striped(x, weight, group, side)
                        if reason:
                            assert output is None and not calls and not edge_log
                            rejected.append(reason)
                            continue
                        assert output.shape == ((rows+7)//8, 8192)
                        style = "all_reduce" if nvls=="1" else "reduce_scatter"
                        assert [(c[0], c[1]) for c in calls] == [("project","main"),(style,"transfer"),("project","main"),(style,"main")]
                        assert edge_log == [("transfer","main"),("main","transfer")]
                        assert not forks.get()
                        # Same preallocated arena in both GEMMs, disjoint element ranges.
                        assert calls[0][4] == calls[2][4] and (calls[0][3],calls[2][3]) == (0,rows*4096)
                        samples.append(dict(rows=rows, capture=capture, approximate_native_method="pynccl" if tc else None,
                                            NVLS=nvls, collective=style))
            for injected in ("second","reduce"):
                group=Group(0,0)
                calls.clear()
                edge_log.clear()
                forks.set(set())
                cap.set(main)
                fail[0]=injected
                try:
                    striped(x, weight, group, side)
                except RuntimeError as exc:
                    assert str(exc)=="injected "+injected
                else:
                    raise AssertionError("injection failed")
                finally:
                    fail[0]=None
                assert edge_log==[("transfer","main"),("main","transfer")] and not forks.get()
                failures.append((rows, injected))
    # Exercise both lowerings of the native CUDA reduce-scatter implementation.
    # PyTorch meta tensors carry native message geometry. The ABI sink receives
    # typed counts on the current stream without starting NCCL.
    abi_calls = []
    for enabled in (False, True):
        class Comm:
            disabled = not enabled
            device = torch.device("meta")
            comm = 71
            def change_state(self, enable):
                from contextlib import nullcontext
                assert enable
                return nullcontext()

        native_args = dict(get_current_device_stream_fast=resolver, buffer_type=int, cudaStream_t=lambda n:n,
                           ReduceOp=NS(SUM="sum"), ncclDataTypeEnum=NS(from_torch=lambda dtype:"BF16"),
                           ncclRedOpTypeEnum=NS(from_torch=lambda op:"SUM"))
        comm = Comm()
        comm._resolve_stream = native(src["pynccl"], "_resolve_stream", native_args, "PyNcclCommunicator").__get__(comm)
        comm.reduce_scatter = native(src["pynccl"], "reduce_scatter", native_args, "PyNcclCommunicator").__get__(comm)
        comm.nccl = NS(ncclReduceScatter=lambda pin,pout,count,dtype,op,handle,stream:
                       abi_calls.append(("PyNCCL", count, stream)))
        main.cuda_stream, side.cuda_stream = "main", "transfer"
        lower_env = dict(reduce_scatter_single=lambda out, inp, group:
                         abi_calls.append(("ProcessGroup", out.numel(), resolver().stream_id)))
        lower = native(src["parallel"], "_reduce_scatter_tensor", lower_env, "GroupCoordinator")
        group = NS(pynccl_comm=comm, device_group=71, is_symmetric_memory_enabled=lambda:False,
                   debug_check_symmetric_mempool=lambda *_:None)
        for name, width, stream in (("parent",8192,main),("first",4096,side),("second",4096,main)):
            full = torch.empty(16384, width, device="meta", dtype=torch.bfloat16)
            reduced = torch.empty(2048, width, device="meta", dtype=torch.bfloat16)
            with select(stream):
                lower(group, reduced, full)
            assert abi_calls[-1] == ("PyNCCL" if enabled else "ProcessGroup",2048*width,stream.stream_id)
    # These are BF16-representable operands. Show a real FP32 accumulator can be order-sensitive.
    singles = torch.tensor([2**24,1.,-2**24,0,0,0,0,0], dtype=torch.bfloat16).float()
    def ordered(order):
        s = torch.zeros((), dtype=torch.float32)
        for i in order:
            s = s + singles[i]
        return float(s.to(torch.bfloat16))
    a, b = ordered(range(8)), ordered([0,2,1,3,4,5,6,7])
    assert (a,b)==(0,1)
    return dict(patch_sha256=cfg["patch_sha256"], tree=cfg["tree"], source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in src.items()},
                torch_version=torch.__version__, eligible=samples, rejected_before_compute_or_collective=len(rejected),
                failing_calls_with_safe_join=failures, native_reduce_scatter_abi_cases=abi_calls, exact_FP32_order_counterexample_BF16_results=[a,b],
                scope="Frozen production striped dispatch, native optional custom-AR size eligibility and native group method resolver, native SUM/scatter selector, native breakable wait hook and current CUDA stream resolver. Large metadata tensors on PyTorch meta. CUDA matmuls, allocator and device collectives themselves are not run.")


if __name__=="__main__":
    print(json.dumps(run(), indent=2))
