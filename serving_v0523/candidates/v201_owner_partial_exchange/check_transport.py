"""Run both model exits with real multiprocess Gloo transport on a CPU host."""
import ast
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


HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) / "python/sglang/srt"
reuse = runpy.run_path(str(next(HERE.parent.glob("v113_*/cpu_check.py"))))
load, install = reuse["load"], reuse["install"]


def ordered_sum(parts):
    total = parts[0].float().clone()
    for part in parts[1:]:
        total.add_(part.float())
    return total.to(torch.bfloat16)


def gather(x, size):
    parts = [torch.empty_like(x, memory_format=torch.contiguous_format) for _ in range(size)]
    dist.all_gather(parts, x.contiguous())
    return parts


def register(fake_impl):
    def decorate(fn):
        op = torch.library.custom_op("owner_fixture::" + fn.__name__, mutates_args=(), device_types="cpu")(fn)
        op.register_fake(fake_impl)
        return op
    return decorate


def worker(rank, size, tmp):
    torch.set_num_threads(1)
    path = Path(tmp)
    dist.init_process_group("gloo", init_method="file://" + str(path / "rdzv"), rank=rank,
                            world_size=size, timeout=timedelta(seconds=90))
    group = reuse["Group"](size, rank)
    comm = group.pynccl_comm
    comm.world_size = size
    comm.all_to_all_single = lambda dst, src: dist.all_to_all_single(dst, src)
    group.use_symmetric_memory = lambda *_: nullcontext()
    group.is_symmetric_memory_enabled = lambda: False
    group.all_reduce = lambda x: ordered_sum(gather(x, size)).to(x.dtype)
    parallel = lambda: NS(tp_group=group)
    runtime = NS(deterministic=NS(enable_deterministic_inference=False))
    install("sglang.srt.environ", {"envs": NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda: False))})
    install("sglang.srt.batch_invariant_ops", {"is_batch_invariant_mode_enabled": lambda: False})

    def finish(local, remote, out, rank, size):
        parts = [local if peer == rank else remote[peer - int(peer > rank), :local.shape[0]]
                 for peer in range(size)]
        out.copy_(ordered_sum(parts))

    ns = dict(torch=torch, os=os, get_parallel=parallel, get_exec=lambda: runtime, register_custom_op=register,
              _finish=finish, _MAX_RECEIVE_BYTES=32*1024**2, _WINDOW_SLACK_BYTES=256*1024)
    load(SRC / "layers/communication/sp_owner_sum.py", ("prepare_owner_sum", "owner_sum_eligible",
         "_exchange", "_owner_shape", "_window_rows", "owner_sum"), ns)
    module = install("sglang.srt.layers.communication.sp_owner_sum", ns)
    live = {}
    row = dict(os=os, torch=torch, get_parallel=parallel, ceil_align=lambda x, s: (x+s-1)//s*s)
    load(SRC / "layers/layernorm_sp.py", ("sp_exit_gather", "reduce_scatter_token_rows", "row_parallel_gbar_matmul"), row)
    row_view = NS(**row, sp_num_tokens=lambda: live["rows"])
    source = ast.parse((SRC / "models/muse_spark.py").read_text())
    method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name=="_forward_sp_replicated_experts")
    model = dict(torch=torch, get_parallel=parallel, layernorm_sp=row_view)
    exec(compile(ast.Module([method], []), "<expert exit>", "exec"), model)
    linear = NS(quant_method=reuse["LinearMethod"](), tp_group=group, _sp_unfused_collective=True)
    tested, entries, elements, sent = 0, 0, 0, 0
    with patch.object(torch.cuda, "get_device_capability", return_value=(9,0)), patch.dict(os.environ, NCCL_NVLS_ENABLE="1"):
        for rejected in ("available", "ranks", "deterministic", "nvls", "comm_rank", "capability"):
            with ExitStack() as ctx:
                fc = NS(tp_group=group)
                if rank == size-1:
                    if rejected=="available": ctx.enter_context(patch.object(comm, "available", False))
                    if rejected=="ranks": fc.tp_group=NS(ranks=group.ranks[::-1])
                    if rejected=="deterministic": ctx.enter_context(patch.object(torch, "are_deterministic_algorithms_enabled", return_value=True))
                    if rejected=="nvls": ctx.enter_context(patch.dict(os.environ, NCCL_NVLS_ENABLE="0"))
                    if rejected=="comm_rank": ctx.enter_context(patch.object(comm, "rank", -1))
                    if rejected=="capability": ctx.enter_context(patch.object(torch.cuda, "get_device_capability", return_value=(8,0)))
                assert not ns["prepare_owner_sum"](group, [fc])
        assert ns["prepare_owner_sum"](group, [linear]) and comm.disabled
        for rows, width, tile in ((size-1,17,1),(size,31,1),(size+1,7,1),(size*6-1,19,2),
                                  (size*9+3,65,4),(size*11,43,5)):
            span = (rows + size - 1)//size
            live["rows"] = rows
            ns["_MAX_RECEIVE_BYTES"] = (size-1)*width*2*tile
            count = min(span, max(0, rows-rank*span))
            gen = torch.Generator().manual_seed(291+rank*3+rows)
            activ = torch.randn(rows, 9, generator=gen).bfloat16()
            linear.weight = torch.randn(width, 9, generator=gen).bfloat16()
            raw = torch.randn(rows, width*2, generator=gen).bfloat16()
            full = raw[:,::2] if rank%2 else raw[:,:width].contiguous()
            before = raw.clone()
            pooled = torch.arange(span*5).reshape(span,5).bfloat16() + rank
            scores = torch.ones(span,3).bfloat16()*(rank+1)
            ids = torch.arange(span*3).reshape(span,3).int()+rank
            exp_pooled, exp_scores, exp_ids = [torch.cat(gather(x, size))[:rows] for x in (pooled,scores,ids)]

            class Experts:
                post_expert_norm = object()
                def __call__(self, x, score, idx, norm):
                    assert norm is self.post_expert_norm
                    assert torch.equal(x,exp_pooled) and torch.equal(score,exp_scores) and torch.equal(idx,exp_ids)
                    return full

            for route in ("direct", "linear", "experts", "parent"):
                value = F.linear(activ,linear.weight) if route=="linear" else full
                expected = F.pad(ordered_sum(gather(value, size)), (0,0,0,(-rows)%size))
                sent_before = comm.sent
                with patch.object(module, "owner_sum_eligible", return_value=True):
                    if route=="direct":
                        actual = ns["owner_sum"](full, size)
                    elif route=="linear":
                        linear._sp_owner_exchange = True
                        actual = row["row_parallel_gbar_matmul"](linear,activ,None)
                    else:
                        layer = NS(experts=Experts(), _sp_owner_exchange=route=="experts")
                        actual = model[method.name](layer,pooled,scores,ids)
                assert torch.equal(actual,expected[rank*span:(rank+1)*span]) and comm.disabled
                assert torch.equal(raw,before)
                expected_bytes = (rows-count)*width*2 if route!="parent" else 0
                assert comm.sent-sent_before == expected_bytes
                assert not any(op[1]==rank for group_shape in comm.group_shapes for op in group_shape)
                comm.group_shapes.clear()
                tested += 1
                entries += int(route in ("linear","experts","parent"))
                elements += actual.numel()
                sent += expected_bytes
    (path/str(rank)).write_text(json.dumps([tested,entries,elements,sent]))
    dist.destroy_process_group()


if __name__=="__main__":
    results = {}
    for size in (2,3,4,8):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(size,tmp), nprocs=size, join=True)
            ranks = [json.loads((Path(tmp)/str(rank)).read_text()) for rank in range(size)]
            results[size] = dict(zip(("rank_cases","model_entry_cases","output_elements","p2p_sent_bytes"),
                                     map(sum,zip(*ranks)),strict=True))
    print(json.dumps(dict(worlds=results,backend="Gloo",dtype="BF16",cpu_finish="rank-ordered FP32, one BF16 cast",
                          startup_vetoes_per_world=6, physical_H200=False),indent=2))
