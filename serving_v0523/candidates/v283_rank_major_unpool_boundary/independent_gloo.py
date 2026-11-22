"""Run staged Muse decode/unpool methods and GroupCoordinator.all_gather with real CPU Gloo."""
import argparse
import ast
import contextlib
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


BASE = "python/sglang/"
GROUP = BASE + "srt/distributed/parallel_state.py"
MODEL = BASE + "srt/models/muse_spark.py"
PRODUCER = BASE + "srt/layers/communication/muse_spark_gemm_ag.py"
NORM = BASE + "kernels/ops/layernorm/muse_spark_norm_residual_jit.py"
HIDDEN = 8192
INNER = 4096


def bind(stage, path, owner, names, env):
    body = ast.parse((stage / path).read_text()).body
    if owner:
        body = next(node.body for node in body if getattr(node, "name", "") == owner)
    for name in names:
        method = next(node for node in body if getattr(node, "name", "") == name)
        exec("from __future__ import annotations\n" + ast.unparse(method), env)
    return {name: env[name] for name in names}


def install(name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    sys.modules[name] = module


def target(rows):
    row = torch.arange(rows, dtype=torch.int32)[:, None]
    col = torch.arange(HIDDEN, dtype=torch.int32)[None, :]
    # Multiplication by an odd word permutes all 65,536 BF16 representations.
    return ((col + row.remainder(8) * HIDDEN) * 2053 + (row // 8) * 1711 + 37).to(torch.int16)


def worker(rank, world, tmp, stage_path):
    torch.set_num_threads(1)
    stage = Path(stage_path)
    path = Path(tmp)
    dist.init_process_group("gloo", init_method="file://" + str(path / f"group{world}"),
                            rank=rank, world_size=world, timeout=timedelta(seconds=150))
    try:
        result = check(stage, rank, world)
        (path / f"{world}_{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def check(stage, rank, world):
    state = NS(dims=[], adapters=[], layouts=[], stages=[], allocations=[], flags=[])
    stripe = HIDDEN // world

    def gather_single(out, local, group):
        # macOS Gloo lacks the native all_gather_into_tensor backend used by NCCL.
        pieces = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(pieces, local.contiguous(), group=group)
        out.view(world, *local.shape).copy_(torch.stack(pieces))
        state.allocations.append(out.data_ptr())

    gather = bind(stage, GROUP, "GroupCoordinator", ["all_gather"], dict(
        torch=torch, is_shm_available=lambda *_: False, all_gather_single=gather_single))["all_gather"]

    class Group:
        world_size = local_size = world
        rank_in_group = rank
        device_group = dist.group.WORLD
        hpu_communicator = None
        ca_comm = NS(world_size=world, rank=rank)
        def is_allocation_symmetric(self):
            return False
        def use_symmetric_memory(self, *args, **kwargs):
            return contextlib.nullcontext()
        def all_gather(self, tensor, dim=-1):
            state.dims.append(dim)
            result = gather(self, tensor, dim)
            same = result.data_ptr() == state.allocations[-1]
            assert same == (dim == 0 or tensor.shape[0] == 1), (tensor.shape, dim, same)
            return result

    group = Group()
    weight = torch.empty((HIDDEN, INNER), dtype=torch.bfloat16, device="meta")

    def gemm(x, w, bias):
        assert x.shape == (state.rows, INNER) and w.shape[1] == INNER and bias is None
        if w.shape[0] == HIDDEN:
            state.stages.append("dense_project")
            return state.expected.view(torch.bfloat16)
        assert w.shape[0] == stripe and w.storage_offset() == rank * stripe * INNER
        state.stages.append("column_project")
        # Noncontiguous output also exercises the producer's contiguity repair.
        return state.local

    install("sglang.srt.layers.quantization.unquant", bf16_gemm_dispatch=gemm)
    producer = bind(stage, PRODUCER, None, ["try_decode_column_gather"], {})["try_decode_column_gather"]

    def run_std(branch, stream, out, new_stream, router, *_args):
        state.layouts.append(list(branch.shape))
        if branch.ndim == 3:
            assert branch.data_ptr() == state.allocations[-1]
            for r in range(world):
                out[:, r*stripe:(r+1)*stripe].view(torch.int16).copy_(branch[r].view(torch.int16))
        else:
            out.view(torch.int16).copy_(branch.view(torch.int16))
        new_stream.copy_(stream)

    def jit_module(h, t, pf, cluster, ranks=1):
        state.adapters.append(ranks)
        assert h == HIDDEN and cluster == 1
        assert (t, pf) == ((256, True) if state.rows < 192 else (512, False))
        if ranks > 1 and state.fail_jit:
            raise OSError("deliberate offline specialized JIT load failure")
        return NS(run_std=run_std)

    config = dict(torch=torch, _jit_module=jit_module)
    for node in ast.parse((stage / NORM).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            config[node.targets[0].id] = ast.literal_eval(node.value)
    config_methods = ["_launch_config", "_cluster_launch_config", "fused_norm_residual"]
    wrapper = bind(stage, NORM, None, config_methods, config)["fused_norm_residual"]
    install("sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit", fused_norm_residual=wrapper)

    class Boundary:
        def jit_kernel_kwargs(self):
            return dict(post_eps=1e-5, pre_eps=1e-5, cluster_size=1)
        def jit_selected(self):
            return state.jit_selected
        def __call__(self, stream, branch, **kwargs):
            assert branch.ndim == 2 and branch.shape == stream.shape, branch.shape
            state.stages.append("ordinary_boundary")
            return branch, stream, None

    def native_gather(comm, branch, stream, **kwargs):
        assert branch.shape == (state.rows, stripe)
        state.stages.append("native_shard_boundary")
        return group.all_gather(branch, dim=-1), stream

    def native_local(comm, x, w):
        state.stages.append("native_local")
        return state.local.contiguous()

    def experts(self, hidden_states, router_input, *, ar_workspace, grouped_decode, column_group):
        state.flags.append(column_group is group)
        return torch.empty((state.rows, INNER), dtype=torch.bfloat16, device="meta")

    def native_ag(comm, x, w):
        state.stages.append("native_complete")
        return group.all_gather(state.local.contiguous(), dim=-1)

    def terminal_gate(stream, branch):
        state.stages.append("terminal_gate")
        assert branch.ndim == 2 and branch.shape == stream.shape
        return branch

    penv = dict(torch=torch, _is_cpu=False, get_forward=lambda: NS(sp_active=state.sp),
                get_parallel=lambda: NS(tp_group=group), is_dp_attention_enabled=lambda:False,
                tensor_model_parallel_all_reduce=lambda x:x, gemm_ag_supported=lambda *_: state.native,
                gemm_local=native_local, gemm_ag=native_ag, bf16_gemm_dispatch=gemm, try_decode_column_gather=producer,
                all_gather_norm_residual=native_gather)
    methods = bind(stage, MODEL, "MuseSpark1xMoE", ["forward", "_forward_tp", "_reduce_and_unpool"], penv)
    methods.update(__call__=methods["forward"], _forward_experts=experts,
                   _defers_unpool_gather=lambda *_: state.defer, finalize_ar_workspace=None, tp_size=world,
                   moe_tp_size=1, post_expert_proj=NS(weight=weight))
    mlp = type("Mlp", (), methods)()
    methods = bind(stage, MODEL, "MuseSpark1xDecoderLayer", ["forward"], penv)
    dec = type("Decoder", (), methods)()
    dec._router_dtype = None
    dec.is_moe_layer = True
    dec.mlp = mlp
    dec.self_attn = lambda pos, x, fb: x
    dec._fused_attn_boundary = lambda stream, branch, **kwargs: (branch, stream, None)
    boundary = Boundary()
    dec.post_feedforward_layernorm = lambda branch: branch
    dec.post_feedforward_residual_gate = terminal_gate
    records = []

    def one(case, rows):
        for key in ("dims", "adapters", "layouts", "stages", "allocations", "flags"):
            getattr(state, key).clear()
        state.rows = rows
        state.expected = target(rows)
        wanted = state.expected[:, rank*stripe:(rank+1)*stripe]
        local = torch.empty((rows, stripe*2), dtype=torch.int16)[:, ::2]
        local.copy_(wanted)
        state.local = local.view(torch.bfloat16)
        state.jit_selected = case != "unselected"
        state.native = case in ("native", "native_complete")
        state.defer = case != "native_complete"
        dec._fused_ffn_boundary = None if case == "terminal" else boundary
        state.sp = case == "sp"
        state.fail_jit = case == "jit_failure"
        dec.operator_dtype = torch.float32 if case == "operator" else torch.bfloat16
        flag = case != "flag_off"
        active_mode = case != "prefill"
        mode = NS(is_decode=lambda: active_mode and case == "decode",
                  is_target_verify=lambda: active_mode and case != "decode")
        batch = NS(muse_decode_column_gather=flag, forward_mode=mode)
        stream = torch.zeros((rows, HIDDEN * (2 if case == "strided" else 1)),
                             dtype=torch.bfloat16 if case == "stream_dtype" else torch.float32)
        if case == "strided":
            stream = stream[:, ::2]
        x = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device="meta")
        returned_stream, actual = dec.forward(None, stream, x, batch)
        if case == "terminal":
            assert actual is None
            actual = returned_stream
        else:
            assert torch.equal(returned_stream, stream)
        assert actual.dtype == torch.bfloat16 and torch.equal(actual.view(torch.int16), state.expected), (case, rows)
        if case in ("verify", "decode", "jit_failure"):
            assert state.dims == [0] and state.adapters == ([world, 1] if state.fail_jit else [world])
            assert state.layouts == ([[rows, HIDDEN]] if state.fail_jit else [[world, rows, stripe]])
            assert state.stages == ["column_project"]
        elif case == "terminal":
            assert state.dims == [-1] and state.stages == ["column_project", "terminal_gate"]
        elif case == "native_complete":
            assert state.dims == [-1] and state.stages == ["native_complete", "ordinary_boundary"]
        elif case == "native":
            assert state.dims == [-1] and state.stages == ["native_local", "native_shard_boundary"]
        elif case in ("strided", "unselected", "operator", "stream_dtype"):
            assert state.dims == [-1] and state.stages == ["column_project", "ordinary_boundary"]
        else:
            assert not state.dims and state.stages == ["dense_project", "ordinary_boundary"]
        assert state.flags == [flag and active_mode and not state.sp]
        return dict(case=case, rows=rows, words=rows*HIDDEN, gather=list(state.dims),
                    jit=list(state.adapters), stages=list(state.stages))

    for rows in (1, 8, 32, 128, 256):
        records.append(one("verify", rows))
        records.append(one("unselected", rows))
    for case, rows in (("decode", 8), ("jit_failure", 32), ("native", 8), ("terminal", 8), ("native_complete", 8),
                       ("strided", 8), ("stream_dtype", 8), ("operator", 8), ("sp", 8),
                       ("flag_off", 8), ("prefill", 8), ("zero", 0), ("too_many", 257)):
        records.append(one(case, rows))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", type=Path)
    args = ap.parse_args()
    stage = args.stage.resolve()
    env = dict(torch=torch, is_dp_attention_enabled=lambda:False)
    calls = bind(stage, MODEL, "MuseSpark1xMoE", ["forward", "_forward_tp"], env)
    marker = object()
    before = NS(_forward_tp=lambda x, router, batch:x)
    assert calls["forward"](before, marker, None, None) is marker
    before = NS(finalize_ar_workspace=None, _forward_experts=lambda x, router, **kwargs:x,
                _reduce_and_unpool=lambda x, *, already_reduced, column_group:x)
    assert calls["_forward_tp"](before, marker, None, None) is marker
    results = {}
    with tempfile.TemporaryDirectory(prefix="v283-gloo-") as td:
        for world in (4, 8):
            mp.spawn(worker, args=(world, td, str(stage)), nprocs=world)
            recs = [json.loads((Path(td) / f"{world}_{rank}.json").read_text()) for rank in range(world)]
            assert all(r == recs[0] for r in recs)
            results[str(world)] = dict(per_rank=recs[0], compared_on_all_ranks=sum(x["words"] for x in recs[0])*world)
    print(json.dumps(dict(torch=torch.__version__, legacy_private_signatures=2, device="CPU Gloo", transport="real Gloo all_gather into list and copy into packed receive; staged GroupCoordinator layout",
                          substituted="GEMM outputs prescribed signed BF16 words; compiled norm and expert math represented by bit-preserving spies",
                          all_bf16_bitpatterns=len(torch.unique(target(8).reshape(-1))), nan_encodings_per_complete_word_set=254,
                          source_sha256={p:sha256((stage/p).read_bytes()).hexdigest() for p in (GROUP, MODEL, PRODUCER, NORM)},
                          results=results), indent=2))


if __name__ == "__main__":
    main()
