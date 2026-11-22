"""Run against a pinned SGLang checkout with the candidate patch applied."""

import argparse
import ast
import contextlib
from datetime import timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
SRT = "python/sglang/srt/"
F = torch.nn.functional


def source(stage, path):
    path = SRT + path
    if (stage / path).exists():
        return (stage / path).read_text()
    return subprocess.check_output(["git", "-C", stage, "show", f"{BASE}:{path}"], text=True)


def selected(code, name, methods=None):
    node = next(n for n in ast.parse(code).body if getattr(n, "name", None) == name)
    if methods is not None:
        node.body = [n for n in node.body if getattr(n, "name", None) in methods
                     or isinstance(n, (ast.Assign, ast.AnnAssign))]
    return ast.unparse(node)


def define(code, namespace):
    exec("from __future__ import annotations\n" + code, namespace)


def module(name, **items):
    value = ModuleType(name)
    value.__dict__.update(items)
    sys.modules[name] = value


class Group:
    def __init__(self, size):
        self.ranks = list(range(size))
        self.world_size = size
        self.cpu_group = dist.group.WORLD
        self.ca_comm = None
        self.calls = []

    def all_gather(self, tensor, dim):
        self.calls.append(("gather", dim, tuple(tensor.shape)))
        outputs = [torch.empty_like(tensor) for _ in self.ranks]
        dist.all_gather(outputs, tensor.contiguous())
        return torch.cat(outputs, dim)

    def all_to_all_single(self, out, inp):
        assert out.is_contiguous() and inp.is_contiguous() and inp.ndim == out.ndim == 1
        self.calls.append(("exchange", 0, tuple(inp.shape)))
        dist.all_to_all_single(out, inp)


def implementation(stage, group, rank, ctx):
    parallel = SimpleNamespace(tp_group=group, attn_tp_group=group, tp_size=group.world_size,
                               attn_tp_size=group.world_size, moe_ep_size=group.world_size,
                               muse_shard_expert_projections=True)
    model = SimpleNamespace(load_format="auto")
    def resolve(name):
        assert name == "tp"
        return group
    def rank_size(g):
        assert g is group
        return rank, g.world_size
    ns = dict(torch=torch, nn=torch.nn, Parameter=torch.nn.Parameter, LinearMethodBase=object,
              _resolve_linear_group=resolve, get_group_rank_size=rank_size,
              divide=lambda n, s: n // s, _is_cpu=False, contextlib=contextlib,
              wrap_method_with_debug_kernel_once=lambda *a, **kw: None,
              get_global_experts_capturer=lambda: None,
              should_skip_weight_loading=lambda n, config: False,
              default_weight_loader=None)
    linear = source(stage, "layers/linear.py")
    support = next(n for n in ast.parse(linear).body if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "WEIGHT_LOADER_V2_SUPPORTED")
    define(ast.unparse(support), ns)
    define(selected(source(stage, "utils/common.py"), "set_weight_attrs"), ns)
    define(selected(source(stage, "layers/quantization/unquant.py"), "UnquantizedLinearMethod",
                    {"create_weights"}), ns)
    module("sglang.srt.layers.quantization.unquant", UnquantizedLinearMethod=ns["UnquantizedLinearMethod"],
           bf16_gemm_dispatch=F.linear)
    define(selected(linear, "LinearBase", {"__init__"}), ns)
    define(selected(linear, "ColumnParallelLinear", {"__init__", "_initialize_partition", "weight_loader"}), ns)
    module("sglang.srt.layers.linear", ColumnParallelLinear=ns["ColumnParallelLinear"])
    module("sglang.srt.runtime_context", get_parallel=lambda: parallel, get_model=lambda: model,
           get_forward=lambda: ctx, get_exec=lambda: SimpleNamespace(
               deterministic=SimpleNamespace(enable_deterministic_inference=False)))
    module("sglang.srt.environ", envs=SimpleNamespace(
        SGLANG_ENABLE_DETERMINISTIC_INFERENCE=SimpleNamespace(get=lambda: False)))
    module("sglang.srt.batch_invariant_ops", is_batch_invariant_mode_enabled=lambda: False)
    define(source(stage, "layers/muse_spark/projection.py"), ns)
    muse = source(stage, "models/muse_spark.py")
    for cls, methods in (("MuseSpark1xMoE", {"_forward_experts", "_reduce_and_unpool"}),
                         ("MuseSparkTextForCausalLM", {"load_weights"})):
        node = next(n for n in ast.parse(muse).body if getattr(n, "name", None) == cls)
        for f in node.body:
            if getattr(f, "name", None) in methods:
                define(ast.unparse(f), ns)
    return ns, parallel, model


def expected_calls(sp, rows, k, n, size):
    width = n // size
    if rows == 0:
        return []
    if not sp and rows < k:
        return [("gather", 1, (rows, width))]
    if sp and rows * (k + width) < k * width:
        return [("gather", 0, (rows, k)), ("exchange", 0, (size * rows * width,))]
    return [("gather", 0, (width, k))]


class Experts(torch.nn.Module):
    use_w4a16 = True
    post_expert_norm = None

    def forward(self, x, *args, **kwargs):
        return x


def run(rank, size, stage, rendezvous, result):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=size, timeout=timedelta(seconds=90))
    try:
        torch.set_default_dtype(torch.bfloat16)
        group = Group(size)
        ctx = SimpleNamespace(sp_active=False)
        ns, parallel, model = implementation(Path(stage), group, rank, ctx)
        gate = ns["use_sharded_expert_projections"]
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
                torch.cuda, "get_device_capability", return_value=(9, 0)):
            assert gate(64, 32)
            model.load_format = "dummy" if rank == size - 1 else "auto"
            assert not gate(64, 32)
            model.load_format = "safetensors"
            with patch.object(torch.cuda, "get_device_capability", return_value=(8, 0) if rank == 0 else (9, 0)):
                assert not gate(64, 32)
            assert not gate(63, 32)
            parallel.muse_shard_expert_projections = False
            assert not gate(64, 32)
            parallel.muse_shard_expert_projections = True
            assert gate(64, 32)
        projection = ns["MuseShardedExpertProjection"]
        network = torch.nn.Module()
        network.mlp = mlp = torch.nn.Module()
        mlp.pre_expert_proj = projection(64, 32, "mlp.pre_expert_proj")
        mlp.post_expert_proj = projection(32, 64, "mlp.post_expert_proj")
        network.nvfp4_checkpoint, network.int4_checkpoint = True, False
        network.config, network.post_load_weights = object(), lambda: None
        mlp.experts, mlp.gate = Experts(), torch.nn.Identity()
        mlp.pre_expert_norm = None
        mlp.tp_size, mlp.moe_tp_size = size, 1
        mlp.sp_replicate_expert_input = mlp.router_in_fp32 = False
        mlp.expert_hidden_size, mlp.top_k, mlp.nvfp4_checkpoint = 32, 1, True
        mlp._defers_unpool_gather = lambda comm, rows: False
        mlp._routing_op = lambda: lambda *a, **kw: (None, None, None)
        mlp._forward_sp_experts = lambda x, *args: x
        counts = {"checkpoint_shards": 0, "projection_pairs": 0, "output_elements": 0, "deferred_boundary_slices": 0}
        generator = torch.Generator()
        for reload in (0, 1):
            generator.manual_seed(1431 + reload)
            full = [torch.randint(-2, 3, shape, generator=generator).bfloat16()
                    for shape in ((32, 64), (64, 32))]
            ns["load_weights"](network, zip(("mlp.pre_expert_proj.weight", "mlp.post_expert_proj.weight"), full))
            for part, whole in zip((mlp.pre_expert_proj, mlp.post_expert_proj), full):
                assert torch.equal(part.weight, whole.chunk(size, dim=0)[rank])
                assert not part.weight.requires_grad and part.weight.dtype == torch.bfloat16
                assert part.weight.untyped_storage().nbytes() == whole.numel() * 2 // size
                counts["checkpoint_shards"] += 1
            for sp in (False, True):
                ctx.sp_active = sp
                for rows in (0, 1, 3, 4, 6, 7, 8, 16, 32, 33, 63, 64, 128, 256):
                    generator.manual_seed(317 * rows + (rank if sp else 0) + reload)
                    x = torch.randint(-2, 3, (rows, 128), generator=generator).bfloat16()[:, ::2]
                    group.calls.clear()
                    projected = ns["_forward_experts"](mlp, x)
                    actual = ns["_reduce_and_unpool"](mlp, projected, already_reduced=True)
                    want = F.linear(x, full[0])
                    assert torch.equal(projected, want) and torch.equal(actual, F.linear(want, full[1]))
                    assert group.calls == (expected_calls(sp, rows, 64, 32, size)
                                           + expected_calls(sp, rows, 32, 64, size))
                    if not sp and rows in (1, 8, 16):
                        mlp._defers_unpool_gather = lambda comm, rows: True
                        group.calls.clear()
                        partial = ns["_reduce_and_unpool"](mlp, projected, already_reduced=True)
                        assert partial.is_contiguous() and not group.calls
                        assert torch.equal(partial, actual.chunk(size, dim=1)[rank])
                        counts["deferred_boundary_slices"] += 1
                        mlp._defers_unpool_gather = lambda comm, rows: False
                    counts["projection_pairs"] += 1
                    counts["output_elements"] += actual.numel() + projected.numel()
        if rank == 0:
            Path(result).write_text(json.dumps({k: v * size for k, v in counts.items()}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    results = {}
    with tempfile.TemporaryDirectory(prefix="okc-proj-gloo-") as tmp:
        for size in (2, 4, 8):
            record = Path(tmp) / f"result-{size}.json"
            mp.spawn(run, args=(size, str(args.stage.resolve()), f"{tmp}/rendezvous-{size}", str(record)),
                     nprocs=size, join=True)
            results[str(size)] = json.loads(record.read_text())
    print(json.dumps({"backend": "CPU Gloo", "torch": torch.__version__, "rank_totals_by_tp": results}, indent=2))
