"""Replay the Muse prompt model, compact grouped experts and token-owner return on Gloo."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
PARENT = HERE.parent / "v163_sparse_prompt_exchange" / "boundary_check.py"


def fixture():
    # Reuse the parent fixture's real graph wrapper, grouped GEMM adapter and finalizer.
    source = ast.parse(PARENT.read_bytes())
    selected = []
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name == "exercise":
            break
        selected.append(node)
    selected.extend(n for n in source.body if getattr(n, "name", None) == "admission")
    f = {"__file__": str(PARENT)}
    exec(compile(ast.Module(selected, []), str(PARENT), "exec"), f)
    f["install"](f["source"](f["SP"]), ["_return_partials"], f["env"])
    return NS(**f)


def bank(f, rank, size, hidden):
    e = f.Expert()
    e.env = f.env
    e.num_local_experts, e.global_num_experts = 2, 2 * size
    e.use_humming = e.use_w4a16 = True
    e._post_expert_norm = e.post_expert_norm = object()
    e._post_expert_norm_weight = torch.arange(1, hidden + 1).bfloat16() / 4
    e._post_expert_norm_eps = 1e-6
    e._post_expert_norm_weight_first = bool(rank % 2)
    e._expert_map = torch.full((2 * size,), -1, dtype=torch.int32)
    e._expert_map[rank::size] = torch.tensor([1, 0], dtype=torch.int32)
    e.intermediate_size_per_partition = hidden
    e.humming_grouped_tunings = {"w13": None, "w2": None}
    e.gemms = []
    rng = torch.Generator().manual_seed(33219 + rank)
    e.weights = {name: torch.randint(-4, 5, (2, hidden, hidden * width), generator=rng).float() / 4
                 for name, width in [("w13", 2), ("w2", 1)]}
    e.w13_global_scale = torch.tensor([0.75, 1.125]).bfloat16()
    e.w2_global_scale = torch.tensor([1.5, 0.875]).bfloat16()
    return e


def exercise(f, rank, size, rows, counts):
    hidden, topk = 12, 8
    env = f.env
    e = bank(f, rank, size, hidden)
    group = NS(rank_in_group=rank, world_size=size, device_group=dist.group.WORLD)
    group.all_gather_into_tensor = lambda dest, value: dist.all_gather_into_tensor(dest, value.contiguous())
    env["get_parallel"] = lambda: NS(tp_group=group, muse_sparse_prompt_exchange=True)
    env["dist"] = dist
    shard_rows = (rows + size - 1) // size
    begin, end = rank * shard_rows, (rank + 1) * shard_rows
    x = torch.empty((shard_rows * size, hidden), dtype=torch.bfloat16)
    ids = torch.empty((len(x), topk), dtype=torch.int32)
    score = torch.empty_like(ids, dtype=torch.float32)
    shard, si, ss = x[begin:end], ids[begin:end], score[begin:end]
    sparse_calls = []

    def sum_rank_group(idx, received, previous, dest, base, first):
        total = torch.zeros_like(dest, dtype=torch.float32) if first else previous.clone()
        for peer in idx:
            live = peer >= 0
            total[live] += received[(peer[live] - base).long()].float()
        dest.copy_(total)
        sparse_calls.append(idx.shape[0])

    env["sum_rank_group"] = sum_rank_group

    def gather(t, count):
        assert count == rows
        if t is ss: return score[:rows]
        if t is si: return ids[:rows]
        assert t is shard and f.capture.get() is None
        return x[:rows]

    def dense_return(t, grp):
        assert t.shape == (rows, hidden) and grp is group
        result = torch.zeros_like(x, dtype=torch.float32)
        result[:rows].copy_(t)
        dist.all_reduce(result)
        counts["dense_collective"] += 1
        return result[begin:end].bfloat16()

    env["layernorm_sp"] = NS(sp_num_tokens=lambda: rows, sp_exit_gather=gather,
                              reduce_scatter_token_rows=dense_return)
    graph = env["BreakableCUDAGraph"]()
    cap = env["BreakableCUDAGraphCapture"](graph)
    cap._begin_new_segment = lambda: None
    cap._end_current_segment = lambda: graph._segments.append(NS(replay=lambda: None))
    # Lower only the test-time route threshold to use small distributed fixtures.
    env["HUMMING_GROUPED_MIN_ROUTES_PER_EXPERT"] = 1
    model = NS(experts=e, num_experts=e.global_num_experts, top_k=topk)
    with cap:
        out = env["_forward_sp_replicated_experts"](model, shard, ss, si)
    assert out.shape == shard.shape and len(graph._segments) == 2 and len(graph._break_fns) == 1
    assert out.is_contiguous() and not out.any()
    address = out.data_ptr()
    rng = torch.Generator().manual_seed(7816 + rows)
    row = torch.arange(len(x), dtype=torch.int32)[:, None]
    slot = torch.arange(topk, dtype=torch.int32)[None, :]
    for mode in ("split", "self", "empty", "fanout", "hook", "hot", "multi", "empty_bank"):
        ids.copy_(((row % size) + (slot % 2) * size).expand_as(ids))
        if mode == "self": ids.copy_((row // shard_rows).expand_as(ids))
        if mode == "empty": ids.fill_(-1)
        if mode == "fanout": ids.copy_(((row + slot) % (size * 2)).expand_as(ids))
        if mode == "multi": ids.copy_((row + slot % max(1, 3 * size // 4)) % size)
        if mode == "hot": ids[:rows * 3 // 4] = 0
        if mode == "empty_bank": ids[ids % size == size - 1] = -1
        if mode in ("split", "empty_bank"): ids[1::3, ::2] = -1
        x.copy_((torch.randint(-12, 13, x.shape, generator=rng).float() / 8).bfloat16())
        score.copy_(torch.randn(score.shape, generator=rng).softmax(-1))
        ref, _, _ = f.reference(e, x[:rows], ids[:rows], score[:rows])
        hooked = mode == "hook" and rank == size - 1
        hook = e.register_forward_hook(lambda mod, args, result: result + 0.25) if hooked else None
        if hooked: ref += 0.25
        parts = [torch.empty_like(ref) for _ in range(size)]
        dist.all_gather(parts, ref)
        expected = torch.zeros_like(x, dtype=torch.float32)
        for item in parts:
            expected[:rows].add_(item)
        f.phase.records.clear()
        sparse_calls.clear()
        out.fill_(torch.nan)
        graph.replay()
        if hook: hook.remove()
        assert out.data_ptr() == address
        assert torch.equal(expected[begin:end].bfloat16().view(torch.int16), out.view(torch.int16)), (mode, rank)
        if mode in ("fanout", "hook"): assert not sparse_calls, (mode, sparse_calls)
        if mode in ("split", "empty", "self"): assert sparse_calls, mode
        if mode == "multi" and size >= 4: assert len(sparse_calls) >= 2
        if sparse_calls: assert sum(sparse_calls) == size
        counts["sparse" if sparse_calls else "dense"] += 1
        counts["owner_elements"] += out.numel()
        counts["return_collectives"] += len(sparse_calls)
        counts["multi_group_rounds"] += len(sparse_calls) > 1
        counts["captured_replays"] += 1
        counts["partial_finalizer_rows"] += sum(item[2].shape[0] for item in f.phase.records if item[0] == "finalize")


def worker(rank, size, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=size,
                            timeout=timedelta(seconds=60))
    f = fixture()
    counts = Counter()
    with torch.no_grad(), patch.object(torch.Tensor, "is_cuda", property(lambda t: True)), \
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
            patch.object(torch.cuda, "is_current_stream_capturing", side_effect=lambda: f.capture.get() is not None):
        for rows in (64, 67, 119, 256):
            exercise(f, rank, size, rows, counts)
    Path(output_dir, f"{size}-{rank}.json").write_text(json.dumps(counts))
    dist.destroy_process_group()


if __name__ == "__main__":
    total = Counter()
    with tempfile.TemporaryDirectory() as scratch:
        for size in (2, 4, 8):
            mp.spawn(worker, (size, str(Path(scratch) / f"gloo-{size}"), scratch), nprocs=size, join=True)
            for rank in range(size):
                total.update(json.loads(Path(scratch, f"{size}-{rank}.json").read_text()))
    gates = fixture().admission()
    print(json.dumps({"counts": total, "admission": gates, "torch": torch.__version__,
                      "scope": "Gloo peers, actual model method, graph callbacks, route sort and finalizer host path. CPU GEMMs, finalizer arithmetic and receiver kernel."}, indent=2))
