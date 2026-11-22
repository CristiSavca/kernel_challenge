"""Exercise the V99 worker's real proposal selection through a two-rank Gloo collective."""

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from independent_counterfactual import Pointer, make_tree, program


def worker_slice(path):
    module = ast.parse(path.read_text())
    method = next(node for node in ast.walk(module)
                  if isinstance(node, ast.FunctionDef) and node.name == "forward_batch_generation"
                  and any(isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "verify_input"
                          for stmt in node.body))
    start = next(i for i, stmt in enumerate(method.body) if isinstance(stmt, ast.Assign)
                 and ast.unparse(stmt.targets[0]) == "draft_tokens"
                 and "_draft_block_tokens_buf" in ast.unparse(stmt.value))
    end = next(i for i in range(start, len(method.body)) if isinstance(method.body[i], ast.Assign)
               and ast.unparse(method.body[i].targets[0]) == "verify_input")
    return compile(ast.Module(method.body[start:end + 1], []), str(path), "exec")


def tree_api(path):
    keep = {"DFlashTree", "select_dflash_tree"}
    module = ast.parse(path.read_text())
    module.body = [node for node in module.body if getattr(node, "name", None) in keep]
    pid, (choose,) = program(path, ("_select_chain_or_tree",))

    class CpuLauncher:
        def __getitem__(self, grid):
            def run(original, chain, advantage, selected, plane, width, block, **_):
                args = [Pointer(tensor.numpy()) for tensor in (original, chain, advantage, selected)]
                for row in range(grid[0]):
                    pid[0] = row
                    choose(*args, plane, width, block)
            return run

    ns = dict(torch=torch, dataclass=dataclass, _select_chain_or_tree=CpuLauncher(),
              triton=SimpleNamespace(next_power_of_2=lambda i: 1 << (i - 1).bit_length()))
    exec(compile(module, str(path), "exec"), ns)
    return ns["DFlashTree"], ns["select_dflash_tree"]


def worker(rank, source, rendezvous, result):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    code = worker_slice(source / "srt/speculative/dflash_worker_v2.py")
    tree_type, select = tree_api(source / "kernels/ops/speculative/dflash_tree.py")
    counts = dict(batches=0, broadcasts=0, reset_batches=0, requests=0)

    class Group:
        world_size = 2
        rank_in_group = rank

        def broadcast(self, tensor, src):
            assert src == 0
            counts["broadcasts"] += 1
            dist.broadcast(tensor, src=src)

    graph_capacity, width = 8, 8
    parents = [-1, 0, 0, 1, 3, 2, 4, 5]
    graph = tree_type(torch.empty((4, graph_capacity, width), dtype=torch.int64))
    chain_buf = torch.empty((graph_capacity, width), dtype=torch.int64)
    group = Group()
    for step, bs in enumerate((8, 3, 1, 5, 7, 2, 4, 8, 1, 6, 3, 8)):
        stage = []
        backend = SimpleNamespace(stage_dflash_tree=lambda depth, mask: stage.append((depth, mask)))
        graph.data.fill_(-12345)
        native = torch.arange(bs * width).reshape(bs, width) + 12345 + step * 100
        reference = torch.empty((4, bs, width), dtype=torch.int64)
        for row in range(bs):
            labels = native[row].clone()
            labels[1:] += 1200
            reference[:, row] = torch.from_numpy(make_tree(labels, parents))
        graph.data[:, :bs].copy_(reference)
        graph.tokens[:bs] += rank * 10000
        local_chain = native + rank * 20000
        advantage = torch.tensor([[-4, 2, 0][(step + r) % 3] for r in range(bs)], dtype=torch.int32)
        prior = None if step % 6 == 0 else advantage if rank == 0 else -advantage
        enabled = step % 6 != 4
        eligible = step % 6 != 3
        allowed_logits = step % 6 != 5
        prefix = torch.arange(bs) * 19 + 32
        positions = (prefix[:, None] + torch.arange(width)).flatten()
        worker_state = SimpleNamespace(_draft_block_tokens_buf=chain_buf, block_size=width,
                                       _tree_verify_backend=backend, _draft_sampler=SimpleNamespace(tree_out=graph),
                                       _can_skip_verify_logits=lambda *args: allowed_logits)
        batch = SimpleNamespace(sampling_info=None, has_grammar=False,
                                reqs=[SimpleNamespace(return_routed_experts=False, return_indexer_topk=False)] * bs)
        ns = dict(self=worker_state, bs=bs, torch=torch, batch=batch, block_ids=local_chain,
                  draft_next=local_chain[:, 1:], draft_out=SimpleNamespace(can_run_graph=eligible),
                  get_dp_tp_group=lambda: group, get_spec=lambda: SimpleNamespace(speculative_dflash_adaptive_tree=enabled),
                  draft_input=SimpleNamespace(tree_policy_advantage=prior), select_dflash_tree=select,
                  positions=positions, prefix_lens=prefix, kv_loc_plan=None, DFlashVerifyInput=SimpleNamespace,
                  CaptureHiddenMode=SimpleNamespace(FULL=1))
        exec(code, ns)
        verify = ns["verify_input"]
        if eligible and allowed_logits:
            expected = reference.clone()
            if enabled and prior is not None:
                for row in range(bs):
                    if advantage[row] > 0:
                        expected[:, row] = torch.from_numpy(make_tree(native[row], np.arange(-1, width - 1)))
            assert torch.equal(ns["tree"].data, expected)
            assert torch.equal(verify.draft_token.reshape(bs, width), expected[0])
            assert torch.equal(verify.positions.reshape(bs, width), prefix[:, None] + expected[1])
            assert torch.equal(stage[0][0], expected[1]) and torch.equal(stage[0][1], expected[3])
            if enabled and rank == 0:
                assert torch.equal(ns["original_tree"].data, reference)
                assert torch.equal(ns["linear_tokens"], native)
            else:
                assert ns["original_tree"] is None
        else:
            assert ns["tree"] is None and stage == [(None, None)]
            assert torch.equal(verify.positions, positions)
            assert torch.equal(verify.draft_token.reshape(bs, width), local_chain)
            counts["reset_batches"] += 1
        assert torch.all(graph.data[:, bs:] == -12345)
        counts["batches"] += 1
        counts["requests"] += bs
    dist.destroy_process_group()
    if rank == 0:
        Path(result).write_text(json.dumps({"backend": "Gloo", "ranks": 2, **counts}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(args.checkout / "python/sglang", Path(directory, "store").as_uri(), args.receipt),
                 nprocs=2, join=True)
