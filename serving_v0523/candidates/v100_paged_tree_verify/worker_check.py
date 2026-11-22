"""Run the worker's actual commit branch and setup vote with CPU tensors and Gloo."""

import argparse
import ast
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def sections(file):
    worker = next(stmt for stmt in ast.parse(file.read_text()).body
                  if isinstance(stmt, ast.ClassDef) and stmt.name == "DFlashWorkerV2")
    methods = {node.name: node for node in worker.body if isinstance(node, ast.FunctionDef)}
    def module(nodes):
        return compile(ast.Module(nodes, []), str(file), "exec")
    setup = module([methods["_prepare_paged_tree_commit"]])
    enable = next(stmt for stmt in methods["init_attention_backends"].body
                  if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "get_spec().speculative_dflash_tree")
    body = methods["forward_batch_generation"].body
    first = next(i for i, stmt in enumerate(body) if isinstance(stmt, ast.Assign)
                 and ast.unparse(stmt.value) == "self._draft_write_ids(kv_loc_plan)")
    last = next(i for i, stmt in enumerate(body[first:], first)
                if isinstance(stmt, ast.If) and "_tree_commit_done.record" in ast.unparse(stmt))
    return setup, module([enable]), module(body[first:last + 1])


def commit_checks(program):
    width, bs = 8, 3
    count = 0
    for page in (1, 16, 128, 160, 256):
        for branched in (False, True):
            lengths = torch.tensor([4, 1, 7], dtype=torch.int32)
            prefix = torch.tensor([page - 1, 2 * page + 3, 7], dtype=torch.int64)
            positions = prefix[:, None] + torch.arange(width)
            slots = torch.tensor([[2, 8, 7, 11, 10, 4, 5, 3], [19, 21, 23, 25, 32, 33, 28, 34],
                                  [43, 38, 40, 41, 39, 44, 36, 37]], dtype=torch.int64)
            if page > 1:
                # Disjoint physical pages, retaining logical in-page offsets even
                # when a request's reserved suffix straddles a page boundary.
                alloc = torch.tensor([[2, 9], [5, 13], [6, 10]])
                local = positions // page - (prefix // page)[:, None]
                slots = alloc.gather(1, local) * page + (positions % page)
            saved = slots.clone()
            order = torch.tensor([[0, 2, 4, 6, 1, 3, 5, 7], [0, 1, 2, 3, 4, 5, 6, 7],
                                  [0, 1, 3, 4, 5, 6, 7, 2]]) if branched else None
            src = saved.gather(1, order) if branched else saved
            event_log = []
            pool = object()
            hidden = torch.arange(bs * width * 2).reshape(bs * width, 2)

            def reorder(table, request_ids, start, loc, permutation):
                assert page == 1 and branched and table is slots
                assert torch.equal(loc, saved) and torch.equal(start, prefix)
                event_log.append("reorder")
                table.copy_(loc.gather(1, permutation))
                return table

            def compact(actual_pool, loc, permutation, commits):
                assert actual_pool is pool and page > 1 and branched
                assert torch.equal(loc, saved) and torch.equal(commits, lengths)
                assert torch.equal(loc.gather(1, permutation), src)
                event_log.append("compact")

            def append(**kw):
                assert torch.equal(kw["target_hidden"], hidden)
                assert torch.equal(kw["positions"], positions.flatten())
                assert kw["tree_order"] is order
                expected = src if page == 1 else saved
                assert torch.equal(kw["cache_loc_2d"], expected)
                assert torch.equal(kw["cache_loc"].reshape(bs, width), expected)
                event_log.append("draft")

            event = SimpleNamespace(record=lambda: event_log.append("fence"))
            runner = SimpleNamespace(token_to_kv_pool=pool, shared_read_done_event=None,
                                     req_to_token_pool=SimpleNamespace(req_to_token=slots))
            w = SimpleNamespace(page_size=page, model_runner=runner, _tree_commit_done=event,
                                _draft_write_ids=lambda p: slots.flatten(), _append_target_hidden_to_draft_kv_by_loc=append)
            def publish(new_lens):
                assert new_lens is next_lengths and runner.shared_read_done_event is event
                event_log.append("publish")
            next_lengths = prefix + lengths
            ns = dict(self=w, kv_loc_plan=None, order=order, batch=SimpleNamespace(req_pool_indices=torch.arange(bs)),
                      prefix_lens=prefix, bs=bs, block_size=width, hidden=hidden, positions=positions.flatten(),
                      verify_out_cache_loc_2d=slots, commit_lens=lengths, new_seq_lens=next_lengths,
                      compact_tree_kv=compact, reorder_dflash_tree_slots=reorder, on_publish=publish)
            exec(program, ns)
            expected = (["reorder" if page == 1 else "compact", "draft", "fence", "publish"]
                        if branched else ["draft"])
            assert event_log == expected
            assert torch.equal(slots, src if page == 1 else saved)
            count += 1
    return count


def vote_check(rank, init, path, outfile):
    dist.init_process_group("gloo", init_method=init, world_size=2, rank=rank, timeout=timedelta(seconds=35))
    setup, enable, commit = sections(path)
    logger = SimpleNamespace(warning=lambda *a: None, info=lambda *a: None)
    group = SimpleNamespace(world_size=2, cpu_group=dist.group.WORLD)
    cases = 0
    for page, fault in ((160, None), (160, "compile"), (160, "layout"), (160, "backend"), (1, None)):
        pool, reqmap = object(), object()
        calls = []
        def supports(p):
            assert p is pool
            return not (fault == "layout" and rank == 1)
        def warm(p, slots, order, lens):
            assert p is pool and torch.count_nonzero(lens) == 0
            assert slots.dtype == torch.int64 and order.tolist() == [list(range(8))]
            calls.append("warm")
            if fault == "compile" and rank == 1:
                raise RuntimeError("synthetic compile failure")
        # The device event itself is GPU-only. Patch only its construction;
        # PyTorch's real CPU Gloo process group handles the capability vote.
        backend = SimpleNamespace(supports_dflash_tree=not (fault == "backend" and rank == 1),
                                  enable_dflash_tree=False)
        runner = SimpleNamespace(attn_backend=backend, req_to_token_pool=reqmap, token_to_kv_pool=pool,
                                 token_to_kv_pool_allocator=object(), lora_manager=None, decode_attn_backend=None)
        w = SimpleNamespace(model_runner=runner, draft_model_runner=SimpleNamespace(req_to_token_pool=reqmap,
                            kv_index_translator=SimpleNamespace(is_translating=False)), page_size=page,
                            block_size=8, device="cpu", _target_tp_rank=rank, selector=SimpleNamespace(top_k=16),
                            _need_mamba_verify_commit=False, use_compact_draft_cache=False, _tree_verify_backend=None)
        ns = dict(torch=torch, logger=logger, get_dp_tp_group=lambda: group, tree_kv_compaction_supported=supports,
                  compact_tree_kv=warm, get_spec=lambda: SimpleNamespace(speculative_dflash_tree=True),
                  self=w, is_swa_req_ring=lambda _:False, SIMULATE_ACC_LEN=0)
        exec(setup, ns)
        w._prepare_paged_tree_commit = lambda eligible: ns["_prepare_paged_tree_commit"](w, eligible)
        from unittest.mock import patch
        with patch.object(torch, "get_device_module", return_value=SimpleNamespace(Event=object)):
            exec(enable, ns)
        assert backend.enable_dflash_tree == (fault is None)
        assert (w._tree_verify_backend is backend) == (fault is None)
        assert bool(calls) == (page > 1 and not (rank == 1 and fault in ("layout", "backend")))
        cases += 1
    dist.destroy_process_group()
    if rank == 0:
        Path(outfile).write_text(json.dumps(dict(gloo_ranks=2, setup_cases=cases,
                                  worker_commit_cases=commit_checks(commit), device="CPU"),indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    path = args.checkout / "python/sglang/srt/speculative/dflash_worker_v2.py"
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(vote_check, args=(Path(directory, "init").as_uri(), path, args.receipt), nprocs=2, join=True)
