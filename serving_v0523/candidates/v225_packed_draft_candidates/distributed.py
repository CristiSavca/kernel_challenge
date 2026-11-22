"""Compare the two real selector paths from a materialized V225 checkout on CPU Gloo."""
import ast
import json
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import sys
import subprocess
import tempfile
from typing import Any, Optional, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.multiprocessing import spawn

from binding import bind

HERE = Path(__file__).resolve().parent
MODEL = "python/sglang/srt/models/dflash.py"
WORKER = "python/sglang/srt/speculative/dflash_worker_v2.py"


def load_nodes(stage, path, names, scope):
    code = ast.parse((stage / path).read_text())
    body = [n for n in code.body if getattr(n, "name", "") in names]
    assert len(body) == len(names)
    for node in body:
        # CPU eager arithmetic; optional compiled CPU check has its own harness.
        node.decorator_list = []
    exec(compile(ast.Module(body, []), path, "exec"), scope)


def load_method(stage, path, cls, name, scope):
    node = next(n for n in ast.parse((stage / path).read_text()).body if getattr(n, "name", "") == cls)
    method = next(n for n in node.body if getattr(n, "name", "") == name)
    exec(compile(ast.Module([method], []), path, "exec"), scope)
    return scope[name]


class CudaMetadataWeight(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda")


def worker(rank, size, temp, stage):
    stage = Path(stage)
    dist.init_process_group("gloo", init_method="file://" + temp + "/gloo", rank=rank, world_size=size)
    try:
        torch.set_num_threads(1)
        scope = dict(torch=torch, nn=nn, F=F, Tuple=Tuple, Any=Any, Optional=Optional,
                     _flashinfer_top_k=None, logger=logging.getLogger(), DFlashDraftModel=nn.Module)
        scope["logger"].setLevel(logging.ERROR)
        calls, setups = [], []

        def gather(t, dim):
            calls.append((str(t.dtype), dim, t.numel()))
            with torch.inference_mode(False):
                dest = [torch.empty_like(t) for _ in range(size)]
                dist.all_gather(dest, t.contiguous())
            return torch.cat(dest, dim=dim)

        def gather_specs(local):
            setups.append(local)
            dest = [None] * size
            dist.all_gather_object(dest, local)
            return dest

        parallel = NS(tp_size=size, tp_group=NS(world_size=size, rank_in_group=rank,
                      cpu_group=dist.group.WORLD, all_gather_object=gather_specs))
        scope.update(tensor_model_parallel_all_gather=gather, get_parallel=lambda: parallel,
                     get_exec=lambda: NS(graph=NS(cuda_graph_config=NS(decode=NS(bs=[1, 4, 16, 32])))),
                     unwrap_lora_layer=lambda x: x, is_cuda=lambda: True,
                     envs=NS(SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=NS(get=lambda: False)),
                     should_apply_lm_head_quant_method=lambda head, method: method is not None,
                     _DENSE_HEAD_DTYPES=(torch.bfloat16, torch.float16, torch.float32))
        load_nodes(stage, MODEL, {"_radix_topk", "_project_candidate_logits", "_pack_candidate_bits", "_rank_ordered_candidate_bits", "_pack_candidate_words",
                   "_candidate_word_scores", "_candidate_word_ids", "_merge_candidate_words", "candidate_topk",
                   "_score_edges", "_follow_maps", "CandidateSelector", "DFlash2DraftModel"}, scope)
        load_nodes(Path(temp) / "target", "python/sglang/srt/speculative/dflash_utils.py", {"is_dense_head_weight"}, scope)
        load_nodes(stage, WORKER, {"_selector_shard_starts", "_selector_lattice", "_SelectorDraftSampler"}, scope)
        make = load_method(stage, WORKER, "DFlashWorkerV2", "_maybe_build_draft_sampler", scope)
        sys.modules["sglang.srt.models.dflash"] = module = ModuleType("sglang.srt.models.dflash")
        module.DFlash2DraftModel = scope["DFlash2DraftModel"]
        model = module.DFlash2DraftModel.__new__(module.DFlash2DraftModel)
        nn.Module.__init__(model)
        torch.manual_seed(13)
        model.candidate_selector = scope["CandidateSelector"](hidden_size=32, vocab_size=512, state_rank=8, top_k=16).bfloat16()
        model.draft_config = NS(output_multiplier=0.75, final_logit_softcapping=19.5)
        with torch.no_grad():
            for param in model.candidate_selector.parameters():
                param.normal_(std=0.15)
        starts = [2, 73, 154, 232]
        counts = [48, 61, 56, 44]
        k, rows, h = 16, counts[rank], 32
        weight = torch.randn(rows, h, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(28 + rank))
        head = NS(shard_indices=NS(org_vocab_start_index=starts[rank], num_org_elements=rows),
                  org_vocab_size=sum(counts), quant_method=None, weight=weight.as_subclass(CudaMetadataWeight))
        w = NS(_target_tp_rank=rank, block_size=8, selector=model.candidate_selector, draft_model=model, device="cpu",
               _selector_sampling_enabled=True, _target_worker=NS(model_runner=NS(model=NS(lm_head=head))))
        original_tensor = torch.tensor

        def host_metadata_tensor(data, *args, **kwargs):
            if kwargs.get("device") == torch.device("cuda"):
                kwargs["device"] = torch.device("cpu")
            return original_tensor(data, *args, **kwargs)

        with patch.object(torch, "tensor", host_metadata_tensor):
            # CUDA metadata presented to the real initializer, Gloo runs on every rank.
            enabled = make(w)
            assert enabled.shard_starts.tolist() == starts and len(setups) == 1
            # Each negative case affects one rank only; other ranks must decline as well.
            old_head = w._target_worker.model_runner.model
            for case in ("width", "negative", "offset64", "dtype", "quant", "custom", "missing", "k"):
                start, width, old_k = head.shard_indices.org_vocab_start_index, head.shard_indices.num_org_elements, model.candidate_selector.top_k
                cuda_weight = head.weight
                if rank == 1:
                    if case == "width":
                        head.shard_indices.num_org_elements = 65537
                    elif case == "negative":
                        head.shard_indices.org_vocab_start_index = -1
                    elif case == "offset64":
                        head.shard_indices.org_vocab_start_index = 2 ** 63 - 1
                    elif case == "dtype":
                        head.weight = cuda_weight.float()
                    elif case == "quant":
                        head.quant_method = object()
                    elif case == "custom":
                        model.compute_candidates = lambda hidden: None
                    elif case == "missing":
                        old_head.lm_head = None
                    elif case == "k":
                        model.candidate_selector.top_k = k - 1
                declined = make(w)
                assert declined is None or declined.shard_starts is None
                head.shard_indices.org_vocab_start_index, head.shard_indices.num_org_elements = start, width
                head.weight, head.quant_method, old_head.lm_head, model.candidate_selector.top_k = cuda_weight, None, head, old_k
                if "compute_candidates" in vars(model):
                    del model.compute_candidates
            assert len(setups) == 9
        head.weight = weight
        model.lm_head = head
        # Read and run V224's separate chooser and candidate_topk functions, using
        # the same head and selector parameters as the contender on this rank.
        baseline_stage = Path(temp) / "baseline"
        reference_scope = dict(scope)
        load_nodes(baseline_stage, MODEL, {"_radix_topk", "_project_candidate_logits", "candidate_topk",
                   "_score_edges", "_follow_maps", "CandidateSelector", "DFlash2DraftModel"}, reference_scope)
        load_nodes(baseline_stage, WORKER, {"_selector_lattice", "_SelectorDraftSampler"}, reference_scope)
        base_model = reference_scope["DFlash2DraftModel"].__new__(reference_scope["DFlash2DraftModel"])
        nn.Module.__init__(base_model)
        base_selector = reference_scope["CandidateSelector"](hidden_size=32, vocab_size=512, state_rank=8, top_k=16).bfloat16()
        base_selector.load_state_dict(model.candidate_selector.state_dict())
        base_model.candidate_selector = base_selector
        base_model.draft_config, base_model.lm_head = model.draft_config, head
        legacy = reference_scope["_SelectorDraftSampler"](draft_model=base_model, max_bs=32, device="cpu",
                                                       block_size=8, sampling_enabled=True)
        counters = dict(tokens=0, ids=0, q=0, bytes_native=0, bytes_packed=0, replays=0)
        with torch.inference_mode():
            for kind in ("random", "all_ties", "boundary_ties"):
                head.weight = weight if kind == "random" else torch.zeros_like(weight)
                if kind == "boundary_ties":
                    head.weight[::2] = 1
                for bs in (1, 4, 16, 32):
                    hidden = torch.randn(bs * 8, h, generator=torch.Generator().manual_seed(bs + 123)).bfloat16()
                    anchors = torch.arange(bs * 8).remainder(256)
                    for sampler in (legacy, enabled):
                        sampler.greedy_mask[:bs] = torch.arange(bs) % 2 == 0
                        sampler.temperatures[:bs] = torch.linspace(0.5, 1.7, steps=bs)
                    calls.clear()
                    torch.manual_seed(bs + 59)
                    legacy(hidden, anchors)
                    assert calls == [("torch.float32", -1, bs*7*k), ("torch.int64", -1, bs*7*k)]
                    calls.clear()
                    torch.manual_seed(bs + 59)
                    enabled(hidden, anchors)
                    assert calls == [("torch.int32", 0, bs*7*k)] and len(setups) == 9
                    for name, n, counter in (("out", bs*7, "tokens"), ("candidate_out", bs, "ids"), ("q_out", bs, "q")):
                        a, b = getattr(legacy, name)[:n], getattr(enabled, name)[:n]
                        assert torch.equal(a, b), (rank, kind, bs, name)
                        counters[counter] += b.numel()
                    counters["replays"] += 1
                    counters["bytes_native"] += 12 * bs * 7 * k
                    counters["bytes_packed"] += 4 * bs * 7 * k
            # Native partition call ignores a supplied capture plan, as required by LiLiCorr.
            def lse_local(x, k):
                vals, ids = torch.topk(x, k)
                return vals, ids, torch.logsumexp(x.float(), -1)
            scope["lilicorr_topk_lse"] = lse_local
            for starts_arg in (None, enabled.shard_starts):
                calls.clear()
                result = scope["candidate_topk"](hidden[:7], head, k, with_partition=True, candidate_shard_starts=starts_arg)
                assert [c[:2] for c in calls] == [("torch.float32", -1), ("torch.int64", -1), ("torch.float32", -1)]
                if starts_arg is None:
                    reference = result
                else:
                    assert all(torch.equal(a, b) for a, b in zip(reference, result))
            # Unsupported native radix output width falls back to the original gather pair.
            radix = scope["_radix_topk"]
            scope["_radix_topk"] = lambda logits, k: (lambda pair: (pair[0].float(), pair[1]))(radix(logits, k))
            calls.clear()
            scope["candidate_topk"](hidden[:7], head, k, candidate_shard_starts=enabled.shard_starts)
            assert [c[:2] for c in calls] == [("torch.float32", -1), ("torch.int64", -1)]
        (Path(temp) / f"rank{rank}.json").write_text(json.dumps(counters))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    stage = Path(sys.argv[1]).resolve()
    bound = bind(stage, sys.argv[2] if len(sys.argv) > 2 else HERE,
                 [MODEL, WORKER, "python/sglang/srt/speculative/dflash_utils.py"])
    with tempfile.TemporaryDirectory() as temp:
        # A sparse integration checkout may omit this shared, unmodified predicate.
        helper = "python/sglang/srt/speculative/dflash_utils.py"
        out = Path(temp) / "target" / helper
        out.parent.mkdir(parents=True)
        out.write_bytes(subprocess.check_output(["git", "-C", str(stage), "show", bound["tree"] + ":" + helper]))
        for path in (MODEL, WORKER):
            ref = Path(temp) / "baseline" / path
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_bytes(subprocess.check_output(["git", "-C", str(stage), "show", bound["baseline_tree"] + ":" + path]))
        spawn(worker, args=(4, temp, str(stage)), nprocs=4, join=True)
        all_counts = [json.loads((Path(temp) / f"rank{i}.json").read_text()) for i in range(4)]
    assert len({tuple(x.values()) for x in all_counts}) == 1
    print(json.dumps(dict(**bound,
        tp=4, torch=torch.__version__, backend="real Gloo, CPU tensors with CUDA metadata in setup",
        batch_sizes=[1,4,16,32], local_vocab=[48,61,56,44], vocab_starts=[2,73,154,232],
        per_rank=all_counts[0], native_calls_per_forward=2, packed_calls_per_forward=1,
        asymmetric_admission_denials=8, per_forward_cpu_votes=0, native_lse_and_wide_score_fallbacks=True,
        cpu_selector_walk=True, actual_FlashInfer=False, CUDA_graph=False, physical_H200=False), indent=2))
