"""Exercise the candidate projection and complete tap-to-KV path on CPU/Gloo."""
import ast
import importlib.util
import json
import logging
import os
import sys
import tempfile
import types
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SOURCE = Path(sys.argv[1]) / "python/sglang/srt"
NS = types.SimpleNamespace


def method(filename, cls, name, globals_):
    tree = ast.parse((SOURCE / filename).read_text())
    owner = next(n for n in tree.body if getattr(n, "name", None) == cls) if cls else tree
    node = next(n for n in owner.body if getattr(n, "name", None) == name)
    block = ast.Module(body=[node], type_ignores=[])
    exec(compile("from __future__ import annotations\n" + ast.unparse(block), str(filename), "exec"), globals_)
    return globals_[name]


def load_projection():
    path = SOURCE / "layers/dflash_context_projection.py"
    spec = importlib.util.spec_from_file_location("actual_fc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Group:
    def __init__(self, size):
        self.world_size = size
        self.cpu_group = dist.group.WORLD
        self.gathers = []

    def all_gather_into_tensor(self, out, local):
        self.gathers.append(tuple(out.shape))
        # Gloo copies to its output on a background thread. Give that thread
        # ordinary storage, then copy back on the inference-mode caller.
        with torch.inference_mode(False), torch.no_grad():
            local = local.detach().clone()
            collected = torch.empty_like(out)
            dist.all_gather_into_tensor(collected, local, group=self.cpu_group)
        out.copy_(collected)

def run(rank, size, rendezvous):
    from unittest.mock import patch

    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=size, timeout=timedelta(seconds=90))
    torch.set_num_threads(1)
    candidate = load_projection()
    group = Group(size)
    f, h = 40, 8
    sp = method("layers/layernorm_sp.py", None, "sp_exit_gather",
                {"torch": torch, "get_parallel": lambda: NS(tp_group=group)})
    pack = method("layers/aux_hidden_states.py", None, "pack_aux_hidden_states",
                  {"torch": torch})
    full = NS(is_full=lambda: True, need_capture=lambda: True)
    capture = method("layers/logits_processor.py", "LogitsProcessor", "_get_hidden_states_to_store",
                     {"pack_aux_hidden_states": pack})
    trim = method("model_executor/runner/prefill_cuda_graph_runner.py", "PrefillCudaGraphRunner",
                  "_trim_logits_output", {"LogitsProcessorOutput": NS})
    logits_weight = torch.arange(24).reshape(3, h).bfloat16()

    class LP(torch.nn.Module):
        keep_last_hidden_states = False
        expected_aux_hidden_states = None
        def forward(self, ids, hidden, head, batch, aux):
            value = capture(self, hidden, None, aux, None, None, None, None, batch)
            return NS(hidden_states=value, next_token_logits=torch.nn.functional.linear(hidden[[1,-1]], logits_weight),
                      dflash_prefill_fc_done=False, last_hidden_states=None, mm_input_embeds=None,
                      input_token_logprobs=None, input_top_logprobs_val=None, input_top_logprobs_idx=None,
                      input_token_ids_logprobs_val=None, input_token_ids_logprobs_idx=None,
                      input_logprobs_copy_done=None, customized_info=None, hidden_states_token_indices=None)
    forward = method("models/muse_spark.py", "MuseSparkTextForCausalLM", "forward", {
        "torch": torch, "pack_aux_hidden_states": pack,
        "get_parallel": lambda: NS(tp_group=group), "get_forward": lambda: NS(set=lambda *a: None),
        "layernorm_sp": NS(sp_exit_gather=sp, runs_sp=lambda mode: mode=="extend"),
        "project_dflash_prefill_shard": candidate.project_dflash_prefill_shard,
    })
    class Model(torch.nn.Module):
        capture_aux_hidden_states = True
        lm_head = None
    target = Model()
    lp = target.logits_processor = LP()
    class Draft(torch.nn.Module):
        quantizable = False
    Draft.project_target_hidden = method("models/dflash.py", "DFlashDraftModel", "project_target_hidden", {"torch": torch})
    draft = Draft()
    fc = draft.fc = torch.nn.Linear(f, h, bias=False).bfloat16()
    w = (torch.arange(h*f).reshape(h, f)%7-3).bfloat16()
    with torch.no_grad():
        fc.weight.copy_(w)
    norm = draft.hidden_norm = torch.nn.RMSNorm(h, eps=1e-5).bfloat16()
    for name, cls_name, cls in [("sglang.srt.models.dflash", "DFlashDraftModel", Draft),
                                ("sglang.srt.layers.logits_processor", "LogitsProcessor", LP)]:
        sys.modules[name] = types.ModuleType(name)
        setattr(sys.modules[name], cls_name, cls)
    assert candidate.prepare_dflash_sp_prefill(draft, target, lp, group) is None
    # Exercise the real group startup checks on CPU, replacing only the two
    # physical GPU capability queries.
    with patch.object(torch.Tensor, "is_cuda", property(lambda _: True)):
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            draft.quantizable = rank == 0
            assert candidate.prepare_dflash_sp_prefill(draft, target, lp, group) is None
            draft.quantizable = False
            context = candidate.prepare_dflash_sp_prefill(draft, target, lp, group)
            assert context is not None

    append = method("speculative/dflash_worker_v2.py", "DFlashWorkerV2",
                    "_append_target_hidden_to_draft_kv_by_loc", {
        "torch": torch, "logger": logging.getLogger("fixture"), "draft_tp_context": lambda _: nullcontext()
    })
    captured = []
    worker = NS(draft_parallel_layout=None, draft_model=draft, model_runner=NS(device=torch.device("cpu")),
                lilicorr=None, _use_fused_kv_materialize=True, _fused_kv_helper=object(),
                _append_target_hidden_fused=lambda **k: captured.append(k))
    cfg = NS(model_runner=NS(spec_algorithm=NS(is_speculative=lambda: True)), _is_full_backend=True, raw_bs=2)
    threshold = 128 * size
    tags = ["short", "boundary", "tail", "bucket", "trim", "local-hook", "global-hook", "output-hook",
            "target-hook", "external-hidden", "expected-external", "late-quantizable", "deterministic", "dtype", "weight-stride", "fc-override",
            "model-override", "replace-fc", "compiled-fc", "wrong-width", "underfilled", "not-full", "no-scope"]
    with torch.inference_mode():
        for tag in tags:
            n = threshold - 1 if tag == "short" else threshold if tag == "boundary" else threshold + 11
            rows = (n + size - 1) // size + (7 if tag == "bucket" else 0)
            valid = n - (9 if tag in ("trim", "bucket") else 0)
            taps = (torch.arange(rows*size*f).reshape(-1, f)%9-4).bfloat16()
            hidden = (torch.arange(rows*size*h).reshape(-1,h)%11-5).bfloat16()
            local_taps, local_hidden = taps.chunk(size)[rank], hidden.chunk(size)[rank]
            target.model = lambda *a, **k: (local_hidden, list(local_taps.split(h, dim=1)))
            batch = NS(forward_mode="extend", capture_hidden_mode=full)
            embeds = torch.zeros((n,h), dtype=torch.bfloat16)
            group.gathers.clear()
            baseline = forward(target, None, None, batch, input_embeds=embeds)
            assert group.gathers == [(rows*size,f),(rows*size,h)]
            assert torch.equal(baseline.hidden_states, taps[:n])
            restore = lambda: None
            if rank == size-1:
                if tag in ("local-hook", "target-hook", "output-hook"):
                    module = {"local-hook":fc, "target-hook":target, "output-hook":lp}[tag]
                    restore = module.register_forward_pre_hook(lambda *a: None).remove
                if tag == "global-hook":
                    restore = torch.nn.modules.module.register_module_forward_hook(lambda *a: None).remove
                if tag == "external-hidden":
                    lp.keep_last_hidden_states = True
                    restore = lambda: setattr(lp, "keep_last_hidden_states", False)
                if tag == "expected-external":
                    lp.expected_aux_hidden_states = 5
                    restore = lambda: setattr(lp, "expected_aux_hidden_states", None)
                if tag == "late-quantizable":
                    draft.quantizable = True
                    restore = lambda: setattr(draft, "quantizable", False)
                if tag == "deterministic":
                    torch.use_deterministic_algorithms(True)
                    restore = lambda: torch.use_deterministic_algorithms(False)
                if tag == "fc-override":
                    fc.forward = lambda x: torch.nn.functional.linear(x,w)
                    restore = lambda: delattr(fc, "forward")
                if tag == "model-override":
                    draft.project_target_hidden = lambda x: norm(fc(x))
                    restore = lambda: delattr(draft, "project_target_hidden")
                if tag == "replace-fc":
                    draft.fc = torch.nn.Linear(f,h,bias=False).bfloat16()
                    restore = lambda: setattr(draft,"fc",fc)
                if tag == "compiled-fc":
                    fc._compiled_call_impl = fc.forward
                    restore = lambda: setattr(fc, "_compiled_call_impl", None)
                if tag == "weight-stride":
                    old = fc.weight
                    fc.weight = torch.nn.Parameter(w.T.contiguous().T)
                    restore = lambda: setattr(fc, "weight", old)
            with candidate.dflash_sp_prefill_scope(None if tag == "no-scope" else context):
                if tag in ("dtype", "wrong-width", "underfilled", "not-full"):
                    probe = local_taps
                    if rank == size-1:
                        probe = {"dtype":probe.float(), "wrong-width":probe[:,:-1],
                                 "underfilled":probe[:1], "not-full":probe}[tag]
                    assert candidate.project_dflash_prefill_shard(probe, n, group, lp,
                        not (rank == size-1 and tag == "not-full")) is None
                group.gathers.clear()
                result = forward(target, None, None, batch, input_embeds=embeds)
            restore()
            enabled = tag in ("boundary","tail","bucket","trim","dtype","wrong-width","underfilled","not-full")
            assert result.dflash_prefill_fc_done is enabled, (tag,rank)
            assert group.gathers == [(rows*size,h if enabled else f),(rows*size,h)]
            assert torch.equal(result.next_token_logits,baseline.next_token_logits)
            expected = torch.nn.functional.linear(taps[:n], w)
            assert torch.equal(result.hidden_states, expected if enabled else taps[:n])
            cfg.raw_num_tokens = valid
            ls = torch.arange(valid-1,-1,-1)
            ps = torch.arange(30, 30+valid)
            for out in (baseline, result):
                trimmed = trim(cfg,out)
                assert trimmed.dflash_prefill_fc_done == out.dflash_prefill_fc_done
                assert torch.equal(trimmed.hidden_states, out.hidden_states[:valid])
                append(worker, target_hidden=trimmed.hidden_states, cache_loc=ls, positions=ps,
                       prefill_fc_done=trimmed.dflash_prefill_fc_done)
            for key in ("ctx_hidden", "ctx_positions", "ctx_cache_loc"):
                assert torch.equal(captured[-1][key], captured[-2][key]), (tag,key)
            assert torch.equal(captured[-1]["ctx_hidden"], norm(expected[:valid]))
            assert torch.equal(captured[-1]["ctx_cache_loc"], ls)
            assert torch.equal(captured[-1]["ctx_positions"], ps)
        try:
            with candidate.dflash_sp_prefill_scope(context):
                with candidate.dflash_sp_prefill_scope(None):
                    assert candidate._prefill_context.get() is None
                assert candidate._prefill_context.get() is context
                raise RuntimeError("scope")
        except RuntimeError:
            assert candidate._prefill_context.get() is None
    dist.destroy_process_group()


def full_k_fixture():
    # Keep the official reduction width. Seven output columns and 259 rows
    # bound the fixture to about 23 MiB of BF16 activations and weights.
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(802)
    rows, width, outputs = 259, 40_960, 7
    x = (torch.randint(-8, 9, (rows,width), generator=gen, dtype=torch.int8).bfloat16()/8)
    w = (torch.randint(-8, 9, (outputs,width), generator=gen, dtype=torch.int8).bfloat16()/8)
    expected = torch.nn.functional.linear(x,w)
    world_results = []
    for size in (2,4,8):
        parts = torch.tensor_split(x, size)
        found = torch.cat([torch.nn.functional.linear(part,w) for part in parts])
        assert torch.equal(expected,found)
        world_results.append(size)
    # Compare against FP32 using dyadic values whose integer sums are exact.
    torch.testing.assert_close(expected.float(), torch.nn.functional.linear(x.float(),w.float()).bfloat16().float(),
                               rtol=0, atol=0)
    return {"M":rows,"N":outputs,"K":width,"tp_sizes":world_results, "bf16_bitwise_equal":True}


if __name__ == "__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with tempfile.TemporaryDirectory() as directory:
        for size in (2, 4, 8):
            mp.spawn(run, args=(size, str(Path(directory)/str(size))), nprocs=size, join=True)
    print(json.dumps({"torch":torch.__version__, "collectives":"actual Gloo all_gather_into_tensor and all_reduce(MIN)",
        "tp_sizes":[2,4,8], "full_K_fixture":full_k_fixture(), "cases_per_size":23, "early_startup_checks_per_size":3,
        "paths":"production FC, tap pack and gather, Muse text model forward, logits FULL capture, replay trim and draft norm/KV boundary",
        "equality":"bit-exact BF16 fixture comparisons across every rank, sampled-position logits, every draft input, positions, reversed slots",
        "boundary_cases":"threshold, overallocated buckets, row trim, one-rank hooks/overrides/compiled/strides/dtype/visibility and deterministic mode, nested scope",
        "h200_executed":False},indent=2))
