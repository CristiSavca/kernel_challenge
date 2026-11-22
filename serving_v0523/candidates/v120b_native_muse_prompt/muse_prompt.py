"""Exercise Muse's real prompt and logit methods on distributed CPU tensors."""

import ast
from contextlib import nullcontext
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp


def source_class(source, name, scope, methods=None):
    cls = next(node for node in ast.parse(source.read_text()).body if getattr(node, "name", None) == name)
    if methods is not None:
        cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
    block = ast.Module([cls], [])
    exec("from __future__ import annotations\n" + ast.unparse(block), scope)
    return scope[name]


def run(rank, world, rendezvous, stage, original_source):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=world, timeout=timedelta(minutes=2))
    path = Path(__file__).resolve().parents[1] / "v108_replicated_sp_draft_context/cpu_check.py"
    sys.argv = [sys.argv[0], str(stage)]
    spec = importlib.util.spec_from_file_location("sp_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    srt = fixture.SOURCE
    corrected = fixture.load_projection()
    original = ModuleType("original")
    exec(original_source, vars(original))
    group = fixture.Group(world)
    mode = NS(is_dllm_extend=lambda: False, is_target_verify=lambda: False, is_decode_or_idle=lambda: False,
              is_draft_extend_v2=lambda: False, is_extend=lambda: True)
    parallel = NS(tp_group=group, tp_size=world, attn_dp_size=1, enable_dp_lm_head=False,
                  enable_tp_lm_head_all_to_all=False)
    feature = NS(enable_fp32_lm_head=False, enable_mis=False, aux_hidden_state_capture=None,
                 return_aux_hidden_state_layer_ids=None)
    env = dict(nn=nn, torch=torch, get_parallel=lambda: parallel,
               get_forward=lambda: NS(set=lambda *_: None), runs_sp=lambda flag: flag is mode)
    env["sp_exit_gather"] = fixture.method("layers/layernorm_sp.py", None, "sp_exit_gather", env)
    sp = NS(sp_exit_gather=env["sp_exit_gather"], runs_sp=env["runs_sp"],
            maybe_exit_gather=fixture.method("layers/layernorm_sp.py", None, "maybe_exit_gather", env))
    pack = fixture.method("layers/aux_hidden_states.py", None, "pack_aux_hidden_states", {"torch": torch})
    scope = dict(env, get_exec=lambda: NS(features=feature, deterministic=NS(rl_on_policy_target=None)),
                 layernorm_sp=sp, pack_aux_hidden_states=pack, _autotune_run_lm_head=None,
                 ForwardBatch=type("ForwardBatch", (), {}),
                 triton_symm_mem_ag=NS(recommended_max_tokens=lambda **_: 0, MultimemAllGatherer=lambda **_: None),
                 InputLogprobProcessor=lambda *args, **kwargs: None,
                 LogitsProcessorOutput=lambda **args: NS(**args, dflash_prefill_fc_done=False))
    base = source_class(srt / "layers/logits_processor.py", "LogitsProcessor", scope,
                        {"__init__", "forward", "_get_pruned_states", "_get_hidden_states_to_store",
                         "_get_last_hidden_states_to_store"})
    # The CPU Torch version lacks CUDA mm(out_dtype=float32). The tiny dyadic
    # head operands give exact FP32 reference products. Logit TP exchange is
    # downstream of tap capture and is omitted from this fixture.
    cpu_mm = lambda a, b, *, out_dtype, out=None: torch.mm(a.float(), b.float()).to(out_dtype)
    muse = source_class(srt / "models/muse_spark.py", "MuseSpark1xLogitsProcessor",
                        dict(OSSLogitsProcessor=base, torch=NS(mm=cpu_mm, float32=torch.float32),
                             use_intel_amx_backend=lambda _: False))
    base._get_logits = lambda self, hidden, head, meta: self._compute_lm_head(hidden, head)
    class Draft(nn.Module):
        quantizable = False
    Draft.project_target_hidden = fixture.method("models/dflash.py", "DFlashDraftModel",
                                                 "project_target_hidden", {"torch": torch})
    for module_name, cls in [("sglang.srt.layers.logits_processor", base), ("sglang.srt.models.muse_spark", muse),
                             ("sglang.srt.models.dflash", Draft)]:
        mod = sys.modules[module_name] = ModuleType(module_name)
        setattr(mod, cls.__name__ if cls is not Draft else "DFlashDraftModel", cls)
    draft = Draft()
    draft.fc = nn.Linear(40, 8, bias=False).bfloat16()
    draft.hidden_norm = nn.RMSNorm(8, eps=1e-5).bfloat16()
    model = nn.Module()
    model.capture_aux_hidden_states = True
    processor = model.logits_processor = muse(NS(vocab_size=13), skip_all_gather=True)
    model.lm_head = nn.Linear(8, 13, bias=False).bfloat16()
    with torch.no_grad():
        draft.fc.weight.copy_((torch.arange(320).view(8, 40) % 7 - 3) / 8)
        model.lm_head.weight.copy_((torch.arange(104).view(13, 8) % 5 - 2) / 4)
    discover = fixture.method("speculative/dflash_worker_v2.py", "DFlashWorkerV2",
                              "_defer_target_verify_gather", {"LogitsProcessor": base})
    worker = NS(target_worker=NS(model_runner=NS(model=model)))
    assert discover(worker) is processor and processor.defer_verify_gather
    with patch.object(torch.Tensor, "is_cuda", property(lambda _: True)), patch.object(
        torch.cuda, "get_device_capability", return_value=(9, 0)
    ):
        assert original.prepare_dflash_sp_prefill(draft, model, processor, group) is None
        handle = corrected.prepare_dflash_sp_prefill(draft, model, processor, group)
        assert handle is not None
        class Custom(muse):
            pass
        alternate = Custom(NS(vocab_size=13), skip_all_gather=True) if rank == world-1 else processor
        assert corrected.prepare_dflash_sp_prefill(draft, model, alternate, group) is None
    forward = fixture.method("models/muse_spark.py", "MuseSparkTextForCausalLM", "forward",
                             dict(env, layernorm_sp=sp, pack_aux_hidden_states=pack,
                                  project_dflash_prefill_shard=corrected.project_dflash_prefill_shard))
    with torch.inference_mode():
        for rows in (128*world - 1, 128*world, 128*world + 3):
            span = (rows + world - 1) // world
            dense = (torch.arange(span*world*40).reshape(-1, 40) % 13 - 6).bfloat16()
            hidden = dense[:, :8].contiguous()
            model.model = lambda *a, **k: (hidden.chunk(world)[rank], dense.chunk(world)[rank])
            lens = torch.tensor([3, rows-3], dtype=torch.int32)
            batch = NS(forward_mode=mode, extend_seq_lens=lens, extend_return_logprob=False,
                       capture_hidden_mode=NS(is_full=lambda: True, need_capture=lambda: True), mm_input_embeds=None)
            expected = draft.hidden_norm(draft.fc(dense[:rows]))
            logits = torch.mm(hidden[[2, rows-1]].float(), model.lm_head.weight.T.float())
            for ids in (None, torch.arange(rows)):
                for alteration in (None, "_get_pruned_states", "_get_hidden_states_to_store",
                                   "_get_last_hidden_states_to_store"):
                    override = rank == world-1 and alteration is not None
                    if override:
                        setattr(processor, alteration, getattr(processor, alteration))
                    try:
                        for context in (None, handle):
                            use = context is not None and rows >= 128*world and alteration is None
                            group.gathers.clear()
                            with corrected.dflash_sp_prefill_scope(context):
                                out = forward(model, ids, None, batch, input_embeds=hidden[:rows])
                            assert out.dflash_prefill_fc_done is use
                            assert group.gathers == [(span*world, 8 if use else 40), (span*world, 8)]
                            normed = draft.hidden_norm(out.hidden_states) if use else draft.project_target_hidden(out.hidden_states)
                            assert torch.equal(normed, expected) and torch.equal(logits, out.next_token_logits)
                    finally:
                        if override:
                            delattr(processor, alteration)
    dist.destroy_process_group()


if __name__ == "__main__":
    stage = Path(sys.argv[1]).resolve()
    parent = Path(__file__).resolve().parents[1] / "v120_phase_parallel_draft_context/validation.json"
    tree = json.loads(parent.read_text())["full_tree"]
    original = subprocess.check_output(["git", "-C", str(stage), "show",
                                        tree + ":python/sglang/srt/layers/dflash_context_projection.py"], text=True)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with tempfile.TemporaryDirectory() as tmp:
        for size in (2, 4, 8):
            mp.spawn(run, args=(size, str(Path(tmp)/str(size)), stage, original), nprocs=size)
    print(json.dumps({"torch": torch.__version__, "tp": [2, 4, 8], "paired_options_per_tp": 24,
                      "native_and_corrected_prompt_calls": 144, "unmodified_V120_rejected_native_Muse_on_all_ranks": True,
                      "short_boundary_uneven_prompt": True, "reference": "BF16 normalized taps and sampled FP32 logits",
                      "H200": False}, indent=2))
