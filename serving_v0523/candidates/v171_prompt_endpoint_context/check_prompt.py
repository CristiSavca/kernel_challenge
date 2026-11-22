"""Exercise Muse's real prompt and logit methods on distributed CPU tensors."""

import ast
from contextlib import ExitStack
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
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


def run(rank, world, rendezvous, stage):
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
    sp.sp_gather_selected_rows = fixture.method("layers/layernorm_sp.py", None, "sp_gather_selected_rows", env)
    class ForwardBatch(NS):
        pass
    pack = fixture.method("layers/aux_hidden_states.py", None, "pack_aux_hidden_states", {"torch": torch})
    scope = dict(env, get_exec=lambda: NS(features=feature, deterministic=NS(rl_on_policy_target=None)),
                 layernorm_sp=sp, pack_aux_hidden_states=pack, _autotune_run_lm_head=None,
                 ForwardBatch=ForwardBatch, LogitsMetadata=NS(from_forward_batch=lambda batch: batch),
                 triton_symm_mem_ag=NS(recommended_max_tokens=lambda **_: 0, MultimemAllGatherer=lambda **_: None),
                 InputLogprobProcessor=lambda *args, **kwargs: None,
                 LogitsProcessorOutput=lambda **args: NS(**args, dflash_prefill_context_done=False))
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
    class RMSNorm(nn.Module):
        _quant_linear = None
        override_orig_dtype = variance_size_override = None
        cast_x_before_out_mul = False
        hidden_size = 8
        variance_epsilon = 1e-5

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter((torch.arange(8).bfloat16() + 1) / 4)
            self._forward_method = None

        def _resolve_forward_method(self):
            return self.forward_native

        def forward(self, x):
            if self._forward_method is None:
                self._forward_method = self._resolve_forward_method()
            return self._forward_method(x)

        def forward_cuda(self, x):
            raise AssertionError("CPU test invoked the CUDA method")

    RMSNorm.forward_native = fixture.method("layers/layernorm.py", "RMSNorm", "forward_native", {"torch": torch})
    for module_name, cls in [("sglang.srt.layers.layernorm", RMSNorm), ("sglang.srt.layers.logits_processor", base), ("sglang.srt.models.muse_spark", muse),
                             ("sglang.srt.models.dflash", Draft)]:
        mod = sys.modules[module_name] = ModuleType(module_name)
        setattr(mod, cls.__name__ if cls is not Draft else "DFlashDraftModel", cls)
    draft = Draft()
    draft.fc = nn.Linear(40, 8, bias=False).bfloat16()
    draft.hidden_norm = RMSNorm()
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
        handle = corrected.prepare_dflash_sp_prefill(draft, model, processor, group)
        assert handle is not None
        class Custom(muse):
            pass
        alternate = Custom(NS(vocab_size=13), skip_all_gather=True) if rank == world-1 else processor
        assert corrected.prepare_dflash_sp_prefill(draft, model, alternate, group) is None
    forward = fixture.method("models/muse_spark.py", "MuseSparkTextForCausalLM", "forward",
                             dict(env, layernorm_sp=sp, pack_aux_hidden_states=pack,
                                  project_dflash_prefill_shard=corrected.project_dflash_prefill_shard))
    def special_output(self, ids, states, head, meta, indices):
        return NS(next_token_logits=states.index_select(0, indices), hidden_states=None,
                  dflash_prefill_context_done=False)

    base.compute_logprobs_at_positions = base.compute_logprobs_for_multi_item_scoring = special_output
    calls = 0
    with torch.inference_mode():
        for rows in (128*world - 1, 128*world, 128*world + 3):
            span = (rows + world - 1) // world
            dense = (torch.arange(span*world*40).reshape(-1, 40) % 13 - 6).bfloat16()
            # The final global rows may be padding. Never include them in the norm or head reference.
            dense[rows:] = float("nan")
            hidden = dense[:, :8].contiguous()
            model.model = lambda *args, **kwargs: (hidden.chunk(world)[rank], dense.chunk(world)[rank])
            expected = draft.project_target_hidden(dense[:rows])
            for scenario in ("ordinary", "owner boundaries", "many endpoints", "custom prune", "norm hook",
                             "changed dispatch", "fused quant", "setwise scoring", "multi-item scoring"):
                lens = [3, rows-3]
                if scenario == "owner boundaries":
                    lens = [0, span, 0, 1, rows-span-1]
                elif scenario == "many endpoints":
                    lens = [1]*(span-1) + [rows-span+1]
                lens = torch.tensor(lens, dtype=torch.int32)
                positions = lens.cumsum(0) - 1
                logits = torch.mm(hidden[:rows][positions].float(), model.lm_head.weight.T.float())
                batch = ForwardBatch(forward_mode=mode, extend_seq_lens=lens, extend_return_logprob=False,
                                     capture_hidden_mode=NS(is_full=lambda: True, need_capture=lambda: True),
                                     mm_input_embeds=None, multi_item_delimiter_indices=None, token_indices_to_pool=None,
                                     is_prefill_only=False)
                is_special = scenario.endswith("scoring")
                if is_special:
                    attr = "token_indices_to_pool" if scenario == "setwise scoring" else "multi_item_delimiter_indices"
                    anchors = torch.tensor([rows - 1, 1, span - 1, 3])
                    setattr(batch, attr, anchors)
                    batch.is_prefill_only = True
                with ExitStack() as stack:
                    if rank == world-1:
                        if scenario == "custom prune":
                            stack.enter_context(patch.object(processor, "_get_pruned_states", wraps=processor._get_pruned_states))
                        elif scenario == "norm hook":
                            stack.enter_context(draft.hidden_norm.register_forward_hook(lambda *unused: None))
                        elif scenario == "changed dispatch":
                            stack.enter_context(patch.object(draft.hidden_norm, "_forward_method", lambda x: draft.hidden_norm.forward_native(x)))
                        elif scenario == "fused quant":
                            stack.enter_context(patch.object(draft.hidden_norm, "_quant_linear", object()))
                    for ids in (None, torch.arange(rows)):
                        for context in (None, handle):
                            use = context is not None and rows >= 128*world and scenario in (
                                "ordinary", "owner boundaries", "many endpoints")
                            selected = use and lens.numel() < span
                            group.gathers.clear()
                            with corrected.dflash_sp_prefill_scope(context):
                                out = forward(model, ids, None, batch, input_embeds=hidden[:rows])
                            assert out.dflash_prefill_context_done is use, (scenario, rank)
                            assert group.gathers == [(span*world, 8 if use else 40),
                                                     (world*lens.numel() if selected else span*world, 8)]
                            if is_special:
                                assert out.hidden_states is None
                                assert torch.equal(out.next_token_logits, hidden.index_select(0, anchors))
                            else:
                                normed = out.hidden_states if use else draft.project_target_hidden(out.hidden_states)
                                assert torch.equal(normed, expected) and torch.equal(logits, out.next_token_logits)
                            calls += 1

        # Check noncontiguous rows with every bit preserved, including signed zero and NaN payloads.
        bits = torch.arange(10*world*16, dtype=torch.int32).to(torch.int16).reshape(10*world, 16)
        bits[:, 0], bits[:, 2], bits[:, 4] = -32768, 32705, -47
        shard = bits.view(torch.bfloat16).chunk(world)[rank][:, ::2]
        total = 10*world-3
        idx = torch.tensor([total-1, 10, 0, -total, 9, -1, 10, -3])
        actual = sp.sp_gather_selected_rows(shard, idx, total)
        assert torch.equal(actual.view(torch.int16), bits[:total, ::2][idx])
        for invalid in (-total-1, total, total+2):
            try:
                sp.sp_gather_selected_rows(shard, torch.tensor([invalid]), total)
            except (IndexError, RuntimeError):
                pass
            else:
                raise AssertionError("out-of-range global row silently accepted")
        assert calls == 108
    dist.destroy_process_group()


if __name__ == "__main__":
    stage = Path(sys.argv[1]).resolve()
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    with tempfile.TemporaryDirectory() as tmp:
        for size in (2, 4, 8):
            mp.spawn(run, args=(size, str(Path(tmp)/str(size)), stage), nprocs=size)
    print(json.dumps({"torch": torch.__version__, "tp": [2, 4, 8], "prompt_forwards": 324,
                      "rank_local_prompt_calls": 1512,
                      "reference": "actual pinned CPU BF16 normalize/project method and Muse head with FP32 accumulation",
                      "startup": "only physical CUDA capability queries simulated",
                      "gloo_collective_shapes_checked": True, "special_scoring_dense_fallback": True,
                      "noncontiguous_rows_nan_payload_signed_zero_order_padding_and_invalid_indices": True,
                      "physical_H200": False}, indent=2))
