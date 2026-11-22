"""Replay the native DFLASH prefill worker, trim and context append on CPU."""
import ast
import hashlib
import importlib.util
import json
import logging
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(sys.argv[1]) / "python/sglang/srt"
DIGESTS = {}


def method(path, owner, name, namespace):
    raw = (ROOT / path).read_bytes()
    DIGESTS[path] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw)
    cls = next(node for node in tree.body if getattr(node, "name", "") == owner)
    node = next(node for node in cls.body if getattr(node, "name", "") == name)
    exec(compile("from __future__ import annotations\n" + ast.unparse(node), str(path), "exec"), namespace)
    return namespace[name]


def load_projection():
    path = "layers/dflash_context_projection.py"
    DIGESTS[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("native_projection", ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


projection = load_projection()


class RMSNorm(nn.Module):
    _quant_linear = None
    forward_native = method("layers/layernorm.py", "RMSNorm", "forward_native", {"torch": torch})
    forward_cuda = forward_native

    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.5, 1.5, hidden).bfloat16())
        self.hidden_size = hidden
        self.variance_size_override = self.override_orig_dtype = None
        self.cast_x_before_out_mul = False
        self.variance_epsilon = 1.e-6
        self.rows = []
        self._forward_method = self.forward_cuda

    def forward(self, x):
        self.rows.append(x.shape[0])
        return self._forward_method(x)


class Draft(nn.Module):
    quantizable = False
    project_target_hidden = method("models/dflash.py", "DFlashDraftModel", "project_target_hidden", {"torch": torch})


class Logits(nn.Module):
    keep_last_hidden_states = False
    expected_aux_hidden_states = None


class MuseLogits(Logits):
    def forward(self, ids, hidden, head, batch, aux, **kwargs):
        return NS(hidden_states=aux, dflash_prefill_context_done=False, sampled_shape=hidden.shape,
                  tokens=kwargs.get("sp_prompt_tokens"))


class Mode:
    def __init__(self, key):
        self.key = key

    def is_extend(self):
        return self.key == "extend"

    def is_idle(self):
        return self.key == "idle"


full = NS(is_full=lambda: True, need_capture=lambda: True)
forward = NS(EXTEND=Mode("extend"))
scope = projection.dflash_sp_prefill_scope
def position(_, prefixes, lengths, total):
    result = torch.cat([torch.arange(int(start), int(start + length)) for start, length in zip(prefixes, lengths)])
    assert len(result) == total
    return result, None

worker_source = "speculative/dflash_worker_v2.py"
native_forward = method(worker_source, "DFlashWorkerV2", "forward_batch_generation", {
    "torch": torch, "ForwardMode": forward, "CaptureHiddenMode": NS(FULL=full),
    "dflash_sp_prefill_scope": scope, "nullcontext": nullcontext,
    "SpecTpSyncSite": NS(DFLASH_TARGET="target"), "async_h2d": lambda rows, **kw: torch.tensor(rows, **kw),
    "compute_position": position, "DFlashDraftInputV2": NS(create_idle_input=lambda device: ("idle", device)),
})
consume = method(worker_source, "DFlashWorkerV2", "_append_target_hidden_to_draft_kv_by_loc", {
    "torch": torch, "draft_tp_context": lambda _: nullcontext(), "logger": logging.getLogger("quiet"),
    "publish_anchor": lambda **kw: kw["ctx_hidden"].clone(),
})
trim = method("model_executor/runner/prefill_cuda_graph_runner.py", "PrefillCudaGraphRunner", "_trim_logits_output",
              {"LogitsProcessorOutput": NS})


model_environment = {
    "torch": torch, "pack_aux_hidden_states": lambda xs: torch.cat(xs, dim=-1),
    "get_forward": lambda: NS(set=lambda *a: None),
    # Exercise each logical rank separately; owner-ordered gathers are composed below.
    "layernorm_sp": NS(runs_sp=lambda mode: mode is forward.EXTEND, sp_exit_gather=lambda x, num_tokens: x),
    "project_dflash_prefill_shard": projection.project_dflash_prefill_shard,
}
model_forward = method("models/muse_spark.py", "MuseSparkTextForCausalLM", "forward", model_environment)


def publish_module(stack, name, key, entity):
    module = ModuleType(name)
    setattr(module, key, entity)
    stack.enter_context(patch.dict(sys.modules, {name: module}))


def check(tag, real, padded, size, fused, graph):
    torch.manual_seed(12)
    width, hidden = 48, 16
    source = (torch.arange(padded * width).reshape(padded, width) % 19 - 9).bfloat16()
    draft = Draft()
    draft.fc = nn.Linear(width, hidden, bias=False).bfloat16()
    with torch.no_grad():
        draft.fc.weight.copy_((torch.arange(width * hidden).reshape(hidden, width) % 7 - 3).bfloat16())
    norm = draft.hidden_norm = RMSNorm(hidden)
    logits = MuseLogits()
    target_model = nn.Module()
    group = NS(world_size=size, cpu_group=None)
    model_environment["get_parallel"] = lambda: NS(tp_group=group)
    store, moves, marks = [], [], []
    batch = NS(forward_mode=forward.EXTEND, is_extend_in_batch=False, return_logprob=tag == "logprob",
               multi_item_delimiter_indices={} if tag=="mis" else None,
               token_indices_to_pool=[] if tag=="setwise" else None, capture_hidden_mode=full,
               return_hidden_states_mode=NS(need_capture=lambda: tag == "external"), seq_lens=torch.tensor([real, real+21]),
               extend_lens=[real//3, real-real//3], prefix_lens=[111, 29])
    batch.out_cache_loc = torch.arange(real, dtype=torch.int32).roll(29)*3+7
    def save(which, **kw):
        store.append((which, kw["ctx_hidden"].clone(), kw["ctx_positions"].clone(), kw["ctx_cache_loc"].clone()))
        if which == "fused" and fused == "failure":
            raise RuntimeError("injected fused launch failure")
    production = NS(model_runner=NS(device=torch.device("cpu"), prefill_attention_backend_str="fa3"),
                    device="cpu", draft_model=draft, draft_parallel_layout=None, _draft_sampler=None,
                    lilicorr=object(), _use_fused_kv_materialize=(fused is not False), _fused_kv_helper=object(),
                    _append_target_hidden_fused=lambda **kw: save("fused", **kw),
                    _append_target_hidden_sequential=lambda **kw: save("sequential", **kw),
                    _draft_write_ids=lambda plan: plan, _logged_padding_trim=True,
                    _validate_phase1_sampling_support=lambda batch: None, _wait_for_fused_hicache_load=lambda batch: None,
                    _tp_sync=NS(sync=lambda site, ids: moves.append(site)),
                    _make_next_draft_input_prefill=lambda **kw: kw)
    production._append_target_hidden_to_draft_kv_by_loc = lambda **kw: consume(production, **kw)
    with ExitStack() as intercept:
        publish_module(intercept, "sglang.srt.layers.layernorm", "RMSNorm", RMSNorm)
        publish_module(intercept, "sglang.srt.models.dflash", "DFlashDraftModel", Draft)
        publish_module(intercept, "sglang.srt.models.muse_spark", "MuseSpark1xLogitsProcessor", MuseLogits)
        publish_module(intercept, "sglang.srt.layers.logits_processor", "LogitsProcessor", Logits)
        intercept.enter_context(patch.object(projection, "_agree", lambda ready, group: bool(ready)))
        intercept.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=(tag=="capture")))
        with patch.object(torch.Tensor, "is_cuda", property(lambda _: True)), patch.object(torch.cuda, "get_device_capability", return_value=(9,0)):
            ctx = projection.prepare_dflash_sp_prefill(draft, target_model, logits, group)
        assert ctx is not None
        production._sp_prefill_context = None if tag=="disabled" else ctx
        if tag == "norm_hook":
            handle = norm.register_forward_hook(lambda mod, args, val: val+1)
            intercept.callback(handle.remove)
        if tag == "fc_hook":
            handle = draft.fc.register_forward_hook(lambda mod, args, val: val+2)
            intercept.callback(handle.remove)
        if tag == "override_prune":
            logits._get_pruned_states = object()
        if tag == "quantized_norm":
            norm._quant_linear = object()
        if tag == "norm_dispatch":
            norm._forward_method = lambda x: norm.forward_native(x) + 3
        expected = draft.project_target_hidden(source[:real])
        norm.rows.clear()
        real_lens = batch.extend_lens
        def target(batch, **kw):
            assert kw["capture_hidden_mode"] is full and kw["return_kv_loc_plan"] is True
            local_rows = (padded+size-1)//size
            tail = torch.nn.functional.pad(source, (0,0,0,local_rows*size-padded))
            rank_outputs = []
            for shard in tail.split(local_rows):
                model = NS(logits_processor=logits, capture_aux_hidden_states=True, lm_head=None,
                           model=lambda *args, **kw: (shard[:, :hidden], shard.tensor_split(3, dim=1)))
                use_embeds = graph == "breakable"
                rank_outputs.append(model_forward(model, None if use_embeds else torch.arange(padded), None,
                                                  batch, source[:, :hidden] if use_embeds else None))
            done = rank_outputs[0].dflash_prefill_context_done
            assert all(out.dflash_prefill_context_done == done for out in rank_outputs)
            assert all(out.tokens == (padded if done else None) for out in rank_outputs)
            marks.append(done)
            features = torch.cat([out.hidden_states for out in rank_outputs])[:padded]
            sampling = torch.tensor([3, 7, 11, 29])[:,None].repeat(1,3)
            init = dict.fromkeys(("input_token_logprobs", "input_top_logprobs_val","input_top_logprobs_idx",
                                  "input_token_ids_logprobs_val","input_token_ids_logprobs_idx","input_logprobs_copy_done",
                                  "customized_info","last_hidden_states","hidden_states_token_indices"),None)
            out = NS(**init, next_token_logits=sampling, hidden_states=features, mm_input_embeds=source,
                     dflash_prefill_context_done=done)
            if graph!="eager":
                run = NS(raw_bs=2, raw_num_tokens=real, _is_full_backend=graph=="full",
                         model_runner=NS(spec_algorithm=NS(is_speculative=lambda: True)))
                out = trim(run, out)
                assert out.hidden_states.shape[0] == real and out.mm_input_embeds.shape[0] == real
                assert out.next_token_logits.shape[0] == (2 if graph=="full" else 4)
            return NS(kv_loc_plan=batch.out_cache_loc,logits_output=out,next_token_ids=torch.tensor([3,7]))
        production.target_worker = NS(forward_batch_generation=target)
        out = native_forward(production,batch,on_publish=lambda lengths: moves.append(("published",lengths)))
        assert out.kv_loc_plan is None and out.logits_output.hidden_states is None
        assert moves[0]=="target" and moves[1][0]=="published" and torch.equal(moves[1][1],batch.seq_lens)
        assert torch.equal(out.next_draft_input["bonus_tokens"],torch.tensor([3,7]))
        assert projection._prefill_context.get() is None
        expected_fast = tag == "normal" and padded >= 128*size
        assert marks == [expected_fast], (tag, marks, expected_fast)
        normal_len = (padded+size-1)//size
        assert norm.rows == ([normal_len]*size if expected_fast else [real]), (tag,norm.rows)
        assert [entry[0] for entry in store] == (["sequential"] if fused is False else
                                                  ["fused","sequential"] if fused=="failure" else ["fused"])
        expected_pos = position(None,torch.tensor(batch.prefix_lens),torch.tensor(real_lens),real)[0]
        for _, val, pos, slots in store:
            assert torch.equal(val, expected), (tag,real,padded,size,val.shape,expected.shape)
            assert torch.equal(pos,expected_pos) and torch.equal(slots,batch.out_cache_loc.to(torch.int64))
        assert torch.equal(production._lilicorr_anchor,expected)
        return expected.numel(), expected_fast


if __name__ == "__main__":
    torch.set_num_threads(1)
    logging.disable(logging.WARNING)
    cases, values, fast = 0, 0, 0
    tags = ["normal","short","disabled","capture","external","logprob","norm_hook","fc_hook","override_prune","quantized_norm","norm_dispatch","mis","setwise"]
    for size in (2,4,8):
        for graph in ("eager","breakable","full"):
            for fused in (True,False,"failure"):
                for tag in tags:
                    rows = 128*size + (1 if graph=="eager" else 9)
                    if tag=="short":
                        rows=128*size-1
                    real = rows - (3 if graph=="eager" else 11)
                    amount, yes=check(tag,real,rows,size,fused,graph)
                    cases+=1;values+=amount;fast+=int(yes)
    print(json.dumps(dict(cases=cases, accelerated=fast, fallback=cases-fast, checked_normalized_bf16_elements=values,
                         sources=DIGESTS, physical_h200=False, torch=torch.__version__, tags=tags,
                         modes=["eager","full","breakable"],group_sizes=[2,4,8],
                         limitations="Host oracle replays the worker and norm using PyTorch CPU; peer ranks and CUDA cache writers are simulated."),indent=2))
