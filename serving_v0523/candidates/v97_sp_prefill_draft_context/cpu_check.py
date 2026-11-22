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
    for name in ("sglang.srt.model_loader.weight_utils", "sglang.srt.utils"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["sglang.srt.model_loader.weight_utils"].sharded_weight_loader = lambda *a, **k: lambda *a: None
    sys.modules["sglang.srt.utils"].set_weight_attrs = lambda *a, **k: None
    path = SOURCE / "layers/dflash_context_projection.py"
    spec = importlib.util.spec_from_file_location("actual_fc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Group:
    def __init__(self, size):
        self.world_size = size
        self.cpu_group = dist.group.WORLD
        self.gathers = 0

    def all_gather_into_tensor(self, out, local):
        self.gathers += 1
        # Gloo copies to its output on a background thread. Give that thread
        # ordinary storage, then copy back on the inference-mode caller.
        with torch.inference_mode(False), torch.no_grad():
            local = local.detach().clone()
            collected = torch.empty_like(out)
            dist.all_gather_into_tensor(collected, local, group=self.cpu_group)
        out.copy_(collected)

    def all_gather(self, local, dim):
        with torch.inference_mode(False), torch.no_grad():
            local = local.detach().clone()
            parts = [torch.empty_like(local) for _ in range(self.world_size)]
            dist.all_gather(parts, local, group=self.cpu_group)
        return torch.cat(parts, dim=dim)


def run(rank, size, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=size, timeout=timedelta(seconds=35))
    torch.set_num_threads(1)
    fc_module = load_projection()
    group = Group(size)
    f, h = 40, 8
    fc = fc_module.DFlashGatheredLinear(f, h, group).to(dtype=torch.bfloat16)
    w = (torch.arange(h * f).reshape(h, f) % 7 - 3).bfloat16()
    with torch.no_grad():
        fc.weight.copy_(w.chunk(size, dim=0)[rank])
    sp = method("layers/layernorm_sp.py", None, "sp_exit_gather", {"torch": torch, "get_parallel": lambda: NS(tp_group=group)})
    pack = lambda x: torch.cat(x, dim=-1) if isinstance(x, list) else x
    full = NS(is_full=lambda: True, need_capture=lambda: True)
    capture = method("layers/logits_processor.py", "LogitsProcessor", "_get_hidden_states_to_store", {"pack_aux_hidden_states": pack})
    logits_weight = torch.arange(24).reshape(3, 8).bfloat16()
    class LP:
        keep_last_hidden_states = False
        def __call__(self, ids, hidden, head, batch, aux):
            value = capture(self, hidden, None, aux, None, None, None, None, batch)
            return NS(hidden_states=value, next_token_logits=torch.nn.functional.linear(hidden[[1,-1]], logits_weight), dflash_prefill_fc_done=False)
    forward = method("models/muse_spark.py", "MuseSparkTextForCausalLM", "forward", {
        "torch": torch, "pack_aux_hidden_states": pack,
        "get_parallel": lambda: NS(tp_group=group),
        "get_forward": lambda: NS(set=lambda *args: None),
        "layernorm_sp": NS(sp_exit_gather=sp, runs_sp=lambda mode: mode == "extend"),
        "project_dflash_prefill_shard": fc_module.project_dflash_prefill_shard,
    })
    norm = torch.nn.RMSNorm(h, eps=1e-5).bfloat16()
    project = method("models/dflash.py", "DFlashDraftModel", "project_target_hidden", {"torch": torch})
    draft = NS(fc=fc, quantizable=False, hidden_norm=norm)
    draft.project_target_hidden = lambda x: project(draft, x)
    append = method("speculative/dflash_worker_v2.py", "DFlashWorkerV2", "_append_target_hidden_to_draft_kv_by_loc", {
        "torch": torch, "logger": logging.getLogger("fixture"), "draft_tp_context": lambda _: nullcontext()
    })
    captured = []
    worker = NS(draft_parallel_layout=None, draft_model=draft, model_runner=NS(device=torch.device("cpu")), lilicorr=None,
                _use_fused_kv_materialize=True, _fused_kv_helper=object(), _append_target_hidden_fused=lambda **k: captured.append(k))
    with torch.inference_mode():
        for n,extra,valid,tag in [(8,0,8,"short"),(9,0,9,"tail"),(19,0,17,"trim"),(32,0,32,"full"),(19,2,19,"bucket"),(19,0,19,"local-hook"),(19,0,19,"global-hook"),(19,0,19,"external"),(19,0,19,"dtype"),(19,0,19,"deterministic")]:
            rows = (n + size - 1) // size + extra
            taps = (torch.arange(rows * size * f).reshape(-1, f) % 9 - 4).bfloat16()
            hidden = (torch.arange(rows * size * h).reshape(-1, h) % 11 - 5).bfloat16()
            local_taps, local_hidden = taps.chunk(size)[rank], hidden.chunk(size)[rank]
            model = NS(model=lambda *a, **k: (local_hidden, list(local_taps.split(h, dim=1))),
                       capture_aux_hidden_states=True, lm_head=None, logits_processor=LP())
            batch = NS(forward_mode="extend", capture_hidden_mode=full)
            embeds = torch.zeros((n,h),dtype=torch.bfloat16)
            baseline = forward(model, None, None, batch, input_embeds=embeds)
            hook = None
            if tag == "local-hook" and rank == 0:
                hook = fc.register_forward_pre_hook(lambda *a: None)
            if tag == "global-hook" and rank == size-1:
                hook = torch.nn.modules.module.register_module_forward_hook(lambda *a: None)
            if tag == "external":
                model.logits_processor.keep_last_hidden_states = True
            if tag == "deterministic" and rank == 0:
                torch.use_deterministic_algorithms(True)
            with fc_module.dflash_sp_prefill_fc_scope(fc):
                if tag == "dtype":
                    assert fc_module.project_dflash_prefill_shard(local_taps.float() if rank == 0 else local_taps,n,group) is None
                result = forward(model, None, None, batch, input_embeds=embeds)
            torch.use_deterministic_algorithms(False)
            if hook:
                hook.remove()
            enabled = n > h and tag not in ("external","local-hook","global-hook","deterministic")
            assert result.dflash_prefill_fc_done is enabled, (n, tag, size)
            assert torch.equal(result.next_token_logits,baseline.next_token_logits)
            assert torch.equal(baseline.hidden_states, taps[:n])
            expected = torch.nn.functional.linear(taps[:n],w)
            assert torch.equal(result.hidden_states,expected if enabled else taps[:n])
            locs = torch.arange(valid-1,-1,-1); positions=torch.arange(30, 30+valid)
            for out in (baseline, result):
                append(worker,target_hidden=out.hidden_states, cache_loc=locs, positions=positions, prefill_fc_done=out.dflash_prefill_fc_done)
            for k in ("ctx_hidden","ctx_positions","ctx_cache_loc"):
                assert torch.equal(captured[-1][k],captured[-2][k]), (tag,k)
            assert torch.equal(captured[-1]["ctx_hidden"],norm(expected[:valid]))
            assert torch.equal(captured[-1]["ctx_cache_loc"],locs)
            assert torch.equal(captured[-1]["ctx_positions"],positions)
        try:
            with fc_module.dflash_sp_prefill_fc_scope(fc):
                with fc_module.dflash_sp_prefill_fc_scope(None):
                    assert fc_module._prefill_fc.get() is None
                assert fc_module._prefill_fc.get() is fc
                raise RuntimeError("probe")
        except RuntimeError:
            assert fc_module._prefill_fc.get() is None
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("GLOO_SOCKET_IFNAME","lo0")
    with tempfile.TemporaryDirectory() as directory:
        for size in (2,4,8):
            mp.spawn(run, args=(size, str(Path(directory) / str(size))), nprocs=size, join=True)
    print(json.dumps({"torch":torch.__version__, "collective":"actual Gloo all_gather_into_tensor, all_gather, all_reduce(MIN)", "tp_sizes":[2,4,8], "cases_per_size":10, "checks":["source MuseSpark forward and SP exit gather","source LogitsProcessor FULL capture","source normalized worker FC/KV boundary", "BF16 logits/FC/normalized KV inputs/positions/slots bit-identical GE integer fixtures","short/tail/bucket/trim/external/hook/deterministic/dtype/nested exception"], "h200_executed":False},indent=2))
