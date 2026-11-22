"""Replay the mixed target, native head, settlement and physical KV writes on CPU."""
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as N

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SRT = Path(sys.argv[1]) / "python/sglang/srt"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


device = load("v202_device", HERE.parent / "v156_mixed_verify_settlement/device_cpu.py")
ref = device.ref
env = ref.env
projection = load("v202_projection", SRT / "layers/dflash_context_projection.py")
env["dflash_sp_prefill_scope"] = projection.dflash_sp_prefill_scope
env["project_dflash_prefill_shard"] = projection.project_dflash_prefill_shard
def reject_paired(*args, **kwargs):
    raise AssertionError("mixed target attempted a raw-KV exchange")


env["project_paired_draft_kv"] = reject_paired
env["get_forward"] = lambda: N(set=lambda *_: None, scoped=lambda **_: nullcontext())
policy = types.ModuleType("sglang.srt.batch_invariant_ops")
policy.is_batch_invariant_mode_enabled = lambda: False
sys.modules[policy.__name__] = policy


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.6, 1.7, width).bfloat16())

    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1.e-6)).to(x.dtype) * self.weight


class Draft(nn.Module):
    quantizable = False
    def __init__(self, original):
        super().__init__()
        self.fc = nn.Linear(40, 8, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            w = torch.arange(320).reshape(8, 40)
            self.fc.weight.copy_(((w * 11 % 17) - 8).float() / 32)
        self.hidden_norm = RMSNorm(8)
        self.layers = original.layers
        self.prepare_context_hidden_for_kv = original.prepare_context_hidden_for_kv
        self.calls = []

    def project_target_hidden(self, rows):
        self.calls.append(rows.shape[0])
        return self.hidden_norm(self.fc(rows))


class NativeHead(nn.Module):
    keep_last_hidden_states = False
    expected_aux_hidden_states = None

    def __init__(self, implementation):
        super().__init__()
        self.implementation = implementation

    def forward(self, *args, **kwargs):
        return self.implementation(*args, **kwargs)


def prepare_case(backend, tails, iteration, lengths):
    # The archived fixture builds real ScheduleBatch views. Substitute its
    # synthetic prompt lengths to cross the unmodified projection threshold.
    source = inspect.getsource(device.prepare)
    tree = ast.parse(source)
    stmt = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "lengths")
    stmt.value = ast.Name(id="fixture_lengths", ctx=ast.Load())
    scope = dict(vars(device), fixture_lengths=lengths)
    exec(compile(ast.fix_missing_locations(tree), "<V156 mixed batch constructor>", "exec"), scope)
    return scope["prepare"](backend, tails, iteration, do_veto=True)


def check(backend, size, lengths, tails, iteration, scenario):
    worker, _, decode, mixed, prompt_fb, execute, _, steps = prepare_case(backend, tails, iteration, lengths)
    target = worker.target_worker
    model = target.model_runner.model
    graph = inspect.getclosurevars(target.model_runner.forward).nonlocals["g"]
    mode = ref.Mode.EXTEND
    head = NativeHead(model.logits_processor)
    model.logits_processor = head
    target_marker = nn.Module()
    draft = Draft(worker.draft_model)
    worker.draft_model = draft
    group = N(world_size=size, rank_in_group=iteration % size, pynccl_comm=None)
    ref.PAR.tp_group = group
    original = (draft, draft.fc, draft.hidden_norm, draft.project_target_hidden, group, target_marker, head)
    worker._sp_prefill_context = None if scenario == "disabled" else original
    projection._agree = lambda ready, _: bool(ready)
    # Represents the previously vetted native RMSNorm backend on a CPU host.
    projection._supports_shard_norm = lambda norm, fc: scenario != "unsupported_norm"
    state = {"selected": [], "issued": [], "scopes": [], "full_gathers": 0}
    q = sum(lengths)
    n = q + 8*tails
    p = len(lengths)
    expected_selections = torch.cat((torch.tensor(lengths).cumsum(0)-1, torch.arange(q, n)))

    def body(ids):
        hidden, taps = ref.body_tokens(ids)
        state["global_h"] = hidden
        state["global_aux"] = torch.cat(taps, dim=-1)
        state["span"] = math.ceil(len(ids) / size)
        return shard(hidden), [shard(tap) for tap in taps]

    def shard(tensor):
        span = state["span"]
        padding = span*size - len(tensor)
        whole = torch.nn.functional.pad(tensor, (0, 0, 0, padding))
        start = group.rank_in_group * span
        return whole[start:start+span]

    def materialize(local, num_tokens):
        state["full_gathers"] += 1
        raw = state["global_aux"]
        if local.shape[1] == 40:
            full = raw
        elif torch.equal(local, shard(state["global_h"])):
            full = state["global_h"]
        else:
            full = draft.hidden_norm(draft.fc(raw))
        assert torch.equal(local, shard(full))
        return full[:num_tokens]

    def select(local, positions, count):
        assert torch.equal(local, shard(state["global_h"]))
        assert torch.equal(positions, expected_selections)
        state["selected"].append(len(positions))
        return state["global_h"][:count][positions]

    env["layernorm_sp"] = N(runs_sp=lambda m: m == mode, sp_exit_gather=materialize,
                            sp_gather_selected_rows=select,
                            maybe_exit_gather=lambda **kw: (state["global_h"][:len(kw["input_ids"])],
                                                           kw["hidden_states_before_norm"]))
    graph.layer_model.forward = lambda ids, *_, **__: body(ids)
    graph.backend.replay = lambda _, fb, **__: body(fb.input_ids)
    attn = draft.layers[0].self_attn
    attn.kv_proj_only = lambda x: (x.repeat(1, 5), (2*x).repeat(1, 5))
    native_target = target.forward_batch_generation

    def call_target(*args, **kwargs):
        assert projection._draft_kv_helper.get() is None
        assert projection._prefill_context.get() is worker._sp_prefill_context
        result = native_target(*args, **kwargs)
        out = result.logits_output
        state["scopes"].append(out.dflash_prefill_context_done)
        state["issued"].append(tuple(out.next_token_logits.shape))
        assert not out.dflash_prefill_kv_projected
        actual_h = ref.body_tokens(mixed.input_ids)[0].float()
        reference_scores = 25 * torch.tanh((actual_h @ ref.WEIGHT.T) * 0.75 / 25)
        reference_scores = reference_scores.index_select(0, expected_selections)
        torch.testing.assert_close(out.next_token_logits, reference_scores, rtol=1.e-6, atol=1.e-6)
        assert torch.equal(out.next_token_logits.argmax(1), reference_scores.argmax(1))
        raw = torch.cat(ref.body_tokens(mixed.input_ids)[1], dim=1)
        expected_hidden = draft.hidden_norm(draft.fc(raw)) if out.dflash_prefill_context_done else raw
        assert torch.equal(out.hidden_states, expected_hidden)
        if scenario == "unexpected_paired":
            out.dflash_prefill_kv_projected = True
        if scenario == "wrong_width":
            out.dflash_prefill_context_done = True
            out.hidden_states = out.hidden_states[:, :7]
        return result

    target.forward_batch_generation = call_target
    cache = worker.draft_model_runner.token_to_kv_pool
    original_k, original_v = cache.k_buffer[0].clone(), cache.v_buffer[0].clone()
    with projection.dflash_sp_prefill_scope(object(), kv_helper="poisoned outer helper"):
        if scenario in ("unexpected_paired", "wrong_width"):
            try:
                execute()
            except ValueError as exc:
                assert "mixed DFLASH settlement" in str(exc)
            else:
                raise AssertionError("invalid hidden representation accepted")
            assert projection._draft_kv_helper.get() == "poisoned outer helper"
            assert steps == ["hicache"]
            assert torch.equal(cache.k_buffer[0], original_k) and torch.equal(cache.v_buffer[0], original_v)
            return None
        result = execute()
        assert projection._draft_kv_helper.get() == "poisoned outer helper"
    assert projection._prefill_context.get() is None
    commits = torch.tensor([1 + (iteration + 3*i) % 8 for i in range(tails)])
    assert torch.equal(result.decode_result.accept_lens, commits.to(result.decode_result.accept_lens.dtype))
    assert state["issued"] == [(p + tails*8, ref.V)]
    assert steps == ["hicache", "native_prompt_sampler", "prompt", "accept", "publish"]
    fast = state["scopes"][0]
    if scenario == "enabled":
        assert fast
    if scenario in ("disabled", "unsupported_norm", "short"):
        assert not fast
    assert draft.calls == ([] if fast else [q, tails*8])
    assert bool(state["selected"]) == (fast and len(expected_selections) < state["span"])
    raw = torch.cat(ref.body_tokens(mixed.input_ids)[1], dim=1)
    normalized = draft.hidden_norm(draft.fc(raw)).repeat(1, 5)
    live = torch.cat((torch.ones(q, dtype=torch.bool), (torch.arange(8) < commits[:,None]).flatten()))
    positions = torch.cat((prompt_fb.positions, (decode.seq_lens[:,None] + torch.arange(8)).flatten()))
    slots = mixed.out_cache_loc[live]
    original_k[slots] = (normalized + positions[:,None].to(normalized.dtype))[live]
    original_v[slots] = (2 * normalized)[live]
    assert torch.equal(cache.k_buffer[0], original_k)
    assert torch.equal(cache.v_buffer[0], original_v)
    return dict(fast=fast, head_narrow=bool(state["selected"]), cache_words=original_k.numel()*2,
                logits_values=(p+tails*8)*ref.V,
                target_rows=n, accepted=commits.tolist(), rejected=int((8-commits).sum()))


def main():
    totals = dict(executions=0, invalid_before_publish=0, normalized=0, raw_fallback=0,
                  narrow_head=0, cache_words=0, target_rows=0, logits_values=0, rejected_tail_slots=0)
    accepted = set()
    with torch.inference_mode():
        for mode in ("eager", "FULL", "BREAKABLE"):
            for size in (2, 4, 8):
                for j, (lengths, tails, scenario) in enumerate((
                    ([131*size + 2, 7, 11], 3, "enabled"),
                    ([129*size + 5, 1], 2, "disabled"),
                    ([130*size + 1, 9], 2, "unsupported_norm"),
                    ([5, 9, 4], 3, "short"),
                    ([1, 3], 18*size, "enabled"),
                    ([130*size + 1, 4], 2, "unexpected_paired"),
                    ([130*size + 1, 7], 2, "wrong_width"),
                )):
                    result = check(mode, size, lengths, tails, j, scenario)
                    if result is None:
                        totals["invalid_before_publish"] += 1
                        continue
                    totals["executions"] += 1
                    totals["normalized" if result["fast"] else "raw_fallback"] += 1
                    totals["narrow_head"] += result["head_narrow"]
                    totals["target_rows"] += result["target_rows"]
                    totals["logits_values"] += result["logits_values"]
                    totals["rejected_tail_slots"] += result["rejected"]
                    totals["cache_words"] += result["cache_words"]
                    accepted.update(result["accepted"])
    assert accepted == set(range(1, 9))
    files = ("models/muse_spark.py", "speculative/dflash_worker_v2.py", "layers/logits_processor.py",
             "layers/dflash_context_projection.py", "model_executor/runner/prefill_cuda_graph_runner.py")
    return dict(**totals, accepted_lengths=sorted(accepted), torch=torch.__version__,
                graph_modes=["eager", "FULL body-replay emulator", "Breakable body-replay emulator"],
                source_sha256={f: hashlib.sha256((SRT/f).read_bytes()).hexdigest() for f in files},
                physical_h200=False)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
