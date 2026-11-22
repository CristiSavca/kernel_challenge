"""Exercise mixed settlement, absolute-position RoPE, partial-write replay and physical prefix scatter on CPU."""
import ast
import hashlib
import importlib.util
import json
import logging
import math
import subprocess
import tempfile
import types
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


HERE = Path(__file__).resolve().parent
fixture = importlib.util.spec_from_file_location(
    "v185_worker_fixture", HERE.parent / "v185_paired_prompt_draft_kv/worker_check.py"
)
native = importlib.util.module_from_spec(fixture)
sys.modules[fixture.name] = native
fixture.loader.exec_module(native)
torch.set_num_threads(1)

W = native.wpath
load = native.load
output_type = NS
scope = dict(native.ws)
scope.update(
    GenerationBatchResult=output_type,
    LogitsProcessorOutput=output_type,
    make_draft_input_v2=load("srt/speculative/draft_worker_common.py", None, "make_draft_input_v2",
                             {"DFlashDraftInputV2": output_type}),
    _commit_accept=load(W, None, "_commit_accept"),
    _is_all_greedy=load(W, None, "_is_all_greedy"),
    compute_dflash_correct_drafts_and_bonus=load("kernels/ops/speculative/dflash.py", None,
                                                 "compute_dflash_correct_drafts_and_bonus"),
    SpecTpSyncSite=NS(DFLASH_TARGET="prompt", DFLASH_ACCEPT_GREEDY="accept"),
    SIMULATE_ACC_LEN=0,
)


init_scope = dict(native.ws, get_parallel=lambda: NS(tp_rank=1), math=math,
                  _get_fused_kv_materialize_helper=lambda: native.Helper,
                  can_dflash_use_fused_qkv_proj=lambda module: (True, None))


class Worker(native.Worker):
    _init_fused_kv_helper = load(W, "DFlashWorkerV2", "_init_fused_kv_helper", init_scope)
    settle_mixed_verify = load(W, "DFlashWorkerV2", "settle_mixed_verify", scope)
    _accept_block = load(W, "DFlashWorkerV2", "_accept_block", scope)
    _prefill_kv_helper = load(W, "DFlashWorkerV2", "_prefill_kv_helper", scope)


def prefix_scatter(k, v, k_bank, v_bank, locations, lengths, **kwargs):
    """Terminal GPU-scatter stand-in; crop each row separately, without reusing the test oracle."""
    for request, length in enumerate(lengths.tolist()):
        indices = locations[request, :length]
        begin = request * locations.shape[1]
        k_bank[indices] = k[begin:begin + length]
        v_bank[indices] = v[begin:begin + length]


gpu_prefix = load(native.mem, "MHATokenToKVPool", "set_kv_buffer_prefix_valid", {
    "_is_cuda": True, "_is_hip": False, "fp8_dtype": torch.float8_e4m3fn,
    "_resolve_fused_scale": load(native.mem, None, "_resolve_fused_scale"),
    "_set_kv_buffer_prefix_valid_impl": prefix_scatter,
})


class Pool(native.Pool):
    def record(self, role, layer):
        self.calls.append((role, layer))
        if self.interrupt == (role, layer):
            self.interrupt = None
            raise RuntimeError("injected after a physical write")

    def _store_kv_layer(self, layer, slots, key, value):
        self.k_buffer[layer][slots] = key
        self.v_buffer[layer][slots] = value
        self.record("prompt", layer)

    def set_kv_buffer_prefix_valid(self, layer, loc, lens, k, v, *scales):
        gpu_prefix(self, layer, loc, lens, k, v, *scales)
        self.record("tail", layer.layer_id)


def build(mode, remap, lens, width, seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(417)
    layers, head_count, dim, hidden = 5, 2, 8, 12
    scaled = mode == "nonunit"
    draft, helper = native.build([None, 1, 1.0, 2.0 if scaled else None, 1], False)
    draft.fc = NS(out_features=hidden)
    lengths = torch.tensor([3, 7, 2], dtype=torch.int32)
    prompt_prefix = torch.tensor([111, 0, 79], dtype=torch.int64)
    prefix_lens = torch.arange(len(lens), dtype=torch.int64) * 31 + 19
    rows = int(lengths.sum()) + len(lens) * width
    positions = torch.cat((native.positions("cpu", prompt_prefix, lengths, int(lengths.sum()))[0],
                           (prefix_lens[:, None] + torch.arange(width)).flatten()))
    normalized = torch.randint(-3, 4, (rows, hidden), generator=g).bfloat16()
    raw = normalized @ helper.flat_kv_weight_t
    capacity = rows * 3 + 11
    plan = native.Plan()
    plan.write_virtual = torch.randperm(capacity, generator=g)[:rows] + 1
    plan._write_ids = {}
    mapping = torch.cat((torch.zeros(1, dtype=torch.long), torch.randperm(capacity, generator=g) + 1))
    space = NS(key="physical" if remap else "native", write=(lambda t: mapping[t]) if remap else None)
    translated = mapping[plan.write_virtual] if remap else plan.write_virtual
    pool = Pool()
    pool.data = torch.full((layers, 2, capacity + 1, head_count, dim), -711., dtype=torch.bfloat16)
    pool.k_buffer, pool.v_buffer = pool.data[:, 0], pool.data[:, 1]
    pool.size, pool.start_layer, pool.page_size = capacity, 0, 1
    pool.row_dim = pool.v_row_dim = head_count * dim
    pool.dtype = pool.store_dtype = torch.bfloat16
    pool.is_quantized_kv_cache = pool.use_hnd = False
    pool.requires_physical_write_loc = True
    pool.calls, pool.interrupt = [], None
    if mode in ("partial_prompt", "partial_tail"):
        pool.interrupt = (mode.split("_")[1], 2)
    worker = Worker()
    worker.device = torch.device("cpu")
    worker.model_runner = NS(device=worker.device)
    worker.draft_model_runner = NS(token_to_kv_pool=pool, kv_index_translator=NS(space=lambda kind: space))
    worker.draft_model, worker.draft_parallel_layout = draft, None
    worker._sp_prefill_context = None if mode == "hardware_fallback" else object()
    worker.lilicorr = worker._selector_sample = worker._target_logits_processor = None
    worker._fused_kv_helper = helper
    worker._use_fused_kv_materialize = mode not in ("per_layer", "nonunit")
    worker._logged_padding_trim = True
    worker.block_size, worker._use_triton_accept_bonus = width, False
    sync = []
    worker._tp_sync = NS(sync=lambda site, value: sync.append(site))
    worker.target_worker = NS(model_runner=NS(sample=lambda logits, batch: logits.next_token_logits.argmax(-1),
                                            model_config=NS(context_len=290)))
    return NS(**locals())


def settle(env, projected, aligned, invalid=None):
    w = env.worker
    prompts, n = len(env.lengths), int(env.lengths.sum())
    w._tp_sync.sync = lambda site, value: env.sync.append(site)
    vocab = 53
    tokens = (torch.arange(len(env.lens) * env.width).view(-1, env.width) * 3 + 11) % vocab
    predictions = (tokens + 17) % vocab
    # At commit length c, the first c-1 draft transitions must agree and the next must differ.
    for i, c in enumerate(env.lens):
        predictions[i, :c-1] = tokens[i, 1:c]
        if c < env.width:
            predictions[i, c-1] = (tokens[i, c] + 1) % vocab
    heads = torch.cat((torch.arange(prompts) + 3, predictions.flatten()))
    logits = torch.full((len(heads), vocab), -19.0)
    logits.scatter_(1, heads[:, None], 31.0)
    context = env.raw if projected else env.normalized
    context = context.clone()
    committed = torch.cat((torch.ones(n, dtype=torch.bool),
                           (torch.arange(env.width)[None, :] < torch.tensor(env.lens)[:, None]).flatten()))
    context[~committed] = float("nan")
    if aligned:
        context = torch.cat((context, torch.full((3, context.shape[-1]), float("nan"), dtype=context.dtype)))
    context_done = True
    if invalid == "context_marker":
        context_done = False
    if invalid == "lilicorr":
        w.lilicorr = object()
    if invalid == "no_layers":
        w.draft_model.layers = []
    if invalid == "wrong_width":
        context = context[:, :-1]
    if invalid == "wrong_rows":
        context = context[:n]
    if invalid == "wrong_rank":
        context = context.flatten()
    if invalid == "wrong_positions":
        env.positions = env.positions[:-1]
    out = NS(kv_loc_plan=env.plan, can_run_cuda_graph=True,
             logits_output=NS(next_token_logits=logits, hidden_states=context,
                              dflash_prefill_context_done=context_done,
                              dflash_prefill_kv_projected=projected))
    prompt = NS(batch_size=prompts, extend_num_tokens=n, extend_seq_lens=env.lengths,
                seq_lens=env.prompt_prefix + env.lengths)
    published = []
    args = dict(target_out=out, prompt_forward_batch=prompt, candidates=tokens,
                prefix_lens=env.prefix_lens, positions=env.positions,
                decode_sampling_info=NS(vocab_size=vocab, is_all_greedy=True),
                on_publish=lambda seqs: published.append(seqs.clone()))
    with patch.object(torch, "mm", wraps=torch.mm) as mm:
        if invalid:
            try:
                w.settle_mixed_verify(**args)
            except ValueError:
                assert not env.pool.calls and not published and not env.sync, invalid
                return
            raise AssertionError(invalid)
        prompt_out, tail_out, next_draft = w.settle_mixed_verify(**args)
    assert out.kv_loc_plan is None and out.logits_output.hidden_states is None
    assert tail_out.accept_lens.tolist() == list(env.lens)
    assert len(published) == 1 and torch.equal(published[0], next_draft.new_seq_lens)
    assert torch.equal(next_draft.new_seq_lens, torch.cat((prompt.seq_lens, env.prefix_lens + torch.tensor(env.lens))))
    assert env.sync == ["prompt", "accept"] and prompt_out.next_token_ids.tolist() == [3, 4, 5]
    assert torch.equal(next_draft.bonus_tokens[prompts:], predictions[torch.arange(len(env.lens)),
                                                                      torch.tensor(env.lens) - 1])
    dead = torch.ones(env.capacity + 1, dtype=torch.bool)
    dead[env.translated[committed]] = False
    assert (env.pool.data[:, :, dead] == -711.).all()
    assert torch.isfinite(env.pool.data[:, :, ~dead]).all()
    assert (env.pool.data[:, :, ~dead] != -711.).all()
    if projected:
        assert mm.call_count == 0
    if env.mode.startswith("partial_"):
        role = env.mode.split("_")[1]
        seen = [layer for observed, layer in env.pool.calls if observed == role]
        assert seen == [0, 1, 2, 0, 1, 2, 3, 4], (env.mode, seen)
        assert w._fused_kv_helper is None and not w._use_fused_kv_materialize
    return env.pool.data.view(torch.int16), dead.sum().item(), mm.call_count


def native_hardware_declines():
    source = native.repo / "python/sglang/srt/layers/dflash_context_projection.py"
    spec = importlib.util.spec_from_file_location("v212_context", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    invariant = types.ModuleType("sglang.srt.batch_invariant_ops")
    invariant.is_batch_invariant_mode_enabled = lambda: False
    x = torch.empty((256, 12), dtype=torch.bfloat16)
    helper = NS(flat_kv_weight_t=torch.ones(12, 20, dtype=torch.bfloat16), n_layers=5, layer_out_dim=4)

    def agree(peers, local, group=None):
        for peer in peers:
            peer.copy_(local)

    with patch.dict(sys.modules, {invariant.__name__: invariant}), module.dflash_sp_prefill_scope(None, helper):
        for ranks in (1, 2):
            assert module.project_paired_draft_kv(x, 256 * ranks, NS(world_size=ranks)) is None
        for ranks in (4, 8):
            group = NS(world_size=ranks, rank_in_group=0, cpu_group=object(), pynccl_comm=None)
            with patch.object(torch.distributed, "all_gather", side_effect=agree) as reduced:
                assert module.project_paired_draft_kv(x, 256 * ranks, group) is None
                assert reduced.call_count == 1
    return dict(unsupported_world_sizes=[1, 2], missing_transport_world_sizes=[4, 8],
                rank_agreement="Modeled unanimous Gloo capability signatures; no P2P transport allowed.")


def nonunit_parent_witness():
    rel = "python/sglang/srt/speculative/dflash_worker_v2.py"
    parent = HERE.parent / "v211_prompt_expert_tiles_mixed_serving"
    metadata = json.loads((parent / "validation.json").read_text())
    assert hashlib.sha256((parent / "patch.diff").read_bytes()).hexdigest() == metadata["patch_sha256"]
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp, rel)
        f.parent.mkdir(parents=True)
        f.write_bytes(subprocess.check_output(["git", "-C", native.repo, "show", native.pin + ":" + rel]))
        subprocess.run(["git", "apply", "--include=" + rel, str(parent / "patch.diff")], cwd=tmp, check=True)
        old_source = f.read_text()
    old_class = next(n for n in ast.parse(old_source).body if getattr(n, "name", None) == "DFlashWorkerV2")
    class PreviousWorker(Worker):
        pass

    tested_methods = {
        "_init_fused_kv_helper", "settle_mixed_verify", "_accept_block", "_draft_write_ids",
        "_append_target_hidden_to_draft_kv_by_loc", "_append_target_hidden_fused",
        "_append_target_hidden_sequential", "_append_paired_prompt_kv",
    }
    names = set()
    for method in old_class.body:
        if getattr(method, "name", None) in tested_methods:
            method_scope = dict(scope, **{"torch": torch})
            if method.name == "_init_fused_kv_helper":
                method_scope.update(init_scope)
            exec("from __future__ import annotations\n" + ast.unparse(method), method_scope)
            setattr(PreviousWorker, method.name, method_scope[method.name])
            names.add(method.name)
    assert names == tested_methods
    configs = ("nonunit", True, (1, 3, 5), 5, 212701)
    previous, current, sequential = (build(*configs) for _ in range(3))
    previous.worker.__class__ = PreviousWorker
    for case in (previous, current):
        case.worker._use_fused_kv_materialize = True
        case.worker._fused_kv_helper = None
        case.worker._init_fused_kv_helper()
    assert previous.worker._fused_kv_helper is not None and previous.worker._use_fused_kv_materialize
    assert current.worker._fused_kv_helper is None and not current.worker._use_fused_kv_materialize
    assert current.worker._prefill_kv_helper() is None
    prev_words = settle(previous, False, True)[0]
    current_words = settle(current, False, True)[0]
    expected = settle(sequential, False, True)[0]
    assert torch.equal(current_words, expected)
    orig_v, wanted_v = prev_words[3, 1].view(torch.bfloat16), expected[3, 1].view(torch.bfloat16)
    touched = wanted_v != -711.
    assert torch.equal(orig_v[touched] * 2, wanted_v[touched])
    wrong = int((orig_v != wanted_v).sum())
    assert wrong > 0
    return dict(v211_patch_sha256=metadata["patch_sha256"], v211_worker_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
                scale=2, cache_quantization_scale=None, affected_layer=3, v211_incorrect_physical_v_words=wrong,
                v212_incorrect_physical_v_words=0, v212_all_KV_words_bitwise_equal_to_sequential=expected.numel(),
                producer="With the ordinary context carrier, exact V211 initialization, mixed settlement and cache writer selected the unscaled fused path. V212 initializer selected ordinary kv_proj_only and applied the model scale.")


def main():
    logging.disable(logging.WARNING)
    count = Counter()
    matrix = {}
    for width, lengths in [(2, (1, 2)), (5, (1, 3, 5)), (8, (8, 1, 4, 7))]:
        for mode in ("fused", "warm", "per_layer", "partial_prompt", "partial_tail", "nonunit", "hardware_fallback"):
            for remap in (False, True):
                for aligned in (False, True):
                    configs = (mode, remap, lengths, width, 212701)
                    actual, baseline = build(*configs), build(*configs)
                    # V212 can receive a raw carrier and lose the helper before settlement.
                    # This is the per-layer-only consumer path.
                    projected = mode not in ("nonunit", "hardware_fallback")
                    if mode == "warm":
                        actual.helper._ensure_workspace(actual.rows + 5, torch.bfloat16)
                        baseline.helper._ensure_workspace(baseline.rows + 5, torch.bfloat16)
                    if mode == "nonunit":
                        actual.worker._use_fused_kv_materialize = True
                        assert actual.worker._prefill_kv_helper() is None
                        actual.worker._init_fused_kv_helper()
                        assert not actual.worker._use_fused_kv_materialize and actual.worker._fused_kv_helper is None
                    if mode == "hardware_fallback":
                        assert actual.worker._prefill_kv_helper() is None
                    if mode not in ("per_layer", "nonunit", "hardware_fallback"):
                        assert actual.worker._prefill_kv_helper() is actual.helper
                    result, dead, candidate_mm = settle(actual, projected, aligned)
                    reference, _, original_mm = settle(baseline, False, aligned)
                    assert torch.equal(result, reference), configs
                    count["cases"] += 1
                    count["cache_bf16_words"] += result.numel()
                    count["untouched_slots"] += dead
                    count["standalone_projection_mms_removed"] += original_mm - candidate_mm
                    matrix[mode] = matrix.get(mode, 0) + 1
    for fault in ("context_marker", "lilicorr", "no_layers", "wrong_width", "wrong_rank", "wrong_rows", "wrong_positions"):
        settle(build("fused", True, (1,3), 4, 212909), True, False, fault)
        count["malformed_rejections_before_sampling_or_writes"] += 1
    return dict(counts=dict(count), cases_by_mode=matrix, nonunit_negative_control=nonunit_parent_witness(),
                hardware_fallbacks=native_hardware_declines(), torch=torch.__version__, methods_from_stage=native.digests,
                kv_heads=2, head_dim=8, layers=5, partial_rotary_dim=6, draft_widths=[2,5,8],
                comparison="Exact staged native mixed settlement and greedy acceptance, retained V211 context carrier versus V212 raw K/V. All physical BF16 cache words compared with stale sentinel, rejected and padded activations poisoned, simultaneous short, intermediate and full acceptance, nonidentity physical remaps.",
                emulated="Host tensors; target FP32 logits and projected/context carriers are supplied. TP sync, terminal GPU NHD/prefix scatter and fused CUDA norm/RoPE are CPU doubles. Native MHATokenToKVPool dispatch, worker writer, helper host code and per-layer norm/RoPE execute. Unsupported hardware fixture supplies no SP context. No H200 timing or accuracy claim.",
                physical_h200=False)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
