"""Run V249 physical ownership checks plus V255 graph/eager serving transitions."""
import argparse
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / "v249_private_draft_swa" / "independent_fa.py"


def exercise_runtime(context):
    be, pool, reader, planner = (context[name] for name in ("be", "pool", "reader", "plan_source"))
    make_batch, bind, env = (context[name] for name in ("batch", "bind", "env"))
    pitch = 641
    raw = torch.arange(1, 2 * pitch + 1, dtype=torch.int32).view(2, pitch)
    ids = torch.arange(2 * pitch + 2)
    reader.req_to_token = planner.req_to_token = be.req_to_token = be.req_to_token_pool.req_to_token = raw
    be.max_context_len = pitch
    be._draft_window_left = 31
    be.init_cuda_graph_state(2, 16)
    graph = make_batch()
    reader.bind_runner_slots(graph)
    be.init_forward_metadata_out_graph(graph, in_capture=True)
    def graph_ptrs():
        md = be.forward_metadata
        return tuple(t.data_ptr() for t in (md.swa_out_cache_loc, md.swa_page_table, md.page_table,
                                            md.cache_seqlens_int32, md.cu_seqlens_k))
    pointers = graph_ptrs()
    columns = {mode: 0 for mode in ("eager", "graph")}
    transitions = []
    for step, prefixes in enumerate(((0, 126), (159, 160), (259, 420), (620, 32), (512, 633))):
        mapping = ((ids * 7 + 41 * step) % 1403).long()
        mapping[0] = 0
        pool.register_mapping(mapping)
        for masked in (False, True, False):
            for mode in ("eager", "graph"):
                fb = make_batch()
                fb.req_pool_indices = torch.tensor((step % 2, 1 - step % 2))
                fb.seq_lens = torch.tensor(prefixes, dtype=torch.int32)
                fb.seq_lens_cpu = fb.seq_lens.clone()
                fb.positions = torch.cat([torch.arange(p, p + 8) for p in prefixes])
                fb.spec_info = NS(draft_token_num=8, num_tokens_per_req=8, ragged_verify_layout=None,
                                  custom_mask=torch.ones(16, dtype=torch.bool) if masked else None)
                writes = torch.cat([raw[r, p:p+8] for r, p in zip(fb.req_pool_indices, prefixes)]).long()
                planner.plan(req_pool_indices=fb.req_pool_indices, seq_lens=fb.seq_lens,
                             seq_lens_cpu=fb.seq_lens_cpu, write_virtual=writes, read_extent=8).bind(fb, reader)
                original_positions = fb.positions.clone()
                if mode == "graph":
                    be.init_forward_metadata_out_graph(fb, in_capture=False)
                    assert graph_ptrs() == pointers
                    be.init_forward_metadata_in_graph(fb)
                else:
                    be.init_forward_metadata(fb)
                md = be.forward_metadata
                assert torch.equal(md.swa_out_cache_loc, mapping[writes]) and torch.equal(fb.positions, original_positions)
                starts = [0 if masked else max(p - 31, 0) // 128 * 128 for p in prefixes]
                lengths = [p + 8 - start for p, start in zip(prefixes, starts)]
                assert md.cache_seqlens_int32.tolist() == lengths
                assert md.cu_seqlens_k.tolist() == [0, lengths[0], sum(lengths)]
                assert md.cu_seqlens_q.tolist() == [0, 8, 16]
                for i, (start, length, row) in enumerate(zip(starts, lengths, fb.req_pool_indices)):
                    virtual = raw[row, start:start+length]
                    assert torch.equal(md.page_table[i, :length], virtual)
                    assert torch.equal(md.swa_page_table[i, :length], mapping[virtual.long()])
                    columns[mode] += length
                if mode == "eager" and not masked:
                    assert md.page_table.shape[1] == 31 + 8 + 127
                transitions.append((mode, masked))
    return dict(metadata_calls=len(transitions), verified_virtual_and_physical_columns=columns,
                startup_negative_controls=policy_checks(be, pool, bind, env),
                graph_stable_addresses=len(pointers), masked_native_then_compact=True,
                exact_absolute_positions_and_write_ids=True)


def policy_checks(be, pool, bind, env):
    model_type = type("DFlash2DraftModel", (torch.nn.Module,), {})
    model = model_type()
    model.block_size = 8
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(5)])
    for layer, window in zip(model.layers, (31, 127, 255, 1023, 2047)):
        layer.self_attn = torch.nn.Module()
        attn = torch.nn.Module()
        attn.head_dim = attn.v_head_dim = 128
        attn.is_cross_attention = False
        attn.attn_type = "enc"
        attn.sliding_window_size = window
        layer.self_attn.attn = attn
    mod = ModuleType("sglang.srt.models.dflash")
    mod.DFlash2DraftModel = model_type
    static = bind("python/sglang/srt/layers/attention/flashattention_backend.py", "_draft_graph_window", "FlashAttentionBackend")
    env.update(get_platform=lambda: NS(is_sm90=True), get_exec=lambda: NS(deterministic=NS(enable_deterministic_inference=False)),
               get_parallel=lambda: NS(attn_dp_enabled=False, attn_dcp_size=1, enable_prefill_cp=False),
               get_spec=lambda: NS(speculative_draft_window_size=None),
               envs=NS(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=NS(get=lambda:False)))
    be.kv_cache_dtype, be.attn_cp_size = torch.bfloat16, 1
    runner = NS(model=model)
    with patch.dict(sys.modules, {mod.__name__:mod}):
        assert static(be, runner) == 2047
        declines = 0
        for obj, attr, value in ((be, "fa_impl_ver", 3), (be, "use_sliding_window_kv_pool", False),
                (pool, "dflash_private_swa", False), (be, "is_draft_runner", False), (be, "attn_cp_size", 2),
                (be, "page_size", 16), (be, "kv_cache_dtype", torch.float32), (be, "_kv_shard_pool", object()),
                (be, "use_mla", True), (be, "topk", 2), (model, "block_size", 16),
                (model.layers[0].self_attn.attn, "attn_type", "decoder"),
                (model.layers[-1].self_attn.attn, "sliding_window_size", None),
                (model.layers[0], "_compiled_call_impl", object())):
            with patch.object(obj, attr, value):
                assert static(be, runner) == -1, attr
            declines += 1
        for install in (model.register_forward_hook, model.layers[0].register_forward_pre_hook,
                        torch.nn.modules.module.register_module_forward_hook,
                        torch.nn.modules.module.register_module_forward_pre_hook):
            handle = install(lambda *args: None)
            try:
                assert static(be, runner) == -1
            finally:
                handle.remove()
            declines += 1
        assert static(be, runner) == 2047
    fixed = be._fixed_draft_window
    for spec, encoder in ((None,None), (NS(draft_token_num=7),None),
                          (NS(draft_token_num=8, ragged_verify_layout=object()),None),
                          (NS(draft_token_num=8, custom_mask=torch.empty(0)),None),
                          (NS(draft_token_num=8),torch.tensor([1]))):
        assert fixed(spec, encoder) == -1
    return declines


def main(stage, candidate):
    # Add candidate-specific backend bindings and a final observer to the parent's
    # linked CPU fixture. Production methods still come from the verified candidate tree.
    script = DONOR.read_text()
    edits = {
        '    be.__dict__.update(device="cpu",': '    be._draft_window_left=7\n'
            '    be._fixed_draft_window=bind(BACK,"_fixed_draft_window","FlashAttentionBackend").__get__(be)\n'
            '    be.__dict__.update(device="cpu",',
        '    return dict(pin=PIN,tree=': '    extension=exercise_runtime(locals())\n'
            '    return dict(extension=extension,pin=PIN,tree=',
    }
    for original, change in edits.items():
        assert script.count(original) == 1
        script = script.replace(original, change)
    namespace = {"__name__": "linked_v249_runtime", "__file__": str(DONOR), "exercise_runtime": exercise_runtime}
    exec(compile(script, str(DONOR), "exec"), namespace)
    namespace["HERE"] = candidate
    torch.set_num_threads(1)
    return namespace["check"](stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    parser.add_argument("--candidate", type=Path, default=HERE)
    args = parser.parse_args()
    print(json.dumps(main(args.stage.resolve(), args.candidate.resolve()), indent=2))
