"""Replay archived H200 request lengths and pinned V241 SWA checkpoint components on CPU."""
import ast
from collections import defaultdict, namedtuple
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = ROOT / "experiments/modal_node/20261113T204343Z-bench"
BASE = HERE.parent / "v241_split_serving_decode/patch.diff"
CHUNK, WINDOW = 16384, 2048


def digest(data):
    return sha256(data).hexdigest()


def pinned_reader():
    path = HERE.parent / "v217_captured_verify_context/independent_model_entry.py"
    spec = importlib.util.spec_from_file_location("v243_pin", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.PARENT = BASE.read_text()
    assert digest(BASE.read_bytes()) == "c1b938cd93f2cfdbaba9b0e6dce7e5d16f95410d811db9a8e8570693739f0acf"
    return mod


def trace(requests):
    conversations = defaultdict(list)
    for req in requests:
        conversations[req["load"], req["rep"], req["user"]].append(req)
    by_load = {}
    for n in sorted({key[0] for key in conversations}):
        cold, warm, deltas = [], [], []
        for (k, _, _), conv in conversations.items():
            if k != n:
                continue
            conv = sorted(conv, key=lambda r: r["turn"])
            assert [r["turn"] for r in conv] == list(range(13))
            cold.append(conv[0])
            warm.extend(conv[1:])
            deltas.extend(b["cached"] - a["prompt"] for a, b in zip(conv, conv[1:]))
        missing = [r["prompt"] - r["cached"] for r in warm]
        assert all(0 < m < WINDOW for m in missing) and set(deltas) == {1}
        chunks, middle = [], []
        for req in cold:
            rows = req["prompt"] - req["cached"]
            parts = [min(CHUNK, rows-i) for i in range(0, rows, CHUNK)]
            chunks.extend(parts)
            middle.extend(parts[:-1])
        by_load[n] = dict(
            cold_requests=len(cold), later_requests=len(warm), cold_uncached=sum(chunks),
            cold_prompt=sum(r["prompt"] for r in cold), cold_modeled_full_middle_chunks=len(middle),
            cold_omitted_keep_2048_per_chunk=sum(max(0, c-WINDOW) for c in chunks),
            cold_omitted_keep_2049_middle_only=sum(c-WINDOW-1 for c in middle),
            later_prompt=sum(r["prompt"] for r in warm), later_cached=sum(r["cached"] for r in warm),
            later_uncached=sum(missing), later_min_new=min(missing), later_max_new=max(missing),
            later_cached_minus_previous_prompt_equals_one=deltas.count(1),
            deficit_rewind_target_rows=sum(max(0, WINDOW-m) for m in missing),
            full_rewind_target_rows=len(warm)*WINDOW)
    return by_load


def fork_witness(reader, src):
    swa = "mem_cache/unified_cache/components/swa.py"
    common, cache = "mem_cache/common.py", "mem_cache/unified_radix_cache.py"
    core, sched = "mem_cache/unified_cache/unified_tree_core.py", "managers/scheduler.py"

    def bind(file, cls, name, env=None):
        return reader.bind(ast.parse(src[file]), cls, name, env or {})

    anchors = [(sched, "self.stash_chunked_request(self.chunked_req)"),
               (common, "tree_cache.checkpoint(req, up_to=req.extend_end)"),
               (cache, "comp.free_out_of_window_slots(req, len(radix_key) - 1, insert_params)"),
               (cache, "result = self.insert(insert_params)"), (swa, "return effective_cache_len"),
               (swa, "def _clip_to_window("), (swa, "self._publish_window(node=fragment"),
               (core, "node, action = self._split_node(child.key, child, prefix_len)"),
               (core, "component.redistribute_on_node_split(new_parent=new_node, child=child)"),
               (sched, "release_kv_cache(req, self.tree_cache, checkpoint=False)")]
    lines = {}
    for f, key in anchors:
        line = next(i for i, s in enumerate(src[f].splitlines(), 1) if key in s)
        lines[f"{f}:{line}"] = key
    text = ast.unparse(next(n for n in ast.walk(ast.parse(src[cache]))
                            if isinstance(n, ast.FunctionDef) and n.name == "checkpoint_into_tree"))
    assert text.index("comp.prepare_for_caching_req") < text.index("comp.free_out_of_window_slots") < text.index("self.insert(")
    assert "not is_finished and envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.get()" in text
    free = bind(common, None, "free_swa_out_of_window_slots", {
        "ComponentType": NS(SWA="swa"), "ceil_align": lambda n, p: (n+p-1)//p*p})
    mint = bind(swa, "SWAComponent", "finalize_match_result_in_tree_core")
    prepare = bind(swa, "SWAComponent", "prepare_for_caching_req")
    clip = bind(swa, "SWAComponent", "_clip_to_window")
    split = bind(swa, "SWAComponent", "redistribute_on_node_split")
    validate = bind(swa, "SWAComponent", "create_match_validator")
    Match = namedtuple("Match", "device_prefix_len host_hit_length full_kv_hit_length best_match_node swa_host_hit_length swa_branching_seqlen")
    root = object()
    tree = NS(root_node=root, page_size=1, is_eagle=False, host_lru_lists={"swa": None})
    comp = NS(component_type="swa", tree_core=tree, cache=NS(is_mamba_enabled=False),
              sliding_window_size=WINDOW, gates_prefix_match=True)
    floor = bind(cache, "UnifiedRadixCache", "swa_retain_floor")(comp.cache, None)
    assert floor is None

    def node(n):
        cd = NS(lock_ref=0, host_lock_ref=1, session_ref=0, session_ids=None,
                metadata={}, value=None, host_value=None)
        return NS(key=range(n), component_data={"swa": cd}, evicted=False, backuped=False)

    def replay(full_hit):
        match = mint(comp, Match(0, 0, full_hit, root, 0, None), None, [], 0)
        state = {}
        kv = NS(cache_protected_len=0, holds_kv=True, req_pool_idx=0, swa_dead_lo=lambda p: 0,
                get_evicted_seqlen=lambda ct: state.get(ct, 0), set_evicted_seqlen=state.__setitem__)
        req = NS(kv=kv, swa_branching_seqlen=match.swa_branching_seqlen)
        cap = prepare(comp, req, NS(set_evicted_seqlen=lambda *a: None), CHUNK, False)
        stop = CHUNK if cap is None else cap
        slots, frees = torch.arange(CHUNK), []
        free(req, stop-1, sliding_window_size=WINDOW, page_size=1, retain_floor=floor,
             req_to_token_pool=NS(req_to_token=slots.unsqueeze(0)), token_to_kv_pool_allocator=None,
             free_segment=lambda rows, start_pos: frees.append((start_pos, rows.tolist())))
        assert frees == [(0, list(range(stop-WINDOW-1)))]
        start = kv.get_evicted_seqlen("swa")
        splits = []
        tree._split_node = lambda key, child, index: (splits.append(index), None)
        assert clip(comp, node=node(stop), start=0, end=stop, window_start=start, cache_actions=[]) == start
        assert splits == [start]
        endpoint = 8192 if full_hit == 0 else 8191
        by_tier = {}
        for tier in ("value", "host_value"):
            leaf = node(stop-start)
            setattr(leaf.component_data["swa"], tier, slots[start:stop])
            validator = validate(comp)
            assert not validator(node(start))
            if endpoint <= start:
                assert not validate(comp)(node(endpoint))
                by_tier[tier] = dict(hierarchy_accepts=False, device_only_accepts=False)
                continue
            parent = node(endpoint-start)
            split(comp, parent, leaf)
            assert torch.equal(getattr(parent.component_data["swa"], tier), slots[start:endpoint])
            ok = validator(parent)
            device = validate(comp, match_device_only=True)
            assert not device(node(start))
            local_ok = device(parent)
            assert ok and local_ok == (tier == "value")
            by_tier[tier] = dict(hierarchy_accepts=ok, device_only_accepts=local_ok)
        # Hugging Face includes the current row in its width; SGLang FA4
        # uses window_left = width - 1 previous positions.
        needed = slots[endpoint-(WINDOW-1-256):endpoint]
        return dict(initial_reusable_boundary=0, initial_full_kv_hit=full_hit,
                    native_swa_branching_seqlen=req.swa_branching_seqlen, effective_checkpoint=stop,
                    native_live_swa_span=[start, stop], later_divergence_endpoint=endpoint,
                    later_uncached_append=256, old_rows_needed_at_next_draft=len(needed),
                    old_rows_present_in_naive_draft_tail=int(torch.isin(needed, slots[-WINDOW:]).sum()),
                    later_acceptance_by_tier=by_tier)
    control, branch = replay(0), replay(8192)
    assert not control["later_acceptance_by_tier"]["value"]["hierarchy_accepts"]
    assert branch["later_acceptance_by_tier"]["value"]["hierarchy_accepts"]
    assert branch["old_rows_present_in_naive_draft_tail"] == 0
    return dict(native_full_checkpoint_control=control, native_branch_checkpoint=branch, source_anchors=lines,
                scope="Original pinned CPU methods for branch mint, preparation, physical eviction, live clipping, split and match. Synthetic inputs, node shells, host mirror and allocator. The branch is possible; archived client lengths do not identify such an event.")


def main():
    reader = pinned_reader()
    files = ["configs/muse_spark.py", "models/muse_spark.py", "speculative/dflash_worker_v2.py", "speculative/dflash_utils.py",
             "mem_cache/common.py", "mem_cache/unified_radix_cache.py", "managers/scheduler.py",
             "mem_cache/unified_cache/components/swa.py", "mem_cache/unified_cache/unified_tree_core.py"]
    src = {f: reader.read_source(f, False) for f in files}
    load_path, info_path, server = [next(ARCHIVE.glob("*v41__"+n)) for n in ("load.json", "server_info.json", "server.log")]
    official = ROOT / "serving_v0523/runs/04b01f958aa5487c.log"
    geom = HERE.parent / "v185_paired_prompt_draft_kv/geometry.py"
    fc_dims = HERE.parent / "v219_verify_graph_sliding_tiles/full_width.json"
    info = json.loads(info_path.read_bytes())["internal_states"][0]
    load = json.loads(load_path.read_bytes())
    assert min(info["max_prefill_tokens"], info["chunked_prefill_size"]) == CHUNK
    assert info["page_size"] == 1 and info["speculative_num_draft_tokens"] == load["gpus"] == 8
    for n in (5, 16, 46):
        assert any(f", {n} layers" in l and "HiCache staging prepared" in l for l in official.read_text().splitlines())
    for a in ("sum(batch.extend_lens)", "cache_loc=self._draft_write_ids(kv_loc_plan)"):
        assert a in src["speculative/dflash_worker_v2.py"]
    by_load = trace(load["requests"])
    assert "return int(sliding_window) - 1" in src["speculative/dflash_utils.py"]
    totals = {k: sum(r[k] for r in by_load.values()) for k in next(iter(by_load.values())) if k not in ("later_min_new", "later_max_new")}
    assert (totals["later_requests"], totals["cold_omitted_keep_2048_per_chunk"],
            totals["deficit_rewind_target_rows"], totals["full_rewind_target_rows"]) == (708, 3906550, 768323, 1449984)
    conf = next(n for n in ast.parse(src["configs/muse_spark.py"]).body if getattr(n, "name", None) == "MuseSparkTextConfig")
    init = next(n for n in conf.body if getattr(n, "name", None) == "__init__").args
    defaults = {a.arg: v for a, v in zip(init.args[-len(init.defaults):], init.defaults)}
    h, layers, q, k, d, w = [ast.literal_eval(defaults[key]) for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "sliding_window")]
    assert (h, layers, q, k, d, w) == (8192, 62, 128, 16, 64, WINDOW)
    assert "self.qkv_proj(hidden_states)" in src["models/muse_spark.py"] and "self.o_proj(output)" in src["models/muse_spark.py"]
    assert json.loads(fc_dims.read_bytes())["weight_shape"] == [h, 5*h]
    assert "SIZE, HIDDEN, CHANNELS, ELEMENT_BYTES = 8, 8192, 1280, 2" in geom.read_text()
    target = 2 * layers * h * (q+k+k+q) * d
    draft = 2 * h * (5*h + 8*1280)
    saved = draft * totals["cold_omitted_keep_2048_per_chunk"]
    added = target * totals["deficit_rewind_target_rows"]
    inputs = [BASE, load_path, info_path, server, official, geom, fc_dims]
    output = dict(pinned_runtime=reader.PIN, source_sha256={f: digest(s.encode()) for f, s in src.items()},
                  raw_sha256={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in inputs},
                  trace=dict(observations="Archived eight-H200 V41 request lengths and match offsets. Prior prompt +1 held for all observed later turns. No physical slots, accepted draft coverage or kernel timings appear in the client trace.",
                             model="Consecutive 16,384-row chunks per individual cold miss. The scheduler may pack or split them differently. A 2,048-row tail in each chunk is an optimistic row count, not a safe write policy.",
                             per_concurrency=by_load, totals=totals),
                  global_dense_flops=dict(draft_fc_and_five_kv_per_row=draft,
                      draft_geometry=dict(fc_out=h, fc_in=5*h, kv_out_per_rank=1280, tp=8),
                      target_qkvo_per_row_lower_bound=target,
                      target_geometry=dict(layers=layers, hidden=h, q_heads=q, kv_heads=k, head_dim=d),
                      target_expression="2 * layers * hidden * (Q_heads + K_heads + V_heads + O_heads) * head_dim",
                      optimistic_saved_cold_draft=saved, added_target_qkvo_deficit_rewind=added,
                      added_target_qkvo_full_rewind=target*totals["full_rewind_target_rows"],
                      added_target_qkvo_vs_saved_draft=added/saved,
                      fa4_exact_deficit_target_rows=totals["deficit_rewind_target_rows"]-totals["later_requests"],
                      fa4_exact_deficit_target_qkvo=added-totals["later_requests"]*target,
                      fa4_exact_target_qkvo_vs_saved_draft=(added-totals["later_requests"]*target)/saved,
                      rewind_convention="The blanket deficit retains 2048 previous rows. Released FA4 needs 2047 previous rows including all newly appended rows, so its exact deficit is one smaller for each of the 708 positive deficits.",
                      scope="Logical multiply-adds across all ranks. Target counts omit attention, extra gates, experts and all other costs. These counts do not measure latency or score."),
                  counterexample=fork_witness(reader, src))
    (HERE/"coverage_replay.json").write_text(json.dumps(output, indent=2)+"\n")
    print("later hits", totals["later_requests"], "modeled omitted rows", totals["cold_omitted_keep_2048_per_chunk"], "target/draft FLOPs", round(added/saved, 5))


if __name__ == "__main__":
    main()
