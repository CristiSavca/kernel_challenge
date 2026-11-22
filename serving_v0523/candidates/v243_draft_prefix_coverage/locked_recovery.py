"""Replay V241's retained-full SWA recovery against private DFLASH KV on CPU."""
import ast
from collections import namedtuple
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS

import torch

from coverage_replay import BASE, HERE, ROOT, pinned_reader


def main():
    reader = pinned_reader()
    sources = {}
    anchors = {}

    def method(path, owner, name, env=None):
        source = sources.setdefault(path, reader.read_source(path, False))
        module = ast.parse(source)
        owner_node = next((n for n in module.body if getattr(n, "name", "") == owner), module)
        node = next(n for n in owner_node.body if getattr(n, "name", "") == name)
        anchors[f"{path}:{node.lineno}"] = name
        return reader.bind(module, owner, name, {"torch": torch, **(env or {})})

    swa = "mem_cache/unified_cache/components/swa.py"
    index = "mem_cache/kv_index_translator.py"
    loc = "mem_cache/kv_loc_plan.py"
    adapter = "mem_cache/rust_tree_core/adapter.py"
    kind = NS(FULL="full", SLIDING_WINDOW="swa")
    recovery = namedtuple("RecoverSWAWithLockedFull", "node_id kept_full incoming_full")
    action_types = {name: type(name, (), {}) for name in (
        "FreeComponentDeviceSlot", "FreeComponentHostSlot", "RebuildFullToSWAMapping",
        "RecoverEvictedSWA", "SWARebuild")}
    action_types["RecoverSWAWithLockedFull"] = recovery
    action_types["BASE_COMPONENT_TYPE"] = "full"
    emit = method(swa, "SWAComponent", "_recover_from_incoming", action_types)
    convert = method(adapter, None, "_cache_action_from_tagged", action_types)
    apply = method(swa, "SWAComponent", "apply_component_action", action_types)
    projection_ids = method("speculative/dflash_worker_v2.py", "DFlashWorkerV2", "_draft_write_ids")
    translator_env = {name: type(name, (), {}) for name in (
        "UnifiedMambaTokenToKVPoolAllocator", "UnifiedSWAAllocatorBase", "BaseSWAKVPool",
        "DeepSeekV4TokenToKVPool")}
    translator_env.update(IdSpaceKind=kind, IdSpace=lambda **kw: NS(write=None, read_v2p=None, **kw), KVIndexTable=NS,
        fused_draft_host_allocator=lambda pool: None, _is_npu=False, get_parallel=lambda: NS())
    translator_type = type("PrivateTranslator", (), {
        name: method(index, "KVIndexTranslator", name, translator_env)
        for name in ("__init__", "space", "read_source", "_reads_translated", "_passthrough_table")})
    window_kind = method(index, "KVIndexTranslator", "window_kv_kind", {"IdSpaceKind": kind})
    plan_type = type("KVLocPlan", (), {name: method(loc, "KVLocPlan", name, {"IdSpaceKind": kind})
        for name in ("__init__", "write_ids", "_window_in", "_cols")})
    class_methods = {name: method(swa, "SWAComponent", name) for name in (
        "_translate_full_to_swa", "_page_pairs", "_transfer_swa_pages", "_set_swa")}
    component_type = type("SWAComponent", (), class_methods)
    translator = translator_type(req_to_token=torch.zeros(1, 1, dtype=torch.long), page_size=1,
        device="cpu", token_to_kv_pool_allocator=object(), token_to_kv_pool=object(), is_draft_worker=True)
    assert not translator.is_translating and window_kind(translator) == kind.FULL
    worker = NS(draft_model_runner=NS(kv_index_translator=translator))

    def full_ids(virtual):
        plan = plan_type(source=translator, req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([virtual.numel()]), seq_lens_cpu=None, write_virtual=virtual)
        ids = projection_ids(worker, plan)
        assert ids is virtual
        return ids

    class Allocator:
        def __init__(self, count):
            self.table = torch.full((count,), -1, dtype=torch.long)
            self.freed = []
            self.swa_attn_allocator = NS(virtual_to_physical=self.table, bind=self.set_full_to_swa_mapping,
                clear_inverse_history=lambda: None)

        def translate_swa_indices_for_transfer(self, ids):
            return self.table[ids]

        def clear_full_to_swa_mapping(self, ids):
            self.table[ids] = -1

        def set_full_to_swa_mapping(self, ids, physical):
            self.table[ids] = physical

        def free_full_segment(self, ids, start_pos):
            assert start_pos == 0
            self.free(ids)

        def free(self, ids):
            assert (self.table[ids] == -1).all()
            self.freed.append(ids.clone())

    cases = []
    generator = torch.Generator().manual_seed(243)
    for rows in (1, 8, 65, 1024, 2048):
        slots = rows * 3 + 1
        for unified in (False, True):
            perm = torch.randperm(slots - 1, generator=generator) + 1
            kept, incoming, swa_physical = perm.split(rows)
            alloc = Allocator(slots)
            alloc.table[incoming] = swa_physical
            published = []
            component = component_type()
            component.cache = NS(token_to_kv_pool_allocator=alloc)
            component.tree_core = NS(page_size=1, set_component_device_value=lambda *args: published.append(args))
            component.component_type = "swa"
            component._unified_allocator = lambda: alloc if unified else None
            node = NS(id=7, component_data={"full": NS(value=kept, lock_ref=1)})
            emitted = []
            emit(component, node=node, incoming_full=incoming, start=0, end=rows, result=None, cache_actions=emitted)
            assert len(emitted) == 1 and type(emitted[0]) is recovery
            from_rust_tag = convert(("recover_swa_with_locked_full", node.id, kept, incoming))
            assert emitted[0].kept_full is from_rust_tag.kept_full and emitted[0].incoming_full is from_rust_tag.incoming_full

            expected = torch.randn((rows, 5, 2, 8), generator=generator).to(torch.bfloat16)
            target = torch.full((slots, 8), float("nan"), dtype=torch.bfloat16)
            target[swa_physical] = expected[:, 0, 1]
            draft = torch.full((slots, 5, 2, 8), float("nan"), dtype=torch.bfloat16)
            draft[full_ids(incoming)] = expected
            dense = draft.clone()
            dense[full_ids(kept)] = expected
            apply(component, from_rust_tag)
            assert len(alloc.freed) == 1 and torch.equal(alloc.freed[0], incoming)
            assert len(published) == 1 and torch.equal(published[0][2], swa_physical)
            assert torch.equal(alloc.table[kept], swa_physical) and torch.equal(target[alloc.table[kept]], expected[:, 0, 1])
            translator.req_to_token = kept.reshape(1, -1).clone()
            table = translator.read_source(None, req_pool_indices=torch.tensor([0]), bs=1, kind=window_kind(translator))
            assert not table.is_translated and table.entry_page_size == 1
            read_ids = table.ids[table.row_ids[0]]
            assert torch.equal(read_ids, kept) and torch.equal(dense[read_ids], expected)
            exposed = draft[read_ids]
            assert torch.isnan(exposed).all()
            # Use ordinary positive attention weights to expose a read with
            # valid target history but omitted private draft values.
            reference_v = dense[kept, 0, 1].float().mean(0)
            omitted_v = exposed[:, 0, 1].float().mean(0)
            assert torch.isfinite(reference_v).all() and torch.isnan(omitted_v).all()
            cases.append(dict(rows=rows, allocator="unified" if unified else "static",
                target_swa_correct=True, dense_draft_correct=True, skipped_draft_poison_words=exposed.numel()))

    # The native H200 Rust core emits the same tuple for both whole and
    # partial overlap. Bind the exact source; this script does not compile Rust.
    rust_path = "rust/sglang-radix-tree/src/components/swa.rs"
    rust = subprocess.check_output(["git", "-C", reader.BASE, "show", f"{reader.PIN}:{rust_path}"]).decode()
    assert rust_path not in json.loads((BASE.parent/"validation.json").read_bytes())["paths"]
    assert rust.count("cache_actions.push(Box::new(RecoverSwaWithLockedFull") == 2
    assert "node.device_lock_ref(FULL) > 0 || preserve_full_ids" in rust
    native_rust_lines = [n for n, s in enumerate(rust.splitlines(), 1)
                         if "cache_actions.push(Box::new(RecoverSwaWithLockedFull" in s]
    path = "mem_cache/layout/fused_draft.py"
    prereq = method(path, "WindowHostKind", "prerequisite", {"WINDOW_RAIL_BACKENDS": {"triton"}})
    assert 'WINDOW_RAIL_BACKENDS = frozenset({"triton"})' in sources[path]
    assert prereq(None, layers=None, ctx=NS(draft_backends=["triton"])) is None
    fa4_decline = prereq(None, layers=None, ctx=NS(draft_backends=["fa4"]))
    assert "no per-step" in fa4_decline
    staging = reader.read_source("mem_cache/hybrid_cache/hybrid_pool_assembler.py", False)
    assert "indices_from_pool=PoolName.KV" in staging and "indices_from_pool=PoolName.SWA" in staging
    assert "draft_swa_pool = draft_kv_pool.swa_kv_pool" in staging
    config = reader.read_source("mem_cache/kv_cache_configurator.py", False)
    assert "if fused_hicache and set(profile.kinds) != {LAYER_FULL}" in config
    log = ROOT/"serving_v0523/runs/04b01f958aa5487c.log"
    report = log.read_text()
    full, swa_count = map(int, re.search(r"full_layer_tokens=(\d+), swa_layer_tokens=(\d+)", report).groups())
    draft_count, draft_gb = re.search(r"draft KV cache allocated. dtype: torch.bfloat16, #tokens: (\d+), KV size: ([0-9.]+) GB", report).groups()
    assert int(draft_count) == full and "RustUnifiedTreeCore" in report
    info = next((ROOT/"experiments/modal_node/20261113T204343Z-bench").glob("*__v41__server_info.json"))
    assert json.loads(info.read_bytes())["internal_states"][0]["speculative_draft_attention_backend"] == "fa4"
    assert "Allocating draft hierarchical KV host pool" in report
    per_row = 2560
    # Reported pool allocation may include alignment and sink slots. The
    # alternate capacity below counts only logical KV payload.
    assert 0 <= float(draft_gb) - full * per_row / 2**30 < 0.01
    output = dict(parent_sha256=sha256(BASE.read_bytes()).hexdigest(), torch=str(torch.__version__),
        native_method_anchors=anchors, cases=cases, total_poison_words=sum(x["skipped_draft_poison_words"] for x in cases),
        rust_core=dict(source=rust_path, lines=native_rust_lines, sha256=sha256(rust.encode()).hexdigest(),
            executed="Native Python component emits the action; same tagged tuple decoded with native Rust adapter and consumed by native Python allocator handler. Rust producer source checked, not run."),
        archive=dict(log=str(log.relative_to(ROOT)), log_sha256=sha256(log.read_bytes()).hexdigest(),
            full_slots=full, target_swa_slots=swa_count, draft_bf16_slots=int(draft_count), reported_draft_gb=float(draft_gb),
            logged_core="RustUnifiedTreeCore", configured_v41_draft_backend="fa4", bf16_draft_bytes_per_row=per_row,
            hypothetical_swa_capacity_difference_gib=round((full-swa_count)*per_row / 2**30, 3)),
        alternative=dict(fa4_window_host_decline=fa4_decline, triton_window_host_prerequisite=None,
            note="A standalone SWA sidecar exists for a genuinely SWA-routed draft pool. The deployed full-addressed private draft's attention window does not establish that storage contract."),
        scope="Counterexample for omitting old private KV on a fresh middle chunk based solely on target SWA tombstones. Existing dense V241 preserves those bytes. Fixture uses five BF16 K/V layers of reduced feature width. No measurement or submission.")
    (HERE/"locked_recovery.json").write_text(json.dumps(output, indent=2)+"\n")
    print(f"{len(cases)} allocator/length pairs, {output['total_poison_words']} exposed BF16 words")


if __name__ == "__main__":
    main()
