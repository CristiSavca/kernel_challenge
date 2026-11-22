"""Bound exact same-forward reuse from native admissions and observed cold prompt lengths."""
import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
CAP = 16384
LEFT = 2047
PER_FULL_LAYER_ROW = 2 * 2 * 64
FULL_LAYERS = 16
HELPER = ROOT / "experiments/iterations/20261119-serving-cold-wave.py"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def function_at(src, name):
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)


def source_checks():
    sglang = ROOT / "serving_v0523/sglang"
    names = ("scheduler.py", "schedule_policy.py")
    sources = {n: subprocess.check_output(["git", "-C", sglang, "show",
        PIN + ":python/sglang/srt/managers/" + n]).decode() for n in names}
    admission = function_at(sources["scheduler.py"], "_get_new_batch_prefill_raw")
    calls = ("self.chunked_req = adder.add_chunked_req", "req.init_next_round_input(self.tree_cache)",
             "res = adder.add_one_req(", "self.chunked_req = adder.new_chunked_req")
    assert [admission.index(x) for x in calls] == sorted(admission.index(x) for x in calls)
    p = sources["schedule_policy.py"]
    old = function_at(p, "add_chunked_req")
    assert "len(req.full_untruncated_fill_ids) - req.prefix_len" in old and "return req if truncated else None" in old
    assert "if admission.is_chunked:\n            self.new_chunked_req = req" in p
    # Both measured builds use these manager files verbatim.
    for v in ("v41_swa_free", "v44_dense_prefill_448"):
        text = (HERE.parent / v / "patch.diff").read_text()
        for n in names:
            assert "diff --git a/python/sglang/srt/managers/" + n + " " not in text
    return {name: sha256(src.encode()).hexdigest() for name, src in sources.items()}


def run(tag, directory, stem):
    parent = ROOT / "experiments/modal_node" / directory
    client, server = [parent / (stem + x) for x in ("load.json", "server.log")]
    data = json.loads(client.read_bytes())
    sections = server.read_text().split("Cache flushed successfully!")[1:]
    reps = [(r["concurrency"], i) for r in data["rungs"] for i in range(r["reps"])]
    assert len(sections) >= len(reps) and data["gpus"] == 8
    waves = []
    for (users, rep), section in zip(reps, sections):
        clients = [c for c in data["requests"] if (c["load"], c["rep"], c["turn"]) == (users, rep, 0)]
        assert len(clients) == users
        ev = cold_wave(section, clients)
        assert all(0 < e["new"] <= CAP and e["seqs"] in (1, 2) for e in ev)
        minimum = min(c["prompt"] for c in clients)
        pairs = [(i, e) for i, e in enumerate(ev) if e["seqs"] == 2]
        assert len(pairs) == users - 1 and minimum > CAP + max(e["cached"] for e in ev)
        assert sum(c["cached"] for c in clients) == sum(e["cached"] for e in ev)
        assert [e["running"] for _, e in pairs] == list(range(users - 1))
        assert all(ev[i + 1]["running"] == e["running"] + 1 for i, e in pairs)
        # Pair: older suffix has t rows, newcomer has n rows and matched p rows.
        # Old begins at L-t and newcomer ends at p+n (exclusive).
        # L-t-(p+n) = L - (t+n) - p. No unknown per-request split or arrival order needed.
        gaps = [minimum - e["new"] - e["cached"] for _, e in pairs]
        assert all(gap > LEFT for gap in gaps)
        incoming_shared = sum(e["cached"] for _, e in pairs)
        waves.append(dict(users=users, rep=rep, prompt_rows=sum(c["prompt"] for c in clients),
            cached_rows=sum(e["cached"] for e in ev), shortest_prompt=minimum,
            prompt_forwards=len(ev), two_prompt_forwards=len(pairs),
            pair_cached_rows_histogram=dict(sorted(Counter(e["cached"] for _, e in pairs).items())),
            minimum_query_gap_lower_bound=min(gaps, default=None),
            minimum_window_gap_lower_bound=min((g - LEFT for g in gaps), default=None),
            identical_position_query_rows_in_same_forward=0, overlapping_sliding_key_positions_in_same_forward=0,
            upper_full_kv_shared_slots_summed_across_boundaries=incoming_shared,
            single_load_full_kv_payload_upper_bytes_per_rank=incoming_shared * PER_FULL_LAYER_ROW * FULL_LAYERS))
    return dict(run=tag, source_log_sha256=digest(server), source_client_sha256=digest(client), waves=waves)


cold_wave = runpy.run_path(str(HELPER))["cold_wave"]
archives = runpy.run_path(str(HERE.parent / "v244_cold_prompt_scaling/check.py"))["ARCHIVES"]
print(json.dumps(dict(pinned_manager_sources=source_checks(), pin=PIN, helper_sha256=digest(HELPER),
    scope="Reconciled cold waves of completed archived clients, one tracked chunk and page-one native radix.",
    first_turn_only=True, chunk_limit=CAP, left_lookback=LEFT, full_layers=FULL_LAYERS,
    full_kv_bytes_per_row_and_layer_per_rank=PER_FULL_LAYER_ROW,
    reports=[run(*a) for a in archives]), indent=2))
