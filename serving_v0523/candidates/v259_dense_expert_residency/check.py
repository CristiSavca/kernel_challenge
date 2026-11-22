"""Evaluate BF16 expert-bank capacity and first-use reuse on V256. Requires only Python and Git."""
import ast
from collections import OrderedDict
from contextlib import contextmanager
from hashlib import sha256
from itertools import combinations, permutations
from math import ceil
import json
import logging
import os
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SERVICE = ROOT / "serving_v0523"
CANDIDATE = HERE.parent / "v256_window_backed_mixed_serving"
CAPACITY = HERE.parent / "v249_private_draft_swa/capacity.json"
TRACE = HERE.parent / "v251_hicache_residency/trace.json"
REACH = HERE.parent / "v197_grouped_prompt_gate_up/independent_reach.json"
HARDWARE = ROOT / "experiments/modal_serving/20261113T065259Z/result.json"
BENCHMARK = ROOT / "tools/serving_moe_ab_worker.py"
SRT = "python/sglang/srt/"
MIB, GIB = 1 << 20, 1 << 30


def load(p):
    return json.loads(p.read_bytes())


@contextmanager
def source_tree():
    manifest = load(CANDIDATE / "validation.json")
    patch = CANDIDATE / "patch.diff"
    assert sha256(patch.read_bytes()).hexdigest() == manifest["patch_sha256"]
    repo = SERVICE / "sglang"
    with TemporaryDirectory(prefix="okc-v259-index-") as tmp:
        objects = Path(tmp) / "objects"
        objects.mkdir()
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"), GIT_OBJECT_DIRECTORY=str(objects),
                   GIT_ALTERNATE_OBJECT_DIRECTORIES=str(repo / ".git/objects"))

        def git(*args):
            return subprocess.check_output(["git", "-C", repo, *args], env=env)

        git("read-tree", manifest["base"])
        git("apply", "--cached", "--whitespace=error-all", patch)
        assert git("write-tree").decode().strip() == manifest["tree"]
        yield lambda path: git("show", manifest["tree"] + ":" + SRT + path).decode()


def bind(source, name, env, cls=None):
    items = ast.parse(source).body
    if cls:
        items = next(node.body for node in items if getattr(node, "name", None) == cls)
    fn = next(node for node in items if getattr(node, "name", None) == name)
    fn.decorator_list = []
    exec("from __future__ import annotations\n" + ast.unparse(fn), env)
    return env[name]


def allocator_replay(files, cap, layer_bank):
    # Execute the released target sizing functions. Backed bytes and headroom are modeled
    # relative to the one measured control point. The unknown pre-capture cap may bind.
    node = ast.parse(files["model_executor/model_runner_components/kv_pool_runtime.py"])
    resize = next(n for n in node.body if getattr(n, "name", None) == "compute_post_capture_kv_resize")
    budget = next(n.value for n in resize.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "budget_bytes" for t in n.targets))
    solve = files["model_executor/pool_configurator.py"]
    env = dict(logger=logging.getLogger(), MemoryPoolConfig=NS)
    cls = "HybridSWAPoolConfigurator"
    names = ("_swa_cell_bytes", "_full_cell_bytes", "_recompute_cell_size", "_solve_pool_sizes",
             "_make_pool_config", "calculate_pool_sizes", "calculate_pool_sizes_from_max_tokens")
    methods = {n: bind(solve, n, env, cls) for n in names}
    methods["validate_swa_pool_size"] = staticmethod(bind(solve, "validate_swa_pool_size", env,
                                                           "MemoryPoolConfigurator"))
    planner = type("Planner", (), methods)()
    planner.__dict__.update(_full_per_token=256, _swa_per_token=256, _full_layers_num=16, _swa_layers_num=46,
                            _draft_swa_full_layers_num=0, _draft_full_layers_num=0, _draft_swa_layers_num=0,
                            _fused_full_entry=None, _fused_swa_entry=None, _draft_cell_size=2560,
                            _swa_full_tokens_ratio=0.25, _enable_unified_memory=False, _sliding_window_size=2047,
                            _page_size=1)
    planner._recompute_cell_size()
    assert planner._cell_size == 9600
    kvc = files["mem_cache/kv_cache_configurator.py"]
    apply_cfg = bind(kvc, "config_from_budget", dict(
        get_schedule=lambda:NS(page_size=1), get_memory=lambda:NS(enable_unified_memory=False)), "KVCacheConfigurator")
    # Its only import is the configurator factory. Supply the already constructed native
    # planner here while retaining the runtime's own token constraints, cap and recomputation.
    imports = {"__import__": lambda *a: NS(create_memory_pool_configurator=lambda _: planner)}
    apply_cfg.__globals__["__builtins__"] = dict(vars(__import__("builtins")), **imports)
    import types
    apply_cfg = types.FunctionType(apply_cfg.__code__, apply_cfg.__globals__)
    expr = compile(ast.Expression(budget), "<V256 post-capture KV budget>", "eval")
    control = cap["full_slots"]
    staged = []
    for surplus in (0, layer_bank * 2):
        for backing in (0, layer_bank, 2 * layer_bank):
            # The V68 capacities determine a calibrated budget. The unobserved slack
            # between this budget and the pre-capture token cap is a separate scenario.
            units = eval(expr, dict(
                free_gb=(control * planner._cell_size + surplus - backing) / GIB,
                headroom_gb=0., mm_reservation_gb=0., pool=NS(post_capture_backed_bytes=0),
                draft_pools=(), canary_workspace_bytes=0))
            out = apply_cfg(NS(_apply_token_constraints=lambda n: n), units, cap_tokens=control)
            staged.append(dict(cap_slack_bytes=surplus, persistent_bank_bytes=backing,
                               full_slots=out.full_max_total_num_tokens, swa_slots=out.swa_max_total_num_tokens,
                               full_slots_lost=control-out.full_max_total_num_tokens))
    return dict(native_target_charge_bytes=planner._cell_size, scenarios=staged)


def exact_factors():
    # Magnitudes of the E2M1 operand used by the native grouped path. The positive E4M3
    # scale is filtered the same way as prepare_humming_nvfp4, before any global scale.
    totals = 0
    for sb in range(127):
        scale = 0.0 if sb < 8 else (1.0 + (sb % 8) / 8.0) * (2.0 ** (sb // 8 - 7))
        for nibble in range(16):
            mag = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[nibble & 7]
            val = (-mag if nibble & 8 else mag) * scale
            word, = struct.unpack(">I", struct.pack(">f", val))
            assert (word & 65535) == 0
            totals += 1
    return totals


def scan_cache(size, capacity, sweeps):
    recent = OrderedDict()
    hits = []
    for _ in range(sweeps):
        n = 0
        for owner in range(size):
            if owner in recent:
                n += 1
                recent.move_to_end(owner)
            else:
                recent[owner] = None
                if len(recent) > capacity:
                    recent.popitem(last=False)
        hits.append(n)
    return hits


def knapsack(blocks, bank_count, costs):
    """Best weighted first-use conversion saving at a sweep boundary, in microseconds."""
    options = [(w13 * costs["w13"] + min(blocks - 2*w13, bank_count) * costs["w2"],
                w13, min(blocks - 2*w13, bank_count))
               for w13 in range(min(blocks//2, bank_count) + 1)]
    saved, w13, w2 = max(options)
    return dict(saved_us=saved, cached_W13=w13, cached_W2=w2)


def check_oracle(costs):
    # Each source weight appears once in a sweep. Enumerate initial cache contents, all
    # request permutations and every optional admission on a miss, using four weights.
    weights = (2, 1, 2, 1)
    value = (costs["w13"], costs["w2"]) * 2
    witnesses = 0
    subsets = [frozenset(s) for k in range(5) for s in combinations(range(4), k)]
    for capacity in range(1, 7):
        fitting = [s for s in subsets if sum(weights[i] for i in s) <= capacity]
        budget = knapsack(capacity, 2, costs)["saved_us"]
        for order in permutations(range(4)):
            states = {s: 0.0 for s in fitting}
            for item in order:
                new = {}
                for have, score in states.items():
                    trial = have | {item}
                    score += value[item] if item in have else 0.
                    for keep in fitting:
                        if keep <= trial:
                            new[keep] = max(new.get(keep, 0.), score)
                states = new
            assert abs(max(states.values()) - budget) < 1e-6
            witnesses += 1
    return witnesses


def main():
    cap, reach, trace, timed = map(load, (CAPACITY, REACH, TRACE, HARDWARE))
    bench = BENCHMARK.read_text()
    assert "H = I = 4096" in bench and "E_GLOBAL, E_LOCAL, RANK, TOPK = 256, 32, 3, 8" in bench
    paths = ("model_executor/pool_configurator.py", "mem_cache/kv_cache_configurator.py",
             "model_executor/model_runner_components/kv_pool_runtime.py", "layers/muse_spark/muse_spark_humming_moe.py",
             "configs/muse_spark.py")
    with source_tree() as load_source:
        files = {p:load_source(p) for p in paths}
    src = files["layers/muse_spark/muse_spark_humming_moe.py"]
    for pattern in ("weight_scale_group_size=16", "scale_bytes.masked_fill(scale_bytes < 8, 0)",
                    "HummingMethod.forward_layer(", "layer.humming_prompt_down", "humming_prompt_gate_up"):
        assert pattern in src
    assert "num_hidden_layers=62" in files["configs/muse_spark.py"]
    cfg = load(CANDIDATE / "validation.json")
    rank_experts, layers, channels = 32, 62, 4096
    full_weights = 3 * rank_experts * channels * channels
    dense, packed = 2 * full_weights, full_weights // 2 + full_weights // 16
    assert dense == 3 * GIB and packed == 864 * MIB
    saved = cap["fixed_slot_device_payload_bytes_saved"]
    net = saved - cap["native_fa_extra_graph_bytes_at_these_dimensions"]
    t = next(r for r in timed["cases"] if r["T"]==16384)
    costs = dict(w13=timed["dequant_us"]["w13_8192x4096"]["eager_us"],
                 w2=timed["dequant_us"]["w2_4096x4096"]["eager_us"])
    conversion = rank_experts * sum(costs.values())
    best = knapsack(net // (32*MIB), layers * rank_experts, costs)
    cases = []
    for row in reach["archive"]:
        if row["users"] not in (16, 32):
            continue
        win = next(w for w in trace["v41"]["windows"] if w["concurrency"]==row["users"] and w["rep"]==row["rep"])
        calls = row["logical_grouped_passes"]
        cases.append(dict(users=row["users"], grouped_sweeps=calls, wall_s=win["wall_s"],
                          all_pinned_conversion_s=conversion * layers * calls / 1e6,
                          feasible_pinned_conversion_s=best["saved_us"] * calls / 1e6,
                          feasible_fraction_of_wall=best["saved_us"] * calls / 1e6 / win["wall_s"],
                          wavefront_conversion_savings_vs_per_chunk_materialization_s={
                              str(k): conversion * layers * (calls - ceil(calls / k)) / 1e6
                              for k in (2, 4, 8)},
                          needed_ms_per_layer_to_gain_5pct=row["net_ms_per_layer_call_for_5pct_raw"],
                          needed_ms_per_layer_to_gain_2_833x=row["net_ms_per_layer_call_for_2_833x_raw"]))
    result = dict(parent_V256_sha256=cfg["patch_sha256"], source_tree=cfg["tree"],
                  source_sha256={p:sha256(v.encode()).hexdigest() for p,v in files.items()},
                  inputs={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                          for p in (CAPACITY, REACH, TRACE, HARDWARE, BENCHMARK)},
                  hardware="Historical single H200 using older Marlin/V42 layouts; V256 untimed",
                  exact_block_scale_E2M1_cases=exact_factors(),
                  per_layer=dict(dense_rank_bytes=dense, packed_rank_bytes=packed,
                                 two_bank_dense_total_bytes=2*dense, multi_layer_dense_bytes=layers*dense,
                                 ideal_logical_nonresident_weight_bytes=packed + 2*dense,
                                 ideal_serialized_transfer_ms=(packed + 2*dense) / 4.8e12 * 1e3,
                                 extra_bytes_vs_ideal_one_packed_read=2*dense),
                  fixed_slot_payload_reduction_bytes=saved, native_graph_metadata_bytes=saved-net,
                  model_headroom_bytes=net, spare_after_two_dense_banks=net-2*dense,
                  sizing=allocator_replay(files,cap,dense),
                  rolling_full_layer_cache_hits=scan_cache(layers,2,4),
                  best_persistent_matrices_against_per_expert_dequant=best,
                  optimal_cache_order_checks=check_oracle(costs),
                  physical_H200=dict(component_local_routes=t["local_routes"],v40_marlin_us=t["v40_us"],
                      V42_dense_us=t["dense384_us"],dense_uniform_baseline=t["dense_ceiling"],dequant_us=costs,
                      all_experts_dequant_us=conversion,serialized_dequant_plus_dense_uniform_us=conversion+t["dense_ceiling"]["us"],
                      older_dense_output_changed_fraction=t["dense384_frac_diff_elems"]),
                  timing_sensitivity=cases)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
