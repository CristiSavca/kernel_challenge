"""Replay captured rank totals through the actual V308/V309 allocation and wave planners."""

import ast
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WAVE = "python/sglang/srt/layers/muse_spark/humming_prefill_waves.py"
K, E, GLOBAL, H, I = 8, 32, 256, 4096, 4096
torch.set_num_threads(1)
triton = SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d,
                         next_power_of_2=lambda n: 1 << (n - 1).bit_length())


def functions(candidate):
    path = next(HERE.parent.glob(candidate + "_*/patch.diff"))
    lines = path.read_text().splitlines()
    first = lines.index(f"+++ b/{WAVE}")
    assert lines[first + 1].startswith("@@ -0,0 +1,")
    end = next(i for i in range(first + 2, len(lines)) if lines[i].startswith("diff --git "))
    assert all(line.startswith("+") for line in lines[first + 2:end])
    source = "\n".join(line[1:] for line in lines[first + 2:end]) + "\n"
    wanted = {"grouped_prefill_waves", "wave_workspace_elements", "wave_capacity", "plan_waves"}
    methods = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    scope = dict(torch=torch, triton=triton)
    exec(compile(ast.Module(methods, []), WAVE, "exec"), scope)
    return scope, sha256(source.encode()).hexdigest()


class GeometryReady(Exception):
    pass


def allocated(functions_, tokens, conditional=True, prepared=True, bind=True):
    """Execute the shipped tensor allocation and both fallback branches on the PyTorch meta device."""
    h = torch.empty((tokens, H), dtype=torch.bfloat16, device="meta")
    ids = torch.empty((tokens, K), dtype=torch.int32, device="meta")
    layer = SimpleNamespace(num_local_experts=E, global_num_experts=GLOBAL,
                            intermediate_size_per_partition=I, humming_conditional_waves=conditional,
                            humming_prompt_gate_up=True if prepared else None)
    layout = {}

    def after_alloc(local_ids, experts, capacity):
        # The planner entry is after every mode's actual allocation, views, and bind decision.
        state = inspect.currentframe().f_back.f_locals
        workspace = state["workspace"]
        assert local_ids is ids and experts == E
        fused = state["project"] is not None
        live_views = {k: state[k] for k in ("packed", "activation", "projected")}
        if not fused:
            live_views["gate_up"] = state["gate_up"]
        assert all(x.untyped_storage()._cdata == workspace.untyped_storage()._cdata for x in live_views.values())
        assert all(x.storage_offset() + x.numel() <= workspace.numel() for x in live_views.values())
        shapes = {k: (x.storage_offset(), x.numel()) for k, x in live_views.items()}
        def disjoint(a, b):
            p, n = shapes[a]
            q, m = shapes[b]
            return p + n <= q or q + m <= p
        assert disjoint("activation", "projected")
        assert disjoint("packed", "activation") if fused else disjoint("packed", "gate_up")
        assert (shapes["packed"] == shapes["projected"]) if fused else disjoint("gate_up", "activation")
        rows = state["rows"]
        limit = triton.cdiv(tokens * K, capacity)
        assert rows >= min(tokens * K, capacity + K - 1) and 0 < limit <= 16
        layout.update(bytes=workspace.nbytes, rows=rows, capacity=capacity, max_waves=limit, fused=fused)
        raise GeometryReady

    names = {"sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton": {"fused_post_expert_norm_weighted_sum": None},
             "sglang.srt.layers.muse_spark.muse_spark_humming_moe": {"_expert_scaled_silu_module": None, "run_grouped_prompt_projection": None},
             "sglang.srt.layers.muse_spark.decode_gate_up": {"bind_prompt_wave": lambda *args: object() if bind else None}}
    modules = {}
    for name, fields in names.items():
        modules[name] = ModuleType(name)
        vars(modules[name]).update(fields)
    cap = functions_["wave_capacity"](tokens, K, E, H, I)
    with patch.dict(sys.modules, modules), patch.dict(functions_, {"plan_waves": after_alloc}):
        try:
            functions_["grouped_prefill_waves"](layer, h, None, ids, cap)
        except GeometryReady:
            pass
    assert layout["fused"] == (prepared and bind)
    return layout


def observed_bounds(plan, tokens, count, layout):
    # The archive retains one rank's total, not token order. Cover both endpoints of the
    # mathematically possible final-token local count. Check the shipped stable sort, edges,
    # and token ranges for both resulting witness layouts. No original route IDs are inferred.
    last_min = max(0, count - (tokens - 1) * K)
    last_max = min(K, count)
    outputs = []
    for last in sorted({last_min, last_max}):
        live = torch.zeros((tokens, K), dtype=torch.bool)
        live[:-1].view(-1)[:count - last] = True
        live[-1, :last] = True
        ids = torch.arange(tokens * K, dtype=torch.int32).view(tokens, K) % E
        ids.masked_fill_(~live, -1)
        bins, slots, edges, bounds = plan(ids, E, layout["capacity"])
        actual = int((bounds[:-1] < tokens).sum())
        expected = min(layout["max_waves"], 1 + (count - last) // layout["capacity"])
        assert actual == expected and bounds[0] == 0 and bounds[-1] == tokens
        assert (bounds[1:] >= bounds[:-1]).all()
        assert torch.equal(slots[:count].sort().values, live.flatten().nonzero().flatten())
        assert (bins[:count] % E == ids.flatten()[slots[:count]]).all()
        for wave, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            start, end = edges[wave * E], edges[(wave + 1) * E]
            routed = slots[start:end]
            assert (routed >= lo * K).all() and (routed < hi * K).all()
            assert routed.numel() == live[lo:hi].sum().item() <= layout["rows"]
        outputs.append(actual)
    return [min(outputs), max(outputs)]


def main():
    parent, parent_hash = functions("v308")
    child, child_hash = functions("v309")
    controls = {}
    trials = []
    inputs = {}
    for path in sorted((ROOT / "experiments/modal_serving").glob("*/result.json")):
        data = json.loads(path.read_bytes())
        if data.get("gpu") != "NVIDIA H200" or data.get("status") != "complete":
            continue
        sampled = False
        for case in data.get("cases", []):
            if case.get("T", 0) * K < GLOBAL * 256 or "stock_us" not in case or "local_routes" not in case:
                continue
            controls.setdefault((case["T"], case["local_routes"]), []).append(path.parent.name)
            sampled = True
        captures = data.get("bcg_dense384", {})
        if "trial2" in captures:
            for i in range(3):
                trial = captures[f"trial{i}"]
                assert trial["replay_equals_eager"] and trial["finite"]
                trials.append((16384, trial["local_routes"], "forced-hot graph replay" if i == 2 else "graph replay", path.parent.name))
            sampled = True
        if sampled:
            inputs[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    observed = [(t, r, "timed control", names) for (t, r), names in sorted(controls.items())]
    observed += [(t, r, label, [run]) for t, r, label, run in trials]
    assert {t for t, r, label, names in observed} == {8192, 12288, 16384}
    report, fixtures = [], []
    for t, r, label, names in observed + [(16384, n, "generated correctness witness", []) for n in (0, 8 * 16384)]:
        old = allocated(parent, t)
        new = allocated(child, t)
        fallback = allocated(child, t, bind=False)
        assert new["bytes"] == fallback["bytes"] < old["bytes"]
        for conditional, prepared in [(False, True), (True, False)]:
            assert allocated(child, t, conditional, prepared) == allocated(parent, t, conditional, prepared)
        row = dict(tokens=t, rank_routes=r, source=label, physical_run_ids=names,
                   v204_fused_live_bf16_bytes=2 * r * (H + I))
        for name, layout, func in [("v308", old, parent), ("v309", new, child), ("v309_bind_declined", fallback, child)]:
            row[name] = dict(**layout, conditional_live_bodies=observed_bounds(func["plan_waves"], t, r, layout))
        (report if names else fixtures).append(row)
    return dict(wave_source_sha256=dict(v308=parent_hash, v309=child_hash), h200_component_archive_sha256=inputs,
                precision="Rank totals were observed on one H200 using generated inputs. CUDA graph replay elapsed times and eight-rank serving histograms are unavailable for these candidates.",
                replay="PyTorch meta allocation and CPU execution of shipped planners. Both final-token occupancy bounds are attained using constructed routes. Counted bodies require conditional CUDA capture; eager warmup runs max_waves.",
                memory="BF16 expert workspace only, excludes metadata, graph objects, allocator rounding, and all other model buffers.",
                recorded=report, synthetic_extremes=fixtures)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
