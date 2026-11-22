"""Verify sparse and dense V164 window seams for the public EP8 / top-eight geometry."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v149_overlapped_expert_route_planning"))
from cpu_check import HUM, MOE, install


def run(stage):
    torch.set_num_threads(1)
    sources = [HUM, MOE, HUM.parent / "sparse_prompt_exchange.py"]
    env = {"torch": torch, "_GROUPED_COMPUTE": None}
    for path, names in zip(sources, [
        ["_compact_prefill_windows", "compact_prefill_and_combine", "humming_grouped_expert_outputs"],
        ["_grouped_humming_and_combine_impl"], ["_group_width"],
    ]):
        install((stage / path).read_text(), names, env)
    n, h, k, ranks = 16384, 8, 8, 8
    r = torch.arange(n)
    x = torch.stack([r % 128, r // 128, *[(r % p) - 7 for p in (7, 11, 13, 17, 19, 23)]], 1).bfloat16()
    slot = torch.arange(k)
    scores = ((r[:, None] + slot) % 31 + 1).float() / 256
    scale1 = torch.linspace(0.25, 1.75, 32).bfloat16()
    scale2 = torch.linspace(0.375, 1.625, 32).bfloat16()
    cases = []
    cap = n + n // 8
    for mode in ("all_to_one", "near_hot_with_gaps", "at_capacity", "one_over", "empty_bank"):
        rows = r[:, None]
        dest = torch.zeros((n, k), dtype=torch.int32)
        if mode == "near_hot_with_gaps":
            dest[(r % 4) == 0] = 1
        if mode in ("at_capacity", "one_over"):
            dest.fill_(1)
            # Permute original tokens before selecting slots to make every seam nontrivial.
            key = ((rows * 143) % n) * k + slot
            dest[key < cap + int(mode == "one_over")] = 0
        if mode == "empty_bank":
            dest.fill_(1)
        local = torch.where(dest == 0, ((rows * 7 + slot // 2) % 32).int(), -1)
        needs = torch.stack([(dest == owner).any(1) for owner in range(ranks)])
        counts = needs.view(ranks, ranks, -1).sum(-1).T.tolist()
        group_width = env["_group_width"](counts, n)
        if mode in ("all_to_one", "empty_bank"):
            assert group_width == 0
        if mode == "near_hot_with_gaps":
            assert group_width > 0
        live = int((local >= 0).sum())
        natural = list(env["_compact_prefill_windows"](local, 32, 256))
        known = list(env["_compact_prefill_windows"](local, 32, 256, live))
        assert natural == known and sum(w[2] for w in known) == live
        assert (len(known) == 1) == (live <= cap)
        assert all(int((local[a:b] >= 0).sum()) == total <= cap for a, b, total in known)
        use = needs[0]
        dense, compact = x, x[use]
        rank_map = use.cumsum(0, dtype=torch.int32) - 1
        layer = SimpleNamespace(num_local_experts=32, global_num_experts=256, intermediate_size_per_partition=h,
            humming_grouped_tunings={"w13": None, "w2": None}, w13_global_scale=scale1,
            w2_global_scale=scale2, _post_expert_norm_weight=None, _post_expert_norm_eps=1e-6,
            _post_expert_norm_weight_first=False)

        def gate(g, y, ids, scales):
            g, u = (g * scales[ids.long(), None]).float().chunk(2, -1)
            y.copy_(torch.nn.functional.silu(g) * u)

        def evaluate(source, remap, scaled, route_count):
            result = torch.full_like(x, float("nan"))
            plans = known if scaled else [(0, n, live)]
            nonempty = iter((a, b) for a, b, total in plans if total)
            actual = []
            current = []
            def gemm(_layer, *, inputs, outputs, expert_layout, sublayer_name, **kw):
                if sublayer_name == "w13":
                    a, b = next(nonempty)
                    current[:] = [a, b]
                    ids = local[a:b].flatten()
                    order = ids.argsort(stable=True)
                    order = order[ids[order] >= 0]
                    original = order // k + a
                    assert torch.equal(inputs, x[original])
                    assert len(inputs) == expert_layout[-1].item()
                    assert torch.equal(torch.diff(expert_layout).long(), ids[order].long().bincount(minlength=32))
                    actual.append(len(inputs))
                    outputs[:, :h] = inputs / 8
                    outputs[:, h:] = (inputs.roll(1, dims=1) + 1) / 8
                else:
                    outputs.copy_(inputs.roll(1, dims=1))
            def final(values, weights, *, out, slot_row_indices, slot_expert_ids, expert_output_scales=None, **kw):
                a, b = current
                assert out.data_ptr() == result[a:b].data_ptr() and torch.equal(weights, scores[a:b])
                ids, mapped = local[a:b], slot_row_indices
                active = ids >= 0
                assert torch.equal(active, mapped >= 0)
                if scaled:
                    assert torch.equal(slot_expert_ids, ids) and expert_output_scales is scale2
                acc = torch.zeros_like(out, dtype=torch.float32)
                for j in range(k):
                    mask = active[:, j]
                    v = values[mapped[mask, j].long()]
                    if scaled:
                        v = (v * scale2[ids[mask, j].long(), None]).bfloat16()
                    f = v.float()
                    acc[mask] += f * torch.rsqrt(f.square().mean(1, keepdim=True) + 1e-6) * weights[mask, j, None]
                out.copy_(acc)
            env["HummingMethod"] = SimpleNamespace(forward_layer=gemm)
            env["fused_post_expert_norm_weighted_sum"] = final
            def stock_silu(g, out):
                a, b = g.float().chunk(2, dim=1)
                out.copy_(torch.nn.functional.silu(a) * b)
            env["silu_and_mul"] = stock_silu
            layer.humming_prefill_silu = SimpleNamespace(run=gate) if scaled else None
            env["_grouped_humming_and_combine_impl"](
                layer, source, local, scores, result, input_row_map=remap, local_route_count=route_count)
            assert actual == [total for _, _, total in plans if total] and torch.isfinite(result).all()
            return result

        reference = evaluate(dense, None, False, None)
        for source, remap in [(dense, None), (compact, rank_map)]:
            for count in (None, live):
                assert torch.equal(evaluate(source, remap, True, count), reference)
            assert torch.equal(evaluate(source, remap, False, live), reference)
        cases.append({"routing": mode, "local_routes": live, "rows_received": len(compact),
                      "transport_group_width": group_width, "routes_by_window": [row[2] for row in known]})
    return {"torch": torch.__version__, "sources": {str(p): hashlib.sha256((stage / p).read_bytes()).hexdigest() for p in sources},
            "shape": [n, k], "ep": ranks, "live_slots_per_window_cap": cap, "cases": cases,
            "bitwise_output_elements_compared": n * h * 6 * len(cases),
            "scope": "Native window, grouped adapter, sorting, remap and finalizer arguments. CPU surrogate GEMM and scaling/finalization. No hardware timing."}


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
