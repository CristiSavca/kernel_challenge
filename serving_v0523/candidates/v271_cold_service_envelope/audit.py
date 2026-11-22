"""Relate the archived closed-loop cold wave to model-shaped TP8/EP8 work."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import runpy
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / "serving_v0523/candidates"
HASHES = {}
TP, LAYERS, CHUNK = 8, 62, 16384
BF, FP8, HBM = 989.5e12, 1979e12, 4.8e12


def checked(path):
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = sha256(data).hexdigest()
    return json.loads(data) if path.suffix == ".json" else runpy.run_path(path)


def cand(prefix, name):
    return checked(next(CAND.glob(f"{prefix}_*/{name}")))


def triangle(n):
    return n * (n + 1) // 2


def pairs(n, window=None):
    if window is None or n <= window:
        return triangle(n)
    return triangle(window) + (n - window) * window


def attention(rows):
    counts = [sum(pairs(r["prompt"], width) - pairs(r["cached"], width) for r in rows)
              for width in (None, 2048)]
    return counts, 4 * 16 * 64 * (16 * counts[0] + 46 * counts[1])


def summarize_window(rows, layer, old_s, peak_s, forced_ratio, targets):
    cold = [r for r in rows if r["turn"] == 0]
    first = min(r["t0"] for r in rows)
    end = max(r["t0"] + r["dur"] for r in rows)
    first_wave = max(r["t0"] + r["ttft"] for r in cold) - first
    assert all(r["t0"] >= first + first_wave for r in rows if r["turn"] > 0)
    assert all(0 <= r["cached"] < r["prompt"] for r in cold)
    n = sum(r["prompt"] - r["cached"] for r in cold)
    calls = LAYERS * n / CHUNK
    work = {op: f * LAYERS * n for op, f in layer.items()}
    # One logical TP-partitioned draft context FC plus all five KV projections.
    work["draft_context_fc_and_kv"] = n * (838860800 // TP)
    counts, fa = attention(cold)
    replicated_draft_fc_extra = n * (TP - 1) * 2 * 40960 * 8192 // TP
    bf_s = sum(work.values()) / BF
    fp_s = fa / FP8
    demand = {str(goal): {
        "required_saved_ms_per_layer_16k_equivalent": 1000 * first_wave * (1 - 1/factor) / calls,
        "minimum_sustained_critical_over_mean_for_rebalance_only":
            1 + first_wave * (1 - 1/factor) / (old_s * calls)}
        for goal, factor in targets.items()}
    return dict(load=cold[0]["load"], rep=cold[0]["rep"], wall_s=end-first,
                first_wave_s=first_wave, cold_summed_request_ttft_s=sum(r["ttft"] for r in cold),
                cold_new_rows=n, cold_cached_rows=sum(r["cached"] for r in cold), later_new_rows=sum(r["prompt"]-r["cached"] for r in rows if r["turn"]>0),
                all_prompt_rows=sum(r["prompt"] for r in rows), all_cached_rows=sum(r["cached"] for r in rows),
                outputs=sum(r["completion"] for r in rows), full_and_windowed_pairs_per_head=counts,
                per_rank_mean_useful_bf16_flops=work, per_rank_useful_qk_av_flops=fa,
                optimistic_selected_tensor_core_seconds=bf_s+fp_s, bf16_peak_equivalent_s=bf_s,
                fp8_peak_equivalent_s=fp_s,
                extra_bf16_seconds_if_draft_fc_replicated=replicated_draft_fc_extra/BF, selected_peak_equivalent_fraction_of_archived_first_wave=(bf_s+fp_s)/first_wave,
                expert_16k_equivalent_layer_calls=calls,
                historical_mean_service_s=old_s*calls, ideal_bf16_expert_s=peak_s*calls,
                conditional_saving_s_if_forced_ratio_on_every_layer=(forced_ratio-1)*old_s*calls,
                conditional_saved_s_if_mean_control_accelerated_to_bf16_peak=(old_s-peak_s)*calls,
                conditional_uniform_phase_demand=demand)


def phase_score(windows, clients, service, multiplier):
    replay = cand("v195", "score_headroom.py")
    calibrate = cand("v265", "audit.py")["calibrate"]
    model, _ = calibrate(service)
    rounds = replay["measured_intervals"](clients)
    base = replay["rerate"](rounds, (1, 1, 1))

    def score(mode, later):
        # Each repetition has its own cold-wave exposure. Keep per-user gaps, output
        # demand and logical throughput numerator. Compound later TTFT and streaming.
        partitioned = {}
        for key, users in rounds.items():
            w = windows[key]
            gain = multiplier(w, mode)
            assert 0 <= gain < 1
            partitioned[key] = [(u[0]*(1-gain), *u[1:]) for u in users]
        new = replay["rerate"](partitioned, (1, later, later))
        return model([new[i][1]/base[i][1] for i in sorted(base)],
                     [new[i][0]/base[i][0] for i in sorted(base)])

    def needed(mode, goal):
        lo, hi = 1., 1000.
        if score(mode, float("inf")) < goal:
            return None
        while score(mode, hi) < goal:
            hi *= 2
        for _ in range(45):
            mid = (lo+hi)/2
            if score(mode, mid) >= goal:
                hi = mid
            else:
                lo = mid
        return hi

    return {mode: dict(frontier_if_later_unchanged=score(mode, 1),
                       frontier_if_later_halved=score(mode, 2),
                       frontier_if_later_zero_limit=(score(mode, float("inf")) if mode != "zero_cold_time_limit" else None),
                       required_later_ttft_and_stream_speedup={str(g):needed(mode, g) for g in (3,4)})
            for mode in ["no_cold_change", "critical_forced_ratio_to_mean",
                         "mean_control_to_peak_bf16", "zero_cold_time_limit"]}


def main():
    checked(ROOT/"tools/okc_node_loadgen.py")
    v268 = cand("v268","validation.json")
    assert "--speculative-dflash-sp-prefill-context" in v268["launch_flags"]
    # Global expert rows = top8 * tokens. Per-rank mean = 1 row per token, not 8.
    layer = dict(qkv_with_gate=2*8192*2304, output=2*1024*8192,
                 pooled_projection=(2*8192*4096)//TP, unpooled_projection=(2*4096*8192)//TP,
                 two_bf16_router_products=(2*2*8192*256)//TP,
                 routed_gate_up_down=2*3*4096**2)
    owned = cand("v250", "check.json")["full_shape_arithmetic"]
    assert [CHUNK*layer[k] for k in ("pooled_projection","unpooled_projection","two_bf16_router_products")] == [
        owned[k] for k in ("pool_flops","unpool_flops","router_flops")]
    draft = cand("v243", "coverage_replay.json")["global_dense_flops"]
    assert draft["draft_fc_and_five_kv_per_row"] == 838860800
    archive = cand("v186", "check.json")["archived_physical"]
    physical = archive["h200_16k"]
    for ref in physical:
        r = checked(ROOT / ref["source"])
        assert r["gpu"] == "NVIDIA H200" and r["status"] == "complete"
    # Normalize each measured 16,224-row duration to 16,384 before applying the
    # linear rate. This is a counterfactual cost curve, not a route histogram.
    old_s = median(r["stock_us"] * 1e-6 * CHUNK / r["routes"] for r in physical)
    forced = max(t["local_routes"] for r in archive["h200_graph_replay"] for t in r["trials"])/CHUNK
    assert not archive["ordinary_checked_all_rank_histograms"]
    peak_s = layer["routed_gate_up_down"] * CHUNK / BF
    service = checked(ROOT / "serving_v0523/runs/abd217baf0ed49fd.json")
    targets = {int(k):v["both_axes"] for k,v in cand("v265","audit.json")["score_scenarios"]["crossings"].items()}
    trace = {}
    for version, directory in [(41,"20261113T204343Z-bench"),(44,"20261113T213641Z-bench")]:
        source = next((ROOT/"experiments/modal_node"/directory).glob(f"*__v{version}__load.json"))
        trace[version] = checked(source)
    results = {}
    by_v41 = {}
    for ver, clients in trace.items():
        grouped = defaultdict(list)
        for row in clients["requests"]:
            grouped[row["load"],row["rep"]].append(row)
        windows = {k:summarize_window(v, layer, old_s, peak_s, forced, targets)
                   for k,v in sorted(grouped.items())}
        results[f"v{ver}"] = list(windows.values())
        if ver == 41:
            by_v41 = windows

    def gain(window, mode):
        return {"no_cold_change": 0,
                "critical_forced_ratio_to_mean":window["conditional_saving_s_if_forced_ratio_on_every_layer"]/window["first_wave_s"],
                "mean_control_to_peak_bf16":window["conditional_saved_s_if_mean_control_accelerated_to_bf16_peak"]/window["first_wave_s"],
                "zero_cold_time_limit":1-1e-12}[mode]

    frontier = phase_score(by_v41, trace[41], service, gain)
    assert abs(frontier["no_cold_change"]["frontier_if_later_unchanged"] - 1.4119) < 5e-5
    # Check against explicit finite causal masks for cached and uncached suffixes.
    for n,p,w in ((1,0,2048),(16,3,2),(30,20,16),(2052,2046,2048)):
        assert pairs(n,w)-pairs(p,w) == sum(min(t+1,w) for t in range(p,n))
    bank = cand("v259","check.json")["per_layer"]["packed_rank_bytes"]
    decode = cand("v262","check.json")["cases"]
    for r in decode:
        m = r["verify_rows"]
        e = 32*(1-(1-8/256)**m)
        assert abs((bank/32)*e*LAYERS/HBM*1000-r["expected_cold_expert_weights_peak_ms_62_layers"]) < 1e-9
    return dict(geometry=dict(layers=LAYERS, ranks=TP, full_layers=16, swa_layers=46, window=2048,
        dense_BF16_peak_TFLOPs=BF/1e12, dense_FP8_peak_TFLOPs=FP8/1e12, HBM_peak_TBps=HBM/1e12,
        active_rank_16k_chunk_bf16_flops_by_layer_family={k:v*CHUNK for k,v in layer.items()},
        all_32_packed_experts_rank_layer_bytes=bank, full_rank_expert_sweep_bytes=bank*LAYERS,
        full_sweep_one_read_hbm_seconds=bank*LAYERS/HBM, full_sweep_uniform_expert_BF16_peak_s=peak_s*LAYERS),
        historical_rank_controls=dict(reference="four independent runs of the same 16,224-route layout",
            normalized_median_complete_marlin_ms=1000*old_s, ideal_16384_route_BF16_ms=1000*peak_s,
            prepared_dense_ms=[r["isolated_bf16_us"]/1000 for r in physical],
            forced_one_rank_ratio_to_node_mean=forced, ordinary_all_rank_histograms=0),
        official_uniform_coupled_factors=targets, windows=results, conditional_four_load_frontier=frontier,
        archive_scope="V41/V44 physical requests, V68 target ownership and V268 sharded long-prompt draft shapes, older Marlin controls, ideal H200 peaks. Phase transfers and expert rank costs are modeled. A V68+ wall time or score is unmeasured.",
        hashes=HASHES)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
