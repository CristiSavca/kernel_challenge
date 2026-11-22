"""Compute a decode sensitivity from three physical archives and a separate client trace."""
import hashlib
import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVING = "serving_v0523/"
EXPERT = "experiments/modal_serving/20261112T191039Z/"
SOURCES = {
    "serving": SERVING + "runs/abd217baf0ed49fd.json",
    "dense": "experiments/runs/b5538a0dd25d46be.json",
    "expert": EXPERT + "result.json",
    "expert_control": EXPERT + "manifest.json",
    "cold_model": SERVING + "candidates/v129_routed_projection_pair/work_model.py",
    "client": "experiments/modal_node/20261113T204343Z-bench/results__20261113T204343Z-bench__v41__load.json",
    "replay": SERVING + "candidates/v195_fused_expert_affine_verify/score_headroom.py",
    "loadgen": "tools/okc_node_loadgen.py",
}


def read(key):
    return json.loads((ROOT / SOURCES[key]).read_bytes())


def main():
    service = read("serving")["result"]
    dense = read("dense")["response"]["result"]
    expert = read("expert")
    assert service["passed"] and service["gpus"] == 8 and dense["gpu"] == "H200" and dense["passed"]
    assert expert["status"] == "complete" and expert["gpu"] == "NVIDIA H200"
    assert any(flag["setting"] == "moe_runner_backend" and flag["build"] == "humming"
               for flag in service["launch_settings"]["launch_flags"])
    cold_model = runpy.run_path(ROOT / SOURCES["cold_model"])["model"]
    replay = runpy.run_path(ROOT / SOURCES["replay"])
    client = read("client")
    rounds = replay["measured_intervals"](client)
    ordinary = replay["rerate"](rounds, (1, 1, 1))
    instant_stream = replay["rerate"](rounds, (1, 1, math.inf))
    score = service["frontier"]["score"]
    results = []
    for users in (1, 32):
        rows = users * 8
        usage = next(r for r in service["rungs"] if r["concurrency"] == users)
        period_ms = 1000 * usage["acceptance"] / usage["tok_s_per_user_p50"]
        target_ms = period_ms * score / 4
        required_ms = period_ms - target_ms
        samples = [r for r in dense["records"] if r["event"] == "benchmark"
                   and r["case"]["load"] == f"verify {users}"]
        native_raw = {r["case"]["layer"]: r["baseline_raw_ns"] / 1000 for r in samples}
        assert set(native_raw) == {"qkv_gate", "o", "pooler", "unpooler", "router"}
        assert all(r["case"]["m"] == rows and r["case"]["weight"] == 62
                   and r["baseline_stats"]["runs"] == 60 for r in samples)
        raw_ms = sum(native_raw.values()) * 62 / 1000
        corrected_ms = sum(r["baseline_ns"] for r in samples) * 62 / 1e6
        m = next(r for r in expert["cases"] if r["T"] == rows)
        assert m["stock_graph_vs_eager_bitwise"] and m["v40_graph_vs_eager_bitwise"]
        stock_ms = m["stock_us"] * 62 / 1000
        v40_ms = m["v40_us"] * 62 / 1000
        total_ms = stock_ms + raw_ms
        floor_us = cold_model(rows)["ideal_cold_weight_floor_us"]
        true_rung = next(r for r in client["rungs"] if r["concurrency"] == users)
        assert abs(ordinary[users][0] - true_rung["tok_s_per_gpu"]) < .06
        assert all(abs(x-y) < .006 for x,y in zip(ordinary[users][2], true_rung["wall_s"]))
        results.append(dict(
            users=users, verify_rows=rows, serving_commits=usage["acceptance"],
            serving_median_user_decode_tok_s=usage["tok_s_per_user_p50"],
            estimated_period_ms=period_ms, target_period_ms=target_ms, required_saving_ms=required_ms,
            native_dense_raw_us=native_raw, dense_62_layer_raw_ms=raw_ms,
            dense_62_layer_overhead_adjusted_ms=corrected_ms,
            expert_live_routes=m["local_routes"], stock_marlin_layer_us=m["stock_us"], v40_marlin_layer_us=m["v40_us"],
            stock_marlin_62_layer_ms=stock_ms, v40_marlin_62_layer_ms=v40_ms,
            two_family_sum_cross_benchmark_ms=total_ms,
            allowed_combined_ms_after_required_saving=total_ms-required_ms,
            expected_cold_expert_weights_peak_us_per_layer=floor_us,
            expected_cold_expert_weights_peak_ms_62_layers=floor_us*62/1000,
            v41_client_wall_s=ordinary[users][2], v41_instant_stream_client_wall_s=instant_stream[users][2],
            v41_instant_stream_throughput_multiplier=instant_stream[users][0]/ordinary[users][0]))
    return dict(v64_official_score=score, uniform_fixed_interactivity_factor_to_four=4/score,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES.values()},
                expert_worker_sha256=read("expert_control")["worker_sha256"],
                dense_repeats_per_case=60, target_layers=62, new_physical_run=False,
                boundaries=[
                    "V64 period divides a mean commit count by a median user decode rate. It is not a step-timer trace.",
                    "The stock dense raw median includes event harness overhead. Dense calls and one EP rank's Marlin call are added as if serial.",
                    "The serving target uses Humming. Standalone timings have different dispatch, routes and overlap.",
                    "Cold weights assume uniform independent distinct top-eight routing and one 4.8 TB/s HBM read per distinct matrix.",
                    "V41 client replay removes only time after first token, keeps measured request gaps, and recomputes the last-finishing user per repetition.",
                    "The challenge scores four frontier points. The uniform fixed-interactivity factor is one sensitivity, not a necessary per-rung result."],
                cases=results)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
