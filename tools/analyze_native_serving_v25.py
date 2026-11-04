#!/usr/bin/env python3
"""Offline V25 component screen. No GPU calls or automatic promotion."""
import argparse
import hashlib
import json

from analyze_wgs_shared_a_result import ROOT, PARENT_SHA, benchmark_rows, gm, read_run, require

PLAN = "experiments/iterations/20261104-native-serving-v25-root-plan.json"


def screen(result, core_name):
    require(core_name == "native_serving_v25", "unexpected core profile marker")
    require(result.get("gpu") == "H200", "unexpected GPU")
    profiles = {p["index"]: p["report"] for p in result["records"] if p["event"] == "profile"}
    rows = []
    for index, row in sorted(benchmark_rows(result).items()):
        require(index in profiles, "missing activation profile")
        # These are semantic family markers, not a fixed launch count. The
        # small-path markers were observed in the retained original-helper
        # H200 profile and correspond to its mask, mean and slot-sum source.
        finalizer_markers = (("where_kernel_impl", "MeanOps", "sum_functor")
                             if row["case"]["bs"] < 32 else
                             ("_post_expert_norm_weighted_sum_kernel",))
        rows.append({"index": index, "M": row["case"]["bs"], "load": row["case"]["load"],
                     "seed": row["case"].get("seed"), "speedup": row["speedup"],
                     "normalized_latency": 1 / row["speedup"],
                     "baseline_us": row["baseline_ns"] / 1000,
                     "candidate_us": row["median_ns"] / 1000,
                     "baseline_raw_us": row["baseline_raw_ns"] / 1000,
                     "candidate_raw_us": row["median_raw_ns"] / 1000,
                     "intended_core_in_profile": core_name in profiles[index],
                     "stock_finalizer_markers": list(finalizer_markers),
                     "stock_finalizer_in_profile": all(n in profiles[index] for n in finalizer_markers)})
    score = gm(r["speedup"] for r in rows)
    gates = {"all_six_GM_speed_at_least_1p15": score >= 1.15,
             "every_case_normalized_latency_at_most_1p02":
                 all(r["normalized_latency"] <= 1.02 for r in rows),
             "all_six_intended_core_profiles": all(r["intended_core_in_profile"] for r in rows),
             "all_six_stock_finalizer_profiles": all(r["stock_finalizer_in_profile"] for r in rows)}
    return {"score": score, "score_ci": result["speedup_ci"], "gates": gates, "cases": rows,
            "fixed_component_screen_passed": all(gates.values())}


def analyze(job, source_sha):
    plan_bytes = (ROOT / PLAN).read_bytes()
    plan = json.loads(plan_bytes)
    require(plan["candidate_sha256"] == source_sha, "plan source mismatch")
    require(plan["fixed_gate"] == {"GM_same_run_speed_min": 1.15,
                                  "every_case_normalized_latency_max": 1.02}, "fixed gate changed")
    require(plan["source_review_status"] == "GO_FOR_OFFICIAL_COMPONENT_CHECK",
            "source review not complete")
    candidate = read_run(job, source_sha, "check")
    report = {"schema": 1, "candidate": candidate["provenance"], "plan": PLAN,
              "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
              "automatic_scored_promotion": False,
              "scope": "Same-run official stock MoE component; not matched actual serving or native incumbent.",
              "unresolved_serving_gates": ["generic rank/empty-route numerical coverage", "two-stream and fresh-graph ownership",
                  "post-load weight adapter", "single-bank full-model memory", "large prefill and TTFT", "official eight-H200 serving"]}
    if candidate["status"] != "PASS":
        return {**report, **{k: v for k, v in candidate.items() if k != "provenance"},
                "fixed_component_screen_passed": False}
    result = candidate["result"]
    parent = read_run("d24dd50196cb4158", PARENT_SHA, "run")
    require(parent["status"] == "PASS", "evaluator anchor missing")
    require(result["evaluator"] == parent["result"]["evaluator"], "evaluator changed")
    require(result["evaluator"]["files"]["problems/muse_moe_w4a16/baseline.py"] == plan["baseline_sha256"],
            "official stock baseline mismatch")
    require(hashlib.sha256((ROOT / "muse_moe_w4a16/baseline.py").read_bytes()).hexdigest()
            == plan["baseline_sha256"], "local baseline changed")
    measured = screen(result, plan["core_profile_name"])
    parent_rows = benchmark_rows(parent["result"])
    for row in measured["cases"]:
        row["speedup_relative_to_public_native"] = row["speedup"] / parent_rows[row["index"]]["speedup"]
    return {**report, **measured,
            "status": "KEEP_COMPONENT_FURTHER_VALIDATION_REQUIRED" if measured["fixed_component_screen_passed"] else "CLOSE_FIXED_ATTEMPT",
            "native_incumbent": parent["provenance"],
            "relative_GM_to_public_native": measured["score"] / parent["result"]["speedup"],
            "native_comparison_limit": "Different runs/seeds; normalized baseline ratios, not direct same-input A/B.",
            "correctness": [r for r in result["records"] if r["event"] == "test"],
            "evaluator": result["evaluator"], "checks": result["checks"], "flags": result["flags"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("source_sha256")
    args = parser.parse_args()
    print(json.dumps(analyze(args.job, args.source_sha256), indent=2))
