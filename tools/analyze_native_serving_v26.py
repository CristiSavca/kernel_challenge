#!/usr/bin/env python3
"""Offline V26 component screen with source-bound abbreviated profiles. No GPU calls or automatic promotion."""
import argparse
import ast
import hashlib
import json

from analyze_wgs_shared_a_result import ROOT, PARENT_SHA, benchmark_rows, gm, read_run, require

PLAN = "experiments/iterations/20261104-native-serving-v26-root-plan.json"


V25_SHA = "3a8c973b737dd074bcf5115f7418f5f0b34139fd4f9f637410ad1765ccbe050d"
V25_JOB = "033a2d2eca374d82"
FINALIZER_MODULE = "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton"
FINALIZER = "fused_post_expert_norm_weighted_sum"
CHAIN_FUNCTIONS = ("kernel_fn", "prepare", "expert_outputs")


def source_chain(candidate_bytes, parent_bytes, baseline_bytes):
    """Reviewed source correspondence, not a proof of arbitrary Python behavior."""
    require(hashlib.sha256(parent_bytes).hexdigest() == V25_SHA, "V25 source anchor changed")
    candidate, parent, baseline = [ast.parse(x) for x in (candidate_bytes, parent_bytes, baseline_bytes)]
    for tree in (candidate, parent, baseline):
        imports = [(n.module, a.name, a.asname) for n in tree.body if isinstance(n, ast.ImportFrom)
                   for a in n.names if a.name == FINALIZER or a.asname == FINALIZER]
        require(imports == [(FINALIZER_MODULE, FINALIZER, None)], "stock finalizer import changed")
    require(not any(isinstance(n, ast.Name) and n.id == FINALIZER and isinstance(n.ctx, (ast.Store, ast.Del))
                    for n in ast.walk(candidate)), "stock finalizer binding overwritten")
    bindings = {}
    for name in CHAIN_FUNCTIONS:
        bodies = []
        for tree in (candidate, parent):
            nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
            require(len(nodes) == 1, "missing or duplicate chain function: " + name)
            bodies.append(ast.dump(nodes[0], include_attributes=False))
        require(bodies[0] == bodies[1], "stock chain function changed: " + name)
        bindings[name] = hashlib.sha256(bodies[0].encode()).hexdigest()
    return {"common_import": FINALIZER_MODULE + "." + FINALIZER, "function_AST_sha256": bindings,
            "V25_source_sha256": V25_SHA,
            "scope": "Exact reviewed call chain plus same qualified import as official baseline; full source review and evaluator/image binding remain required."}


def profile_evidence(report, m, core_name, source_chain_verified):
    require(bool(core_name), "empty core profile marker")
    core = core_name in report
    if m < 32:
        families = ("at::native::elementwise_kernel", "at::native::vectorized_elementwise_kernel",
                    "at::native::unrolled_elementwise_kernel")
        observed = "at::native::reduce_kernel" in report and any(x in report for x in families)
        method = "Exact common stock-helper source chain plus visible ATen reduction/elementwise families; full template names are abbreviated."
    else:
        observed = "_post_expert_norm_weighted_sum_kernel" in report
        method = "Exact common stock-helper source chain plus visible original fused finalizer name."
    return {"intended_core_in_profile": core, "stock_finalizer_observable_evidence": observed,
            "stock_finalizer_chain_established": source_chain_verified is True and observed,
            "activation_evidence_method": method}


def screen(result, core_name, source_chain_verified):
    require(core_name == "native_serving_v26", "unexpected core profile marker")
    require(result.get("gpu") == "H200", "unexpected GPU")
    profiles = {p["index"]: p["report"] for p in result["records"] if p["event"] == "profile"}
    rows = []
    for index, row in sorted(benchmark_rows(result).items()):
        require(index in profiles, "missing activation profile")
        rows.append({"index": index, "M": row["case"]["bs"], "load": row["case"]["load"],
                     "seed": row["case"].get("seed"), "speedup": row["speedup"],
                     "normalized_latency": 1 / row["speedup"],
                     "baseline_us": row["baseline_ns"] / 1000,
                     "candidate_us": row["median_ns"] / 1000,
                     "baseline_raw_us": row["baseline_raw_ns"] / 1000,
                     "candidate_raw_us": row["median_raw_ns"] / 1000,
                     **profile_evidence(profiles[index], row["case"]["bs"], core_name, source_chain_verified)})
    score = gm(r["speedup"] for r in rows)
    gates = {"all_six_GM_speed_at_least_1p15": score >= 1.15,
             "every_case_normalized_latency_at_most_1p02":
                 all(r["normalized_latency"] <= 1.02 for r in rows),
             "all_six_intended_core_profiles": all(r["intended_core_in_profile"] for r in rows),
             "all_six_stock_finalizer_chains": all(r["stock_finalizer_chain_established"] for r in rows)}
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
    closed_v25 = read_run(V25_JOB, V25_SHA, "check")
    require(closed_v25["status"] == "PASS", "V25 anchor unavailable")
    require(result["evaluator"] == closed_v25["result"]["evaluator"], "V25 evaluator mismatch")
    chain = source_chain((ROOT / candidate["provenance"]["source"]).read_bytes(),
                         (ROOT / closed_v25["provenance"]["source"]).read_bytes(),
                         (ROOT / "muse_moe_w4a16/baseline.py").read_bytes())
    require(chain == plan["stock_chain_source_binding"], "source chain differs from frozen plan")
    measured = screen(result, plan["core_profile_name"], True)
    v25_rows = benchmark_rows(closed_v25["result"])
    parent_rows = benchmark_rows(parent["result"])
    for row in measured["cases"]:
        row["speedup_relative_to_V25"] = row["speedup"] / v25_rows[row["index"]]["speedup"]
        row["speedup_relative_to_public_native"] = row["speedup"] / parent_rows[row["index"]]["speedup"]
    return {**report, **measured,
            "status": "KEEP_COMPONENT_FURTHER_VALIDATION_REQUIRED" if measured["fixed_component_screen_passed"] else "CLOSE_FIXED_ATTEMPT",
            "stock_chain_source_binding": chain,
            "closed_V25": closed_v25["provenance"],
            "relative_GM_to_V25": measured["score"] / closed_v25["result"]["speedup"],
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
