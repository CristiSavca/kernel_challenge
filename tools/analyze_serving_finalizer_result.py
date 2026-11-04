#!/usr/bin/env python3
"""Apply the frozen v22 component gate to retained official evidence, offline."""
import hashlib
import json

from analyze_wgs_shared_a_result import ROOT, benchmark_rows, gm, read_run, require

JOB = "cffa322caeff4aa2"
SOURCE = "fc2fc6b51822e0e91eb6905790476ced13aa4c8599954dac2bb20edbf45fd39f"
PLAN = "experiments/iterations/20261104-serving-finalizer-root-plan.json"


def analyze():
    plan_bytes = (ROOT / PLAN).read_bytes()
    plan = json.loads(plan_bytes)
    require(plan["candidate_sha256"] == SOURCE, "plan source mismatch")
    candidate = read_run(JOB, SOURCE, "check")
    require(candidate["status"] == "PASS", "official evidence failed")
    result = candidate["result"]
    parent = read_run("d24dd50196cb4158",
                      "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0", "run")
    require(parent["status"] == "PASS", "evaluator anchor unavailable")
    require(result["evaluator"] == parent["result"]["evaluator"], "evaluator changed")
    require(result["gpu"] == "H200", "unexpected GPU")
    require(result["evaluator"]["files"]["problems/muse_moe_w4a16/baseline.py"]
            == plan["baseline_sha256"], "stock baseline mismatch")
    require(hashlib.sha256((ROOT / "muse_moe_w4a16/baseline.py").read_bytes()).hexdigest()
            == plan["baseline_sha256"], "local baseline changed")
    gate = plan["fixed_gate"]
    profiles = {p["index"]: p["report"] for p in result["records"] if p["event"] == "profile"}
    rows = []
    for index, row in sorted(benchmark_rows(result).items()):
        m = row["case"]["bs"]
        affected = m in gate["affected_M"]
        require(affected or m in gate["control_M"], "unexpected partition")
        names = ("_small_post_expert_inverse_kernel", "_small_post_expert_combine_kernel")
        selected = (all(name in profiles[index] for name in names) if affected else
                    "_post_expert_norm_weighted_sum_kernel" in profiles[index]
                    and all(name not in profiles[index] for name in names))
        rows.append({"M": m, "load": row["case"]["load"], "affected": affected,
                     "seed": row["case"]["seed"], "speedup": row["speedup"],
                     "normalized_latency": 1 / row["speedup"],
                     "baseline_us": row["baseline_ns"] / 1000,
                     "candidate_us": row["median_ns"] / 1000,
                     "baseline_raw_us": row["baseline_raw_ns"] / 1000,
                     "candidate_raw_us": row["median_raw_ns"] / 1000,
                     "expected_profile_path_present": selected})
    affected = [r for r in rows if r["affected"]]
    controls = [r for r in rows if not r["affected"]]
    affected_speed = gm(r["speedup"] for r in affected)
    control_latency = gm(r["normalized_latency"] for r in controls)
    lo, hi = gate["control_GM_normalized_latency_range"]
    gates = {
        "affected_GM": affected_speed >= gate["affected_GM_same_run_speed_min"],
        "every_affected": all(r["speedup"] >= gate["every_affected_speed_min"] for r in affected),
        "two_affected_at_least_1p05": sum(r["speedup"] >= 1.05 for r in affected)
            >= gate["affected_cases_speed_at_least_1p05_min"],
        "control_GM": lo <= control_latency <= hi,
        "every_control": all(r["normalized_latency"] <= gate["every_control_normalized_latency_max"]
                             for r in controls),
        "profiles": all(r["expected_profile_path_present"] for r in rows),
    }
    return {
        "candidate": candidate["provenance"], "plan": PLAN,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "status": "KEEP_SERVING_COMPONENT_VALIDATION_PENDING" if all(gates.values()) else "CLOSE_FIXED_ATTEMPT",
        "gates": gates, "all_fixed_gates_pass": all(gates.values()),
        "score": result["speedup"], "score_ci": result["speedup_ci"],
        "affected_GM_speedup": affected_speed,
        "affected_GM_latency_reduction_pct": 100 * (1 - 1 / affected_speed),
        "control_GM_normalized_latency": control_latency,
        "cases": rows, "evaluator": result["evaluator"],
        "flags": result["flags"], "checks": result["checks"],
        "correctness": [r for r in result["records"] if r["event"] == "test"],
        "resources": {"official_private_checks": 1, "Modal_allocations": 0, "out_of_pocket_usd": 0},
        "automatic_scored_promotion": False,
        "limits": [
            "Same-run stock-Marlin component comparison, not a gain over native247888 or whole serving.",
            "Profiles confirm kernel activation; profile durations are not benchmark measurements.",
            "Official GPU coverage is H4096, K8, BF16 input, stock Marlin routes; generic guard breadth needs GPU tests.",
            "Separate multi-stream/graph ownership and full eight-GPU serving validation remain required.",
        ],
        "next_experiment": "Focused generic finalizer GPU correctness and two-stream/CUDA-graph validation before serving integration.",
    }


if __name__ == "__main__":
    report = analyze()
    output = ROOT / "experiments/iterations/20261104-serving-finalizer-root-result.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "gates", "score", "affected_GM_speedup", "control_GM_normalized_latency")}, indent=2))
