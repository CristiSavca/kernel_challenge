#!/usr/bin/env python3
"""Offline application of the fixed v21 gate. Never polls or promotes."""
import json
from pathlib import Path

from analyze_wgs_shared_a_result import PARENTS, PARENT_SHA, ROOT, compare, read_run

JOB = "1a2a2a31f96f43e3"
SOURCE = "4fcf108fc860865b0284031075c9e35461b761eb11f5cd4681917e44165871da"


def analyze():
    candidate = read_run(JOB, SOURCE, "check")
    report = {
        "candidate": candidate["provenance"],
        "fixed_plan": "tools/WGS_STEADY_LG_V21_PLAN.md",
        "automatic_promotion_allowed": False,
        "status": candidate["status"],
        "limitations": [
            "Cross-run, fresh-seed comparisons are not paired candidate/incumbent timing.",
            "An official pass or gain does not prove a smaller hot instruction footprint or its cause.",
            "No full-service speedup or untested decoder-port claim follows from this standalone result.",
        ],
    }
    if candidate["status"] != "PASS":
        report.update({k: v for k, v in candidate.items() if k not in ("result", "provenance")})
        return report
    result = candidate["result"]
    report.update(score=result["speedup"], score_ci=result["speedup_ci"],
                  evaluator=result["evaluator"], flags=result["flags"], checks=result["checks"])
    comparisons = []
    for job, mode in PARENTS:
        parent = read_run(job, PARENT_SHA, mode)
        assert parent["status"] == "PASS"
        comparison = compare(result, parent["result"])
        # Shared parser's old SS-A thresholds do not apply to this experiment.
        comparison.pop("gates")
        comparison.pop("performance_gate_passed")
        comparison["parent"] = parent["provenance"]
        comparison["gates"] = {
            "MAIN_GM_speed_min_1p01": comparison["affected_speedup_ratio_GM"] >= 1.01,
            "every_MAIN_latency_max_1p02": all(
                row["normalized_latency_ratio"] <= 1.02
                for row in comparison["cases"] if row["affected"]),
            "control_GM_latency_0p98_to_1p02":
                .98 <= comparison["control_normalized_latency_ratio_GM"] <= 1.02,
        }
        comparisons.append(comparison)
    jointly_positive = [
        row["M"] for row in comparisons[0]["cases"] if row["affected"]
        and all(next(r for r in comp["cases"] if r["M"] == row["M"])["speedup_ratio"] > 1
                for comp in comparisons)
    ]
    passed = len(jointly_positive) >= 2 and all(
        all(comp["gates"].values()) for comp in comparisons)
    report.update(comparisons=comparisons, jointly_positive_MAIN=jointly_positive,
                  performance_gate_passed=passed,
                  status="SCREEN_PASS_PAIRED_CONFIRMATION_PENDING" if passed else "REJECT_FIXED_V21",
                  next_action="Confirm paired gain before considering a decoder port."
                  if passed else "Close this attempt; no identical retry, unroll sweep, promotion or decoder port.")
    return report


if __name__ == "__main__":
    report = analyze()
    output = ROOT / "experiments/iterations/20261104-wgs-steady-lg-root-result.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
