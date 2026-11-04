#!/usr/bin/env python3
"""Reproduce the offline KPERM audit and explicitly hypothetical ROI scenarios."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/iterations"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(job):
    receipt_path = ROOT / f"experiments/runs/{job}.json"
    receipt = json.loads(receipt_path.read_text())
    profile_path = Path(receipt["response"]["profile"]["json"])
    source_path = ROOT / receipt["source"]
    assert sha(source_path) == receipt["source_sha256"]
    result = receipt["response"]["result"]
    assert receipt["response"]["status"] == "done" and result["passed"]
    return receipt, json.loads(profile_path.read_text()), {
        "job": job,
        "receipt": str(receipt_path.relative_to(ROOT)),
        "receipt_sha256": sha(receipt_path),
        "profile": str(profile_path.relative_to(ROOT)),
        "profile_sha256": sha(profile_path),
        "source_sha256": sha(source_path),
        "evaluator_sha256": result["evaluator"]["code_sha256"],
        "node": result["machine"]["node"],
        "speedup": result["speedup"],
        "speedup_ci": result["speedup_ci"],
    }


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def audit_kperm():
    candidate, candidate_profile, provenance = load("bcb761dfca264db0")
    result = candidate["response"]["result"]
    counts = Counter(row["event"] for row in result["records"])
    assert counts == {"setup": 1, "test": 5, "benchmark": 6, "profile": 6}
    comparisons = []
    for job in ("143f8587286e4384", "3c15b408fdb04c44"):
        _, parent_profile, parent_provenance = load(job)
        rows = []
        for new, old in zip(candidate_profile["benchmarks"], parent_profile["benchmarks"]):
            assert new["case"]["bs"] == old["case"]["bs"]
            rows.append({
                "tokens": new["case"]["bs"],
                "changed_path": new["case"]["bs"] > 16,
                "candidate_seed": new["case"]["seed"],
                "parent_seed": old["case"]["seed"],
                "speedup_ratio": new["speedup"] / old["speedup"],
                "candidate_normalized_us": new["median_ns"] / 1000,
                "parent_normalized_us": old["median_ns"] / 1000,
                "candidate_raw_us": new["median_raw_ns"] / 1000,
                "parent_raw_us": old["median_raw_ns"] / 1000,
                "baseline_time_ratio": new["baseline_ns"] / old["baseline_ns"],
            })
        comparisons.append({
            "parent": parent_provenance,
            "cases": rows,
            "all_cases_speedup_ratio_geomean": geomean([x["speedup_ratio"] for x in rows]),
            "changed_cases_speedup_ratio_geomean": geomean([x["speedup_ratio"] for x in rows if x["changed_path"]]),
            "unchanged_controls_speedup_ratio_geomean": geomean([x["speedup_ratio"] for x in rows if not x["changed_path"]]),
        })
    report = {
        "candidate": provenance,
        "record_counts": dict(counts),
        "correctness": "5 explicit tests and 6 timed cases passed; setup also passed. No flags or rejections.",
        "comparisons": comparisons,
        "critic_verdict": {
            "score_now": "NO_GO: aggregate gains 1.24–1.43% have overlapping reported confidence intervals and heterogeneous unchanged controls.",
            "repeat_standalone": "NO_GO: preserve this result; no identical retry or parameter sweep.",
            "isolated_decoder_private_check": "CONDITIONAL_GO: one exact isolated integration after source/layout/dispatch/precision review. This tests a different workload and whole-decoder impact; it is not promotion or certification of a gain.",
            "interpretation": "All three changed sizes improve their SGLang-relative ratios versus both parent runs, plausibly indicating a modest WGS benefit. Fresh seeds/nodes/routing and baseline shifts prevent causal or precise cross-job gain claims. A ~3% WGS gain is not a credible standalone route to the ~15% whole-decoder gap.",
        },
    }
    (OUT / "20261104-moe-kperm3-official-critic.json").write_text(json.dumps(report, indent=2) + "\n")


def decoder_scenarios():
    _, profile, provenance = load("4b216098993e4112")
    cases = []
    for benchmark, capture in zip(profile["benchmarks"], profile["profiles"]):
        groups = dict.fromkeys(("moe", "attention", "dense", "boundary_route"), 0.0)
        kernels = []
        for line in capture["report"].splitlines():
            match = re.match(r"^(.*?)\s+(\d+\.\d+) µs\s+[\d.]+%\s+(\d+)\s*$", line)
            if not match:
                continue
            name, duration, calls = match.groups()
            group = "moe" if "moe_fused" in name else "attention" if "attn_kernel" in name else "boundary_route" if name.startswith(("_boundary", "_route")) else "dense"
            groups[group] += float(duration)
            kernels.append({"name": name, "duration_us": float(duration), "calls": int(calls), "group": group})
        assert kernels
        cases.append({"index": benchmark["index"], "case": benchmark["case"], "benchmark_us": benchmark["median_ns"] / 1000, "groups_us": groups, "profile_row_sum_us": sum(groups.values()), "kernel_count": sum(x["calls"] for x in kernels), "kernels": kernels})

    def fraction(case, group):
        return case["groups_us"][group] / case["profile_row_sum_us"]

    definitions = {
        "20pct_lower_attention_rows_only_indices9_11": lambda c: .2 * fraction(c, "attention") if c["index"] in (9, 11) else 0,
        "20pct_lower_MoE_rows_only_tokens_over16": lambda c: .2 * fraction(c, "moe") if c["case"]["batch"] * c["case"].get("q_len", 1) > 16 else 0,
        "25pct_lower_dense_rows_all_cases": lambda c: .25 * fraction(c, "dense"),
        "20pct_lower_MoE_rows_all_cases": lambda c: .2 * fraction(c, "moe"),
        "25pct_lower_dense_and20pct_lower_MoE_rows_all_cases": lambda c: .25 * fraction(c, "dense") + .2 * fraction(c, "moe"),
        "remove_all_boundary_and_route_rows_impossible_ideal": lambda c: fraction(c, "boundary_route"),
    }
    scenarios = {}
    for name, reduction in definitions.items():
        loads = {}
        for case in cases:
            load_entry = loads.setdefault(case["case"]["load"], {"incumbent_weighted_us": 0., "scenario_weighted_us": 0.})
            weighted = case["benchmark_us"] * case["case"]["weight"]
            load_entry["incumbent_weighted_us"] += weighted
            load_entry["scenario_weighted_us"] += weighted * (1 - reduction(case))
        assert len(loads) == 6
        ratio = geomean([x["scenario_weighted_us"] / x["incumbent_weighted_us"] for x in loads.values()])
        scenarios[name] = {"normalized_latency_reduction": 1 - ratio, "hypothetical_score": provenance["speedup"] / ratio, "loads": loads}
    report = {
        "provenance": provenance,
        "method": "For each case, reduce benchmark median_ns by the assumed fraction of the rounded profile-row sum. Combine sliding/full with case weights 46/16, then take the geometric mean of the six scenario/incumbent load-time ratios. Dense includes fused ub2 and cuBLAS split-K reduce rows.",
        "limitations": ["These are profile-share scenarios, NOT performance bounds or predictions.", "PDL row durations contain dependency waits and may overlap; profile sums are not wall time or exclusive compute time.", "Profiles and benchmark timings come from different captures and may differ materially.", "No evidence establishes the assumed reductions; precision, bandwidth, resource admission, dependency ordering, and implementation cost can prevent them.", "A 20% latency reduction is different from a 1.2x speedup; these scenarios use the former."],
        "cases": cases,
        "scenarios": scenarios,
        "strategic_decision": "Prioritize a broad MoE plus dense portfolio. A full-Q8-only win is secondary. Do not replace measured critical-path evidence with megakernel headline gains or fused-launch counts.",
    }
    (OUT / "20261104-decoder-research-roi-scenarios.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    audit_kperm()
    decoder_scenarios()
    print("Saved KPERM critic audit and explicitly hypothetical decoder ROI scenarios.")
