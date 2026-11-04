#!/usr/bin/env python3
"""Offline independent REG112 result audit. No polling or GPU operations."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from analyze_kperm_and_decoder_roi import ROOT, OUT, load, sha
from analyze_dense_escape_result import kernel_rows


def gm(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def main():
    receipt, profile, provenance = load("42527fa2c64d4b42")
    assert provenance["source_sha256"] == "31127a37ce35616b555e2ea5a5df673719aa03c422f9edc1a2be75320b3b9f65"
    result = receipt["response"]["result"]
    counts = Counter(r["event"] for r in result["records"])
    assert counts == {"setup": 1, "test": 5, "benchmark": 6, "profile": 6}
    assert result["returncode"] == 0 and result["passed"] and result["error"] is None
    assert result["flags"] == [] and result["checks"] == {"flagged": [], "ok": True, "rejected": []}
    assert all(r.get("status", "pass") == "pass" for r in result["records"])
    log_path = Path(receipt["response"]["output"]["log"])
    log = log_path.read_text()
    assert result["log_tail"].startswith(log)
    extra = result["log_tail"][len(log):].splitlines()
    assert len(extra) == 12 and all(x.startswith("USDT:") for x in extra)
    marker = re.search(r"OKC_WGS_WAIT2_REG112 resources intended_initial=96 producer=32 consumer=112 threads=(\d+) dynamic=(\d+) occupancy=(\d+)/(\d+) regs=(\d+)/(\d+) local=(\d+)/(\d+) static=(\d+)/(\d+)", log)
    assert marker and tuple(map(int, marker.groups())) == (640, 231168, 1, 1, 96, 96, 0, 0, 0, 0)
    compiler = []
    lines = log.splitlines()
    for big in (0, 1):
        symbol = f"_ZN3okc9moe_fusedILb{big}EEEvNS_6ParamsE"
        warning = [{"line": n + 1, "text": line} for n, line in enumerate(lines)
                   if "C7511" in line and symbol in line]
        assert len(warning) == 1 and "insufficient register resources" in warning[0]["text"]
        p = lines.index("ptxas info    : Function properties for " + symbol)
        assert lines[p + 1].strip() == "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"
        assert lines[p + 2].strip() == "ptxas info    : Used 96 registers, used 16 barriers"
        compiler.append({"BIG": bool(big), "serialization": warning[0], "registers": 96,
                         "stack_bytes": 0, "spill_store_bytes": 0, "spill_load_bytes": 0, "barriers": 16})
    comparisons = []
    for job, kind, expected in (
        ("d24dd50196cb4158", "scored_incumbent", "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"),
        ("bcb761dfca264db0", "private_incumbent", "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"),
        ("f685c2533ebc4214", "rejected_wait2_base", "97ffccc30a90ce8400cddef800d932889822eaf876fbdedaec18b1013e8f66b6"),
    ):
        _, old, parent = load(job)
        assert parent["source_sha256"] == expected
        assert parent["evaluator_sha256"] == provenance["evaluator_sha256"]
        assert len(profile["benchmarks"]) == len(old["benchmarks"]) == len(profile["profiles"]) == len(old["profiles"]) == 6
        cases = []
        for n, o, np, op in zip(profile["benchmarks"], old["benchmarks"], profile["profiles"], old["profiles"]):
            assert n["index"] == o["index"] == np["index"] == op["index"]
            assert {k: v for k, v in n["case"].items() if k != "seed"} == {k: v for k, v in o["case"].items() if k != "seed"}
            nr, pr = kernel_rows(np), kernel_rows(op)
            assert len(nr) == len(pr) == 1 and "moe_fused" in nr[0]["name"] and "moe_fused" in pr[0]["name"]
            cases.append({
                "index": n["index"], "tokens": n["case"]["bs"], "affected": n["case"]["bs"] > 16,
                "candidate_seed": n["case"]["seed"], "parent_seed": o["case"]["seed"],
                "speedup_ratio": n["speedup"] / o["speedup"],
                "inverse_speedup_ratio": o["speedup"] / n["speedup"],
                "harness_median_ratio": n["median_ns"] / o["median_ns"],
                "raw_median_ratio": n["median_raw_ns"] / o["median_raw_ns"],
                "baseline_median_ratio": n["baseline_ns"] / o["baseline_ns"],
                "candidate_harness_us": n["median_ns"] / 1000, "parent_harness_us": o["median_ns"] / 1000,
                "candidate_raw_us": n["median_raw_ns"] / 1000, "parent_raw_us": o["median_raw_ns"] / 1000,
                "candidate_profile": nr, "parent_profile": pr,
            })
        groups = {}
        for name, affected in (("affected", True), ("controls", False)):
            rows = [x for x in cases if x["affected"] is affected]
            assert len(rows) == 3
            groups[name] = {key + "_geomean": gm([x[key] for x in rows]) for key in
                            ("speedup_ratio", "inverse_speedup_ratio", "harness_median_ratio", "raw_median_ratio", "baseline_median_ratio")}
        aggregate = provenance["speedup"] / parent["speedup"]
        assert math.isclose(aggregate, gm([x["speedup_ratio"] for x in cases]), rel_tol=1e-12)
        comparisons.append({"comparison_kind": kind, "parent": parent, "cases": cases, "groups": groups,
                            "aggregate_speedup_ratio": aggregate,
                            "affected_geomean_passes_0p95_gate": groups["affected"]["inverse_speedup_ratio_geomean"] <= 0.95,
                            "every_affected_case_passes_1p03_gate": all(x["inverse_speedup_ratio"] <= 1.03 for x in cases if x["affected"]),
                            "reported_aggregate_intervals_overlap": max(provenance["speedup_ci"][0], parent["speedup_ci"][0]) <= min(provenance["speedup_ci"][1], parent["speedup_ci"][1]),
                            "same_gpu_node": provenance["node"] == parent["node"]})
    report = {
        "candidate": provenance, "record_counts": dict(counts), "flags": result["flags"], "checks": result["checks"],
        "resources": {"log": str(log_path.relative_to(ROOT)), "log_sha256": sha(log_path), "log_bytes": log_path.stat().st_size,
                      "marker": marker.group(), "compiler_entries": compiler,
                      "saved_log_exact_prefix_of_receipt_tail": True,
                      "receipt_suffix": "12 USDT profiler start/stop lines only",
                      "truncation_limit": "Visible output covers build steps1/3 through3/3, both entry properties and resource marker. Full original process stdout is not independently certified. Positive C7511 in both kernels is established regardless of any unretained diagnostics.",
                      "quota_limit": "Host initial96 and occupancy1 do not prove retained dynamic consumer112/producer32 instructions or achieved occupancy. No PTX/SASS artifact exposes actual USETMAXREG in this receipt."},
        "comparisons": comparisons,
        "verdict": {
            "decision": "REJECT_REG112_NO_SCORE_NO_DECODER_INTEGRATION_NO_QUOTA_RESCUE_NO_IDENTICAL_RETRY",
            "primary_reason": "Both BIG variants retain explicit C7511 insufficient-register WGMMA serialization. The fixed plan's compiler kill gate fails even though all correctness and resource admission gates pass.",
            "performance_gate": "Compare the three affectedM32/128/256 cases to scored247888. Required GM(incumbent_speedup/candidate_speedup)<=0.95 and no case>1.03; do not use failedWAIT2 as a substitute incumbent.",
            "causal_limit": "Different GPU nodes, fresh input seeds and baseline drift limit causal timing claims. The aggregate interval overlaps the scored incumbent and WAIT2, but lies below the private incumbent interval. Unchanged small cases are controls. Positive serialization proves the intended mechanism failed, not its exact contribution to elapsed time.",
            "next_action": "Close this32/112 allocation and preserve incumbent247888/decoder97af. Current small-WGMMA feasibility and dense-codec review remain distinct research, not automatic implementation approval."
        },
        "gpu_used_by_critic": False, "official_polled": False, "candidate_edited": False,
    }
    output = OUT / "20261104-wgs-reg112-official-critic.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "source": provenance,
                      "comparisons": [{"kind": c["comparison_kind"], "aggregate": c["aggregate_speedup_ratio"], "groups": c["groups"],
                                       "cases": [{"tokens": r["tokens"], "relative_latency_ratio": r["inverse_speedup_ratio"], "harness_ratio": r["harness_median_ratio"], "raw_ratio": r["raw_median_ratio"]} for r in c["cases"]]}
                                      for c in comparisons]}, indent=2))


if __name__ == "__main__":
    main()
