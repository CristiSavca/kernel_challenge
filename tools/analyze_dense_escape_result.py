#!/usr/bin/env python3
"""Offline exact-receipt audit; never submits, polls, or imports GPU code."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from analyze_kperm_and_decoder_roi import OUT, ROOT, geomean, load, sha


def kernel_rows(capture):
    rows = []
    for line in capture["report"].splitlines():
        match = re.match(r"^(.*?)\s+(\d+\.\d+) µs\s+([\d.]+)%\s+(\d+)\s*$", line)
        if match:
            name, duration, share, calls = match.groups()
            rows.append({"name": name, "duration_us": float(duration), "share_percent": float(share), "calls": int(calls)})
    assert rows
    return rows


def main():
    receipt, profile, provenance = load("d4c26ebb65e94a84")
    assert provenance["source_sha256"] == "b0387be327a480c84227985566bf086c70d894394f95ad9b0589144c902e260b"
    result = receipt["response"]["result"]
    counts = Counter(row["event"] for row in result["records"])
    assert counts == {"setup": 1, "test": 13, "benchmark": 12, "profile": 12}
    assert all(row.get("status", "pass") == "pass" for row in result["records"])
    assert not result["flags"] and result["checks"] == {"flagged": [], "ok": True, "rejected": []}
    new_kernels = {row["index"]: kernel_rows(row) for row in profile["profiles"]}
    comparisons = []
    for job in ("f8e991f1a2c2469e", "4b216098993e4112"):
        _, parent_profile, parent_provenance = load(job)
        assert provenance["evaluator_sha256"] == parent_provenance["evaluator_sha256"]
        old_kernels = {row["index"]: kernel_rows(row) for row in parent_profile["profiles"]}
        cases, loads = [], {}
        for new, old in zip(profile["benchmarks"], parent_profile["benchmarks"]):
            nc, oc = new["case"], old["case"]
            assert {k: v for k, v in nc.items() if k != "seed"} == {k: v for k, v in oc.items() if k != "seed"}
            ni, oi = new["index"], old["index"]
            assert ni == oi
            active = any("sk2_global_escape_kernel" in row["name"] for row in new_kernels[ni])
            assert active == (ni < 8)
            assert any("sk2_kernel" in row["name"] for row in old_kernels[oi]) == active
            cases.append({
                "index": ni, "case": nc, "parent_seed": oc["seed"], "has_sk2_profile_rows": active,
                "speedup_ratio": new["speedup"] / old["speedup"],
                "candidate_normalized_us": new["median_ns"] / 1000,
                "parent_normalized_us": old["median_ns"] / 1000,
                "candidate_raw_us": new["median_raw_ns"] / 1000,
                "parent_raw_us": old["median_raw_ns"] / 1000,
                "baseline_time_ratio": new["baseline_ns"] / old["baseline_ns"],
                "candidate_profile_kernels": new_kernels[ni], "parent_profile_kernels": old_kernels[oi],
            })
            entry = loads.setdefault(nc["load"], {
                "has_sk2_profile_rows": active, "candidate_weighted_baseline_ns": 0.,
                "parent_weighted_baseline_ns": 0., "candidate_weighted_time_ns": 0.,
                "parent_weighted_time_ns": 0.,
            })
            assert entry["has_sk2_profile_rows"] == active
            for label, bench in (("candidate", new), ("parent", old)):
                entry[f"{label}_weighted_baseline_ns"] += bench["baseline_ns"] * nc["weight"]
                entry[f"{label}_weighted_time_ns"] += bench["median_ns"] * nc["weight"]
        for entry in loads.values():
            entry["candidate_speedup"] = entry["candidate_weighted_baseline_ns"] / entry["candidate_weighted_time_ns"]
            entry["parent_speedup"] = entry["parent_weighted_baseline_ns"] / entry["parent_weighted_time_ns"]
            entry["speedup_ratio"] = entry["candidate_speedup"] / entry["parent_speedup"]
        assert math.isclose(geomean([v["candidate_speedup"] for v in loads.values()]), provenance["speedup"], rel_tol=1e-9)
        assert math.isclose(geomean([v["parent_speedup"] for v in loads.values()]), parent_provenance["speedup"], rel_tol=1e-9)
        comparisons.append({
            "parent": parent_provenance, "cases": cases, "loads": loads,
            "all_loads_speedup_ratio_geomean": geomean([v["speedup_ratio"] for v in loads.values()]),
            "sk2_loads_speedup_ratio_geomean": geomean([v["speedup_ratio"] for v in loads.values() if v["has_sk2_profile_rows"]]),
            "no_sk2_control_loads_speedup_ratio_geomean": geomean([v["speedup_ratio"] for v in loads.values() if not v["has_sk2_profile_rows"]]),
        })
    log = Path(receipt["response"]["output"]["log"])
    report = {
        "candidate": provenance, "record_counts": dict(counts), "flags": result["flags"],
        "checks": result["checks"], "comparisons": comparisons,
        "output_visibility": {"saved_log": str(log.relative_to(ROOT)), "saved_log_bytes": log.stat().st_size,
                              "saved_log_sha256": sha(log), "log_tail_chars": len(result["log_tail"]),
                              "compiler_diagnostics_complete": False},
        "aggregation": "Official sliding/full weights46/16, six-load geometric mean; reproduces candidate and both parents exactly. Fresh runs are not paired against each other.",
        "verdict": {
            "decision": "REJECT this exact candidate; no score, no integration, no identical retry.",
            "evidence": "All12 case speedup ratios and all6 load ratios fall below both exact80a4 parent runs. Aggregate relative speedup is0.969163 versus private and0.970934 versus public. The four affected SK2 loads lose approximately3-5% each; this does not support the ring-decoupling performance hypothesis.",
            "controls": "Profiles0..7 execute renamed SK2 kernels. Profiles8..11 (M128/M256) contain no SK2 in candidate or parents and are unchanged-path controls. Their raw times improve while SGLang-relative ratios regress because reference baselines also shift; do not treat that as a measured SK2 effect.",
            "causal_limits": "Dense autotuning selected different configurations/backends for some small workloads. Renamed uncompressed SK2 rows also exist although the new branch activates only COMP&&CL1. Truncated symbols do not always expose CL. PDL profile durations include waiting/overlap, and captures have fresh seeds/nodes; no exact attribution to sparse global loads, correction stalls or individual projections follows.",
            "compiler_limits": "Successful compile diagnostics are not retained completely. No visible serialization warning is not proof of no forced serialization, and the profile report contains no register/localBytes/achieved occupancy evidence.",
            "next_direction": "Preserve rejection and wait for the already submitted compact mechanism. Do not tune this failed rewrite through a blind parameter sweep; any dense follow-up needs a different falsifiable mechanism and evidence of a recoverable bottleneck."
        }
    }
    path = OUT / "20261104-dense-escape-global-official-critic.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(path.relative_to(ROOT)), "comparison_ratios": [{k: v for k, v in c.items() if k.endswith("geomean")} for c in comparisons]}, indent=2))


if __name__ == "__main__":
    main()
