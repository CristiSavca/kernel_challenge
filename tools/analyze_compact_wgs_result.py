#!/usr/bin/env python3
"""Read-only offline compact-WGS result audit; no polling or GPU code."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from analyze_dense_escape_result import kernel_rows
from analyze_kperm_and_decoder_roi import OUT, ROOT, geomean, load, sha


def main():
    receipt, profile, provenance = load("94e63981c631474e")
    assert provenance["source_sha256"] == "cf09ba584cc1537802636bb905ac08cda2095f83a163358e1c9bb716490130b8"
    result = receipt["response"]["result"]
    counts = Counter(row["event"] for row in result["records"])
    assert counts == {"setup": 1, "test": 5, "benchmark": 6, "profile": 6}
    assert all(row.get("status", "pass") == "pass" for row in result["records"])
    assert result["returncode"] == 0 and not result["error"]
    assert not result["flags"] and result["checks"] == {"flagged": [], "ok": True, "rejected": []}
    log_path = Path(receipt["response"]["output"]["log"])
    log = log_path.read_text()
    pattern = r"OKC_COMPACT_WGS resources threads=(\d+) dynamic=(\d+) occupancy=(\d+)/(\d+) regs=(\d+)/(\d+) local=(\d+)/(\d+)"
    match = re.search(pattern, log)
    assert match and match.group(0) in result["log_tail"]
    threads, dynamic, o0, o1, r0, r1, l0, l1 = map(int, match.groups())
    assert (threads, dynamic) == (320, 98048) and min(o0, o1) >= 2
    comparisons = []
    for job in ("143f8587286e4384", "3c15b408fdb04c44", "bcb761dfca264db0"):
        _, parent_profile, parent_provenance = load(job)
        assert provenance["evaluator_sha256"] == parent_provenance["evaluator_sha256"]
        cases = []
        for new, old, np, op in zip(profile["benchmarks"], parent_profile["benchmarks"], profile["profiles"], parent_profile["profiles"]):
            nc, oc = new["case"], old["case"]
            assert {k: v for k, v in nc.items() if k != "seed"} == {k: v for k, v in oc.items() if k != "seed"}
            cases.append({
                "index": new["index"], "case": nc, "parent_seed": oc["seed"], "changed_path": nc["bs"] > 16,
                "speedup_ratio": new["speedup"] / old["speedup"],
                "candidate_normalized_us": new["median_ns"] / 1000,
                "parent_normalized_us": old["median_ns"] / 1000,
                "candidate_raw_us": new["median_raw_ns"] / 1000,
                "parent_raw_us": old["median_raw_ns"] / 1000,
                "baseline_time_ratio": new["baseline_ns"] / old["baseline_ns"],
                "candidate_profile_kernels": kernel_rows(np), "parent_profile_kernels": kernel_rows(op),
            })
        assert math.isclose(geomean([b["speedup"] for b in profile["benchmarks"]]), provenance["speedup"], rel_tol=1e-9)
        assert math.isclose(geomean([b["speedup"] for b in parent_profile["benchmarks"]]), parent_provenance["speedup"], rel_tol=1e-9)
        affected = geomean([c["speedup_ratio"] for c in cases if c["changed_path"]])
        comparisons.append({
            "parent": parent_provenance,
            "relationship": "direct_parent" if job != "bcb761dfca264db0" else "separate_KPERM_option_not_direct_parent",
            "cases": cases,
            "all_cases_speedup_ratio_geomean": geomean([c["speedup_ratio"] for c in cases]),
            "affected_cases_speedup_ratio_geomean": affected,
            "affected_cases_normalized_latency_increase": 1 / affected - 1,
            "unchanged_controls_speedup_ratio_geomean": geomean([c["speedup_ratio"] for c in cases if not c["changed_path"]]),
        })
    report = {
        "candidate": provenance, "record_counts": dict(counts),
        "tests": [r for r in result["records"] if r["event"] == "test"],
        "checks": result["checks"], "flags": result["flags"],
        "resources": {
            "artifact": str(log_path.relative_to(ROOT)), "artifact_sha256": sha(log_path),
            "artifact_bytes": log_path.stat().st_size, "retained_line": match.group(0),
            "threads": threads, "dynamic_smem_bytes": dynamic,
            "BIGfalse": {"occupancy_max_active_blocks_per_SM": o0, "registers_per_thread": r0, "local_size_bytes_per_thread": l0},
            "BIGtrue": {"occupancy_max_active_blocks_per_SM": o1, "registers_per_thread": r1, "local_size_bytes_per_thread": l1},
            "interpretation": "Actual compiled kernel admission query reports two blocks/SM for both specializations. The exercised WGS marker resolves fallback ambiguity. Local64B/thread is adverse resource evidence, but cudaFuncAttributes.localSizeBytes alone does not distinguish spills from stack/local arrays, quantify traffic, or identify its contribution to latency.",
            "compiler_diagnostics_complete": False,
            "compiler_limit": "No forced-serialization warning is visible in retained output. Complete successful compiler output was not captured, so absence of C7512/C7510 is unverified. Do not infer absence of serialization or exact spill-load/store counts."
        },
        "comparisons": comparisons,
        "verdict": {
            "decision": "REJECT this exact architecture; no scored submission, decoder integration or identical retry.",
            "performance": "Official aggregate2.0472155356x is13.11% below the public parent. Affected M32/M128/M256 ratios are0.747144/0.731956/0.749699 against public, geometric0.742891: about34.61% more normalized latency. Unchanged controls improve1.63% geometrically. All three affected raw times and profile captures regress strongly, so heterogeneous cross-job noise does not plausibly explain away this broad failure.",
            "target": "Fails the prespecified target of at least10% lower affected aggregate latency with no more than3% loss on any affected case. Every affected case instead regresses materially.",
            "causality": "Two-block admission succeeded but was insufficient for performance. Hypothesized adverse contributors include smaller per-SM weight rings, duplicated activation requests, more stage/fixup events, resource pressure and final Y rereads. The retained data do not isolate their costs; do not claim64B local memory or any unobserved serialization alone caused the regression.",
            "correctness_scope": "Five explicit tests and six timed shapes passed, including nearby M33/M257 and skewed M128 in the explicit tests. This does not itself prove every valid M384 histogram or every possible graph transition; those CPU/static invariants remain separate evidence.",
            "next_action": "Preserve exact failure evidence and incumbent. Reject speculative FT128 follow-up absent actual decoder reuse evidence. Any new broad MoE proposal must address a supported bottleneck rather than assuming higher occupancy wins."
        }
    }
    path = OUT / "20261104-compact-wgs-official-critic.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(path.relative_to(ROOT)), "resources": report["resources"], "comparison_ratios": [{k: v for k, v in c.items() if k.endswith("geomean") or k.endswith("increase")} for c in comparisons]}, indent=2))


if __name__ == "__main__":
    main()
