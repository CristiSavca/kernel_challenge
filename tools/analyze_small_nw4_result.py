#!/usr/bin/env python3
"""Offline evidence for the single NW4 check; never polls or submits a job."""
import json
import math
import re
from collections import Counter
from pathlib import Path

from analyze_dense_escape_result import kernel_rows
from analyze_kperm_and_decoder_roi import ROOT, OUT, geomean, load, sha

JOB = "f46b93032b5548c1"
SOURCE = "34ee99c242c4ffb5028ec3511fd8740c5f15461e1c73e9c1343ae6611573e3e3"
PARENT = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"


def main():
    receipt = json.loads((ROOT / f"experiments/runs/{JOB}.json").read_text())
    response = receipt["response"]
    assert receipt["source_sha256"] == SOURCE
    assert sha(ROOT / receipt["source"]) == SOURCE
    if response.get("status") not in ("done", "failed", "rejected"):
        print(json.dumps({"job": JOB, "status": response.get("status"), "result_pending": True}))
        return
    result = response.get("result") or {}
    if not result.get("passed"):
        report = {"job": JOB, "source_sha256": SOURCE, "passed": False,
                  "response": response, "decision": "No promotion; inspect exact failure before any further work."}
    else:
        receipt, profile, provenance = load(JOB)
        counts = Counter(r["event"] for r in result["records"])
        assert counts == {"setup": 1, "test": 5, "benchmark": 6, "profile": 6}
        assert all(r.get("status", "pass") == "pass" for r in result["records"])
        assert result["returncode"] == 0 and not result["error"]
        assert not result["flags"] and result["checks"] == {"flagged": [], "ok": True, "rejected": []}
        assert math.isclose(geomean([r["speedup"] for r in profile["benchmarks"]]), provenance["speedup"], rel_tol=1e-9)
        log_path = Path(response["output"]["log"])
        log = log_path.read_text()
        pattern = r"OKC_SMALL_NW4 resources threads=(\d+) dynamic=(\d+) occupancy=(\d+)/(\d+) regs=(\d+)/(\d+) local=(\d+)/(\d+)"
        resources = []
        for match in re.finditer(pattern, log):
            threads, dynamic, o0, o1, r0, r1, l0, l1 = map(int, match.groups())
            assert match.group(0) in result["log_tail"]
            assert (threads, dynamic) == (160, 101632) and min(o0, o1) >= 2
            resources.append(dict(line=match.group(0), threads=threads, dynamic_smem_bytes=dynamic,
                                  occupancy=[o0, o1], registers=[r0, r1], local_bytes=[l0, l1]))
        assert resources, "No actual NW4 resource marker retained; mechanism not established"
        comparisons = []
        for parent_job in ("d24dd50196cb4158", "bcb761dfca264db0"):
            _, pp, parent = load(parent_job)
            assert parent["source_sha256"] == PARENT
            assert parent["evaluator_sha256"] == provenance["evaluator_sha256"]
            rows = []
            assert len(profile["benchmarks"]) == len(pp["benchmarks"]) == 6
            for new, old, np, op in zip(profile["benchmarks"], pp["benchmarks"], profile["profiles"], pp["profiles"]):
                assert {k:v for k,v in new["case"].items() if k != "seed"} == {k:v for k,v in old["case"].items() if k != "seed"}
                ratio = new["speedup"] / old["speedup"]
                rows.append(dict(tokens=new["case"]["bs"], changed_path=new["case"]["bs"] <= 16,
                    candidate_case=new["case"], parent_case=old["case"], speedup_ratio=ratio,
                    baseline_relative_latency_ratio=1 / ratio,
                    candidate_normalized_us=new["median_ns"] / 1000, parent_normalized_us=old["median_ns"] / 1000,
                    candidate_raw_us=new["median_raw_ns"] / 1000, parent_raw_us=old["median_raw_ns"] / 1000,
                    baseline_time_ratio=new["baseline_ns"] / old["baseline_ns"],
                    candidate_profile_kernels=kernel_rows(np), parent_profile_kernels=kernel_rows(op)))
            affected = [r for r in rows if r["changed_path"]]
            controls = [r for r in rows if not r["changed_path"]]
            latency = geomean([r["baseline_relative_latency_ratio"] for r in affected])
            comparisons.append(dict(parent=parent, cases=rows,
                small_case_relative_latency_reduction=1-latency,
                worst_small_case_relative_latency_ratio=max(r["baseline_relative_latency_ratio"] for r in affected),
                controls_speedup_ratio_geomean=geomean([r["speedup_ratio"] for r in controls]),
                meets_prespecified_small_case_target=latency <= .85 and all(r["baseline_relative_latency_ratio"] <= 1.03 for r in affected)))
        report = dict(candidate=provenance, passed=True, record_counts=dict(counts), resources=resources,
            resource_log_sha256=sha(log_path), comparisons=comparisons,
            limits="Cross-job ratios use fresh routes and baseline measurements; intervals, controls and raw times require review. Local bytes do not uniquely identify spills. Occupancy is capacity, not achieved occupancy. No automatic publication or integration.")
    out = OUT / "20261104-small-nw4-official-analysis.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(out.relative_to(ROOT)), "passed": report["passed"],
        "resources": report.get("resources"), "comparisons": [
            {k:v for k,v in c.items() if k not in ("parent", "cases")} for c in report.get("comparisons", [])]}, indent=2))


if __name__ == "__main__":
    main()
