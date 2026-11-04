#!/usr/bin/env python3
"""Offline dense-V10 receipt/profile audit. Never contacts an official service."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOB = "d4c26ebb65e94a84"
PARENTS = ("4b216098993e4112", "f8e991f1a2c2469e")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(job):
    rp = ROOT / "experiments/runs" / f"{job}.json"
    receipt = json.loads(rp.read_text())
    response = receipt["response"]
    assert response["status"] == "done" and response["result"]["passed"]
    source = ROOT / receipt["source"]
    assert digest(source) == receipt["source_sha256"]
    pp = Path(response["profile"]["json"])
    lp = Path(response["output"]["log"])
    result = response["result"]
    provenance = {
        "job": job, "source_sha256": digest(source),
        "receipt": str(rp.relative_to(ROOT)), "receipt_sha256": digest(rp),
        "profile": str(pp.relative_to(ROOT)), "profile_sha256": digest(pp),
        "log_bytes": lp.stat().st_size, "log_sha256": digest(lp),
        "speedup": result["speedup"], "speedup_ci": result["speedup_ci"],
        "node": result["machine"]["node"],
        "evaluator_sha256": result["evaluator"]["code_sha256"],
        "events": dict(Counter(x["event"] for x in result["records"])),
        "flags": result["flags"], "checks": result["checks"],
    }
    return json.loads(pp.read_text()), provenance


def kernels(capture):
    rows = []
    for line in capture["report"].splitlines():
        match = re.match(r"^(.*?)\s+(\d+\.\d+) µs\s+[\d.]+%\s+(\d+)\s*$", line)
        if not match:
            continue
        name, duration, calls = match.groups()
        row = {"name": name, "us": float(duration), "calls": int(calls)}
        sk = re.search(r"sk2_(?:global_escape_)?kernel<((?:\d+,\s*){7})(true|false)", name)
        if sk:
            vals = [int(v) for v in sk[1].split(",") if v.strip()]
            row.update(template_prefix=vals, compressed=sk[2] == "true")
            mt, tokr, nt, nwr, wsk, upw, ns = vals
            shape = (nt, nwr, wsk, upw)
            role = "unknown"
            if not row["compressed"]:
                role = "router_uncompressed_unchanged"
            elif shape == (3, 1, 4, 2):
                role = "qkv"
            elif shape == (4, 1, 8, 1):
                role = "pooler_fused_router"
            elif shape == (1, 2, 2, 4):
                role = "pooler_separate"
            elif shape == (4, 1, 4, 1):
                role = "o" if ns == 4 else "unpooler"
            elif shape == (4, 2, 2, 2):
                role = "unpooler"
            elif shape == (2, 2, 2, 2):
                role = "o" if ns == 3 else "pooler_separate"
            row["role"] = role
            row["logical_path_changed"] = row["compressed"]  # all matched compressed configs are CL1
        rows.append(row)
    return rows


def main():
    candidate, provenance = load(JOB)
    assert provenance["source_sha256"] == "b0387be327a480c84227985566bf086c70d894394f95ad9b0589144c902e260b"
    comparisons = []
    for job in PARENTS:
        parent, pp = load(job)
        cases, loads = [], {}
        for new, old, nc, oc in zip(candidate["benchmarks"], parent["benchmarks"], candidate["profiles"], parent["profiles"]):
            assert new["index"] == old["index"] and new["case"]["load"] == old["case"]["load"]
            new_rows, old_rows = kernels(nc), kernels(oc)
            by_key = {tuple(x["template_prefix"]): x for x in old_rows if "template_prefix" in x}
            matched = []
            for row in new_rows:
                if "template_prefix" not in row:
                    continue
                prior = by_key.get(tuple(row["template_prefix"]))
                if prior:
                    matched.append({"role": row["role"], "prefix": row["template_prefix"],
                                    "changed": row["logical_path_changed"], "parent_us": prior["us"],
                                    "candidate_us": row["us"], "delta_us": row["us"] - prior["us"]})
            tokens = new["case"]["batch"] * new["case"].get("q_len", 1)
            cases.append({"index": new["index"], "load": new["case"]["load"], "layer": new["case"]["layer"],
                          "tokens": tokens, "small_T_target": tokens <= 32,
                          "candidate_seed": new["case"]["seed"], "parent_seed": old["case"]["seed"],
                          "candidate_normalized_us": new["median_ns"] / 1000,
                          "parent_normalized_us": old["median_ns"] / 1000,
                          "candidate_raw_us": new["median_raw_ns"] / 1000,
                          "parent_raw_us": old["median_raw_ns"] / 1000,
                          "speedup_ratio": new["speedup"] / old["speedup"],
                          "baseline_time_ratio": new["baseline_ns"] / old["baseline_ns"],
                          "candidate_kernels": new_rows, "parent_kernels": old_rows, "matched_SK2": matched})
            sums = loads.setdefault(new["case"]["load"], [0., 0., 0., 0.])
            w = new["case"]["weight"]
            for i, value in enumerate((new["baseline_ns"], new["median_ns"], old["baseline_ns"], old["median_ns"])):
                sums[i] += w * value
        load_results = {k: {"speedup_ratio": (cb / ct) / (pb / pt), "normalized_latency_ratio": ct / pt}
                        for k, (cb, ct, pb, pt) in loads.items()}
        def gm(target):
            values = [x["speedup_ratio"] for x in cases if x["small_T_target"] == target]
            return math.exp(sum(map(math.log, values)) / len(values))
        comparisons.append({"parent": pp, "aggregate_speedup_ratio": provenance["speedup"] / pp["speedup"],
                            "target_case_ratio_geomean": gm(True), "control_case_ratio_geomean": gm(False),
                            "load_results": load_results, "cases": cases})
    report = {
        "candidate": provenance, "decision": "REJECT promotion; preserve incumbent. No rescue variant or GPU run.",
        "comparisons": comparisons,
        "interpretation": [
            "Every candidate case/load speedup is below both parents; all eight small-T normalized latencies rise.",
            "26 compressed renamed SK2 launches in small-T profiles exercise the changed CL1 branch; six renamed uncompressed router launches do not.",
            "T1/T8 dispatch loses the parent UV/fused pooler-router path, adding separate router and boundary1; T32 o moves to cuBLAS.",
            "Matched qkv durations increase in all eight small-T captures; this is stronger mechanism evidence than comparing differently dispatched pooler rows.",
            "Large-T captures contain no SK2, yet speedup ratios also fall; nodes, fresh seeds, route counts, baseline times and tuning differ.",
            "No device evidence isolates sparse global sectors, correction-tail waits, table scans or register pressure as the cause.",
            "Local candidate log is empty and result log_tail contains profiler messages; no compiler/register/spill/pmax measurements are available.",
        ],
        "source_scan_analysis": {
            "E": "exact escape count for one CTA", "N": "NESC, active changed profiles use3 or4",
            "parent": "N*E successful pointer increments, not E: every warp scans each entry; owner reads int2, nonowners inspect x. Stage boundaries add lookahead reads.",
            "candidate": "N*E int2 source accesses, plus per-entry stage/range/modulo tests. Compiler may sink the delta load into owned branch.",
            "source_level_payload_bytes_excluding_parent_lookahead": {"parent": "4*(N+1)*E", "candidate_full_int2": "8*N*E", "N3": "16E versus24E", "N4": "20E versus32E"},
            "not_measured_traffic": "Warp-uniform shared reads broadcast; multiply neither formula by32. Instructions, bank transactions and global reads require device evidence.",
            "partition_feasibility": "Stable plane grouping preserves original per-plane order, but requires KST/NESC-specific metadata. Pooler same table uses NESC4 atTOKR8/16 andNESC3 at32.",
            "alternative_directory": "Immutable per-stage start/end offsets preserve original table and support variableNESC:4*(K/KST+1)B perCTA,20–132B across active geometries. Each warp visits only its assigned stage intervals; no implementation authorized.",
        },
    }
    path = ROOT / "experiments/iterations/20261104-dense-escape-global-mechanism-analysis.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(path.relative_to(ROOT)), "speedup": provenance["speedup"],
                      "ratios": {x["parent"]["job"]: x["aggregate_speedup_ratio"] for x in comparisons}}))


if __name__ == "__main__":
    main()
