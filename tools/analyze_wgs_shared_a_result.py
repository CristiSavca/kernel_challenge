#!/usr/bin/env python3
"""Offline SS-A performance gate; no polling, GPU work or automatic promotion.

The caller supplies the eventual job and frozen source hash. Resource admission
and selected implementation are separate reviews. Local bytes or compiler
serialization warnings are not performance rejection rules.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0"
PARENTS = (("d24dd50196cb4158", "run"), ("bcb761dfca264db0", "check"))
PROBLEM = "muse_moe_w4a16"
CASES = (("decode 1", 1), ("decode 16", 16), ("decode 32", 32),
         ("verify 1", 8), ("verify 16", 128), ("verify 32", 256))
COUNTS = {"setup": 1, "test": 5, "benchmark": 6, "profile": 6}
PENDING = {"received", "queued", "running", "pending"}
TERMINAL = {"done", "failed", "rejected", "error", "cancelled", "infra_error", "timeout", "duplicate"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            "invalid positive numeric field: " + label)
    return value


def gm(values):
    values = list(values)
    require(bool(values), "empty geometric mean")
    return math.exp(sum(math.log(positive(v, "ratio")) for v in values) / len(values))


def bounded_file(path, root=ROOT):
    root = Path(root).resolve()
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(root), "artifact outside repository")
    require(path.stat().st_size <= 16 * 1024 * 1024, "artifact too large")
    return path, path.read_bytes()


def pass_gate(result):
    require(result.get("passed") is True and result.get("returncode") == 0
            and result.get("status") == "ok" and result.get("error") is None,
            "official result did not pass")
    require(result.get("flags") == [] and result.get("checks") ==
            {"ok": True, "flagged": [], "rejected": []}, "official checks not clean")
    records = result.get("records", [])
    require(Counter(r.get("event") for r in records) == COUNTS, "record coverage mismatch")
    for row in records:
        require(row.get("status", "pass" if row["event"] == "profile" else None) == "pass",
                "missing or failed event status")
        require(not row.get("flags", []), "flagged event")
    score = positive(result.get("speedup"), "speedup")
    interval = result.get("speedup_ci")
    require(isinstance(interval, list) and len(interval) == 2, "missing interval")
    lo, hi = (positive(v, "interval") for v in interval)
    require(lo <= score <= hi, "invalid interval ordering")


def benchmark_rows(result):
    pass_gate(result)
    rows = {r["index"]: r for r in result["records"] if r["event"] == "benchmark"}
    require(set(rows) == set(range(6)), "duplicate or missing benchmark index")
    for index, row in rows.items():
        case = row["case"]
        require((case["load"], case["bs"]) == CASES[index]
                and case["weight"] == 60 and case["sharding"] == "ep8", "unexpected case metadata")
        for field in ("baseline_ns", "median_ns", "median_raw_ns", "speedup"):
            positive(row.get(field), field)
        require(math.isclose(row["baseline_ns"] / row["median_ns"], row["speedup"], rel_tol=1e-9),
                "case speedup does not reproduce")
    require(math.isclose(gm(r["speedup"] for r in rows.values()), result["speedup"], rel_tol=1e-9),
            "six-load aggregate does not reproduce")
    return rows


def compare(candidate, parent):
    require(candidate["evaluator"] == parent["evaluator"], "evaluator mismatch")
    require(candidate["gpu"] == parent["gpu"] == "H200", "GPU class mismatch")
    current, baseline = benchmark_rows(candidate), benchmark_rows(parent)
    cases = []
    for index, row in sorted(current.items()):
        old = baseline[index]
        clean = lambda c: {k: v for k, v in c.items() if k != "seed"}
        require(clean(row["case"]) == clean(old["case"]), "case metadata mismatch")
        ratio = row["speedup"] / old["speedup"]
        cases.append({"index": index, "M": row["case"]["bs"], "load": row["case"]["load"],
                      "affected": row["case"]["bs"] > 16, "speedup_ratio": ratio,
                      "normalized_latency_ratio": 1 / ratio,
                      "candidate_seed": row["case"].get("seed"), "parent_seed": old["case"].get("seed"),
                      "harness_median_ratio": row["median_ns"] / old["median_ns"],
                      "raw_median_ratio": row["median_raw_ns"] / old["median_raw_ns"],
                      "baseline_median_ratio": row["baseline_ns"] / old["baseline_ns"],
                      "candidate_harness_us": row["median_ns"] / 1000,
                      "parent_harness_us": old["median_ns"] / 1000,
                      "candidate_raw_us": row["median_raw_ns"] / 1000,
                      "parent_raw_us": old["median_raw_ns"] / 1000})
    affected = [r for r in cases if r["affected"]]
    controls = [r for r in cases if not r["affected"]]
    require(len(affected) == len(controls) == 3, "affected/control partition")
    target = gm(r["speedup_ratio"] for r in affected)
    control = gm(r["normalized_latency_ratio"] for r in controls)
    aggregate = candidate["speedup"] / parent["speedup"]
    require(math.isclose(aggregate, gm(r["speedup_ratio"] for r in cases), rel_tol=1e-9),
            "relative aggregate mismatch")
    gates = {"affected_GM_at_least_3pct_faster": target >= 1.03,
             "every_affected_case_at_most_3pct_slower":
                 all(r["normalized_latency_ratio"] <= 1.03 for r in affected),
             "control_GM_within_3pct": .97 <= control <= 1.03}
    ci, pi = candidate["speedup_ci"], parent["speedup_ci"]
    return {"cases": cases, "affected_speedup_ratio_GM": target,
            "control_normalized_latency_ratio_GM": control, "aggregate_speedup_ratio": aggregate,
            "gates": gates, "performance_gate_passed": all(gates.values()),
            "candidate_reported_CI": ci, "parent_reported_CI": pi,
            "reported_CIs_overlap": max(ci[0], pi[0]) <= min(ci[1], pi[1]),
            "same_reported_node": (candidate.get("machine", {}).get("node") ==
                                   parent.get("machine", {}).get("node"))
                if candidate.get("machine", {}).get("node") and parent.get("machine", {}).get("node") else None,
            "inference_limit": "Fresh seeds/different nodes are not direct same-input A/B. Confidence intervals alone do not prove attribution or a compiler mechanism."}


def read_run(job, source_sha, mode, root=ROOT):
    root = Path(root).resolve()
    require(bool(re.fullmatch(r"[0-9a-f]{16}", job)), "invalid job ID")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", source_sha)), "invalid source SHA")
    receipt_path, raw = bounded_file(f"experiments/runs/{job}.json", root)
    receipt = json.loads(raw)
    response = receipt["response"]
    status = response["status"]
    ids = [response[k] for k in ("id", "job_id") if k in response]
    require(receipt["job"] == job and bool(ids) and all(v == job for v in ids), "receipt job mismatch")
    require(receipt["problem"] == PROBLEM, "board mismatch")
    for field, expected in (("board", PROBLEM), ("mode", mode)):
        require((field not in response and status == "received") or response.get(field) == expected,
                field + " mismatch")
    require(receipt["source_sha256"] == source_sha, "receipt source mismatch")
    source_path, source = bounded_file(receipt["source"], root)
    require(digest(source) == source_sha, "frozen source bytes mismatch")
    provenance = {"job": job, "source_sha256": source_sha, "receipt_sha256": digest(raw),
                  "receipt": str(receipt_path.relative_to(root)), "source": str(source_path.relative_to(root))}
    if status in PENDING:
        return {"status": "PENDING", "saved_status": status, "provenance": provenance,
                "server_identity_confirmed": "board" in response and "mode" in response}
    require(status in TERMINAL, "unrecognized job status")
    result = response.get("result") or {}
    try:
        pass_gate(result)
    except ValueError as error:
        return {"status": "OFFICIAL_GATE_FAILED", "reason": str(error),
                "official_error": result.get("error"), "provenance": provenance}
    require(status == "done", "passing result has inconsistent terminal status")
    pp, data = bounded_file(response["profile"]["json"], root)
    profile = json.loads(data)
    require(profile["job_id"] == job and profile["problem"] == PROBLEM, "profile identity mismatch")
    _, profile_source = bounded_file(profile["submission"], root)
    require(digest(profile_source) == source_sha, "profile source mismatch")
    for field, event in (("benchmarks", "benchmark"), ("profiles", "profile")):
        # The official CLI drops absent/empty flags from extracted benchmark
        # records. Normalize only that documented-in-our-archives difference.
        normalized = lambda rows: [{**r, "flags": r.get("flags", [])} for r in rows]
        require(normalized(profile[field]) == normalized(
                    [r for r in result["records"] if r["event"] == event]),
                "profile and receipt records disagree")
    benches = benchmark_rows(result)
    profiles = {r["index"]: r for r in profile["profiles"]}
    require(set(profiles) == set(benches), "profile index coverage")
    require(all(profiles[i]["case"] == benches[i]["case"] for i in benches),
            "profile/benchmark case mismatch")
    provenance.update(profile=str(pp.relative_to(root)), profile_sha256=digest(data),
                      evaluator_sha256=result["evaluator"]["code_sha256"])
    return {"status": "PASS", "provenance": provenance, "result": result}


def analyze(job, source_sha, root=ROOT):
    candidate = read_run(job, source_sha, "check", root)
    report = {"schema": 1, "candidate": candidate["provenance"], "automatic_promotion_allowed": False,
              "method": "Six-load geometric score; affectedM32/128/256 GM and controlM1/8/16 GM separately.",
              "unresolved_gates": ["frozen_source_review", "actual_launch_admission", "intended_MAIN_activation"],
              "diagnostic_policy": "Local bytes and compiler serialization warnings are informational; actual validity, correctness and measured speed decide."}
    if candidate["status"] != "PASS":
        return {**report, **{k: v for k, v in candidate.items() if k != "provenance"}, "comparisons": []}
    require(source_sha != PARENT_SHA, "incumbent is not an SS-A candidate")
    comparisons = []
    for parent_job, mode in PARENTS:
        parent = read_run(parent_job, PARENT_SHA, mode, root)
        require(parent["status"] == "PASS", "incumbent evidence incomplete")
        comparisons.append({"parent": parent["provenance"], **compare(candidate["result"], parent["result"])})
    passed = all(r["performance_gate_passed"] for r in comparisons)
    return {**report, "status": "PERFORMANCE_GATE_PASS_OTHER_GATES_PENDING" if passed else "PERFORMANCE_GATE_FAILED",
            "performance_gate_passed": passed, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = analyze(args.job, args.source_sha256)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        report = {"status": "EVIDENCE_REJECTED", "error": str(error), "automatic_promotion_allowed": False}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
