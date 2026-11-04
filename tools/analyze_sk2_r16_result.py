#!/usr/bin/env python3
"""Offline R16 performance gate. No polling, GPU work or automatic promotion.

Resource, activation and source-review gates are deliberately separate. This tool
cannot authorize submission even when its numerical performance gates pass.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2"
PARENTS = (("7a8c4941741645ee", "run"), ("b2a773e2d7194de8", "check"))
LOAD_TOKENS = {"decode 1": 1, "decode 16": 16, "decode 32": 32,
               "verify 1": 8, "verify 16": 128, "verify 32": 256}
COUNTS = {"setup": 1, "test": 13, "benchmark": 12, "profile": 12}


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
    return math.exp(sum(math.log(positive(x, "ratio")) for x in values) / len(values))


def bounded_file(path, root=ROOT):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(root.resolve()), "artifact outside repository")
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
        if row["event"] == "profile":
            require(row.get("status", "pass") == "pass", "failed profile")
        else:
            require(row.get("status") == "pass", "missing or failed explicit event status")
        require(not row.get("flags", []), "flagged event")
    score = positive(result.get("speedup"), "speedup")
    interval = result.get("speedup_ci")
    require(isinstance(interval, list) and len(interval) == 2, "missing interval")
    lo, hi = (positive(x, "interval") for x in interval)
    require(lo <= score <= hi, "invalid interval ordering")


def load_scores(result):
    """Reproduce the official weighted six-load score, validating every case."""
    pass_gate(result)
    benches = {r["index"]: r for r in result["records"] if r["event"] == "benchmark"}
    require(set(benches) == set(range(12)), "duplicate or missing benchmark index")
    loads = {}
    for row in benches.values():
        case = row["case"]
        name, layer = case["load"], case["layer"]
        require(name in LOAD_TOKENS and layer in ("sliding", "full"), "unknown load/layer")
        tokens = case["batch"] * case.get("q_len", 1)
        require(tokens == LOAD_TOKENS[name], "unexpected token shape")
        weight = 46 if layer == "sliding" else 16
        require(case["weight"] == weight, "unexpected load weight")
        for field in ("baseline_ns", "median_ns", "median_raw_ns", "speedup"):
            positive(row.get(field), field)
        require(math.isclose(row["baseline_ns"] / row["median_ns"],
                             row["speedup"], rel_tol=1e-9), "case speedup does not reproduce")
        item = loads.setdefault(name, {"tokens": tokens, "layers": [],
                                      "baseline_ns": 0., "candidate_ns": 0.})
        require(layer not in item["layers"], "duplicate load layer")
        item["layers"].append(layer)
        item["baseline_ns"] += weight * row["baseline_ns"]
        item["candidate_ns"] += weight * row["median_ns"]
    require(set(loads) == set(LOAD_TOKENS), "load coverage mismatch")
    for item in loads.values():
        require(set(item["layers"]) == {"sliding", "full"}, "incomplete load")
        item["speedup"] = item["baseline_ns"] / item["candidate_ns"]
        item["affected"] = item["tokens"] <= 32
    require(math.isclose(gm(v["speedup"] for v in loads.values()),
                         result["speedup"], rel_tol=1e-9), "aggregate speedup does not reproduce")
    return benches, loads


def compare(candidate, parent):
    require(candidate["evaluator"]["code_sha256"] == parent["evaluator"]["code_sha256"],
            "evaluator mismatch")
    nb, nl = load_scores(candidate)
    pb, pl = load_scores(parent)
    cases = []
    for index, row in sorted(nb.items()):
        old = pb[index]
        clean = lambda c: {k: v for k, v in c.items() if k != "seed"}
        require(clean(row["case"]) == clean(old["case"]), "case metadata mismatch")
        ratio = row["speedup"] / old["speedup"]
        cases.append({"index": index, "load": row["case"]["load"],
                      "layer": row["case"]["layer"],
                      "affected": LOAD_TOKENS[row["case"]["load"]] <= 32,
                      "candidate_seed": row["case"].get("seed"), "parent_seed": old["case"].get("seed"),
                      "speedup_ratio": ratio, "normalized_latency_ratio": 1 / ratio,
                      "harness_median_ratio": row["median_ns"] / old["median_ns"],
                      "raw_median_ratio": row["median_raw_ns"] / old["median_raw_ns"],
                      "baseline_median_ratio": row["baseline_ns"] / old["baseline_ns"],
                      "candidate_harness_us": row["median_ns"] / 1000,
                      "parent_harness_us": old["median_ns"] / 1000,
                      "candidate_raw_us": row["median_raw_ns"] / 1000,
                      "parent_raw_us": old["median_raw_ns"] / 1000})
    loads = {}
    for name, row in nl.items():
        old = pl[name]
        ratio = row["speedup"] / old["speedup"]
        loads[name] = {"affected": row["affected"], "candidate_speedup": row["speedup"],
                       "parent_speedup": old["speedup"], "speedup_ratio": ratio,
                       "normalized_latency_ratio": 1 / ratio,
                       "harness_weighted_time_ratio": row["candidate_ns"] / old["candidate_ns"],
                       "baseline_weighted_time_ratio": row["baseline_ns"] / old["baseline_ns"]}
    aggregate = candidate["speedup"] / parent["speedup"]
    require(math.isclose(aggregate, gm(x["speedup_ratio"] for x in loads.values()),
                         rel_tol=1e-9), "relative aggregate mismatch")
    affected = gm(x["speedup_ratio"] for x in loads.values() if x["affected"])
    control = gm(x["normalized_latency_ratio"] for x in loads.values() if not x["affected"])
    gates = {"aggregate_at_least_2pct": aggregate >= 1.02,
             "affected_four_loads_improve": affected > 1,
             "every_affected_case_at_most_3pct_slower":
                 all(x["normalized_latency_ratio"] <= 1.03 for x in cases if x["affected"]),
             "two_control_load_drift_within_3pct": .97 <= control <= 1.03}
    ci, pi = candidate["speedup_ci"], parent["speedup_ci"]
    return {"cases": cases, "loads": loads, "aggregate_speedup_ratio": aggregate,
            "affected_four_load_speedup_ratio_GM": affected,
            "control_two_load_normalized_latency_ratio_GM": control,
            "gates": gates, "performance_gate_passed": all(gates.values()),
            "candidate_reported_CI": ci, "parent_reported_CI": pi,
            "reported_CIs_overlap": max(ci[0], pi[0]) <= min(ci[1], pi[1]),
            "same_reported_node": candidate.get("machine", {}).get("node") ==
                                  parent.get("machine", {}).get("node"),
            "inference_limit": "Different runs/seeds are not a same-input pairing; interval overlap or disjointness alone is not a causal mechanism test."}


def read_run(job, source_sha, mode, root=ROOT):
    root = Path(root).resolve()
    require(bool(re.fullmatch(r"[0-9a-f]{16}", job)), "invalid job ID")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", source_sha)), "invalid source SHA")
    receipt_path, raw = bounded_file(f"experiments/runs/{job}.json", root)
    receipt = json.loads(raw)
    response = receipt["response"]
    status = response["status"]
    server_ids = [response[k] for k in ("id", "job_id") if k in response]
    require(receipt["job"] == job and bool(server_ids)
            and all(value == job for value in server_ids), "receipt job mismatch")
    require(receipt["problem"] == "muse_decode_layer", "board mismatch")
    # The submit endpoint initially returns only job_id/status. Missing server
    # identity is allowed solely for that acknowledgement, never terminal data.
    for key, expected in (("board", "muse_decode_layer"), ("mode", mode)):
        require((key not in response and status == "received")
                or response.get(key) == expected, key + " mismatch")
    require(receipt["source_sha256"] == source_sha, "receipt source mismatch")
    source_path, source = bounded_file(receipt["source"], root)
    require(digest(source) == source_sha, "frozen source bytes mismatch")
    provenance = {"job": job, "source_sha256": source_sha, "receipt_sha256": digest(raw),
                  "receipt": str(receipt_path.relative_to(root)), "source": str(source_path.relative_to(root))}
    if status in ("queued", "running", "received", "pending"):
        confirmed = "board" in response and "mode" in response
        return {"status": "PENDING", "saved_status": status, "provenance": provenance,
                "server_identity_confirmed": confirmed,
                **({"note": "Acknowledged; complete board/mode identity not yet returned."}
                   if not confirmed else {})}
    require(status in ("done", "failed", "rejected", "error", "cancelled",
                       "infra_error", "timeout", "duplicate"), "unrecognized job status")
    result = response.get("result") or {}
    try:
        pass_gate(result)
    except ValueError as error:
        return {"status": "OFFICIAL_GATE_FAILED", "reason": str(error),
                "official_error": result.get("error"), "provenance": provenance}
    require(status == "done", "passing result has inconsistent terminal status")
    pp, data = bounded_file(response["profile"]["json"], root)
    profile = json.loads(data)
    require(profile["job_id"] == job and profile["problem"] == "muse_decode_layer", "profile identity mismatch")
    _, profile_source = bounded_file(profile["submission"], root)
    require(digest(profile_source) == source_sha, "profile source mismatch")
    for key, event in (("benchmarks", "benchmark"), ("profiles", "profile")):
        require(profile[key] == [r for r in result["records"] if r["event"] == event],
                "profile and receipt records disagree")
    provenance.update(profile=str(pp.relative_to(root)), profile_sha256=digest(data),
                      evaluator_sha256=result["evaluator"]["code_sha256"])
    return {"status": "PASS", "provenance": provenance, "result": result}


def analyze(job, source_sha, root=ROOT):
    root = Path(root).resolve()
    candidate = read_run(job, source_sha, "check", root)
    report = {"schema": 1, "candidate": candidate["provenance"], "automatic_promotion_allowed": False,
              "unresolved_gates": ["frozen_source_review", "compiled_resource_admission",
                                   "selected_activation_and_preserved_fusion"],
              "method": "46/16 weighted sliding/full load ratios; six-load geometric score; four affected and two control loads."}
    if candidate["status"] != "PASS":
        return {**report, **{k: v for k, v in candidate.items() if k != "provenance"}, "comparisons": []}
    require(source_sha != PARENT_SHA, "incumbent is not the new R16 candidate")
    comparisons = []
    for parent_job, mode in PARENTS:
        parent = read_run(parent_job, PARENT_SHA, mode, root)
        require(parent["status"] == "PASS", "incumbent evidence is incomplete")
        comparisons.append({"parent": parent["provenance"],
                            **compare(candidate["result"], parent["result"])})
    passed = all(x["performance_gate_passed"] for x in comparisons)
    report.update(status="PERFORMANCE_GATE_PASS_OTHER_GATES_PENDING" if passed else "PERFORMANCE_GATE_FAILED",
                  performance_gate_passed=passed, comparisons=comparisons)
    return report


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
