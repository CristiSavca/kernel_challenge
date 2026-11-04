#!/usr/bin/env python3
"""Offline fixed V23 decision. No polling, allocations, retries or promotion."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from analyze_sk2_r16_result import ROOT, bounded_file, digest, gm, positive, require

JOB = "0193a093baa446f3"
SOURCE = "c6b90f168bfdc21a8583b6a4bd03cebc0e6c4e03f994f74759c54ab5b677335c"
PARENT = "87a9bddc1b013ff3b31a56aacaf53cb0bea500153b0304a9d4d3f90f6eb87f0c"
PARENTS = [("8f9e5971f95841a0", "run"), ("c21d391dbf604bb5", "check")]
PROBLEM = "muse_attention_decode"
COUNTS = {"test": 16, "benchmark": 12, "profile": 12}  # Both parents have no setup event.
LOADS = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]
BATCHES = [1, 16, 32, 1, 16, 32]
AFFECTED = frozenset(range(1, 6))
SPECIALIZATIONS = [(1, False, 13), (1, False, 11), (1, False, 1), (1, True, 0),
                   (1, False, 1), (1, True, 0), (8, False, 15), (8, False, 11),
                   (8, False, 3), (8, True, 0), (8, False, 3), (8, True, 0)]
PLAN = "tools/ATTENTION_EXACT_RESCALE_V23_PLAN_20261104.md"
PLAN_SHA = "70c4e49bf08a58894d84007d18f7260cf4fd41c87bcba96591e1986d65804bda"


def pass_gate(result):
    require(result.get("passed") is True and result.get("returncode") == 0
            and result.get("status") == "ok" and result.get("error") is None, "official result did not pass")
    require(result.get("flags") == [] and result.get("checks") ==
            {"ok": True, "flagged": [], "rejected": []}, "official checks not clean")
    records = result.get("records", [])
    require(Counter(r.get("event") for r in records) == COUNTS, "record coverage mismatch")
    for kind, count in COUNTS.items():
        rows = [r for r in records if r["event"] == kind]
        require({r["index"] for r in rows} == set(range(count)), "duplicate/missing " + kind + " index")
    for row in records:
        require(row.get("status", "pass" if row["event"] == "profile" else None) == "pass", "event did not pass")
        require(row.get("flags", []) == [], "invalid/nonempty event flags")
    score = positive(result.get("speedup"), "score")
    ci = result.get("speedup_ci")
    require(isinstance(ci, list) and len(ci) == 2, "missing confidence interval")
    require(positive(ci[0], "CI low") <= score <= positive(ci[1], "CI high"), "CI ordering")


def clean_rows(rows):
    for row in rows:
        require(row.get("flags", []) == [], "exported flags not empty")
    return [{k: v for k, v in row.items() if k != "flags"} for row in rows]


def read_run(job, sha, mode):
    path, raw = bounded_file(f"experiments/runs/{job}.json")
    receipt = json.loads(raw); response = receipt["response"]; status = response["status"]
    require(receipt["job"] == job and receipt["problem"] == PROBLEM
            and receipt["source_sha256"] == sha, "local receipt identity mismatch")
    ids = [response[k] for k in ("id", "job_id") if k in response]
    require(bool(ids) and all(x == job for x in ids), "server job identity mismatch")
    for key, expected in [("board", PROBLEM), ("mode", mode)]:
        require(response.get(key) == expected or (key not in response and status == "received"), "server identity mismatch")
    source_path, source = bounded_file(receipt["source"])
    require(digest(source) == sha, "source bytes mismatch")
    provenance = {"job": job, "mode": mode, "source_sha256": sha, "receipt_sha256": digest(raw),
                  "receipt": str(path.relative_to(ROOT)), "source": str(source_path.relative_to(ROOT))}
    if status in ("received", "queued", "running", "pending"):
        return {"status": "PENDING", "saved_status": status, "provenance": provenance}
    require(status in ("done", "failed", "rejected", "error", "cancelled", "infra_error", "timeout", "duplicate"),
            "unknown terminal state")
    result = response.get("result") or {}
    try:
        pass_gate(result)
    except ValueError as exc:
        return {"status": "OFFICIAL_GATE_FAILED", "reason": str(exc), "provenance": provenance,
                "official_error": result.get("error"), "saved_status": status}
    require(status == "done", "inconsistent passing terminal state")
    profile_path, profile_raw = bounded_file(response["profile"]["json"])
    profile = json.loads(profile_raw)
    require(profile["job_id"] == job and profile["problem"] == PROBLEM, "profile identity mismatch")
    _, profile_source = bounded_file(profile["submission"])
    require(digest(profile_source) == sha, "profile source mismatch")
    for field, event in [("benchmarks", "benchmark"), ("profiles", "profile")]:
        require(clean_rows(profile[field]) == clean_rows([r for r in result["records"] if r["event"] == event]),
                "exported profile/receipt disagreement")
    summary = profile["summary"]
    require(summary.get("tests") == summary.get("tests_passed") == 16, "exported test summary mismatch")
    for key in ("passed", "error", "status", "score_us", "speedup", "speedup_ci"):
        require(summary[key] == result[key], "exported summary disagreement")
    provenance.update(profile=str(profile_path.relative_to(ROOT)), profile_sha256=digest(profile_raw),
        evaluator_inventory_sha256=digest(json.dumps(result["evaluator"], sort_keys=True, separators=(",", ":")).encode()))
    return {"status": "PASS", "provenance": provenance, "result": result}


def scores(result):
    pass_gate(result)
    rows = {r["index"]: r for r in result["records"] if r["event"] == "benchmark"}
    profiles = {r["index"]: r for r in result["records"] if r["event"] == "profile"}
    loads = {}
    for i, row in sorted(rows.items()):
        c = row["case"]; load = LOADS[i // 2]; layer = "sliding" if i % 2 == 0 else "full"
        require(c["load"] == load and c["layer"] == layer, "indexed case mismatch")
        require(c["batch"] == BATCHES[i // 2] and c.get("q_len", 1) == (1 if i < 6 else 8), "attention shape mismatch")
        require(c["weight"] == (46 if layer == "sliding" else 16), "load weight mismatch")
        require(profiles[i]["case"] == c, "profile/benchmark case mismatch")
        for key in ("median_ns", "median_raw_ns", "baseline_ns", "speedup"):
            positive(row[key], key)
        require(math.isclose(row["speedup"], row["baseline_ns"] / row["median_ns"], rel_tol=1e-9), "case score mismatch")
        loads.setdefault(load, {})[layer] = row
    weighted = {}
    for load, pair in loads.items():
        weighted[load] = sum(w * pair[layer]["baseline_ns"] for layer, w in [("sliding",46),("full",16)]) / \
                         sum(w * pair[layer]["median_ns"] for layer, w in [("sliding",46),("full",16)])
    require(math.isclose(gm(weighted.values()), result["speedup"], rel_tol=1e-9), "six weighted-load score mismatch")
    return rows, weighted


def activation(result):
    rows = []
    for r in sorted((r for r in result["records"] if r["event"] == "profile"), key=lambda r:r["index"]):
        report = r.get("report", "")
        kernels = [line for line in report.splitlines() if re.search(r"\s[0-9.]+\s+µs\s", line) and not line.startswith("total")]
        matches = re.findall(r"::attn_kernel<(\d+), (true|false), (\d+)>", report)
        sig = [(int(a), b == "true", int(c)) for a,b,c in matches]
        passed = len(kernels) == 1 and sig == [SPECIALIZATIONS[r["index"]]] and bool(re.search(r"\s1\s*$", kernels[0]))
        rows.append({"index":r["index"],"affected":r["index"] in AFFECTED,
                     "expected_specialization":list(SPECIALIZATIONS[r["index"]]),
                     "visible_specializations":[list(x) for x in sig],"pass":passed})
    return {"pass":len(rows)==12 and all(r["pass"] for r in rows),"cases":rows,
            "limit":"Visible selected specialization plus frozen source pin; branch skip frequency and compiler instruction removal are unknown."}


def compare(candidate, parent):
    require(candidate["evaluator"] == parent["evaluator"], "full evaluator inventory/image mismatch")
    require(candidate["gpu"] == parent["gpu"] == "H200", "GPU model mismatch")
    cb, cl = scores(candidate); pb, pl = scores(parent)
    strip_seed = lambda c:{k:v for k,v in c.items() if k != "seed"}
    for kind in ("test", "benchmark", "profile"):
        cr = {r["index"]:r for r in candidate["records"] if r["event"] == kind}
        pr = {r["index"]:r for r in parent["records"] if r["event"] == kind}
        require(all(strip_seed(cr[i]["case"]) == strip_seed(pr[i]["case"]) for i in cr), "case contract mismatch beyond seed")
    cases = []
    for i in range(12):
        ratio = cb[i]["speedup"] / pb[i]["speedup"]
        cases.append({"index":i,"load":cb[i]["case"]["load"],"layer":cb[i]["case"]["layer"],
                      "affected":i in AFFECTED,"speedup_ratio":ratio,"normalized_latency_ratio":1/ratio,
                      "harness_median_ratio":cb[i]["median_ns"]/pb[i]["median_ns"],
                      "raw_median_ratio":cb[i]["median_raw_ns"]/pb[i]["median_raw_ns"],
                      "baseline_median_ratio":cb[i]["baseline_ns"]/pb[i]["baseline_ns"],
                      "candidate_seed":cb[i]["case"].get("seed"),"parent_seed":pb[i]["case"].get("seed"),
                      "candidate_harness_us":cb[i]["median_ns"]/1000,"parent_harness_us":pb[i]["median_ns"]/1000,
                      "candidate_raw_us":cb[i]["median_raw_ns"]/1000,"parent_raw_us":pb[i]["median_raw_ns"]/1000})
    loads = {load: {"candidate_speedup":cl[load],"parent_speedup":pl[load],"speedup_ratio":cl[load]/pl[load]} for load in LOADS}
    aggregate = candidate["speedup"] / parent["speedup"]
    require(math.isclose(aggregate, gm(v["speedup_ratio"] for v in loads.values()), rel_tol=1e-9), "relative aggregate mismatch")
    affected = gm(r["speedup_ratio"] for r in cases if r["affected"])
    control = gm(r["normalized_latency_ratio"] for r in cases if not r["affected"])
    ci, pi = candidate["speedup_ci"], parent["speedup_ci"]
    gates = {"six_load_score_ratio_at_least_1p01":aggregate>=1.01,
             "affected_five_case_speed_GM_at_least_1p03":affected>=1.03,
             "every_affected_latency_at_most_1p02":all(r["normalized_latency_ratio"]<=1.02 for r in cases if r["affected"]),
             "seven_control_latency_GM_0p98_to_1p02":.98<=control<=1.02}
    return {"cases":cases,"weighted_loads":loads,"aggregate_speedup_ratio":aggregate,
            "affected_speedup_ratio_GM":affected,"control_normalized_latency_ratio_GM":control,"gates":gates,
            "candidate_score_CI":ci,"parent_score_CI":pi,"reported_CIs_overlap":max(ci[0],pi[0])<=min(ci[1],pi[1]),
            "candidate_node":candidate.get("machine",{}).get("node"),"parent_node":parent.get("machine",{}).get("node"),
            "test_seeds":[{"index":r["index"],"candidate":r["case"].get("seed"),
                           "parent":next(p for p in parent["records"] if p["event"]=="test" and p["index"]==r["index"])["case"].get("seed")}
                          for r in candidate["records"] if r["event"]=="test"]}


def analyze():
    _, plan = bounded_file(PLAN); require(digest(plan)==PLAN_SHA,"fixed plan changed")
    run = read_run(JOB,SOURCE,"check")
    report = {"candidate":run["provenance"],"fixed_plan":PLAN,"fixed_plan_sha256":PLAN_SHA,
              "automatic_promotion_allowed":False,"comparisons":[],"fixed_gate_passed":False,
              "limits":["Cross-run comparisons use fresh seeds and possibly different nodes, not candidate/incumbent same-input pairing.",
                        "Confidence intervals and overlap do not prove significance or causal instruction savings.",
                        "No branch-frequency, SASS, whole-decoder or serving performance claim follows."]}
    if run["status"]!="PASS":
        return {**report,**{k:v for k,v in run.items() if k not in ("provenance","result")}}
    result=run["result"]; comparisons=[]; parent_activations=[]
    for job,mode in PARENTS:
        old=read_run(job,PARENT,mode);require(old["status"]=="PASS","parent not complete")
        act=activation(old["result"]);require(act["pass"],"parent selected specialization mismatch")
        parent_activations.append({"job":job,**act})
        comparisons.append({"parent":old["provenance"],**compare(result,old["result"])})
    joint=[i for i in sorted(AFFECTED) if all(c["cases"][i]["speedup_ratio"]>1 for c in comparisons)]
    act=activation(result)
    passed=act["pass"] and len(joint)>=3 and all(all(c["gates"].values()) for c in comparisons)
    return {**report,"status":"FIXED_SCREEN_PASS" if passed else "REJECT_FIXED_V23","fixed_gate_passed":passed,
            "score":result["speedup"],"score_ci":result["speedup_ci"],"counts":COUNTS,"setup_events":0,
            "evaluator":result["evaluator"],"comparisons":comparisons,"jointly_positive_affected_indices":joint,
            "joint_positive_gate_passed":len(joint)>=3,"activation":act,"parent_activations":parent_activations,
            "next_action":"Independent result review before any decoder integration or public promotion."
                          if passed else "Close fixed attempt; no noise retry, threshold retune or decoder port."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    try: report=analyze()
    except (ValueError,KeyError,TypeError,OSError) as exc:
        report={"status":"EVIDENCE_REJECTED","reason":str(exc),"comparisons":[],"fixed_gate_passed":False,"automatic_promotion_allowed":False}
    text=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.output: args.output.write_text(text)
    print(text,end="")
