#!/usr/bin/env python3
"""Read saved receipts/profiles only; never query or modify official jobs."""
import ast
from collections import Counter
import json
import math
import re
from pathlib import Path

from analyze_dense_escape_global_v10 import ROOT, digest, load

JOB = "4b9f33ffa6bc4eed"
PARENTS = ("4b216098993e4112", "f8e991f1a2c2469e", "7a8c4941741645ee", "b2a773e2d7194de8")
SHA = "15a7188b3e35b469565cc2d7eb9b37c0804e341dab5f1a4939a16563efdaa172"


def kernels(capture):
    rows = []
    for line in capture["report"].splitlines():
        m = re.match(r"^(.*?)\s+(\d+\.\d+) µs\s+[\d.]+%\s+(\d+)\s*$", line)
        if not m:
            continue
        name, duration, calls = m.groups()
        row = dict(name=name, us=float(duration), calls=int(calls))
        sk = re.search(r"sk2_(?:stage_directory_)?kernel<((?:\d+,\s*){7})(tru|fal)", name)
        if sk:
            vals = [int(x) for x in sk[1].split(",") if x.strip()]
            mt, tokr, nt, nwr, wsk, upw, ns = vals
            compressed = sk[2] == "tru"
            shape = nt, nwr, wsk, upw
            if not compressed:
                role = "router"
            elif shape == (3, 1, 4, 2):
                role = "qkv"
            elif shape == (4, 1, 8, 1):
                role = "pooler_fused_router"
            elif shape == (1, 2, 2, 4):
                role = "pooler_separate_R16"
            elif shape == (4, 1, 4, 1):
                role = "o" if ns == 4 else "unpooler_R32"
            elif shape == (4, 2, 2, 2):
                role = "unpooler_R64"
            elif shape == (2, 2, 2, 2):
                role = "o" if ns == 3 else "pooler_separate_R32"
            else:
                raise AssertionError((vals, name))
            row.update(prefix=vals, compressed=compressed, role=role)
        rows.append(row)
    return rows


def selections(log, source):
    tree = ast.parse(source.read_bytes())
    constants = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ("bfp_SK2_CONFIGS", "bfp_SK2_K"):
                constants[n.targets[0].id] = ast.literal_eval(n.value)
    records = []
    for line in log.splitlines():
        if not line.startswith("SK2_STAGE_DIRECTORY "):
            continue
        fields = line.split()[1:]
        record = {k: ast.literal_eval(v) for k, v in zip(fields[::2], fields[1::2])}
        configs = [c for c in constants["bfp_SK2_CONFIGS"]
                   if (c[2], c[3], 64*c[6]*c[7]) == (record["rows"], record["R"], record["KST"])
                   and 16*((4*(constants["bfp_SK2_K"][c[1]]//record["KST"]+1)+15)//16) == record["tail_bytes"]]
        layers = set(c[1] for c in configs)
        assert len(layers) == 1
        layer = layers.pop()
        record.update(layer=layer, config_names=[c[0] for c in configs])
        stages = constants["bfp_SK2_K"][layer] // record["KST"]
        record["stages"] = stages
        record["logical_scan_work_saved_bytes"] = 4*(record["NESCmin"]-1)*record["escapes"]
        record["logical_directory_cost_bytes"] = record["ctas"]*(record["tail_bytes"]+8*stages)
        record["logical_work_surplus_bytes"] = record["logical_scan_work_saved_bytes"]-record["logical_directory_cost_bytes"]
        record["total_appended_bytes"] = record["ctas"]*record["tail_bytes"] if record["selected"] else 0
        assert record["selected"] == (record["fits"] and record["logical_work_surplus_bytes"] > 0)
        records.append(record)
    assert len(records) == 9 and sum(x["selected"] for x in records) == 4
    return records


def main():
    cand, provenance = load(JOB)
    assert provenance["source_sha256"] == SHA
    receipt = json.loads((ROOT/"experiments/runs"/f"{JOB}.json").read_text())
    result = receipt["response"]["result"]
    log = Path(receipt["response"]["output"]["log"]).read_text()
    assert [s for s in log.splitlines() if s.startswith("SK2_STAGE_DIRECTORY ")] == [
        s for s in result["log_tail"].splitlines() if s.startswith("SK2_STAGE_DIRECTORY ")]
    activation = selections(log, ROOT/receipt["source"])
    selected_roles = {"qkv", "pooler_fused_router", "pooler_separate_R32", "unpooler_R64"}
    comparisons = []
    for job in PARENTS:
        parent, pp = load(job)
        cases, sums = [], {}
        for new, old, nc, oc in zip(cand["benchmarks"], parent["benchmarks"], cand["profiles"], parent["profiles"]):
            assert new["index"] == old["index"] and new["case"]["load"] == old["case"]["load"]
            nr, pr = kernels(nc), kernels(oc)
            matching = {tuple(r["prefix"]): r for r in pr if "prefix" in r}
            matched = []
            for r in nr:
                if "prefix" not in r:
                    continue
                r["directory_selected_from_prepare"] = r["role"] in selected_roles
                oldr = matching.get(tuple(r["prefix"]))
                if oldr:
                    matched.append(dict(role=r["role"], prefix=r["prefix"], selected=r["directory_selected_from_prepare"],
                        candidate_us=r["us"], parent_us=oldr["us"], delta_us=r["us"]-oldr["us"]))
            def dispatch(rows):
                return sorted((str((r["prefix"], r["compressed"])) if "prefix" in r else r["name"], r["calls"]) for r in rows)
            tokens = new["case"]["batch"]*new["case"].get("q_len", 1)
            cases.append(dict(index=new["index"], load=new["case"]["load"], layer=new["case"]["layer"], tokens=tokens,
                target=tokens<=32, candidate_seed=new["case"]["seed"], parent_seed=old["case"]["seed"],
                candidate_normalized_us=new["median_ns"]/1000, parent_normalized_us=old["median_ns"]/1000,
                speedup_ratio=new["speedup"]/old["speedup"], baseline_ratio=new["baseline_ns"]/old["baseline_ns"],
                same_reported_dispatch=dispatch(nr)==dispatch(pr), candidate_kernels=nr, parent_kernels=pr, matched_SK2=matched))
            s=sums.setdefault(new["case"]["load"], [0.,0.,0.,0.])
            for i,x in enumerate((new["baseline_ns"],new["median_ns"],old["baseline_ns"],old["median_ns"])):
                s[i]+=new["case"]["weight"]*x
        def gm(target):
            return math.exp(sum(math.log(x["speedup_ratio"]) for x in cases if x["target"]==target)/sum(x["target"]==target for x in cases))
        comparisons.append(dict(parent=pp, aggregate_speedup_ratio=provenance["speedup"]/pp["speedup"],
            target_case_ratio_geomean=gm(True), control_case_ratio_geomean=gm(False),
            load_results={k:dict(speedup_ratio=(cb/ct)/(pb/pt),normalized_latency_ratio=ct/pt)
                          for k,(cb,ct,pb,pt) in sums.items()}, cases=cases))
    launch_counts=Counter()
    for c in comparisons[0]["cases"]:
        for k in c["candidate_kernels"]:
            if "prefix" in k:
                launch_counts[(k["role"], k["directory_selected_from_prepare"])]+=k["calls"]
    report=dict(candidate=provenance, preparation_selections=activation,
        observed_launch_counts=[dict(role=k[0],directory_selected=k[1],calls=v) for k,v in launch_counts.items()],
        comparisons=comparisons,
        decision="REJECT promotion and combination with KPERM: mechanism active but no useful whole-decoder gain.",
        limitations=["Fresh weights, seeds, machine nodes, baseline timings and autotuning differ between jobs.",
                     "Kernel names are truncated and omit later template arguments; dispatch equality means only the names/calls retained in official reports.",
                     "Profiler rows can include PDL dependency waits and cannot be summed as independent latency savings.",
                     "Preparation records prove selected table geometry and enlarged pmax, not measured register counts, spills, occupancy or memory transactions.",
                     "Logical work surplus is an immutable source heuristic, not measured bytes or a latency bound."])
    path=ROOT/"experiments/iterations/20261104-dense-stage-directory-result-audit.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(saved=str(path.relative_to(ROOT)),candidate_score=provenance["speedup"],
        launch_counts=report["observed_launch_counts"],
        comparisons=[dict(job=x["parent"]["job"],score=x["parent"]["speedup"],ratio=x["aggregate_speedup_ratio"],target=x["target_case_ratio_geomean"],control=x["control_case_ratio_geomean"],loads=x["load_results"],
            cases=[{k:v for k,v in c.items() if k in ("index","tokens","layer","candidate_normalized_us","parent_normalized_us","speedup_ratio","same_reported_dispatch")} for c in x["cases"]]) for x in comparisons]),indent=2))


if __name__ == "__main__":
    main()
