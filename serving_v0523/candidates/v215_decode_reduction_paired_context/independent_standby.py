"""Exercise the private V215 selection using archived schemas and a local mock only."""

import copy
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile


HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs"
STAGE = RUNS / "history/nov20_v215_staging"
sys.path[:0] = [str(STAGE), str(RUNS)]
import submit_nov20_v187_check as worker
import test_nov20_v187_check as incumbent
import v215_selection as selector


receipt = selector.V204_RECEIPT + ".submit.txt"
job = selector.V204_JOB
frozen = json.loads((RUNS / "04b01f958aa5487c.json").read_text())
created = datetime.fromisoformat("2026-11-19T00:03:10.589038+00:00").timestamp()


def improved():
    data = copy.deepcopy(frozen)
    data.update(id=job, serving_flags=worker.V204.flags, created_at=created,
                started_at=created+60, finished_at=created+3600)
    result = data["result"]
    result.update(serving_flags=worker.V204.flags, patch={"sha256": selector.V204_SHA})
    rung = result["rungs"][0]
    norm = result["vs_stock"]["rungs"][0]
    rung["tok_s_per_gpu"] = 18400.0
    norm.update(at_stock_point=1.490, tok_s_per_gpu=1.496)
    return data


def change(data, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        data = data[int(key) if isinstance(data, list) else key]
    data[keys[-1]] = value


def run_case(label, *, mutate=None, expect, prepare=None):
    with tempfile.TemporaryDirectory(prefix="okc-v215-offline-") as name:
        judge = incumbent.Judge(name)
        shutil.copyfile(RUNS / receipt, judge.runs / receipt)
        judge.jobs[job] = improved()
        if mutate:
            change(judge.jobs[job], *mutate)
        if prepare:
            prepare(judge)
        def poll():
            return worker.run_once(okc="mock", candidate=worker.V215, runs=judge.runs,
                                   clock=lambda: judge.now, execute=judge.execute)
        result = poll()
        candidate = None
        if judge.attempts:
            command, = judge.attempts
            details = json.loads(judge.receipt.read_text().splitlines()[0])
            candidate = details["candidate"]
            registered = {"v215": worker.V215, "v198": worker.V198}[candidate]
            expected = ["mock","serve","check",str(worker.ROOT/"serving_v0523/candidates"/registered.directory/"patch.diff"),"--users","16"]
            for key, val in sorted(registered.flags.items()):
                expected.extend(("--flag", key[2:] + ("=" + val if val else "")))
            assert command == expected + ["--no-wait"], label
            assert details["sha256"] == registered.sha256 and details["flags"] == registered.flags
            assert ("v204_comparison" in details["evidence"]) == (candidate == "v215")
            assert judge.events[-3:-1] == [["serve","status","--json"], ["serve","status",incumbent.IDS[2],"--json"]]
            assert poll() == "attempt recorded" and len(judge.attempts) == 1
        else:
            assert not judge.receipt.exists(), (label, result)
        assert candidate == expect, (label, candidate, expect, result)
        return candidate or "held"


def main():
    cases = {}
    def check(label, **kwargs):
        cases[label] = run_case(label, **kwargs)

    check("clear_win", expect="v215")
    check("different_node_same_cluster", mutate=("result.node", "another-dedicated-node"), expect="v215")
    check("negative_memory_delta", mutate=("result.gpu_memory.over_stock_gb", -1), expect="v215")
    check("memory_limit_inclusive", mutate=("result.gpu_memory.over_stock_gb", 2), expect="v215")
    neg = [
        ("raw_tie","result.rungs.0.tok_s_per_gpu",17798.36027),
        ("normalized_tie","result.vs_stock.rungs.0.at_stock_point",1.4472112),
        ("raw_improvement_only","result.vs_stock.rungs.0.at_stock_point",1.420),
        ("normalized_improvement_only","result.rungs.0.tok_s_per_gpu",17000),
        ("correctness_fail","result.correctness.passed",False),
        ("later_incomplete","result.later_turns.incomplete",True),
        ("short_generations","result.short_outputs",1),
        ("truncated","result.rungs.0.truncated",True),
        ("wrong_repetition","result.rungs.0.reps",3),
        ("wrong_weightset","result.weights","other"),
        ("different_image","result.image","other"),
        ("wrong_gpus","result.gpus",4),
        ("another_cluster","cluster","other"),
        ("shared_resource","dedicated",False),
        ("missing_physical_node","result.node"," "),
        ("different_corpus_geometry","result.shape.turns",14),
        ("missing_stock_match","result.vs_stock.rungs.0.stock",{}),
        ("no_memory_delta","result.gpu_memory.over_stock_gb",None),
        ("memory_overage","result.gpu_memory.over_stock_gb",2.01),
        ("limit_overridden","result.gpu_memory.limit_gb",3),
        ("bad_delta_type","result.gpu_memory.over_stock_gb",True),
        ("infinite_raw_rate","result.rungs.0.tok_s_per_gpu",math.inf),
        ("nan_raw_rate","result.rungs.0.tok_s_per_gpu",math.nan),
        ("infinite_at_stock","result.vs_stock.rungs.0.at_stock_point",math.inf),
        ("nan_at_stock","result.vs_stock.rungs.0.at_stock_point",math.nan),
        ("first_ttft_ratio_over","result.vs_stock.rungs.0.ttft_first_p90",2.001),
        ("later_ttft_ratio_over","result.vs_stock.rungs.0.ttft_later_p90",2.001),
        ("first_ttft_seconds_over","result.rungs.0.ttft_first_p90_s",170),
        ("later_ttft_seconds_over","result.rungs.0.ttft_later_p90_s",0.714),
        ("failed","status","failed"),("duplicate","duplicate_of","other"),
        ("refunded","refunded",True),("wrong_job","id","1"*16),
        ("wrong_patch","result.patch.sha256","a"*64),("bad_flags","serving_flags",{}),
        ("score_instead_of_private","mode","run"),("other_team","team","other"),
        ("future_finish","finished_at",worker.END.timestamp()+300),
    ]
    for label, path, val in neg:
        check(label, mutate=(path,val), expect="v198")
    for metric,path,limit,reported_floor,reported_pass in [
        ("raw","result.rungs.0.tok_s_per_gpu",17798.36027,17798.3,17798.4),
        ("normalized","result.vs_stock.rungs.0.at_stock_point",1.4472112,1.447,1.448),
    ]:
        for label,val,want in (
                ("one_ulp_below", math.nextafter(limit,-math.inf), "v198"),
                ("one_ulp_above", math.nextafter(limit,math.inf), "v215"),
                ("official_rounded_below",reported_floor,"v198"),
                ("official_rounded_above",reported_pass,"v215")):
            check(metric+"_"+label,mutate=(path,val),expect=want)
    for state in ("queued","running"):
        def pending(judge, state=state):
            judge.jobs[job].update(status=state, result=None, finished_at=None)
        check(state,prepare=pending,expect="v198")
    check("missing_receipt",prepare=lambda j:(j.runs/receipt).unlink(),expect="v198")
    check("conflicting_receipt",prepare=lambda j:shutil.copyfile(j.runs/receipt,j.runs/"v204_copy_nov19.submit.txt"),expect="v198")
    for field,val in (("used",1),("left",2),("per_day",4),("used",True)):
        check("wrong_quota_"+field+"_"+str(val),
              prepare=lambda j,field=field,val=val:j.quota["limits"]["checks"].update({field:val}),expect=None)
    def quota_flip(judge):
        judge.on_quota=lambda i:judge.quota["limits"]["checks"].update(used=3,left=0) if i==2 else None
    check("last_quota_consumed", prepare=quota_flip,expect=None)
    def control_flip(judge):
        judge.on_quota=lambda i:judge.jobs[incumbent.IDS[2]].update(status="done") if i==2 else None
    check("v120b_finished_at_last_quota",prepare=control_flip,expect=None)
    check("ambiguous_peer_receipt",prepare=lambda j:(j.runs/(worker.COMPETITORS[0][2]+".submit.txt")).write_text("pending"),expect=None)
    check("outside_window",prepare=lambda j:setattr(j,"now",worker.END),expect=None)
    def selection_flip(judge):
        judge.on_quota=lambda i:judge.jobs[job]["result"]["rungs"][0].update(tok_s_per_gpu=16000) if i==1 else None
    check("v204_win_disappeared_during_preflight",prepare=selection_flip,expect=None)
    def remove_receipt_at_last_quota(judge):
        judge.on_quota=lambda i:(judge.runs/receipt).unlink() if i==2 else None
    check("v204_receipt_lost_at_last_quota",prepare=remove_receipt_at_last_quota,expect=None)
    inputs = [Path(worker.__file__),Path(selector.__file__),RUNS/receipt,
              RUNS/"04b01f958aa5487c.json",RUNS/"abd217baf0ed49fd.json",HERE/"patch.diff"]
    print(json.dumps({"source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                      "physical_result":"Simulated V204 alongside archived genuine V68/V64 schemas.",
                      "outcomes":cases,"total":len(cases)},indent=2))


if __name__=="__main__":
    main()
