#!/usr/bin/env python3
"""Offer one matched V173 check after V163 used the otherwise terminal second-slot branch."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path

import submit_nov20_v163_check as fallback

slots, first, measured = fallback.slots, fallback.first, fallback.measured
nov19 = slots.previous
ROOT, RUNS = fallback.ROOT, fallback.RUNS
START, END, RESET = fallback.START, fallback.END, fallback.RESET
V168 = nov19.V168
V173 = measured.Candidate("v173", "v173_medium_expert_tma",
    "7d1bffd4526ea77fc2567479d4f569f673ba82062c342a01817c5b5562f11c4a", V168.flags)
PARENT_SHA = "95816b34347ec23816b3def52e99265449ce0c8df523e0a36f587603ae57b574"
NAME = "v173_nov20_check16"
OWN = (V173, 16, NAME, START, END)
PARENT = (V168, 16, nov19.prefix(V168), nov19.START, nov19.END)
REQUIRED = (*fallback.REQUIRED, fallback.OWN, PARENT)


def plan(root=ROOT):
    base = root/"serving_v0523/candidates"
    if V168.sha256 != PARENT_SHA or V173.flags != fallback.V163.flags:
        raise ValueError("V168 parent SHA or matched serving flags changed")
    for candidate in (V173, *(req[0] for req in REQUIRED)):
        if first.digest(base/candidate.directory/"patch.diff") != candidate.sha256:
            raise ValueError("frozen patch mismatch for " + candidate.name)
    manifests = [json.loads((base/c.directory/"validation.json").read_text()) for c in (V168,V173)]
    for candidate, manifest in zip((V168,V173), manifests, strict=True):
        if manifest["patch_sha256"] != candidate.sha256 or manifest["launch_flags"] != V173.flags:
            raise ValueError("frozen manifest mismatch for " + candidate.name)
    parent, child = manifests
    if (child["parent_sha256"] != PARENT_SHA or child["parent_tree"] != parent["tree"]
            or child["strict_apply_reverse_full_delta"] is not True or child["checked_flag_patterns"] != 76):
        raise ValueError("V173 source does not match the frozen V168 lineage")
    return dict(candidate=V173.name, sha256=V173.sha256, flags=V173.flags, mode="check", users=16,
        first_two_private_checks=[REQUIRED[0][2],fallback.NAME], terminal_failed_controls=[q[2] for q in REQUIRED[1:3]],
        measured_parent=dict(receipt=PARENT[2],sha256=PARENT_SHA,users=16,require="passed correct, complete nontruncated rung"),
        requires_unattempted=[q[2] for q in fallback.PEERS], required_quota=slots.THIRD, maximum_private_checks=1,
        start_utc=START.isoformat(),end_utc_exclusive=END.isoformat(),resets_at=RESET,
        arbitration_lock=slots.LOCK, receipt=NAME+".submit.txt")


def local_position(runs):
    # local_position enforces every competing slot-two/three waiter is unattempted.
    base = fallback.local_position(runs)
    if base is None:
        return None
    preceding = (*base, slots.bound_receipt(runs,fallback.OWN), slots.bound_receipt(runs,PARENT))
    if not all(preceding) or len(set(preceding)) != len(preceding):
        return None
    evidence = json.loads((runs/(fallback.NAME+".submit.txt")).read_text().splitlines()[0]).get("evidence")
    if (not isinstance(evidence,dict) or type(evidence.get("slot")) is not int or evidence["slot"] != 2
            or evidence.get("v129_job") != base[0]
            or evidence.get("failed_controls") != dict(v120b=base[1],v139=base[2])):
        return None
    return preceding


def official_evidence(runs, ids, okc, execute):
    if ids is None:
        return None
    times = {}
    for req, job in zip(REQUIRED, ids, strict=True):
        cmd = [okc,"serve","status",job,"--json"]
        response = execute(cmd, capture_output=True, text=True, timeout=60)
        if response.returncode:
            return None
        data = json.loads(response.stdout)
        if not isinstance(data,dict):
            return None
        # Both preceding jobs must still count toward the November 20 counter.
        if req in (REQUIRED[0],fallback.OWN,PARENT) and data.get("refunded",False) is not False:
            return None
        identity = fallback.official(req,job,okc,lambda *a,**kw:response,runs)
        if identity is None or req in REQUIRED[1:3] and not identity[1]:
            return None
        times[job] = identity[2]
        if req == PARENT:
            if data["status"] != "done" or identity[0] != "passed":
                return None
            result = data["result"]
            rungs, correctness = result.get("rungs"), result.get("correctness")
            if not isinstance(correctness,dict) or correctness.get("passed") is not True:
                return None
            if "short_outputs" in result and (type(result["short_outputs"]) is not int or result["short_outputs"]!=0):
                return None
            if "later_turns" in result:
                later = result["later_turns"]
                if (not isinstance(later,dict) or later.get("ran") is not True or later.get("passed") is not True
                        or later.get("incomplete",False) is not False or later.get("insufficient",False) is not False):
                    return None
            if not isinstance(rungs,list) or len(rungs) != 1 or not isinstance(rungs[0],dict):
                return None
            rung = rungs[0]
            if (type(rung.get("concurrency")) is not int or rung["concurrency"] != 16
                    or rung.get("truncated") is not False or type(rung.get("reps")) is not int or rung["reps"]<1):
                return None
            metrics = {k:rung.get(k) for k in ("tok_s_per_gpu","tok_s_per_user_e2e","tok_s_per_user_p50")}
            rates_and_times = [*metrics.values(), *(v for k,v in rung.items()
                               if k.startswith(("tok_s_", "ttft_")) and k not in metrics)]
            if not all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in rates_and_times):
                return None
    if times[ids[0]] > times[ids[3]]:
        return None
    return dict(slot=3,v129_job=ids[0],failed_controls=dict(v120b=ids[1],v139=ids[2]),v163_job=ids[3],
                comparison_job=ids[4],comparison_sha256=PARENT_SHA,comparison_users=16,
                comparison_reps=rung["reps"],comparison_metrics=metrics,comparison_rung=rung,
                comparison_node=result.get("node"),comparison_cluster=data.get("cluster"),
                comparison_vs_stock=result.get("vs_stock"))


def run_once(*,okc,root=ROOT,runs=RUNS,clock=first.utc_now,execute=subprocess.run):
    def inside():
        return START <= clock() < END
    def denied(cmd):
        return subprocess.CompletedProcess(cmd,1,"","V173 private slot or comparator unavailable")
    def bounded(cmd,**kw):
        now = clock()
        if not START <= now < END:
            return denied(cmd)
        return execute(cmd,**(kw|{"timeout":min(kw["timeout"],(END-now).total_seconds())}))
    if not inside():
        return "outside submission window"
    with slots.arbitrate(runs) as acquired:
        if not acquired:
            return "another allocation poll is active"
        def attempted(exclude_own=False):
            return slots.prior_attempt(runs,V173,NAME,exclude_own)
        if attempted():
            return "attempt recorded"
        plan(root)
        ids = local_position(runs)
        proof = official_evidence(runs,ids,okc,bounded)
        if proof is None:
            return "waiting for V163 in slot two and measured V168/16"
        patch = root/"serving_v0523/candidates"/V173.directory/"patch.diff"
        status = [okc,"serve","status","--json"]
        expected = [okc,"serve","check",str(patch),"--users","16"]
        for key,value in sorted(V173.flags.items()):
            expected.extend(("--flag",key[2:] + ("="+value if value else "")))
        expected.append("--no-wait")
        def fresh_quota(**kw):
            response = bounded(status,**kw)
            if response.returncode:
                return response
            data = json.loads(response.stdout)
            quota = data["limits"]["checks"]
            if (data.get("resets_at") != RESET or not isinstance(quota,dict)
                    or any(type(quota.get(k)) is not int or quota[k]!=v for k,v in slots.THIRD.items())):
                return denied(status)
            return response

        def still_local():
            return inside() and not attempted() and local_position(runs)==ids

        def guarded(cmd,**kw):
            if cmd == status:
                response = fresh_quota(**kw)
                if (response.returncode or not still_local()
                        or official_evidence(runs,ids,okc,bounded)!=proof or not still_local()):
                    return denied(cmd)
                plan(root)
                # All five official dependencies are settled before the last
                # aggregate read. Only local checks follow it before the request.
                response = fresh_quota(**kw)
                return response if not response.returncode and still_local() else denied(cmd)
            if cmd != expected:
                raise ValueError("unexpected V173 operation")
            if not inside() or attempted(True) or local_position(runs)!=ids or first.digest(patch)!=V173.sha256:
                return denied(cmd)
            return bounded(cmd,**kw)
        result = measured.submit_once((NAME,V173,16,proof),okc=okc,now=clock(),root=root,runs=runs,
                    execute=guarded,start=START,end=END,reset=RESET,reserve_checks=0)
        if result.startswith("submitted "):
            new_job = result.split()[1]
            if new_job in ids or slots.bound_receipt(runs,OWN)!=new_job:
                return "attempt needs manual reconciliation; no retry"
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    args = parser.parse_args()
    frozen = plan()
    if not args.run:
        print(json.dumps(frozen,indent=2))
        return
    okc = os.environ.get("OKC",str(Path.home()/".okc/bin/okc"))
    with (RUNS/"nov20_v173_runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        print(json.dumps(dict(armed_at=first.utc_now().isoformat(),**frozen)),flush=True)
        last = None
        while first.utc_now()<END:
            if first.utc_now()>=START:
                try:
                    state = run_once(okc=okc)
                except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
                    state = f"{type(exc).__name__}: {exc}"
                if state != last:
                    print(json.dumps(dict(at=first.utc_now().isoformat(),outcome=state)),flush=True)
                    last = state
                if state.startswith(("submitted ","attempt ","prior attempt","outside submission window")):
                    break
            time.sleep(min(60,max(0,(END-first.utc_now()).total_seconds())))
        print(json.dumps(dict(finished_at=first.utc_now().isoformat())),flush=True)


if __name__=="__main__":
    main()
