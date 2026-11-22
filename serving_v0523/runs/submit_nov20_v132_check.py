#!/usr/bin/env python3
"""Offer V132 in an available November 20 slot after the measured V120b control passes."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path

from submit_nov17_integrated_checks import Candidate, ROOT, RUNS, submit_once
from submit_nov19_decode_checks import V120b, prefix
import submit_nov19_decode_checks as previous
from nov20_check_slots import accepted_receipt
import submit_nov20_v129_check as first
import nov20_check_slots as slots

START, END, RESET = first.START, first.END, first.RESET
V132 = Candidate("v132","v132_verify_scheduler_metadata",
                 "45c62ddfc33bb0060b836a235367f6bbb6dba08fd4c98eddcaa6c880fcbbc4d0",
                 V120b.flags | {"--speculative-dflash-share-verify-schedule":""})
NAME = "v132_nov20_check32"
PEER = (slots.V141, 32, slots.V141_NAME, slots.GRACE, END)
PARENTS = ((first.V129,1,first.NAME),(V120b,32,prefix(V120b)))
PARENT_CHECKS = ((*PARENTS[0],START,END), (*PARENTS[1],previous.START,previous.END))


def prior_attempt(runs, exclude_own=False):
    return slots.prior_attempt(runs, V132, NAME, exclude_own)


def allocation(runs):
    peer_job = slots.dated_receipt(runs, PEER)
    if peer_job:
        return 3, peer_job
    if slots.prior_attempt(runs, slots.V141, slots.V141_NAME):
        return None
    return 2, None


def positive(value):
    return type(value) in (float,int) and math.isfinite(value) and value>0


def live_gate(candidate,users,job,okc,execute,*,runs=None):
    response=execute([okc,"serve","status",job,"--json"],capture_output=True,text=True,timeout=60)
    if response.returncode:
        return False
    is_first=candidate==first.V129
    req=PARENT_CHECKS[0] if is_first else PARENT_CHECKS[1]
    if (candidate,users)!=req[:2]:
        return False
    state=slots.official_state(req,job,okc,lambda *a,**kw:response,runs=runs)
    if is_first:
        return state in ("pending","passed","failed")
    if state!="passed":
        return False
    data=json.loads(response.stdout)
    result=data.get("result") or {}
    rungs=result.get("rungs")
    if not isinstance(rungs,list) or len(rungs)!=1 or not isinstance(rungs[0],dict):
        return False
    rung=rungs[0]
    correctness=result.get("correctness")
    return (data["status"]=="done" and isinstance(correctness,dict) and correctness.get("passed") is True
            and type(rung.get("concurrency")) is int and rung["concurrency"]==32
            and rung.get("truncated") is False and type(rung.get("reps")) is int and rung["reps"]>0
            and all(positive(rung.get(k)) for k in ("tok_s_per_gpu","tok_s_per_user_e2e","tok_s_per_user_p50")))


def plan(root=ROOT,runs=RUNS):
    base=root/"serving_v0523/candidates"
    for candidate in (first.V129,V120b,V132):
        directory=base/candidate.directory
        val=json.loads((directory/"validation.json").read_text())
        if (first.digest(directory/"patch.diff")!=candidate.sha256 or val["patch_sha256"]!=candidate.sha256
                or val["launch_flags"]!=candidate.flags):
            raise ValueError("frozen patch or flags mismatch: "+candidate.name)
    if val["parent_sha256"]!=V120b.sha256 or val["fixed_flag_patterns"]!=76:
        raise ValueError("V132 parent or fixed-flag validation mismatch")
    return dict(mode="check",candidate=V132.name,users=32,sha256=V132.sha256,flags=V132.flags,
                required_first=first.NAME,required_first_sha256=first.V129.sha256,
                required_passing_control=prefix(V120b),required_control_sha256=V120b.sha256,
                start_utc=START.isoformat(),end_utc_exclusive=END.isoformat(),resets_at=RESET,
                maximum_private_checks=1,required_quota={"per_day":3,"used":1,"left":2},
                reserved_private_checks=1, alternate_last_slot={"quota":slots.THIRD,"reserve_checks":0,
                    "requires_accepted":slots.V141_NAME,"sha256":slots.V141.sha256,
                    "earliest_peer_utc":slots.GRACE.isoformat()}, arbitration_lock=slots.LOCK, receipt=NAME+".submit.txt")


def run_once(*,okc,root=ROOT,runs=RUNS,clock=first.utc_now,execute=subprocess.run):
    if not START <= clock() < END:
        return "outside submission window"
    with slots.arbitrate(runs) as acquired:
        if not acquired:
            return "another allocation poll is active"
        return _offer(okc=okc, root=root, runs=runs, clock=clock, execute=execute)


def _offer(*, okc, root, runs, clock, execute):
    def in_window():
        return START<=clock()<END

    def bounded(cmd,**kwargs):
        remaining=(END-clock()).total_seconds()
        if remaining<=0 or not in_window():
            return subprocess.CompletedProcess(cmd,1,"","check window closed")
        return execute(cmd,**(kwargs|{"timeout":min(remaining,kwargs["timeout"])}))

    if not in_window():
        return "outside submission window"
    if prior_attempt(runs):
        return "attempt recorded"
    plan(root,runs)
    position=allocation(runs)
    if position is None:
        return "waiting for an exact V141 acceptance receipt"
    slot, peer_job = position
    quota_required = slots.SECOND if slot == 2 else slots.THIRD
    def receipt_ids():
        return tuple(slots.bound_receipt(runs,req) for req in PARENT_CHECKS)
    ids=receipt_ids()
    if not all(ids) or ids[0]==ids[1] or peer_job in ids:
        return "waiting for distinct confirmed V129 and V120b receipts"
    for (candidate,users,_),job in zip(PARENTS,ids):
        if not live_gate(candidate,users,job,okc,bounded,runs=runs):
            return "waiting for "+candidate.name+" official evidence"
    def eligible():
        return (allocation(runs) == position and (not peer_job or
                slots.official_state(PEER, peer_job, okc, bounded, runs=runs) in ("pending","passed","failed")))
    if not eligible():
        return "waiting for confirmed V141 private allocation"
    patch=root/"serving_v0523/candidates"/V132.directory/"patch.diff"
    args=[a for k,v in sorted(V132.flags.items()) for a in ("--flag",k[2:]+("="+v if v else ""))]
    expected=[okc,"serve","check",str(patch),"--users","32",*args,"--no-wait"]

    def guarded(cmd,**kwargs):
        if cmd==[okc,"serve","status","--json"]:
            response=bounded(cmd,**kwargs)
            if response.returncode:
                return response
            try:
                quota=json.loads(response.stdout)
                counts=quota["limits"]["checks"]
                valid=(quota["resets_at"]==RESET and all(type(counts[k]) is int and counts[k]==n
                       for k,n in quota_required.items()))
            except (KeyError,TypeError,ValueError):
                valid=False
            if (valid and in_window() and ids==receipt_ids() and not prior_attempt(runs) and eligible()
                    and live_gate(V120b,32,ids[1],okc,bounded,runs=runs)):
                return response
            return subprocess.CompletedProcess(cmd,1,"","V132 allocation not confirmed")
        if cmd!=expected:
            raise ValueError("command outside the private V132/32 allocation")
        if (prior_attempt(runs,exclude_own=True) or ids!=receipt_ids() or allocation(runs)!=position
                or first.digest(patch)!=V132.sha256):
            return subprocess.CompletedProcess(cmd,1,"","frozen evidence changed before submission")
        return bounded(cmd,**kwargs)

    result=submit_once((NAME,V132,32,{"first_check_job":ids[0],"comparison_job":ids[1],
                                     "comparison_patch":V120b.sha256} | ({"v141_job": peer_job, "slot": 3} if peer_job else {})),
                       okc=okc,now=clock(),runs=runs,root=root,execute=guarded,start=START,end=END,
                       reset=RESET,reserve_checks=3-slot)
    if result.startswith("submitted ") and accepted_receipt(runs,V132,32,NAME)!=result.split()[1]:
        return "attempt needs manual reconciliation; no retry"
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    run=parser.parse_args().run
    frozen=plan()
    if not run:
        print(json.dumps(frozen,indent=2))
        return
    okc=os.environ.get("OKC",str(Path.home()/".okc/bin/okc"))
    with (RUNS/"nov20_v132_runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another V132 runner owns the lock.",flush=True)
            return
        print(json.dumps({"armed_at":first.utc_now().isoformat(),**frozen}),flush=True)
        previous=None
        while first.utc_now()<END:
            if first.utc_now()>=START:
                try:
                    state=run_once(okc=okc)
                except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as exc:
                    state=f"{type(exc).__name__}: {exc}"
                if state!=previous:
                    print(json.dumps({"at":first.utc_now().isoformat(),"result":state}),flush=True)
                    previous=state
                if state.startswith(("attempt ","prior attempt","submitted ")) or state=="outside submission window":
                    break
            time.sleep(min(60,max(0,(END-first.utc_now()).total_seconds())))
        print(json.dumps({"finished_at":first.utc_now().isoformat()}),flush=True)


if __name__=="__main__":
    main()
