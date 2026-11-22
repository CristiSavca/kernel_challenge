#!/usr/bin/env python3
"""Offer V141 in the third November 20 slot or after the second-slot grace period."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from submit_nov17_integrated_checks import ROOT, RUNS, V64, submit_once
import submit_nov19_integrated_verify as parent
import submit_nov20_v129_check as first
import submit_nov20_v132_check as second
import nov20_check_slots as slots


START, END, RESET = first.START, first.END, first.RESET
V141, NAME = slots.V141, slots.V141_NAME
REQUIRED = ((first.V129, 1, first.NAME, START, END),
            (second.V132, 32, second.NAME, START, END),
            (parent.V139, 4, parent.NAME, parent.START, parent.END))



def plan(root=ROOT, runs=RUNS):
    base = root / "serving_v0523/candidates"
    for c in (V64, first.V129, second.V132, parent.V139, V141):
        directory = base / c.directory
        if first.digest(directory / "patch.diff") != c.sha256:
            raise ValueError("frozen patch mismatch for " + c.name)
        if c != V64:
            val = json.loads((directory / "validation.json").read_text())
            if val["patch_sha256"] != c.sha256 or val["launch_flags"] != c.flags:
                raise ValueError("frozen identity or flags mismatch for " + c.name)
    if (val["parent_sha256"] != parent.V139.sha256 or val["scored_anchor_sha256"] != V64.sha256
            or val["fixed_flag_patterns"] != 76 or not first.parent_gate(runs)):
        raise ValueError("V141 lineage or confirmed V64 reference mismatch")
    scored = json.loads((runs / (first.PARENT_JOB + ".json")).read_text())["result"]["rungs"]
    points = [r for r in scored if r["concurrency"] == 32 and r["truncated"] is False and second.positive(r["tok_s_per_gpu"])]
    if len(points) != 1:
        raise ValueError("missing complete 32-user V64 reference")
    return dict(mode="check", candidate=V141.name, sha256=V141.sha256, flags=V141.flags, users=32,
                comparison_job=first.PARENT_JOB, comparison_sha256=V64.sha256,
                v64_32user_tokens_s_gpu=points[0]["tok_s_per_gpu"], start_utc=START.isoformat(),
                end_utc_exclusive=END.isoformat(), resets_at=RESET, maximum_private_checks=1,
                required_quota=dict(per_day=3, used=2, left=1),
                preceding_reservations=[dict(candidate=c.name, sha256=c.sha256, users=u, receipt=n+".submit.txt")
                                        for c,u,n,_,_ in REQUIRED[:2]],
                parent=dict(candidate=parent.V139.name, sha256=parent.V139.sha256, users=4,
                            states=["queued","running","passed"], receipt=parent.NAME+".submit.txt"),
                alternate_second_slot={"start_utc":slots.GRACE.isoformat(),"quota":slots.SECOND,"reserve_checks":1,
                    "requires_unattempted":second.NAME,"prefer_V132_if_comparator_passed":True},
                arbitration_lock=slots.LOCK, receipt=NAME+".submit.txt")


def prior_attempt(runs, exclude_own=False):
    return slots.prior_attempt(runs, V141, NAME, exclude_own)


def receipt_ids(runs, required=REQUIRED):
    jobs = tuple(slots.bound_receipt(runs, req) for req in required)
    return jobs if all(jobs) and len(set(jobs)) == len(jobs) else None


def allocation(runs, now):
    ids = receipt_ids(runs)
    if ids:
        return 3, REQUIRED, ids
    if now < slots.GRACE or second.prior_attempt(runs):
        return None
    required = (REQUIRED[0], REQUIRED[2])
    ids = receipt_ids(runs, required)
    return (2, required, ids) if ids else None


def run_once(*, okc, root=ROOT, runs=RUNS, clock=first.utc_now, execute=subprocess.run):
    if not START <= clock() < END:
        return "outside submission window"
    with slots.arbitrate(runs) as acquired:
        if not acquired:
            return "another allocation poll is active"
        return _offer(okc=okc, root=root, runs=runs, clock=clock, execute=execute)


def _offer(*, okc, root, runs, clock, execute):
    def bounded(command, **kwargs):
        now = clock()
        if not START <= now < END:
            return subprocess.CompletedProcess(command,1,"","submission window closed")
        return execute(command, **(kwargs | {"timeout":min(kwargs["timeout"], (END-now).total_seconds())}))

    if not START <= clock() < END:
        return "outside submission window"
    if prior_attempt(runs):
        return "attempt recorded"
    plan(root,runs)
    position = allocation(runs, clock())
    if not position:
        return "waiting for eligible slot and exact acceptance receipts"
    slot, required, ids = position
    quota_required = slots.THIRD if slot == 3 else slots.SECOND

    def secondary_available():
        if slot == 3:
            return True
        candidate, users, name = second.PARENTS[1]
        comparator = second.accepted_receipt(runs, candidate, users, name)
        if not comparator:
            # A missing control does not reserve this slot; an unresolved actual attempt does.
            return not slots.prior_attempt(runs, candidate, name)
        if comparator in ids:
            return False
        req = (candidate, users, name, second.previous.START, second.previous.END)
        state = slots.official_state(req, comparator, okc, bounded, runs=runs)
        # A passing comparator has priority. Missing or ambiguous live evidence is retried.
        return state in ("pending", "failed")

    for req, job in zip(required,ids):
        state = slots.official_state(req,job,okc,bounded,runs=runs)
        if state is None:
            return "waiting for official identity of " + req[0].name
        if req[0] == parent.V139 and state == "failed":
            return "terminal V139 failure"
    if not secondary_available():
        return "waiting for V132 priority or confirmed control state"
    patch = root/"serving_v0523/candidates"/V141.directory/"patch.diff"
    expected = [okc,"serve","check",str(patch),"--users","32",*first.FLAGS,"--no-wait"]

    def guarded(cmd, **kwargs):
        if cmd == [okc,"serve","status","--json"]:
            response = bounded(cmd,**kwargs)
            if response.returncode:
                return response
            try:
                data = json.loads(response.stdout)
                checks = data["limits"]["checks"]
                valid = (data["resets_at"] == RESET and all(type(checks[k]) is int and checks[k] == v
                         for k,v in quota_required.items()))
            except (ValueError,TypeError,KeyError):
                valid = False
            if (valid and START <= clock() < END and allocation(runs, clock())==position and secondary_available() and not prior_attempt(runs)
                    and slots.official_state(REQUIRED[-1],ids[-1],okc,bounded,runs=runs) in ("passed","pending")):
                return response
            return subprocess.CompletedProcess(cmd,1,"","V141 allocation or parent unconfirmed")
        if cmd != expected:
            raise ValueError("unexpected command outside V141 private allocation")
        if (prior_attempt(runs,exclude_own=True) or allocation(runs, clock())!=position or first.digest(patch)!=V141.sha256
                or not first.parent_gate(runs)):
            return subprocess.CompletedProcess(cmd,1,"","frozen evidence changed")
        return bounded(cmd,**kwargs)

    result = submit_once((NAME,V141,32,dict(v129_job=ids[0],v132_job=ids[1] if slot == 3 else None,v139_job=ids[-1],slot=slot,
                         comparison_job=first.PARENT_JOB,comparison_sha256=V64.sha256)),
                         okc=okc,now=clock(),root=root,runs=runs,execute=guarded,start=START,end=END,reset=RESET,reserve_checks=3-slot)
    if result.startswith("submitted ") and second.accepted_receipt(runs,V141,32,NAME)!=result.split()[1]:
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
    with (RUNS/"nov20_v141_runner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(dict(armed_at=first.utc_now().isoformat(),**frozen)),flush=True)
        previous=None
        okc=os.environ.get("OKC",str(Path.home()/".okc/bin/okc"))
        while first.utc_now()<END:
            if first.utc_now()>=START:
                try:
                    state=run_once(okc=okc)
                except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as exc:
                    state=f"{type(exc).__name__}: {exc}"
                if state!=previous:
                    print(json.dumps(dict(at=first.utc_now().isoformat(),outcome=state)),flush=True)
                    previous=state
                if state.startswith(("submitted ","attempt ","prior attempt","terminal V139 failure","outside submission window")):
                    break
            time.sleep(min(60,max(0,(END-first.utc_now()).total_seconds())))
        print(json.dumps(dict(finished_at=first.utc_now().isoformat())),flush=True)


if __name__=="__main__":
    main()
