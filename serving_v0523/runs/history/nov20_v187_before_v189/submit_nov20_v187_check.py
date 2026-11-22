#!/usr/bin/env python3
"""Offer the final private slot while V132's comparator is still queued or running after 04:00 UTC."""
import argparse
import fcntl
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import time

import submit_nov20_v180_check as v180

earlier, slots, first, measured = v180.earlier, v180.slots, v180.first, v180.measured
sibling = v180.fallback.previous
ROOT, RUNS, RESET = first.ROOT, first.RUNS, first.RESET
START, END = first.START.replace(hour=4, minute=0), first.END.replace(hour=8, minute=0)
V187 = measured.Candidate("v187", "v187_paged_verify_paired_prompt",
    "b11c67af71472e8d899fa1b97d103a65bc4b1d8d59cd60991f87fe9c6a9f5d2a",
    v180.nov19.V68.flags | {"--speculative-dflash-sp-prefill-context": ""})
TREE = "179d3985bd13196ffee8787a0d034c4c4876836f"
PARENT = "35b2086ec900233547ca35a6c2103ec9d5201d2f82818af91015421476d8ca30"
# Both names are intentional: existing V132 and V154c workers honor the durable
# slot claim if a CLI times out, including after restart. The header is always V187.
NAME = "v187_in_v132_standby_slot_nov20_check16"
OWN = (V187, 16, NAME, START, END)
REQUIRED = (sibling.FIRST, sibling.V141, sibling.V120B, v180.REFERENCE)
COMPETITORS = (sibling.V132, sibling.OWN, v180.fallback.OWN, v180.peer.OWN, earlier.OWN, v180.OWN)


def plan(root=ROOT):
    base = root / "serving_v0523/candidates"
    for candidate in (V187, *(r[0] for r in REQUIRED)):
        if first.digest(base / candidate.directory / "patch.diff") != candidate.sha256:
            raise ValueError("frozen patch mismatch for " + candidate.name)
    manifest = json.loads((base/V187.directory/"validation.json").read_text())
    if (manifest.get("tree") != TREE or manifest.get("patch_sha256") != V187.sha256
            or manifest.get("parent_sha256") != PARENT or manifest.get("launch_flags") != V187.flags
            or manifest.get("strict_apply_reverse_full_delta") is not True
            or manifest.get("strict_named_donor_reconstruction") is not True):
        raise ValueError("V187 export or launch flags changed")
    comparison = v180._reference(json.loads((root/"serving_v0523/runs"/(v180.nov19.V68_JOB+".json")).read_text()))
    if comparison is None:
        raise ValueError("V68 archived private comparison changed")
    return dict(candidate=V187.name, flags=V187.flags, tree=TREE, sha256=V187.sha256, mode="check", users=16,
                start_utc=START.isoformat(), end_utc_exclusive=END.isoformat(), resets_at=RESET,
                required_quota=slots.THIRD, maximum_private_checks=1, arbitration_lock=slots.LOCK,
                last_observations=["fresh aggregate private quota", "exact still-pending V120b"],
                max_preflight_seconds=120, max_final_observations_seconds=30,
                priority="Any V120b completion before the final parent observation yields the slot.",
                preceding_in_order=[r[2] for r in REQUIRED[:2]], pending_control=REQUIRED[2][2],
                requires_unattempted=[r[2] for r in COMPETITORS], receipt=NAME+".submit.txt", comparison=comparison)


def unexpected_attempt(runs):
    allowed = {runs/(r[2]+".submit.txt") for r in (*REQUIRED[:2], OWN)}
    for path in runs.rglob("*"):
        if path in allowed or not path.is_file():
            continue
        name=path.name.lower()
        if name.endswith((".submit.txt",".submit.json",".submitted_at.txt")):
            if "nov20" in name:
                return True
            try:
                with path.open() as f:
                    stamp=datetime.fromisoformat(json.loads(f.readline())["attempted_at"])
                if stamp.tzinfo is not None and first.START.replace(hour=0,minute=0)<=stamp<END:
                    return True
            except (KeyError, TypeError, ValueError):
                pass
    return False


def local_position(runs):
    # The exact CLI receipts authenticate users when the live queued API omits it.
    ids = tuple(slots.bound_receipt(runs,r) for r in REQUIRED)
    if not all(ids) or len(set(ids)) != len(ids) or ids[-1] != v180.nov19.V68_JOB or unexpected_attempt(runs):
        return None
    for r in COMPETITORS:
        if slots.prior_attempt(runs, r[0], NAME, exclude_own=True):
            return None
    try:
        attempt_times = []
        for req in REQUIRED[:2]:
            with (runs/(req[2]+".submit.txt")).open() as receipt:
                attempt_times.append(datetime.fromisoformat(json.loads(receipt.readline())["attempted_at"]))
        if not attempt_times[0] <= attempt_times[1]:
            return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    evidence = earlier.receipt_evidence(runs, REQUIRED[1])
    if (not isinstance(evidence,dict) or type(evidence.get("slot")) is not int or evidence["slot"] != 2
            or evidence.get("v129_job") != ids[0] or "v132_job" not in evidence or evidence["v132_job"] is not None):
        return None
    return ids


def live_evidence(runs, ids, okc, execute, clock):
    if ids is None:
        return None
    stamps = {}
    comparison = None
    # Read the pending control *last* so that V132 gets priority if it finishes
    # while we are checking the two consumed slots or the frozen comparator.
    for i in (0, 1, 3, 2):
        req, job = REQUIRED[i], ids[i]
        res = execute([okc,"serve","status",job,"--json"],capture_output=True,text=True,timeout=60)
        if res.returncode:
            return None
        data = json.loads(res.stdout)
        if not isinstance(data,dict) or data.get("refunded",False) is not False:
            return None
        identity = v180.fallback.official(req, job, okc, lambda *args,**kwargs:res, runs)
        if identity is None or identity[2] > clock().timestamp():
            return None
        stamps[i] = identity[2]
        if i == 2 and (identity[0] != "pending" or data.get("status") not in ("queued","running")
                       or data.get("result") is not None):
            return None
        if i == 3:
            comparison = v180._reference(data)
            if comparison is None:
                return None
    if not stamps[0] <= stamps[1]:
        return None
    return dict(slot=3,v129_job=ids[0],v141_job=ids[1],pending_v120b_job=ids[2],
                previously_consumed=[dict(job=ids[i], created_at=stamps[i]) for i in (0,1)], **comparison)


def run_once(*,okc,root=ROOT,runs=RUNS,clock=first.utc_now,execute=subprocess.run):
    began=clock()
    deadline=min(END,began+timedelta(seconds=120))
    final_deadline=None
    def inside():
        now=clock()
        return START <= now < deadline and (final_deadline is None or now < final_deadline)
    def deny(cmd):
        return subprocess.CompletedProcess(cmd,1,"","V187 standby unavailable")
    def bounded(cmd,**kw):
        now=clock()
        if not inside():
            return deny(cmd)
        limit=deadline if final_deadline is None else min(deadline, final_deadline)
        if now >= limit:
            return deny(cmd)
        return execute(cmd,**(kw | {"timeout":min(kw["timeout"],(limit-now).total_seconds())}))
    if not inside():
        return "outside submission window"
    with slots.arbitrate(runs) as held:
        if not held:
            return "another allocation poll is active"
        def prior(own=False):
            return slots.prior_attempt(runs,V187,NAME,own)
        if prior():
            return "attempt recorded"
        position=local_position(runs)
        proof=live_evidence(runs,position,okc,bounded,clock)
        if proof is None:
            return "waiting for V129 and V141 acceptances, V68 pass and pending V120b"
        plan(root)
        patch=root/"serving_v0523/candidates"/V187.directory/"patch.diff"
        status=[okc,"serve","status","--json"]
        expected=[okc,"serve","check",str(patch),"--users","16"]
        for key,value in sorted(V187.flags.items()):
            expected.extend(("--flag",key[2:]+("="+value if value else "")))
        expected.append("--no-wait")

        def quota(**kw):
            res=bounded(status, **kw)
            if res.returncode:
                return res
            data=json.loads(res.stdout)
            counts=data["limits"]["checks"]
            if (data.get("resets_at")!=RESET or not isinstance(counts,dict) or
                    any(type(counts.get(k)) is not int or counts[k]!=v for k,v in slots.THIRD.items())):
                return deny(status)
            return res

        def unchanged(own=False):
            return inside() and not prior(own) and local_position(runs)==position

        def guarded(cmd,**kw):
            nonlocal final_deadline
            if cmd==status:
                res=quota(**kw)
                if res.returncode or not unchanged() or live_evidence(runs,position,okc,bounded,clock)!=proof:
                    return deny(cmd)
                plan(root)
                if not unchanged():
                    return deny(cmd)
                final_deadline=min(deadline,clock()+timedelta(seconds=30))
                res=quota(**kw)
                if res.returncode:
                    return deny(cmd)
                quota_time=clock()
                control_job=position[2]
                live=bounded([okc,"serve","status",control_job,"--json"],**kw)
                if live.returncode:
                    return deny(cmd)
                data=json.loads(live.stdout)
                identity=v180.fallback.official(REQUIRED[2],control_job,okc,lambda *a,**k:live,runs)
                if (identity is None or identity[0]!="pending" or identity[2]>clock().timestamp()
                        or data.get("status") not in ("queued","running") or data.get("result") is not None
                        or data.get("refunded",False) is not False or not unchanged()):
                    return deny(cmd)
                proof.update(final_quota={"observed_at":quota_time.isoformat(),"checks":dict(slots.THIRD),"resets_at":RESET},
                             final_control={"observed_at":clock().isoformat(),"job":control_job,"status":data["status"],
                                            "created_at":identity[2]})
                return res
            if cmd!=expected:
                raise ValueError("unexpected V187 operation")
            if not unchanged(True) or first.digest(patch)!=V187.sha256:
                return deny(cmd)
            now=clock()
            if not inside():
                return deny(cmd)
            # Freshness limits the moment we authorize the command. An already
            # authorized CLI can use its usual delivery timeout up to the cutoff.
            return execute(cmd,**(kw | {"timeout":min(kw["timeout"],(END-now).total_seconds())}))

        result=measured.submit_once((NAME,V187,16,proof),okc=okc,root=root,runs=runs,now=clock(),
                                    execute=guarded,start=START,end=END,reset=RESET,reserve_checks=0)
        if result.startswith("submitted "):
            job=result.split()[1]
            if job in position or slots.bound_receipt(runs,OWN)!=job:
                return "attempt needs manual reconciliation; no retry"
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    frozen=plan()
    if not args.run:
        print(json.dumps(frozen,indent=2))
        return
    okc=os.environ.get("OKC",str(Path.home()/".okc/bin/okc"))
    with (RUNS/"nov20_v187_runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        print(json.dumps(dict(armed_at=first.utc_now().isoformat(),**frozen)),flush=True)
        last=None
        while first.utc_now()<END:
            if first.utc_now()>=START:
                try:
                    state=run_once(okc=okc)
                except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as err:
                    state=f"{type(err).__name__}: {err}"
                if state!=last:
                    print(json.dumps(dict(at=first.utc_now().isoformat(),outcome=state)),flush=True)
                    last=state
                if state.startswith(("submitted ","attempt ","prior attempt","outside submission window")):
                    break
            time.sleep(min(60,max(0,(END-first.utc_now()).total_seconds())))
        print(json.dumps(dict(finished_at=first.utc_now().isoformat())),flush=True)


if __name__=="__main__":
    main()
