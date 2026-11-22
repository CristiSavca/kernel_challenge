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
import sys
from types import SimpleNamespace

# Also resolves the established gate modules when this file is run from its staging directory.
if not (Path(__file__).parent / "submit_nov20_v180_check.py").is_file():
    sys.path.insert(1, str(Path(__file__).resolve().parents[2]))
sys.path.insert(2, str(Path(__file__).resolve().parent.parent / 'nov20_v229_staging'))
import v229_selection
import scored_receipt

import submit_nov20_v180_check as v180

earlier, slots, first, measured = v180.earlier, v180.slots, v180.first, v180.measured
sibling = v180.fallback.previous
ROOT, RUNS, RESET = first.ROOT, first.RUNS, first.RESET
START, END = first.START.replace(hour=4, minute=0), first.END.replace(hour=8, minute=0)
V187 = measured.Candidate("v187", "v187_paged_verify_paired_prompt",
    "b11c67af71472e8d899fa1b97d103a65bc4b1d8d59cd60991f87fe9c6a9f5d2a",
    v180.nov19.V68.flags | {"--speculative-dflash-sp-prefill-context": ""})
V189 = measured.Candidate("v189", "v189_decode_grouped_mma", "18cf2315fa01dc568e28b178f92dad69386191a23ba4834a6616b65fe6eac597", V187.flags)
V198 = measured.Candidate("v198", "v198_segmented_verify_prompt_experts", "fa91157f4e5e3c9704664047ba177af4ddabf2f5d2ab0af2560e1320902a59e4", V187.flags)
MIXED_FLAGS = V198.flags | {"--speculative-dflash-mixed-prefill": ""}
V204 = measured.Candidate("v204", "v204_mixed_prompt_weight_pipeline", v229_selection.V204_SHA, MIXED_FLAGS)
V229 = measured.Candidate("v229", "v229_sharded_decode_serving", v229_selection.V229_SHA, MIXED_FLAGS)
CANDIDATES = {candidate.name: candidate for candidate in (V187, V189, V198, V229)}
# The prior selection may already have an ambiguous or renamed receipt.
V215 = measured.Candidate("v215", "v215_decode_reduction_paired_context",
    "bbf69a796f2e2ae2edb2b5d913f0e0dc799037d555c10dfe58d00eb71f7cd7b6", MIXED_FLAGS)
PRIOR_CHOICES = (V187, V189, V198, V215, V229)
V198_TREE = "927369031684bbe39bd46354b552a1a32cbfbb2f"
V197_SHA = "0974f9542bd481e7c15fbce9331cd041966ebf4e4813e7d1640defcc728b2f0e"
V197_TREE = "957fdd918906a7656f14e0a3336120546ea55b24"
V195_TREE = "0481ee37be4bff6d675a4a8fb6144ab73091977e"
V189_TREE = "2092bc5535d827fdf836172bff2f18cb4b594e3e"
TREE = "179d3985bd13196ffee8787a0d034c4c4876836f"
PARENT = "35b2086ec900233547ca35a6c2103ec9d5201d2f82818af91015421476d8ca30"
# All candidate choices use one durable O_EXCL claim and one process lock.
# V132 is in its filename so existing V132/V154c pollers honor even a timeout.
LEGACY_NAME = "v187_in_v132_standby_slot_nov20_check16"
NAME = "v187_or_v189_in_v132_standby_slot_nov20_check16"
OWN = (V187, 16, NAME, START, END)
PROCESS_LOCK = "nov20_v187_runner.lock"
REQUIRED = (sibling.FIRST, sibling.V141, sibling.V120B, v180.REFERENCE)
COMPETITORS = (sibling.V132, sibling.OWN, v180.fallback.OWN, v180.peer.OWN, earlier.OWN, v180.OWN)


def plan(root=ROOT, candidate=V187):
    if candidate not in CANDIDATES.values():
        raise ValueError("unregistered standby candidate")
    base = root / "serving_v0523/candidates"
    for artifact in (V187, candidate, *(r[0] for r in REQUIRED)):
        if first.digest(base / artifact.directory / "patch.diff") != artifact.sha256:
            raise ValueError("frozen patch mismatch for " + artifact.name)
    manifest = json.loads((base/V187.directory/"validation.json").read_text())
    if (manifest.get("tree") != TREE or manifest.get("patch_sha256") != V187.sha256
            or manifest.get("parent_sha256") != PARENT or manifest.get("launch_flags") != V187.flags
            or manifest.get("strict_apply_reverse_full_delta") is not True
            or manifest.get("strict_named_donor_reconstruction") is not True):
        raise ValueError("V187 export or launch flags changed")
    if candidate == V229:
        fallback_plan = plan(root, V198)
        _, margin = v229_selection.frozen_inputs(root, candidate, v180.nov19.V68_JOB)
        return fallback_plan | dict(candidate=V229.name, tree=v229_selection.V229_TREE, sha256=V229.sha256,
                                    flags=V229.flags, alternative_candidates=[c.name for c in PRIOR_CHOICES],
                                    conditional_parent=v229_selection.V204_JOB, fallback=V198.name,
                                    selection_margin=margin, candidate_lineage=v229_selection.plan_lineage())
    tree = TREE
    if candidate == V189:
        tree = V189_TREE
        child = json.loads((base/V189.directory/"validation.json").read_text())
        if (child.get("tree") != tree or child.get("patch_sha256") != V189.sha256
                or child.get("parent") != "V187" or child.get("parent_sha256") != V187.sha256
                or child.get("parent_tree") != TREE or child.get("launch_flags") != V187.flags
                or child.get("strict_apply_reverse_full_delta") is not True):
            raise ValueError("V189 export or launch flags changed")
    if candidate == V198:
        tree = V198_TREE
        child = json.loads((base / V198.directory / "validation.json").read_text())
        required = dict(candidate="V198", tree=tree, patch_sha256=V198.sha256, parent="V197",
                        parent_sha256=V197_SHA, parent_tree=V197_TREE, common_parent="V195",
                        common_tree=V195_TREE, attention_donor="V196", launch_flags=V187.flags)
        verified = ("strict_apply_reverse_full_delta", "strict_both_donor_orders", "source_and_modes_verified")
        if any(child.get(k) != v for k, v in required.items()) or any(child.get(k) is not True for k in verified):
            raise ValueError("V198 export, joint reconstruction or launch flags changed")
    comparison = v180._reference(json.loads((root/"serving_v0523/runs"/(v180.nov19.V68_JOB+".json")).read_text()))
    if comparison is None:
        raise ValueError("V68 archived private comparison changed")
    return dict(candidate=candidate.name, flags=candidate.flags, tree=tree, sha256=candidate.sha256, mode="check", users=16,
                start_utc=START.isoformat(), end_utc_exclusive=END.isoformat(), resets_at=RESET,
                required_quota=slots.THIRD, maximum_private_checks=1, arbitration_lock=slots.LOCK,
                process_lock=PROCESS_LOCK, alternative_candidates=[V187.name, V189.name, V198.name] if candidate == V198 else [V187.name,V189.name],
                legacy_claim=LEGACY_NAME+".submit.txt",
                last_observations=["fresh aggregate private quota", "exact still-pending V120b"],
                max_preflight_seconds=120, max_final_observations_seconds=30,
                priority="Any V120b completion before the final parent observation yields the slot.",
                preceding_in_order=[r[2] for r in REQUIRED[:2]], pending_control=REQUIRED[2][2],
                requires_unattempted=[r[2] for r in COMPETITORS], receipt=NAME+".submit.txt", comparison=comparison)


def local_position(runs):
    # The exact CLI receipts authenticate users when the live queued API omits it.
    ids = tuple(slots.bound_receipt(runs,r) for r in REQUIRED)
    if not all(ids) or len(set(ids)) != len(ids) or ids[-1] != v180.nov19.V68_JOB:
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


def run_once(*,okc,candidate=V187,root=ROOT,runs=RUNS,clock=first.utc_now,execute=subprocess.run,monotonic=time.monotonic):
    read_utc = clock
    began = previous_utc = read_utc()
    tick_deadline = monotonic() + 120
    utc_deadline = min(END, began + timedelta(seconds=120))
    rolled_back = False

    def clock():
        nonlocal previous_utc, rolled_back
        current = read_utc()
        if current < previous_utc:
            rolled_back = True
        previous_utc = current
        return current

    def seconds_left():
        current = clock()
        if rolled_back or current < START:
            return 0.0
        return min((utc_deadline - current).total_seconds(), tick_deadline - monotonic())

    def inside():
        return seconds_left() > 0

    def deny(cmd):
        return subprocess.CompletedProcess(cmd,1,"","standby unavailable")

    def bounded(cmd,**kw):
        remaining = seconds_left()
        if remaining <= 0:
            return deny(cmd)
        return execute(cmd,**(kw | {"timeout":min(kw["timeout"],remaining)}))

    if not START <= began < END:
        return "outside submission window"
    if not inside():
        return "waiting for a fresh standby window"
    with slots.arbitrate(runs) as held:
        if not held:
            return "another allocation poll is active"
        def prior(own=False):
            return any(slots.prior_attempt(runs,variant,NAME,own) for variant in PRIOR_CHOICES)
        if prior():
            return "attempt recorded"
        claim_path=runs/(NAME+".submit.txt")
        allowed={runs/(req[2]+".submit.txt") for req in REQUIRED[:2]}
        snapshot=scored_receipt.inventory(runs,claim_path)
        scores=scored_receipt.local_score(runs,root,snapshot,allowed,START.replace(hour=0,minute=0),END)
        position=local_position(runs)
        requested = candidate
        config = SimpleNamespace(V204=V204, V229=V229, V198=V198, slots=slots, v180=v180, earlier=earlier)
        def selection():
            if (position is None or scores is None or any(row[0] in position for row in scores)
                    or not inside() or scored_receipt.inventory(runs,claim_path)!=snapshot
                    or not scored_receipt.confirmed(scores,okc,bounded,clock)
                    or scored_receipt.inventory(runs,claim_path)!=snapshot):
                return requested,None
            data = live_evidence(runs, position, okc, bounded, clock)
            if data is not None and scores:
                data["independent_scored_run"] = dict(job=scores[0][0],mode="run",board="serving",
                    team=scored_receipt.TEAM,patch_sha256=scores[0][3])
            if requested != V229 or data is None:
                return requested, data
            selected, evidence = v229_selection.select(config, runs, root, data, okc, bounded, clock)
            if evidence is not None:
                data["v204_comparison"] = evidence
            return selected, data

        candidate, proof = selection()
        if proof is None:
            return "waiting for V129 and V141 acceptances, V68 pass and pending V120b"
        plan(root,candidate)
        patch=root/"serving_v0523/candidates"/candidate.directory/"patch.diff"
        status=[okc,"serve","status","--json"]
        expected=[okc,"serve","check",str(patch),"--users","16"]
        for key,value in sorted(candidate.flags.items()):
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
            return (inside() and not prior(own) and scored_receipt.inventory(runs,claim_path)==snapshot
                    and local_position(runs)==position
                    and (candidate != V229 or slots.bound_receipt(runs, v229_selection.parent_request(config)) == v229_selection.V204_JOB))

        def guarded(cmd,**kw):
            nonlocal utc_deadline, tick_deadline
            if cmd==status:
                res=quota(**kw)
                if res.returncode or not unchanged() or selection()!=(candidate,proof):
                    return deny(cmd)
                plan(root,candidate)
                if not unchanged():
                    return deny(cmd)
                utc_deadline = min(utc_deadline, clock() + timedelta(seconds=30))
                tick_deadline = min(tick_deadline, monotonic() + 30)
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
                return res if inside() else deny(cmd)
            if cmd!=expected:
                raise ValueError("unexpected private standby operation")
            if not unchanged(True) or first.digest(patch)!=candidate.sha256:
                return deny(cmd)
            now=clock()
            if not inside():
                return deny(cmd)
            # Freshness limits the moment we authorize the command. An already
            # authorized CLI can use its usual delivery timeout up to the cutoff.
            return execute(cmd,**(kw | {"timeout":min(kw["timeout"],(END-now).total_seconds())}))

        result=measured.submit_once((NAME,candidate,16,proof),okc=okc,root=root,runs=runs,now=clock(),
                                    execute=guarded,start=START,end=END,reset=RESET,reserve_checks=0)
        if result == "outside submission window" and rolled_back:
            return "waiting for a fresh standby window"
        if result.startswith("submitted "):
            job=result.split()[1]
            if job in position or any(row[0]==job for row in scores) or slots.bound_receipt(runs,(candidate,16,NAME,START,END))!=job:
                return "attempt needs manual reconciliation; no retry"
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--candidate",choices=tuple(CANDIDATES),default="v187")
    args=parser.parse_args()
    candidate=CANDIDATES[args.candidate]
    frozen=plan(candidate=candidate)
    if not args.run:
        print(json.dumps(frozen,indent=2))
        return
    okc=os.environ.get("OKC",str(Path.home()/".okc/bin/okc"))
    with (RUNS/PROCESS_LOCK).open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return
        print(json.dumps(dict(armed_at=first.utc_now().isoformat(),**frozen)),flush=True)
        last=None
        while first.utc_now()<END:
            if first.utc_now()>=START:
                try:
                    state=run_once(okc=okc,candidate=candidate)
                except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as err:
                    state=f"{type(err).__name__}: {err}"
                if state!=last:
                    print(json.dumps(dict(at=first.utc_now().isoformat(),outcome=state)),flush=True)
                    last=state
                if (state.startswith(("submitted ","attempt ","prior attempt"))
                        or (state=="outside submission window" and first.utc_now()>=END)):
                    break
            time.sleep(min(60,max(0,(END-first.utc_now()).total_seconds())))
        print(json.dumps(dict(finished_at=first.utc_now().isoformat())),flush=True)


if __name__=="__main__":
    main()
