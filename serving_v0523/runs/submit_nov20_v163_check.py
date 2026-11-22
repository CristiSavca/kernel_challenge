#!/usr/bin/env python3
"""Offer V163 at 16 users in the second private slot after both Nov19 controls fail."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path

import nov20_check_slots as slots
import submit_nov17_integrated_checks as measured
import submit_nov20_v154c_check as previous

ROOT, RUNS, V64 = measured.ROOT, measured.RUNS, measured.V64
first = previous.first
START, END, RESET = first.START, first.END, first.RESET
V163 = measured.Candidate("v163", "v163_sparse_prompt_exchange",
    "d4840308ea25166a1a3db0b4358beb6d59b9de97faaa43ef8a3795241bdc33a1",
    V64.flags | {"--muse-sparse-prompt-exchange": ""})
NAME = "v163_nov20_check16"
OWN = (V163, 16, NAME, START, END)
REQUIRED = (previous.FIRST, previous.V120B, previous.V139)
PEERS = (previous.V132, previous.V141, previous.OWN)
REFERENCE = previous.REFERENCE
FROZEN_REPORT_SHA256 = "545dd1af119639f8ff6cf6fc997e2007926110980f673969d1452c4537cae0aa"


def plan(root=ROOT):
    base = root/"serving_v0523/candidates"
    for candidate in (V64, V163, *(req[0] for req in REQUIRED)):
        if first.digest(base/candidate.directory/"patch.diff") != candidate.sha256:
            raise ValueError("frozen patch mismatch for " + candidate.name)
    manifest = json.loads((base/V163.directory/"validation.json").read_text())
    if (manifest["parent_sha256"] != V64.sha256 or manifest["patch_sha256"] != V163.sha256
            or manifest["launch_flags"] != V163.flags or manifest["fixed_flag_patterns"] != 76
            or manifest["tree"] != "aff28f2cc94bd57228886740f691d7543f9a13ac"
            or manifest["strict_roundtrips"] is not True or manifest["new_bare_flag_parsed"] is not True):
        raise ValueError("V163 export or flags mismatch")
    path = root/"serving_v0523/runs"/(REFERENCE+".json")
    reference = json.loads(path.read_text())
    result = reference["result"]
    rungs = result["rungs"]
    if ((reference["id"], reference["board"], reference["mode"], reference["status"], reference["duplicate_of"]) !=
            (REFERENCE, "serving", "check", "done", None) or reference["serving_flags"] != V64.flags
            or result.get("serving_flags") != V64.flags or result["patch"]["sha256"] != V64.sha256
            or result["status"] != "ok" or result["passed"] is not True or result["correctness"]["passed"] is not True
            or result["concurrency"] != [16] or len(rungs) != 1 or rungs[0]["concurrency"] != 16
            or rungs[0]["truncated"] is not False or rungs[0]["reps"] != 1 or rungs[0]["tok_s_per_gpu"] != 17349.9):
        raise ValueError("V64 matched comparator does not match the frozen 16-user result")
    return dict(candidate=V163.name, users=16, mode="check", sha256=V163.sha256, flags=V163.flags,
        comparison_job=REFERENCE, comparison_sha256=V64.sha256, comparison_tokens_s_gpu=17349.9,
        frozen_comparison_report_sha256=FROZEN_REPORT_SHA256, current_comparison_report_sha256=first.digest(path),
        first=REQUIRED[0][2], both_controls_failed=[r[2] for r in REQUIRED[1:]],
        requires_unattempted=[r[2] for r in PEERS], start_utc=START.isoformat(), end_utc_exclusive=END.isoformat(),
        resets_at=RESET, required_quota=slots.SECOND, reserve_checks=1, maximum_private_checks=1,
        arbitration_lock=slots.LOCK, receipt=NAME+".submit.txt")


def local_position(runs):
    if any(slots.prior_attempt(runs, candidate, name) for candidate, _, name, _, _ in PEERS):
        return None
    ids = tuple(slots.bound_receipt(runs, req) for req in REQUIRED)
    return ids if all(ids) and len(set(ids)) == len(ids) else None


def official(req, job, okc, execute, runs):
    cmd = [okc, "serve", "status", job, "--json"]
    response = execute(cmd, capture_output=True, text=True, timeout=60)
    if response.returncode:
        return None
    data = json.loads(response.stdout)
    if not isinstance(data, dict):
        return None
    # A refunded first check cannot explain the single consumed November 20 slot.
    if req == REQUIRED[0] and data.get("refunded", False) is not False:
        return None
    identity = previous.official_identity(req, job, okc, lambda *a, **kw: response, runs=runs)
    if identity is not None:
        return identity
    # The service can time out before producing a benchmark result. In the
    # archived service schema the error has no patch, user count or passed field.
    # The exact CLI acceptance binds all missing fields to this one live job ID.
    candidate, users, _, begin, end = req
    stamp, declared, result = data.get("created_at"), data.get("users"), data.get("result")
    if ((data.get("id"), data.get("board"), data.get("mode"), data.get("status")) !=
            (job, "serving", "check", "failed") or data.get("duplicate_of") is not None
            or data.get("serving_flags") != candidate.flags or type(stamp) not in (float, int)
            or not math.isfinite(stamp) or not begin.timestamp() <= stamp < end.timestamp()
            or declared is not None and (type(declared) is not int or declared != users)
            or not isinstance(result, dict) or result.get("status") != "timeout"
            or not isinstance(result.get("error"), str) or not result["error"].strip()
            or result.get("rungs") != [] or set(result) != {"error", "rungs", "status"}
            or slots.bound_receipt(runs, req) != job):
        return None
    return "failed", True, stamp


def live_position(runs, ids, okc, execute):
    if ids is None:
        return False
    for req, job in zip(REQUIRED[1:], ids[1:], strict=True):
        identity = official(req, job, okc, execute, runs)
        if identity is None or not identity[1]:
            return False
    return official(REQUIRED[0], ids[0], okc, execute, runs) is not None


def run_once(*, okc, root=ROOT, runs=RUNS, clock=first.utc_now, execute=subprocess.run):
    def inside():
        return START <= clock() < END
    def deny(cmd):
        return subprocess.CompletedProcess(cmd, 1, "", "V163 slot or identity unavailable")
    def bounded(cmd, **kw):
        now = clock()
        if not START <= now < END:
            return deny(cmd)
        return execute(cmd, **(kw | {"timeout": min(kw["timeout"], (END-now).total_seconds())}))
    if not inside():
        return "outside submission window"
    with slots.arbitrate(runs) as acquired:
        if not acquired:
            return "another allocation poll is active"
        def attempted(exclude_own=False):
            return slots.prior_attempt(runs, V163, NAME, exclude_own)
        if attempted():
            return "attempt recorded"
        plan(root)
        ids = local_position(runs)
        if not live_position(runs, ids, okc, bounded):
            return "waiting for V129 acceptance and both terminal control failures"
        patch = root/"serving_v0523/candidates"/V163.directory/"patch.diff"
        status = [okc, "serve", "status", "--json"]
        expected = [okc, "serve", "check", str(patch), "--users", "16"]
        for key, value in sorted(V163.flags.items()):
            expected += ["--flag", key[2:] + ("="+value if value else "")]
        expected.append("--no-wait")
        def quota():
            response = bounded(status, capture_output=True, text=True, timeout=60)
            if response.returncode:
                return response
            data = json.loads(response.stdout)
            counts = data["limits"]["checks"]
            if data["resets_at"] != RESET or any(type(counts[k]) is not int or counts[k] != v for k,v in slots.SECOND.items()):
                return deny(status)
            return response
        def eligible():
            return (inside() and not attempted() and local_position(runs) == ids
                    and live_position(runs, ids, okc, bounded) and inside())
        def guarded(cmd, **kw):
            if cmd == status:
                # All recoverable remote reads precede the permanent attempt.
                # Hold the other candidates' allocation lock throughout both reads.
                check = quota()
                if check.returncode or not eligible():
                    return deny(cmd)
                plan(root)
                check = quota()
                return check if not check.returncode and eligible() else deny(cmd)
            if cmd != expected:
                raise ValueError("unexpected V163 check payload")
            if not inside() or attempted(True) or local_position(runs) != ids or first.digest(patch) != V163.sha256:
                return deny(cmd)
            return bounded(cmd, **kw)
        proof = dict(slot=2, v129_job=ids[0], failed_controls=dict(v120b=ids[1],v139=ids[2]),
                     comparison_job=REFERENCE, comparison_sha256=V64.sha256, comparison_tokens_s_gpu=17349.9,
                     frozen_comparison_report_sha256=FROZEN_REPORT_SHA256)
        outcome = measured.submit_once((NAME, V163, 16, proof), okc=okc, root=root, runs=runs, now=clock(),
            execute=guarded, start=START, end=END, reset=RESET, reserve_checks=1)
        if outcome.startswith("submitted ") and slots.dated_receipt(runs, OWN) != outcome.split()[1]:
            return "attempt needs manual reconciliation; no retry"
        return outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    frozen = plan()
    if not args.run:
        print(json.dumps(frozen, indent=2))
        return
    okc = os.environ.get("OKC", str(Path.home()/".okc/bin/okc"))
    with (RUNS/"nov20_v163_runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        print(json.dumps(dict(armed_at=first.utc_now().isoformat(), **frozen)), flush=True)
        last = None
        while first.utc_now() < END:
            if first.utc_now() >= START:
                try:
                    state = run_once(okc=okc)
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                    state = f"{type(exc).__name__}: {exc}"
                if state != last:
                    print(json.dumps(dict(at=first.utc_now().isoformat(), outcome=state)), flush=True)
                    last = state
                if state.startswith(("submitted ", "attempt ", "prior attempt", "outside submission window")):
                    break
            time.sleep(min(60,max(0,(END-first.utc_now()).total_seconds())))
        print(json.dumps(dict(finished_at=first.utc_now().isoformat())), flush=True)


if __name__ == "__main__":
    main()
