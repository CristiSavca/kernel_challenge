#!/usr/bin/env python3
"""Check the combined V87 candidate at four users with a spare slot, then the matched V72a/V76a pair after V68 passes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from submit_nov17_integrated_checks import Candidate, ROOT, RUNS, submit_once


START = datetime(2026, 11, 18, 0, 3, tzinfo=timezone.utc)
END = datetime(2026, 11, 18, 23, 30, tzinfo=timezone.utc)
RESET = "2026-11-19T00:00:00Z"
V68_JOB = "04b01f958aa5487c"
V68_SHA = "6248a58842e1a22c1cda5a957d5ab464a6ec824efcdf8308bdcc9c6a1046e505"
BASE_FLAGS = {
    "--enable-layernorm-sp": "",
    "--moe-runner-backend": "humming",
    "--speculative-moe-runner-backend": "marlin",
    "--swa-full-tokens-ratio": "0.25",
}
TREE_FLAGS = BASE_FLAGS | {"--speculative-dflash-tree": ""}
V72a = Candidate(
    "v72a", "v72a_tree_multicast_decode",
    "ad27be6e346c1667521ccbcba8f36b4c282d2017424a0fadb31e1f9460516cba",
    TREE_FLAGS,
)
V76a = Candidate(
    "v76a", "v76a_sharded_draft_target_decode",
    "12518a74131666e6da2f2485d2a8bc1d4884b4944f16be748d2831ba06b8a691",
    TREE_FLAGS | {"--speculative-dflash-tp-context-projection": ""},
)
V79a = Candidate(
    "v79a", "v79a_wide_selector_tree",
    "28690547fa3a3d52a05c1d4385c48906979b21ac52b4533d42e2cdbc4cfd94b1",
    V76a.flags | {"--speculative-num-draft-tokens": "12"},
)
V80a = Candidate("v80a", "v80a_fused_experts_sharded_draft_filters",
                "9c6cf15326e1fbda08159c6d20d6dc1c23150695738453df15d60ace3b2d3864",
                V76a.flags | {"--speculative-dflash-tp-conv-projection": ""})
V84a = Candidate("v84a", "v84a_routed_decode_committed_context",
                "9e3ed9204b56015d69420d220375c4d2bddeb7e32f8b92ca3442fd6dcd1acecd",
                V80a.flags | {"--speculative-dflash-compact-context": ""})
V87 = Candidate("v87", "v87_fused_draft_head_skinny_experts",
                "c04ffda8c63ad2ac672b41f74bd7ba11f3e171abd501410a1bcc58ac24142684",
                V84a.flags | {"--speculative-dflash-fused-candidate-head": ""})
V88 = Candidate("v88", "v88_v64_fused_draft_head",
                "4829c222939a0cb362975aa701c985836874f62fb6fd8e1907001ada4cd3b52f",
                BASE_FLAGS | {"--speculative-dflash-fused-candidate-head": ""})


def v68_failure(reason):
    return reason in (f"judge job {V68_JOB} is failed",
                      f"judge job {V68_JOB} completed without passing")


# Consider V87 first so all three may run in the same window, preserving
# both slots for the matched comparison when another submission used one.
CANDIDATES = (V87, V72a, V76a)
POLL_SECONDS = 300


def utc_now():
    return datetime.now(timezone.utc)


def check_load(candidate):
    return 4 if candidate in (V87, V88) else 16


def receipt_name(candidate):
    return f"{candidate.name}_nov18_check{check_load(candidate)}"


def decisions(runs=RUNS):
    candidates = CANDIDATES
    skipped = runs / f"{receipt_name(V72a)}.skip.json"
    if skipped.is_file():
        try:
            reason = json.loads(skipped.read_text()).get("reason")
        except (OSError, ValueError, AttributeError):
            reason = None
        if v68_failure(reason):
            candidates = (*candidates, V88)
    return [
        (receipt_name(candidate), candidate, check_load(candidate),
         {"kind": "private decode experiment", "v68_job": V68_JOB,
          "requires_v68_pass": candidate in (V72a, V76a),
          "requires_v68_failure": candidate is V88})
        for candidate in candidates
        if not (runs / f"{receipt_name(candidate)}.submit.txt").exists()
        and not (runs / f"{receipt_name(candidate)}.skip.json").exists()
    ]


def job_state(job, sha256, flags, *, okc, execute, users=16):
    response = execute(
        [okc, "serve", "status", job, "--json"],
        capture_output=True, text=True, timeout=60,
    )
    if response.returncode:
        return "waiting", "judge status request failed"
    try:
        data = json.loads(response.stdout)
        if data["id"] != job or data["board"] != "serving" or data["mode"] != "check":
            return "waiting", "judge status identity mismatch"
        if data.get("users") is not None and data["users"] != users:
            return "waiting", "judge status load mismatch"
        state = data["status"]
        if state in ("queued", "running"):
            return state, f"judge job {job} is {state}"
        if state in ("failed", "cancelled", "canceled", "withdrawn"):
            return "failed", f"judge job {job} is {state}"
        if state != "done":
            return "waiting", f"judge job {job} is {state}"
        result = data["result"]
        if result.get("passed") is False or result.get("status") != "ok":
            return "failed", f"judge job {job} completed without passing"
        if result.get("passed") is not True:
            return "waiting", "completed judge job has no positive result"
        if result.get("concurrency") != [users]:
            return "failed", "judge result has unexpected load"
        if result["patch"]["sha256"] != sha256 or result["serving_flags"] != flags:
            return "failed", f"judge job {job} source or flags mismatch"
    except (KeyError, TypeError, ValueError, AttributeError):
        return "waiting", "judge status response incomplete"
    return "passed", f"judge job {job} passed with exact source and flags"


def status_from_receipt(candidate, name, runs, okc, execute):
    if (runs / f"{name}.skip.json").exists():
        return "failed", f"{name} skipped"
    receipt = runs / f"{name}.submit.txt"
    if not receipt.exists():
        return "waiting", f"{name} has no receipt"
    contents = receipt.read_text()
    matches = set(re.findall(r"\bjob ([0-9a-f]{16})\b", contents))
    if len(matches) != 1 or not contents.rstrip().endswith("CLI exit code: 0"):
        return "waiting", f"{name} has an unconfirmed receipt"
    try:
        meta = json.loads(contents.splitlines()[0])
        if (meta["candidate"], meta["sha256"], meta["flags"], meta["users"]) != (
            candidate.name, candidate.sha256, candidate.flags, check_load(candidate)
        ):
            return "failed", f"{name} receipt mismatches the frozen check"
    except (IndexError, ValueError, KeyError, TypeError):
        return "waiting", f"{name} has incomplete metadata"

    return job_state(matches.pop(), candidate.sha256, candidate.flags, okc=okc,
                     execute=execute, users=check_load(candidate))


def dependency_gate(candidate, *, okc, runs, execute):
    state, reason = job_state(V68_JOB, V68_SHA, BASE_FLAGS, okc=okc, execute=execute)
    if candidate is V88:
        if state == "failed" and v68_failure(reason):
            return "passed", reason
        return "waiting", "V88 requires a confirmed V68 failure"
    admitted = ("passed", "queued", "running")
    if candidate is V87:
        if state not in admitted:
            return state, reason
        for parent in (V72a, V76a):
            name = receipt_name(parent)
            if any((runs / f"{name}.{suffix}").exists() for suffix in ("submit.txt", "skip.json")):
                state, reason = status_from_receipt(parent, name, runs, okc, execute)
                if state not in admitted:
                    return state, reason
        return "passed", reason
    if state != "passed" or candidate is V72a:
        return state, reason
    state, reason = status_from_receipt(V72a, receipt_name(V72a), runs, okc, execute)
    if state in admitted:
        return "passed", reason
    return state, reason


def skip(candidate, reason, *, runs, now):
    path = runs / f"{receipt_name(candidate)}.skip.json"
    try:
        with path.open("x") as output:
            json.dump({"skipped_at": now.isoformat(), "candidate": candidate.name,
                       "reason": reason}, output)
            output.write("\n")
    except FileExistsError:
        pass


class DependencyChanged(RuntimeError):
    def __init__(self, state, reason):
        self.state = state
        super().__init__(reason)


def run_offer(offer, *, okc, root=ROOT, runs=RUNS, execute=subprocess.run, clock=utc_now):
    now = clock()
    if not START <= now < END:
        return "outside submission window"
    candidate = offer[1]
    state, reason = dependency_gate(candidate, okc=okc, runs=runs, execute=execute)
    if state == "failed":
        skip(candidate, reason, runs=runs, now=now)
        return "skipped " + reason
    if state != "passed":
        return "waiting " + reason

    def gated_execute(command, **kwargs):
        if command[1:] == ["serve", "status", "--json"]:
            response = execute(command, **kwargs)
            # This runs before submit_once exclusively creates an attempt receipt.
            # Recheck dependencies immediately after the quota read.
            state, reason = dependency_gate(candidate, okc=okc, runs=runs, execute=execute)
            if state != "passed":
                raise DependencyChanged(state, reason)
            return response
        if command[1:3] != ["serve", "check"]:
            raise ValueError("unexpected runner command")
        if not START <= clock() < END:
            raise DependencyChanged("expired", "submission deadline reached before CLI launch")
        return execute(command, **kwargs)

    try:
        return submit_once(
            offer, okc=okc, now=now, root=root, runs=runs, execute=gated_execute,
            start=START, end=END, reset=RESET, reserve_checks=2 if candidate is V87 else 0,
        )
    except DependencyChanged as changed:
        if changed.state in ("failed", "expired"):
            skip(candidate, str(changed), runs=runs, now=clock())
            return "skipped " + str(changed)
        return "waiting " + str(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan = {
        "start_utc": START.isoformat(), "end_utc": END.isoformat(),
        "reset_utc": RESET, "v68_job": V68_JOB,
        "standby_on_v68_failure": {"candidate": V88.name, "users": check_load(V88),
                                    "sha256": V88.sha256, "flags": V88.flags},
        "checks": [{"candidate": c.name, "users": check_load(c), "sha256": c.sha256,
                    "flags": c.flags, "minimum_checks_left": 3 if c is V87 else 1,
                    "requires_v68_pass": c is not V87} for c in CANDIDATES],
    }
    if not args.run:
        print(json.dumps(plan, indent=2))
        return
    okc = os.environ.get("OKC", str(Path.home() / ".okc/bin/okc"))
    with (RUNS / "nov18_decode_runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another Nov18 decode runner owns the lock.", flush=True)
            return
        print(json.dumps({"armed_at": utc_now().isoformat(), **plan}), flush=True)
        next_poll = START.timestamp()
        previous = None
        while utc_now() < END:
            now = utc_now()
            if now.timestamp() < next_poll:
                time.sleep(min(60, next_poll - now.timestamp()))
                continue
            offers = decisions()
            if not offers:
                break
            results = []
            for offer in offers:
                try:
                    result = run_offer(offer, okc=okc)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    result = f"status unavailable {type(exc).__name__}: {exc}"
                results.append((offer[1].name, result))
            if results != previous:
                print(json.dumps({"at": utc_now().isoformat(), "results": results}), flush=True)
                previous = results
            next_poll = utc_now().timestamp() + POLL_SECONDS
        print(json.dumps({"finished_at": utc_now().isoformat(), "remaining":
                          [offer[1].name for offer in decisions()]}), flush=True)


if __name__ == "__main__":
    main()
