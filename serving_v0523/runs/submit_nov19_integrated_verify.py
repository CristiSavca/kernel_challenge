#!/usr/bin/env python3
"""Run the first two existing Nov19 allocations and one four-user V139 private check."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

import submit_nov19_decode_checks as previous
import submit_nov20_v129_check as control
from submit_nov17_integrated_checks import Candidate, ROOT, RUNS, V64, submit_once


V139 = Candidate("v139", "v139_integrated_experts_verify",
                 "c4428099805f991512ff14783368efdada1985b6a4b15c039a17aa54ac4782ed", V64.flags)
NAME = "v139_nov19_check4"
OLD_MEMBERS = (*previous.TREE_ALTERNATIVES, *previous.LEGACY_TREE_ALTERNATIVES)
RETAINED = (*previous.ALTERNATIVES, *previous.PREFILL_ALTERNATIVES)
START, END, RESET = previous.START, previous.END, previous.RESET


def checked_plan(root=ROOT, runs=RUNS):
    patch = root / "serving_v0523/candidates" / V139.directory / "patch.diff"
    if control.digest(patch) != V139.sha256 or not control.parent_gate(runs):
        raise ValueError("V139 patch or official V64 control mismatch")
    record = json.loads((runs / (control.PARENT_JOB + ".json")).read_text())["result"]
    rungs = [r for r in record["rungs"] if r["concurrency"] == 4 and r["truncated"] is False]
    if len(rungs) != 1 or not math.isfinite(rungs[0]["tok_s_per_gpu"]) or rungs[0]["tok_s_per_gpu"] <= 0:
        raise ValueError("missing complete four-user control")
    plan = previous.plan()
    plan["checks"][2] = dict(candidate="v139", users=4, sha256=V139.sha256, flags=V139.flags,
                             comparison_job=control.PARENT_JOB, measured_tokens_per_s_gpu=rungs[0]["tok_s_per_gpu"],
                             superseded_receipts=[previous.prefix(c) for c in OLD_MEMBERS])
    return plan


def prior_receipt(runs, exclude_own=False):
    own = runs / (NAME + ".submit.txt")
    if own.exists() and not exclude_own:
        return own.name
    for c in OLD_MEMBERS:
        p = runs / (previous.prefix(c) + ".submit.txt")
        if p.exists():
            return p.name
    for p in runs.iterdir():
        if p == own or not p.is_file():
            continue
        if p.name.endswith((".submit.txt", ".submit.json")):
            raw = p.read_text()
            if V139.sha256 in raw or V139.directory + "/patch.diff as a check run" in raw:
                return p.name
        elif control.JUDGE_FILE.fullmatch(p.name) and V139.sha256 in p.read_text():
            return p.name
    return None


def offer(*, okc, root=ROOT, runs=RUNS, clock=previous.utc_now, execute=subprocess.run):
    if not START <= clock() < END:
        return "outside submission window"
    with (runs / "nov19_tree_alternative.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "waiting for alternative lock"
        if prior_receipt(runs):
            return "attempt recorded"
        patch = root / "serving_v0523/candidates" / V139.directory / "patch.diff"
        if not control.parent_gate(runs) or control.digest(patch) != V139.sha256:
            return "unconfirmed patch or parent"
        expected = [okc, "serve", "check", str(patch), "--users", "4", *control.FLAGS, "--no-wait"]
        quota = [okc, "serve", "status", "--json"]

        def guarded(cmd, **kwargs):
            now = clock()
            if not START <= now < END:
                return subprocess.CompletedProcess(cmd, 1, "", "closed")
            kwargs["timeout"] = min(kwargs["timeout"], (END-now).total_seconds())
            if cmd == quota:
                response = execute(cmd, **kwargs)
                if response.returncode:
                    return response
                status = json.loads(response.stdout)
                counts = status["limits"]["checks"]
                if (status["resets_at"] == RESET
                    and all(type(counts[k]) is int for k in ("per_day","used","left"))
                    and counts["per_day"] == 3 and counts["used"] >= 0
                    and counts["left"] > 0 and counts["left"] + counts["used"] == 3):
                    return response
                return subprocess.CompletedProcess(cmd, 1, "", "quota mismatch")
            if cmd != expected:
                raise ValueError("unexpected judge command")
            if prior_receipt(runs, exclude_own=True) or control.digest(patch) != V139.sha256:
                return subprocess.CompletedProcess(cmd, 1, "", "prior attempt or source changed")
            return execute(cmd, **kwargs)

        return submit_once((NAME, V139, 4, dict(comparison_job=control.PARENT_JOB,
                            comparison_sha256=V64.sha256, comparison_load=4)),
                           okc=okc, now=clock(), root=root, runs=runs, execute=guarded,
                           start=START, end=END, reset=RESET)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    run = parser.parse_args().run
    plan = checked_plan()
    if not run:
        print(json.dumps(plan, indent=2))
        return
    with (RUNS/"nov19_decode_runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(dict(armed_at=previous.utc_now().isoformat(), **plan)), flush=True)
        okc = os.environ.get("OKC", str(Path.home()/".okc/bin/okc"))
        previous_outcomes = None
        while previous.utc_now() < END:
            if previous.utc_now() < START:
                time.sleep(min(60,max(0,(START-previous.utc_now()).total_seconds())))
                continue
            outcomes = {}
            for c in (*RETAINED, V139):
                try:
                    outcomes[c.name] = offer(okc=okc) if c == V139 else previous.run_candidate(c, okc=okc)
                except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                    outcomes[c.name] = f"{type(exc).__name__}: {exc}"
            if outcomes != previous_outcomes:
                print(json.dumps(dict(at=previous.utc_now().isoformat(), outcomes=outcomes)), flush=True)
                previous_outcomes = outcomes
            if all(value == "attempt recorded" or value.startswith(("submitted ","attempt needs","attempt timed"))
                   for value in outcomes.values()):
                return
            time.sleep(min(60,max(0,(END-previous.utc_now()).total_seconds())))


if __name__ == "__main__":
    main()
