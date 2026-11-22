#!/usr/bin/env python3
"""Offer one private V129 check at the measured V64 one-user load."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from submit_nov17_integrated_checks import Candidate, ROOT, RUNS, V64, submit_once

START = datetime(2026, 11, 20, 0, 3, tzinfo=timezone.utc)
END = datetime(2026, 11, 20, 7, 59, tzinfo=timezone.utc)
RESET = "2026-11-21T00:00:00Z"
PARENT_JOB = "abd217baf0ed49fd"
PARENT_RECEIPT_HASH = "f7104580a17cccaf8f1a284e9eca261c38a41d231dadb8e58b4502c62941e015"
V129 = Candidate("v129", "v129_routed_projection_pair",
                 "f20be13ffbc4d3ae2da7d473b3f8ea4b5ccd19fb42ea55a46ba954efced751e2", V64.flags)
NAME = "v129_nov20_check1"
TAG = re.compile(r"(?<![a-z0-9])v129(?![a-z0-9])", re.I)
JUDGE_FILE = re.compile(r"[0-9a-f]{16}(?:\.status)?\.json\Z")
FLAGS = [arg for key, value in sorted(V64.flags.items())
         for arg in ("--flag", key[2:] + ("=" + value if value else ""))]


def utc_now():
    return datetime.now(timezone.utc)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_gate(runs):
    data = json.loads((runs / f"{PARENT_JOB}.json").read_text())
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != PARENT_RECEIPT_HASH:
        return False
    result = data["result"]
    return ((data["id"], data["board"], data["mode"], data["status"], data["serving_flags"])
            == (PARENT_JOB, "serving", "run", "done", V64.flags)
            and result["passed"] is True and result["status"] == "ok"
            and result["patch"]["sha256"] == V64.sha256 and result["serving_flags"] == V64.flags
            and result["concurrency"] == [1, 4, 16, 32]
            and any(r["concurrency"] == 1 and r["truncated"] is False
                    and r["reps"] > 0 and r["tok_s_per_user_p50"] > 0 for r in result["rungs"]))


def prior_attempt(runs, exclude_own=False):
    for path in sorted(runs.iterdir()):
        name = path.name
        if exclude_own and name == NAME + ".submit.txt":
            continue
        is_submission = name.endswith((".submit.txt", ".submit.json", ".submitted_at.txt"))
        named = TAG.search(name)
        if named and (is_submission or ("check" in name or ".skip." in name) and name.endswith((".txt", ".json"))):
            return name
        if not (is_submission or JUDGE_FILE.fullmatch(name)):
            continue
        raw = path.read_text()
        if is_submission:
            first = raw.splitlines()[0] if raw else ""
            try:
                meta = json.loads(first)
            except ValueError:
                meta = None
            if isinstance(meta, dict):
                if (TAG.search(str(meta.get("candidate", ""))) or meta.get("sha256") == V129.sha256
                        or TAG.search(str(meta.get("directory", "")))):
                    return name
                # An unrelated candidate may cite V129 as a comparator in its own
                # metadata. Inspect the CLI output separately for conflicting evidence.
                if meta.get("candidate") and meta.get("sha256"):
                    raw = raw.partition("\n")[2]
            if TAG.search(raw) or V129.sha256 in raw:
                return name
        elif V129.sha256 in raw or TAG.search(raw):
            # Official per-job files carry the patch hash under result.patch when complete.
            # A damaged or unidentified receipt mentioning this payload also blocks a retry.
            return name
    return None


def plan(root=ROOT, runs=RUNS):
    base = root / "serving_v0523/candidates"
    if digest(base/V129.directory/"patch.diff") != V129.sha256 or digest(base/V64.directory/"patch.diff") != V64.sha256:
        raise ValueError("frozen candidate or parent patch hash mismatch")
    if not parent_gate(runs):
        raise ValueError("unconfirmed V64 scored one-user reference")
    validation = json.loads((base/V129.directory/"validation.json").read_text())
    if (validation["parent_sha256"], validation["patch_sha256"], validation["launch_flags"]) != (
            V64.sha256, V129.sha256, V64.flags):
        raise ValueError("frozen V129 flags or parent mismatch")
    return dict(mode="check", candidate=V129.name, users=1, sha256=V129.sha256,
                flags=V64.flags, comparison_job=PARENT_JOB, comparison_sha256=V64.sha256,
                start_utc=START.isoformat(), end_utc_exclusive=END.isoformat(), resets_at=RESET,
                max_private_checks=1, reserved_private_checks=2, receipt=NAME+".submit.txt")


def run_once(*, okc, root=ROOT, runs=RUNS, clock=utc_now, execute=subprocess.run):
    def in_window():
        return START <= clock() < END

    if not in_window():
        return "outside submission window"
    old = prior_attempt(runs)
    if old:
        return "attempt recorded in " + old
    frozen = plan(root, runs)
    patch = root/"serving_v0523/candidates"/V129.directory/"patch.diff"
    expected = [okc, "serve", "check", str(patch), "--users", "1", *FLAGS, "--no-wait"]

    def guarded(cmd, **kwargs):
        if not in_window():
            return subprocess.CompletedProcess(cmd, 1, "", "submission window closed")
        if cmd == [okc, "serve", "status", "--json"]:
            kwargs["timeout"] = min(kwargs["timeout"], max(.001, (END-clock()).total_seconds()))
            resp = execute(cmd, **kwargs)
            if resp.returncode:
                return resp
            try:
                quota = json.loads(resp.stdout)
                checks = quota["limits"]["checks"]
                valid = all(type(checks[k]) is int and checks[k] == n
                            for k,n in (("per_day",3),("used",0),("left",3)))
                valid = valid and quota["resets_at"] == RESET
            except (ValueError, TypeError, KeyError):
                valid = False
            if not valid or not in_window() or prior_attempt(runs) or not parent_gate(runs):
                return subprocess.CompletedProcess(cmd, 1, "", "unconfirmed first slot, duplicate, parent, or window")
            return resp
        if cmd != expected:
            raise ValueError("command outside the private V129/1 allocation")
        if prior_attempt(runs, exclude_own=True) or not parent_gate(runs) or digest(patch) != V129.sha256:
            return subprocess.CompletedProcess(cmd, 1, "", "frozen evidence changed before submission")
        remaining = (END-clock()).total_seconds()
        if remaining <= 0 or not in_window():
            return subprocess.CompletedProcess(cmd, 1, "", "submission window closed")
        return execute(cmd, **(kwargs | {"timeout":min(kwargs["timeout"],remaining)}))

    result = submit_once((NAME,V129,1,{"comparison_job":PARENT_JOB,"comparison_sha256":V64.sha256}),
                         okc=okc, now=clock(), runs=runs, root=root, execute=guarded,
                         start=START, end=END, reset=RESET, reserve_checks=frozen["reserved_private_checks"])
    if result.startswith("submitted "):
        receipt = (runs/(NAME+".submit.txt")).read_text()
        ids = set(re.findall(r"\bjob ([0-9a-f]{16})\b", receipt))
        if ids != {result.split()[1]} or not receipt.rstrip().endswith("CLI exit code: 0"):
            return "attempt needs manual reconciliation; no retry"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    run = parser.parse_args().run
    frozen = plan()
    if not run:
        print(json.dumps(frozen,indent=2))
        return
    okc = os.environ.get("OKC", str(Path.home()/".okc/bin/okc"))
    with (RUNS/"nov20_v129_runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another Nov20 V129 runner owns the lock.", flush=True)
            return
        print(json.dumps({"armed_at":utc_now().isoformat(), **frozen}), flush=True)
        while utc_now() < START:
            time.sleep(min(60, max(0,(START-utc_now()).total_seconds())))
        previous = None
        while utc_now() < END:
            try:
                result = run_once(okc=okc)
            except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
                result = f"{type(exc).__name__}: {exc}"
            if result != previous:
                print(json.dumps({"at":utc_now().isoformat(),"result":result}), flush=True)
                previous = result
            if result.startswith(("attempt ", "prior attempt", "submitted ")) or result == "outside submission window":
                break
            time.sleep(min(60, max(0,(END-utc_now()).total_seconds())))
        print(json.dumps({"finished_at":utc_now().isoformat()}), flush=True)


if __name__ == "__main__":
    main()
