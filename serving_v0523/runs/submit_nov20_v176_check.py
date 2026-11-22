#!/usr/bin/env python3
"""Offer V176/16 in the last private slot if the Nov19 V168 alternative was never chosen."""
import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import submit_nov20_v173_check as peer

fallback, slots, first, measured, nov19 = peer.fallback, peer.slots, peer.first, peer.measured, peer.nov19
ROOT, RUNS, START, END, RESET = peer.ROOT, peer.RUNS, peer.START, peer.END, peer.RESET
V176 = measured.Candidate(
    "v176", "v176_integrated_prompt_decode", "08e5e126fe79b5fb37e706a8cbf8c8fdc449ef6794500200928f350462c44eb6",
    peer.V168.flags | {"--speculative-dflash-share-verify-schedule": "", "--speculative-dflash-sp-prefill-context": ""},
)
TREE = "f1ed4dd4f1d567819e193b8c437717424ace0e9d"
NAME = "v176_nov20_check16"
OWN = (V176, 16, NAME, START, END)
ALTERNATIVES = tuple((c, 16, nov19.prefix(c), nov19.START, nov19.END) for c in (nov19.V98, nov19.V95))
SLOT_HISTORY = (*ALTERNATIVES, *((c, 16, nov19.prefix(c), nov19.START, nov19.END) for c in nov19.LEGACY_PREFILL_ALTERNATIVES))
PRECEDING = (*fallback.REQUIRED, fallback.OWN)


def plan(root=ROOT):
    fallback.plan(root)
    base = root / "serving_v0523/candidates"
    for candidate in (V176, peer.V168, *(r[0] for r in ALTERNATIVES)):
        if first.digest(base / candidate.directory / "patch.diff") != candidate.sha256:
            raise ValueError("frozen patch mismatch for " + candidate.name)
    manifest = json.loads((base / V176.directory / "validation.json").read_text())
    if (manifest["tree"] != TREE or manifest["launch_flags"] != V176.flags or manifest["patch_sha256"] != V176.sha256
            or manifest["parent_sha256"] != peer.PARENT_SHA or manifest["strict_apply_reverse_full_delta"] is not True
            or manifest["checked_flag_patterns"] != 76 or nov19.PREFILL_ALTERNATIVES != (nov19.V98, peer.V168, nov19.V95)):
        raise ValueError("V176 source, flags or Nov19 branch contract changed")
    return dict(candidate=V176.name, sha256=V176.sha256, tree=TREE, flags=V176.flags,
                mode="check", users=16, start_utc=START.isoformat(), end_utc_exclusive=END.isoformat(),
                resets_at=RESET, arbitration_lock=slots.LOCK, maximum_private_checks=1, required_quota=slots.THIRD,
                receipt=NAME + ".submit.txt", nov19_selected_one_of=[r[2] for r in ALTERNATIVES],
                required_failed_controls=[r[2] for r in PRECEDING[1:3]], first_two=[PRECEDING[0][2], fallback.NAME],
                requires_unattempted=[r[2] for r in (*fallback.PEERS, peer.OWN, peer.PARENT)],
                comparison=dict(candidate=fallback.V163.name, sha256=fallback.V163.sha256, users=16,
                                reference=fallback.NAME, scope="whole-package comparison with a passing ancestor",
                                stock_private_reference=fallback.REFERENCE))


def receipt_evidence(runs, req):
    try:
        with (runs / (req[2] + ".submit.txt")).open() as receipt:
            header = json.loads(receipt.readline())
        return header.get("evidence") if isinstance(header, dict) else None
    except (OSError, ValueError):
        return None


def alternate_conflict(runs, selected, selected_job):
    """Find orphan, renamed or duplicate Nov19 checks for either alternative."""
    tags = {req[0].name: re.compile(r"(?<![a-z0-9])" + req[0].name + r"(?![a-z0-9])", re.I)
            for req in SLOT_HISTORY}
    canonical = runs / (selected[2] + ".submit.txt")
    for p in runs.rglob("*"):
        submitted = p.name.endswith((".submit.txt", ".submit.json", ".submitted_at.txt", ".skip.json"))
        judge = first.JUDGE_FILE.fullmatch(p.name)
        if p == canonical or not p.is_file() or not (submitted or judge):
            continue
        raw = p.read_text()
        line, _, tail = raw.partition("\n")
        try:
            data = json.loads(raw if judge else line)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            data = {}
        if submitted and data.get("candidate") and data.get("sha256"):
            # Evidence in other governed receipt headers can refer to these candidates.
            text = " ".join((p.name, str(data["candidate"]), str(data["sha256"]), tail))
        else:
            text = p.name + " " + raw
        for candidate, users, name, begin, end in SLOT_HISTORY:
            if not (tags[candidate.name].search(text) or candidate.sha256 in text):
                continue
            if judge:
                if candidate == selected[0] and p.name == selected_job + ".json":
                    continue
                stamp = data.get("created_at")
                result = data.get("result")
                verified = (data.get("id") == p.stem and data.get("board") == "serving"
                            and isinstance(result, dict) and isinstance(result.get("patch"), dict)
                            and result["patch"].get("sha256") == candidate.sha256)
                loads = result.get("concurrency") if verified else None
                other_load = (isinstance(loads, list) and len(loads) == 1
                              and type(loads[0]) is int and loads[0] > 0 and loads[0] != users)
                if verified and (data.get("mode") == "run" or other_load or type(stamp) in (float, int) and math.isfinite(stamp)
                                 and not begin.timestamp() <= stamp < end.timestamp()):
                    continue
            elif p.name.endswith(".submit.txt"):
                # A different dated, successfully accepted check can be unrelated to the Nov19 choice.
                record_name = p.name.removesuffix(".submit.txt")
                load = data.get("users")
                if type(load) is int and load > 0 and load != users and slots.accepted_receipt(p.parent, candidate, load, record_name):
                    continue
                if slots.accepted_receipt(p.parent, candidate, users, record_name) and not slots.accepted_receipt(
                    p.parent, candidate, users, record_name, (begin, end)
                ):
                    try:
                        attempted = datetime.fromisoformat(data["attempted_at"])
                    except (ValueError, TypeError, KeyError):
                        return True
                    if attempted.tzinfo and not begin <= attempted < end:
                        continue
            return True
    return False


def local_position(runs):
    if not (runs / (fallback.NAME + ".submit.txt")).is_file():
        return None
    selected = [req for req in ALTERNATIVES if (runs / (req[2] + ".submit.txt")).is_file()]
    if len(selected) != 1:
        return None
    alternate = selected[0]
    if (nov19.prefill_attempt(runs, exclude=runs / (alternate[2] + ".submit.txt"))
            or any(slots.prior_attempt(runs, req[0], req[2]) for req in (peer.OWN, peer.PARENT))):
        return None
    ids = tuple(slots.bound_receipt(runs, req) for req in (*PRECEDING, alternate))
    if not all(ids) or len(set(ids)) != len(ids) or alternate_conflict(runs, alternate, ids[4]) or any(
        slots.prior_attempt(runs, req[0], req[2]) for req in fallback.PEERS
    ):
        return None
    second = receipt_evidence(runs, fallback.OWN)
    branch = receipt_evidence(runs, alternate)
    if (not isinstance(second, dict) or type(second.get("slot")) is not int or second["slot"] != 2
            or second.get("v129_job") != ids[0] or second.get("failed_controls") != dict(v120b=ids[1], v139=ids[2])
            or not isinstance(branch, dict) or type(branch.get("check_load")) is not int or branch["check_load"] != 16
            or branch.get("comparison_receipt") != nov19.prefix(nov19.V93) or branch.get("comparison_job") != nov19.V93_JOB
            or branch.get("exclusive_alternatives") != [c.name for c in nov19.PREFILL_ALTERNATIVES]):
        return None
    return alternate, ids


def passing_rung(result):
    quality = result.get("correctness")
    if not isinstance(quality, dict) or quality.get("passed") is not True:
        return None
    if "short_outputs" in result and (type(result["short_outputs"]) is not int or result["short_outputs"] != 0):
        return None
    if "later_turns" in result:
        later = result["later_turns"]
        if (not isinstance(later, dict) or later.get("ran") is not True or later.get("passed") is not True
                or later.get("incomplete", False) is not False or later.get("insufficient", False) is not False):
            return None
    rungs = result.get("rungs")
    if not isinstance(rungs, list) or len(rungs) != 1 or not isinstance(rungs[0], dict):
        return None
    rung = rungs[0]
    if (type(rung.get("concurrency")) is not int or rung["concurrency"] != 16
            or rung.get("truncated") is not False or type(rung.get("reps")) is not int or rung["reps"] < 1):
        return None
    rates = [rung.get(k) for k in ("tok_s_per_gpu", "tok_s_per_user_e2e", "tok_s_per_user_p50")]
    rates.extend(v for k, v in rung.items() if k.startswith(("ttft_", "tok_s_")))
    return rung if all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in rates) else None


def official_evidence(runs, position, okc, execute):
    if position is None:
        return None
    alternate, ids = position
    checks = (*PRECEDING, alternate)
    stamps = {}
    # Pending V163 normally needs just this one query.
    for i in (3, 4, 1, 2, 0):
        req, job = checks[i], ids[i]
        response = execute([okc, "serve", "status", job, "--json"], capture_output=True, text=True, timeout=60)
        if response.returncode:
            return None
        data = json.loads(response.stdout)
        if not isinstance(data, dict) or i in (0, 3, 4) and data.get("refunded", False) is not False:
            return None
        identity = fallback.official(req, job, okc, lambda *a, **kw: response, runs)
        if identity is None or i in (1, 2) and not identity[1]:
            return None
        stamps[i] = identity[2]
        if i == 3:
            if identity[0] != "passed" or data["status"] != "done":
                return None
            result = data["result"]
            rung = passing_rung(result)
            if rung is None:
                return None
            comparison = dict(comparison_job=job, comparison_sha256=req[0].sha256, comparison_users=16,
                              comparison_rung=rung, comparison_node=result.get("node"), comparison_cluster=data.get("cluster"),
                              comparison_vs_stock=result.get("vs_stock"))
        if i == 4 and data["status"] not in ("queued", "running", "done", "failed"):
            return None
    if stamps[0] > stamps[3] or stamps[4] >= stamps[0]:
        return None
    return dict(slot=3, v129_job=ids[0], failed_controls=dict(v120b=ids[1], v139=ids[2]), v163_job=ids[3],
                nov19_selected=alternate[0].name, nov19_job=ids[4], nov19_sha256=alternate[0].sha256, **comparison)


def run_once(*, okc, root=ROOT, runs=RUNS, clock=first.utc_now, execute=subprocess.run):
    def inside():
        return START <= clock() < END

    def deny(cmd):
        return subprocess.CompletedProcess(cmd, 1, "", "V176 private slot or comparison unavailable")

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
            return slots.prior_attempt(runs, V176, NAME, exclude_own)

        if attempted():
            return "attempt recorded"
        position = local_position(runs)
        proof = official_evidence(runs, position, okc, bounded)
        if proof is None:
            return "waiting for settled Nov19 alternative, failed controls and passing V163/16 in slot two"
        plan(root)
        patch = root / "serving_v0523/candidates" / V176.directory / "patch.diff"
        status = [okc, "serve", "status", "--json"]
        expected = [okc, "serve", "check", str(patch), "--users", "16"]
        for key, value in sorted(V176.flags.items()):
            expected.extend(("--flag", key[2:] + ("=" + value if value else "")))
        expected.append("--no-wait")

        def quota(**kw):
            response = bounded(status, **kw)
            if response.returncode:
                return response
            data = json.loads(response.stdout)
            limits = data["limits"]["checks"]
            if (data.get("resets_at") != RESET or not isinstance(limits, dict)
                    or any(type(limits.get(k)) is not int or limits[k] != v for k, v in slots.THIRD.items())):
                return deny(status)
            return response

        def still_local():
            return inside() and not attempted() and local_position(runs) == position

        def guarded(cmd, **kw):
            if cmd == status:
                response = quota(**kw)
                if (response.returncode or not still_local()
                        or official_evidence(runs, position, okc, bounded) != proof or not still_local()):
                    return deny(cmd)
                plan(root)
                response = quota(**kw)
                return response if not response.returncode and still_local() else deny(cmd)
            if cmd != expected:
                raise ValueError("unexpected V176 operation")
            if not inside() or attempted(True) or local_position(runs) != position or first.digest(patch) != V176.sha256:
                return deny(cmd)
            return bounded(cmd, **kw)

        outcome = measured.submit_once((NAME, V176, 16, proof), okc=okc, root=root, runs=runs, now=clock(),
                                      execute=guarded, start=START, end=END, reset=RESET, reserve_checks=0)
        if outcome.startswith("submitted "):
            job = outcome.split()[1]
            if job in position[1] or slots.bound_receipt(runs, OWN) != job:
                return "attempt needs manual reconciliation; no retry"
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    frozen = plan()
    if not args.run:
        print(json.dumps(frozen, indent=2))
        return
    okc = os.environ.get("OKC", str(Path.home() / ".okc/bin/okc"))
    with (RUNS / "nov20_v176_runner.lock").open("a") as lock:
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
            time.sleep(min(60, max(0, (END-first.utc_now()).total_seconds())))
        print(json.dumps(dict(finished_at=first.utc_now().isoformat())), flush=True)


if __name__ == "__main__":
    main()
