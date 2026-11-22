#!/usr/bin/env python3
"""Offer a matched 16-user V180 check only after a terminal, nonrefunded hardware failure of Nov19 V168."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import submit_nov20_v176_check as earlier

peer, fallback, slots, first, measured = earlier.peer, earlier.fallback, earlier.slots, earlier.first, earlier.measured
ROOT, RUNS, START, END, RESET, nov19 = earlier.ROOT, earlier.RUNS, earlier.START, earlier.END, earlier.RESET, earlier.nov19
NAME = "v180_nov20_check16"
V180 = measured.Candidate("v180", "v180_paged_verify_direct_prompt",
    "c7a76bfd0cd6d8f2ea2cbf29eeec4f4197c1d27245c45575f1528aa27d341cc6", nov19.V68.flags)
TREE = "208b625ae43f820d9f06f36e97c2b1e00484d42c"
OWN = (V180, 16, NAME, START, END)
REFERENCE = (nov19.V68, 16, "v68_nov17_check16", measured.START, measured.END)
CHECKS = (*peer.REQUIRED, REFERENCE)


def _reference(data):
    result = data.get("result")
    if (not isinstance(result, dict) or data.get("id") != nov19.V68_JOB or data.get("status") != "done"
            or data.get("board") != "serving" or data.get("mode") != "check" or data.get("duplicate_of") is not None
            or data.get("refunded", False) is not False or data.get("serving_flags") != nov19.V68.flags
            or result.get("passed") is not True or result.get("status") != "ok" or result.get("concurrency") != [16]
            or result.get("serving_flags") != nov19.V68.flags or not isinstance(result.get("patch"), dict)
            or result["patch"].get("sha256") != nov19.V68.sha256):
        return None
    rung = earlier.passing_rung(result)
    stock = result.get("vs_stock")
    if not isinstance(stock, dict) or stock.get("indicative") is not True:
        return None
    measurement = [result.get("node"), data.get("cluster"), result.get("rungs"), stock.get("rungs"), stock.get("ttft_ratio_limit")]
    digest = hashlib.sha256(json.dumps(measurement, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (rung is None or rung["reps"] != 1 or rung["tok_s_per_gpu"] != 17512.9
            or digest != "2517f13656653128811d7c4fee6e6ad192801115a0bf21e91b10f8ce86abdd8d"):
        return None
    return dict(comparison_job=nov19.V68_JOB, comparison_sha256=nov19.V68.sha256, comparison_users=16,
                comparison_rung=rung, comparison_node=result.get("node"), comparison_cluster=data.get("cluster"),
                comparison_vs_stock=result.get("vs_stock"))


def plan(root=ROOT):
    fallback.plan(root)
    base = root / "serving_v0523/candidates"
    for candidate in (V180, nov19.V68, peer.V168):
        if first.digest(base / candidate.directory / "patch.diff") != candidate.sha256:
            raise ValueError("frozen source mismatch for " + candidate.name)
    manifest = json.loads((base / V180.directory / "validation.json").read_text())
    if (manifest["tree"] != TREE or manifest["patch_sha256"] != V180.sha256
            or manifest["launch_flags"] != nov19.V68.flags or nov19.V68_JOB != "04b01f958aa5487c"
            or manifest["parent_sha256"] != nov19.V68.sha256 or manifest["measured_parent_job"] != nov19.V68_JOB
            or manifest["strict_apply_reverse_full_delta"] is not True):
        raise ValueError("V180 source, launch flags or comparator identity changed")
    proof = _reference(json.loads((root / "serving_v0523/runs" / (nov19.V68_JOB + ".json")).read_text()))
    if proof is None:
        raise ValueError("physical V68 hardware receipt mismatch")
    return dict(candidate=V180.name, sha256=V180.sha256, tree=TREE, flags=V180.flags, mode="check", users=16,
                start_utc=START.isoformat(), end_utc_exclusive=END.isoformat(), resets_at=RESET, receipt=NAME+".submit.txt",
                maximum_private_checks=1, required_quota=slots.THIRD, arbitration_lock=slots.LOCK,
                first_two=[CHECKS[0][2], fallback.NAME], terminal_failed_controls=[x[2] for x in CHECKS[1:3]],
                required_terminal_hardware_failure=peer.PARENT[2], comparison=proof,
                requires_unattempted=[r[2] for r in (*fallback.PEERS, peer.OWN, earlier.OWN)])


def local_position(runs):
    if not (runs / (peer.PARENT[2]+".submit.txt")).is_file() or not (runs/(fallback.NAME+".submit.txt")).is_file():
        return None
    base = fallback.local_position(runs)
    if base is None:
        return None
    ids = (*base, slots.bound_receipt(runs,fallback.OWN), slots.bound_receipt(runs,peer.PARENT))
    second = earlier.receipt_evidence(runs,fallback.OWN)
    if (not all(ids) or len(set(ids)) != len(ids) or not isinstance(second,dict)
            or type(second.get("slot")) is not int or second["slot"] != 2 or second.get("v129_job") != ids[0]
            or second.get("failed_controls") != dict(v120b=ids[1],v139=ids[2])
            or any(slots.prior_attempt(runs,r[0],r[2]) for r in (peer.OWN,earlier.OWN))
            or fallback.previous.unique_receipt(runs,peer.PARENT) != ids[4]
            or earlier.alternate_conflict(runs,peer.PARENT,ids[4])
            or slots.bound_receipt(runs,REFERENCE) != nov19.V68_JOB or nov19.V68_JOB in ids):
        return None
    evidence = earlier.receipt_evidence(runs,peer.PARENT)
    if (not isinstance(evidence,dict) or evidence.get("decision_job") != nov19.V93_JOB
            or evidence.get("decision_receipt") != nov19.prefix(nov19.V93)
            or evidence.get("comparison_job") != nov19.V64_PRIVATE_JOB
            or evidence.get("comparison_sha256") != nov19.V64_SHA
            or evidence.get("comparison_report_sha256") != nov19.V64_PRIVATE_REPORT_SHA
            or evidence.get("scored_parent_job") != nov19.V64_JOB
            or evidence.get("exclusive_alternatives") != [c.name for c in nov19.PREFILL_ALTERNATIVES]
            or type(evidence.get("check_load")) is not int or evidence["check_load"] != 16):
        return None
    return "v168_failed_on_hardware", (*ids, nov19.V68_JOB)


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def _candidate_crash(result):
    log, crash = result.get("startup_log"), result.get("crash")
    if (not isinstance(crash, dict) or not isinstance(log, dict)
            or type(log.get("lines")) is not int or log["lines"] <= 0):
        return False
    if crash.get("captured") is True:
        frames = crash.get("frames")
        return (isinstance(crash.get("type"), str) and bool(crash["type"].strip())
                and isinstance(crash.get("message"), str) and bool(crash["message"].strip())
                and isinstance(frames, list) and any(isinstance(f, dict)
                and isinstance(f.get("file"), str) and f["file"].startswith("sglang/")
                and type(f.get("line")) is int and f["line"] > 0
                and isinstance(f.get("function"), str) and bool(f["function"].strip()) for f in frames))
    # The archived event-only failure completed startup, logged the process exit
    # after the first request, and classified the server exception as TypeError.
    return (crash.get("captured") is False and log.get("cut") == "first request"
            and _number(result.get("startup_s")) and result["startup_s"] > 0
            and result.get("error") in ("TypeError", "NameError", "AttributeError", "AssertionError", "IndexError")
            and isinstance(crash.get("events"), list) and "A server process exited with an error." in crash["events"])


def _measured_failure(result):
    quality = result.get("correctness")
    if (isinstance(quality, dict) and quality.get("passed") is False
            and type(quality.get("references")) is int and quality["references"] > 0
            and isinstance(quality.get("bounds"), dict)
            and type(quality["bounds"].get("min_references")) is int
            and 0 < quality["bounds"]["min_references"] <= quality["references"]):
        bounds = quality["bounds"]
        for metric, limit, maximum in (("agreement", "agreement_min", False), ("prefill_top1", "prefill_top1_min", False),
                ("prefill_kl_mean", "prefill_kl_mean", True), ("decode_kl_mean", "decode_kl_mean", True)):
            value, boundary = quality.get(metric), bounds.get(limit)
            if (_number(value) and _number(boundary) and value >= 0 and boundary > 0
                    and (maximum or max(value, boundary) <= 1)
                    and (value > boundary if maximum else value < boundary)):
                return True
    if result.get("status") != "ok":
        return False
    memory = result.get("gpu_memory")
    if (isinstance(memory, dict) and _number(memory.get("limit_gb")) and memory["limit_gb"] == 2
            and _number(memory.get("over_stock_gb")) and memory["over_stock_gb"] > memory["limit_gb"]):
        return True
    vs_stock, measured_rungs = result.get("vs_stock"), result.get("rungs")
    if (not isinstance(vs_stock, dict) or vs_stock.get("indicative") is not True
            or not _number(vs_stock.get("ttft_ratio_limit")) or vs_stock["ttft_ratio_limit"] != 2
            or not isinstance(vs_stock.get("rungs"), list) or len(vs_stock["rungs"]) != 1
            or not isinstance(measured_rungs, list) or len(measured_rungs) != 1):
        return False
    paired, measured = vs_stock["rungs"][0], measured_rungs[0]
    if (not all(isinstance(r, dict) and type(r.get("concurrency")) is int and r["concurrency"] == 16
                for r in (paired, measured)) or not isinstance(paired.get("stock"), dict)
            or type(measured.get("reps")) is not int or measured["reps"] < 1 or measured.get("truncated") is not False):
        return False
    for key in ("ttft_first_p90", "ttft_later_p90"):
        elapsed, stock, ratio = measured.get(key + "_s"), paired["stock"].get(key + "_s"), paired.get(key)
        if (all(_number(v) and v > 0 for v in (elapsed, stock, ratio))
                and elapsed > 2 * stock and ratio > 2 and abs(elapsed / stock - ratio) <= 0.02):
            return True
    return False


def hardware_failure(data):
    result = data.get("result")
    if (data.get("status") != "done" or data.get("refunded", False) is not False
            or not isinstance(result, dict) or result.get("passed") is not False
            or type(result.get("gpus")) is not int or result["gpus"] != 8
            or not isinstance(result.get("node"), str) or not result["node"].strip()
            or not _number(result.get("pod_s")) or result["pod_s"] <= 0):
        return False
    if result.get("status") == "crashed":
        return _candidate_crash(result)
    return result.get("status") in ("ok", "incorrect") and _measured_failure(result)


def official_evidence(runs, position, okc, execute):
    if position is None:
        return None
    ids = position[1]
    stamps = {}
    for i in (4,1,2,0,3,5):
        req,job=CHECKS[i],ids[i]
        response=execute([okc,"serve","status",job,"--json"],capture_output=True,text=True,timeout=60)
        if response.returncode:
            return None
        data=json.loads(response.stdout)
        if not isinstance(data,dict) or i in (0,3,4,5) and data.get("refunded",False) is not False:
            return None
        replay = lambda *a, **kw: response
        if i == 4:
            result = data.get("result")
            if (slots.official_state(req, job, okc, replay, runs=runs) != "failed" or not hardware_failure(data)
                    or not isinstance(result.get("patch"), dict) or result["patch"].get("sha256") != peer.V168.sha256
                    or result.get("serving_flags") != peer.V168.flags or result.get("concurrency") != [16]):
                return None
            identity = ("failed", True, data["created_at"])
        else:
            identity = fallback.official(req, job, okc, replay, runs)
        if identity is None or i in (1,2) and not identity[1]:
            return None
        if i in (0,3) and data.get("status") not in ("queued","running","done","failed"):
            return None
        if i == 4:
            failure=data["result"]
            details=dict(job=job, sha256=peer.V168.sha256, status=failure["status"], node=failure["node"],
                         gpus=failure["gpus"], pod_s=failure["pod_s"])
        if i == 5:
            comparison=_reference(data)
            if comparison is None:
                return None
        stamps[i]=identity[2]
    if stamps[0] > stamps[3] or stamps[4] >= stamps[0]:
        return None
    return dict(slot=3,v129_job=ids[0],v163_job=ids[3],failed_controls=dict(v120b=ids[1],v139=ids[2]),
                v168_hardware_failure=details,**comparison)


def run_once(*, okc, root=ROOT, runs=RUNS, clock=first.utc_now, execute=subprocess.run):
    def inside():
        return START <= clock() < END

    def deny(cmd):
        return subprocess.CompletedProcess(cmd, 1, "", "V180 private slot or comparison unavailable")

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
            return slots.prior_attempt(runs, V180, NAME, exclude_own)

        if attempted():
            return "attempt recorded"
        position = local_position(runs)
        proof = official_evidence(runs, position, okc, bounded)
        if proof is None:
            return "waiting for confirmed Nov19 V168 hardware failure, V129 and V163, and measured V68/16"
        plan(root)
        patch = root / "serving_v0523/candidates" / V180.directory / "patch.diff"
        status = [okc, "serve", "status", "--json"]
        expected = [okc, "serve", "check", str(patch), "--users", "16"]
        for key, value in sorted(V180.flags.items()):
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
                raise ValueError("unexpected V180 operation")
            if not inside() or attempted(True) or local_position(runs) != position or first.digest(patch) != V180.sha256:
                return deny(cmd)
            return bounded(cmd, **kw)

        outcome = measured.submit_once((NAME, V180, 16, proof), okc=okc, root=root, runs=runs, now=clock(),
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
    with (RUNS / "nov20_v180_runner.lock").open("a") as lock:
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
