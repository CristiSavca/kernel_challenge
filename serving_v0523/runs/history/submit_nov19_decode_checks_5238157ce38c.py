#!/usr/bin/env python3
"""Select up to three private serving comparisons on November 19."""

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

from submit_nov17_integrated_checks import Candidate, ROOT, RUNS, V68, submit_once
from submit_nov18_decode_checks import (
    BASE_FLAGS, V68_JOB, V72a, V76a, V79a, V80a, V84a, V87, job_state,
    receipt_name as nov18_receipt_name, status_from_receipt, utc_now,
)

START = datetime(2026, 11, 19, 0, 3, tzinfo=timezone.utc)
END = datetime(2026, 11, 19, 22, tzinfo=timezone.utc)
RESET = "2026-11-20T00:00:00Z"
V89 = Candidate(
    "v89", "v89_paired_expert_dequant",
    "e20ea10ed16c9b4dec6e0d4c834ef44c086f5cbf468ee6574a5a258800da6e18", V87.flags,
)
V116 = Candidate(
    "v116", "v116_staged_active_experts",
    "341ad47ffba45fb7521fa0a4d1795140f65509a9591dfc363772ac434e19beeb", V89.flags,
)
V116b = Candidate(
    "v116b", "v116b_staged_active_experts_regcap",
    "8d20646d565613a685baea176e49a91ae5c395ec0bb477600a8f3180f07517b8", V116.flags,
)
V93_JOB = "75557c21595347de"
V93 = Candidate(
    "v93", "v93_expert_argmax_dense_prefill",
    "995c3f3c78750bbf99e6c3db3f8ab0fa8353be87cbffb07b1d15c49e6a4c0fdb",
    V89.flags | {"--speculative-dflash-reduced-target-argmax": ""},
)
V98 = Candidate(
    "v98", "v98_graph_grouped_prefill",
    "c1b8ffcd56c3fbb3c80ae008ab0f10bfe8f9d601ac6822f8d1a27d5c7aad2525", V93.flags,
)
V95 = Candidate(
    "v95", "v95_v64_dense_fp8_prefill",
    "e9359287a5e70dff46449743557bf4622c121ae7adb0c36c119a1b2d7ca311e9", BASE_FLAGS,
)
V99 = Candidate(
    "v99", "v99_adaptive_draft_tree",
    "b481fac41f1b182a0552ebb381e3305f201a13b9005639906094c6655c595bb4", V93.flags | {"--speculative-dflash-adaptive-tree": ""},
)
V101b = Candidate(
    "v101b", "v101b_endpoint_guarded_spine",
    "5efac32facda3a20a0e7ba5fb1b1724454eba4e2439c4fe95da11cba67b1e39e",
    V99.flags | {"--speculative-dflash-adaptive-spine": ""},
)
V102 = Candidate(
    "v102", "v102_v64_graph_grouped_prefill",
    "ee8bf58ecb98548d236b54f774e948139fff0659c3479b6586b3243142daa5b6", BASE_FLAGS,
)
V87_JOB = "00c134c978104d65"
V64_JOB = "abd217baf0ed49fd"
V64_SHA = "f6aaf93ce06ccfce9461e06bc3f71dea05dd901c01497c8be31bd2c515da689a"
V100c = Candidate(
    "v100c", "v100c_v64_paged_prefill_control",
    "335845f55f6e584e938dcb5bba72c18f89677f83f78a1374692525306cd635ed",
    BASE_FLAGS | {"--page-size": "160", "--chunked-prefill-size": "16320",
                  "--max-prefill-tokens": "16320", "--cuda-graph-max-bs-prefill": "16384"},
)
V106 = Candidate(
    "v106", "v106_paged_swa_suffix_prefill",
    "376d808919cba9ba365fca58147802d03d836ab26f96a521d0bf6b1cee5bf14c", V100c.flags,
)
V111 = Candidate(
    "v111", "v111_paged_prefill_verify",
    "3b899d0739a395a8a96bbcaf4a97794b6b2b07c8e0ba2a5a646ab836f5db943a",
    V106.flags | {"--speculative-dflash-sp-prefill-context": ""},
)
V120 = Candidate(
    "v120", "v120_phase_parallel_draft_context",
    "06ab5d1589ea8030947eeba6c78c361c4b072a0aef9fd08219356302b0ae1c35",
    V111.flags | {"--speculative-dflash-direct-prefix-kv": "",
                  "--speculative-dflash-parallel-verify-context": ""},
)
V120b = Candidate(
    "v120b", "v120b_native_muse_prompt",
    "0cb0a90248b98b07047075964ee6bf7200d9ebaadcf9b2eb4b10fd6098f9f7b7", V120.flags,
)
LEGACY_EXPERT_ALTERNATIVES = (V100c, V106, V89, V116, V111, V120)
ALTERNATIVES = (V116b, V120b, V80a)
PREFILL_ALTERNATIVES = (V98, V102, V95)
V125 = Candidate(
    "v125", "v125_v68_clustered_expert_mlp",
    "3564873a4327719c83fa88a3d1eae5cf3f06d6c42b8505c45a7403a883d658d3", V68.flags,
)
V126 = Candidate(
    "v126", "v126_split_cluster_expert_mlp",
    "eeba21105433a3bc908b943eb3dfd04b1c8213f474a84d556db654c2f0670c27", V68.flags,
)
LEGACY_TREE_ALTERNATIVES = (V99, V125)
TREE_ALTERNATIVES = (V101b, V79a, V126)
CANDIDATES = (*ALTERNATIVES, *PREFILL_ALTERNATIVES, *TREE_ALTERNATIVES)


def check_load(candidate):
    if candidate == V126:
        return 1
    if candidate in (V100c, V106, V111, V120, V120b):
        return 32
    return 4 if candidate in (V87, V89, V116, V116b) else 16


def prefix(candidate):
    day = "nov18" if candidate in (V72a, V76a, V87, V93) else "nov19"
    return f"{candidate.name}_{day}_check{check_load(candidate)}"


def expert_choice(runs, okc, execute):
    """Return the eligible alternative only when V87's submission and outcome are identifiable."""
    name = prefix(V87)
    receipt = runs / f"{name}.submit.txt"
    skipped = runs / f"{name}.skip.json"
    if skipped.exists():
        data = json.loads(skipped.read_text())
        if (not receipt.exists() and isinstance(data, dict) and data.get("candidate") == V87.name
                and isinstance(data.get("reason"), str) and data["reason"]):
            return V80a, "V87 was skipped"
        return None, "inconsistent V87 skip and attempt evidence"
    if not receipt.exists():
        return None, "V87 has no confirmed receipt"
    text = receipt.read_text()
    ids = set(re.findall(r"\bjob ([0-9a-f]{16})\b", text))
    if ids != {V87_JOB} or not text.rstrip().endswith("CLI exit code: 0"):
        return None, "V87 has an unconfirmed CLI receipt"
    meta = json.loads(text.splitlines()[0])
    if not isinstance(meta, dict) or (meta.get("candidate"), meta.get("sha256"), meta.get("flags"),
                                     meta.get("users")) != (V87.name, V87.sha256, V87.flags, 4):
        return None, "V87 submission metadata mismatch"
    return parent_choice(V87, V87_JOB, 4, V116b, V80a, okc, execute, pending=V120b,
                         crashed=V120b)


def v64_gate(runs):
    try:
        data = json.loads((runs / f"{V64_JOB}.json").read_text())
    except (OSError, ValueError):
        return "waiting", "V64 scored receipt unavailable"
    if not isinstance(data, dict) or (
        data.get("id"), data.get("board"), data.get("mode"), data.get("status"), data.get("serving_flags")
    ) != (V64_JOB, "serving", "run", "done", BASE_FLAGS):
        return "waiting", "V64 scored receipt identity mismatch"
    result = data.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("patch"), dict) or (
        result.get("status"), result.get("serving_flags"), result.get("concurrency"), result["patch"].get("sha256")
    ) != ("ok", BASE_FLAGS, [1, 4, 16, 32], V64_SHA) or result.get("passed") is not True:
        return "waiting", "V64 completed result does not confirm the frozen four-load parent"
    return "passed", "V64 passed its scored four-load run"


def parent_choice(parent, job, users, healthy, failed, okc, execute, *, pending=None, crashed=None):
    response = execute([okc, "serve", "status", job, "--json"],
                       capture_output=True, text=True, timeout=60)
    if response.returncode:
        return None, f"{parent.name.upper()} status unavailable"
    data = json.loads(response.stdout)
    if not isinstance(data, dict) or (data.get("id"), data.get("board"), data.get("mode")) != (
            job, "serving", "check"):
        return None, f"{parent.name.upper()} job identity mismatch"
    if (data.get("users") not in (None, users) or
            data.get("serving_flags") is not None and data["serving_flags"] != parent.flags):
        return None, f"{parent.name.upper()} live load or flags mismatch"
    result = data.get("result")
    if result is not None:
        if not isinstance(result, dict):
            return None, f"{parent.name.upper()} result is malformed"
        patch = result.get("patch")
        if (patch is not None and (not isinstance(patch, dict) or patch.get("sha256") != parent.sha256)
                or result.get("serving_flags", parent.flags) != parent.flags
                or result.get("concurrency", [users]) != [users]):
            return None, f"{parent.name.upper()} judge payload mismatch"
    state = data.get("status")
    if state in ("queued", "running"):
        return (healthy if pending is None else pending), f"{parent.name.upper()} is {state}"
    if state in ("failed", "canceled", "cancelled", "withdrawn"):
        return failed, f"{parent.name.upper()} is {state}"
    if state != "done" or not result:
        return None, f"{parent.name.upper()} has no conclusive state"
    if crashed is not None and result.get("status") == "crashed":
        # A runtime crash with no measured rung leaves the shared decode branch unresolved.
        if (result.get("passed") is False and result.get("concurrency") == [users]
                and result.get("serving_flags") == parent.flags
                and isinstance(result.get("patch"), dict) and result["patch"].get("sha256") == parent.sha256):
            return crashed, f"{parent.name.upper()} crashed on the confirmed payload"
        return None, f"{parent.name.upper()} crash evidence is incomplete"
    if result.get("passed") is False or result.get("status") in ("failed", "error", "timeout"):
        return failed, f"{parent.name.upper()} failed"
    if (result.get("passed") is True and result.get("status") == "ok"
            and isinstance(result.get("patch"), dict) and result["patch"].get("sha256") == parent.sha256
            and result.get("serving_flags") == parent.flags and result.get("concurrency") == [users]):
        return healthy, f"{parent.name.upper()} passed at {users} users"
    return None, f"{parent.name.upper()} result is incomplete"


def v93_choice(runs, okc, execute, healthy, failed, *, pending=None):
    name = prefix(V93)
    receipt = runs / f"{name}.submit.txt"
    if not receipt.is_file() or (runs / f"{name}.skip.json").exists():
        return None, "missing or conflicting V93 submission evidence"
    try:
        text = receipt.read_text()
        meta = json.loads(text.splitlines()[0])
        ids = set(re.findall(r"\bjob ([0-9a-f]{16})\b", text))
        if (ids != {V93_JOB} or not text.rstrip().endswith("CLI exit code: 0")
                or not isinstance(meta, dict)
                or (meta.get("candidate"), meta.get("sha256"), meta.get("flags"), meta.get("users"))
                != (V93.name, V93.sha256, V93.flags, 16)):
            return None, "unconfirmed or mismatched V93 receipt"
        return parent_choice(V93, V93_JOB, 16, healthy, failed, okc, execute, pending=pending)
    except (ValueError, TypeError, IndexError):
        return None, "malformed V93 evidence"


def prefill_gate(candidate, runs, okc, execute):
    v93, reason = v93_choice(runs, okc, execute, V98, V95, pending=V102)
    if v93 is None:
        return "waiting", reason
    try:
        v68, detail = parent_choice(V68, V68_JOB, 16, V98, V95, okc, execute)
        if v68 is None:
            return "waiting", detail
    except (ValueError, TypeError, IndexError):
        return "waiting", "malformed V68 evidence"
    choice = V95 if V95 in (v93, v68) else v93
    if choice != candidate:
        return "waiting", f"{choice.name} is eligible for the conditional slot"
    if choice == V102:
        return v64_gate(runs)
    if choice == V98:
        state, parent_detail = optional_parent_gate(runs, okc, execute)
        if state != "passed":
            return "waiting", parent_detail
    return "passed", f"{reason}; {detail}"


def optional_parent_gate(runs, okc, execute):
    for parent in (V72a, V76a):
        name = prefix(parent)
        if not any((runs / f"{name}.{end}").exists() for end in ("submit.txt", "skip.json")):
            continue
        state, detail = status_from_receipt(parent, name, runs, okc, execute)
        if state not in ("passed", "queued", "running"):
            return state, detail
    return "passed", "shared parent checks are eligible"


def dependency(candidate, runs, okc, execute):
    if candidate in PREFILL_ALTERNATIVES:
        return prefill_gate(candidate, runs, okc, execute)
    if candidate in ALTERNATIVES:
        try:
            choice, reason = expert_choice(runs, okc, execute)
        except (OSError, ValueError, TypeError, IndexError, subprocess.SubprocessError):
            return "waiting", "V87 evidence unavailable"
        if choice != candidate:
            return "waiting", reason
        if candidate == V120b:
            return v64_gate(runs)
    state, detail = job_state(V68_JOB, V68.sha256, V68.flags, okc=okc, execute=execute)
    if state not in ("passed", "queued", "running"):
        return state, detail
    if candidate != V126:
        parent_state, detail = optional_parent_gate(runs, okc, execute)
        if parent_state != "passed":
            return parent_state, detail
    if candidate not in ALTERNATIVES:
        # V68 validates the shared serving path. The V64 score supplies the
        # one-user reference for the complete candidate.
        fallback = V126 if state == "passed" else V79a
        try:
            selected, reason = v93_choice(runs, okc, execute, V101b, fallback, pending=fallback)
        except (OSError, subprocess.SubprocessError):
            selected, reason = None, "V93 status unavailable"
        if selected is None and fallback == V79a:
            selected = V79a
        if selected != candidate:
            return "waiting", reason
        if candidate == V126:
            return v64_gate(runs)
    return "passed", reason


def run_candidate(candidate, *, runs=RUNS, root=ROOT, okc, clock=utc_now, execute=subprocess.run):
    if not START <= clock() < END:
        return "outside submission window"
    if candidate not in CANDIDATES:
        raise ValueError("candidate is outside the frozen Nov19 allocation")
    if candidate in ALTERNATIVES:
        lock_name, members = "nov19_expert_alternative.lock", (*ALTERNATIVES, *LEGACY_EXPERT_ALTERNATIVES)
    elif candidate in PREFILL_ALTERNATIVES:
        # An old runner's V84a attempt occupies this same slot across updates.
        lock_name, members = "nov19_prefill_alternative.lock", (*PREFILL_ALTERNATIVES, V84a)
    else:
        lock_name, members = "nov19_tree_alternative.lock", (*TREE_ALTERNATIVES, *LEGACY_TREE_ALTERNATIVES)
    # Keep alternate processes and restarts from claiming both candidates.
    # submit_once exclusively writes its candidate's durable receipt before invoking the CLI.
    with (runs / lock_name).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "waiting for alternative lock"
        if candidate in TREE_ALTERNATIVES and (runs / "v139_nov19_check4.submit.txt").exists():
            return "attempt recorded"
        if any((runs / f"{prefix(other)}.submit.txt").exists() for other in members):
            return "attempt recorded"
        return _offer(candidate, runs, root, okc, clock, execute)


def _offer(candidate, runs, root, okc, clock, execute):
    name = prefix(candidate)
    skip = runs / f"{name}.skip.json"
    # An older Nov18 process could have recorded the same frozen width experiment.
    prior = runs / f"{nov18_receipt_name(V79a)}.submit.txt"
    if (runs / f"{name}.submit.txt").exists() or skip.exists() or (candidate == V79a and prior.exists()):
        return "attempt recorded"
    if not START <= clock() < END:
        return "outside submission window"

    def bounded(command, **kwargs):
        remaining = (END - clock()).total_seconds()
        if remaining <= 0:
            raise TimeoutError("submission window closed")
        return execute(command, **(kwargs | {"timeout": min(kwargs["timeout"], remaining)}))

    def gate():
        state, detail = dependency(candidate, runs, okc, bounded)
        if state == "failed":
            with skip.open("x") as output:
                json.dump({"candidate": candidate.name, "at": clock().isoformat(),
                           "reason": detail}, output)
                output.write("\n")
        return state, detail

    state, detail = gate()
    if state != "passed":
        return f"{state} {detail}"

    def guarded(command, **kwargs):
        if command[1:] == ["serve", "status", "--json"]:
            response = bounded(command, **kwargs)
            if response.returncode:
                return response
            # Reconcile the parent outcomes again immediately after reading the quota.
            state, reason = gate()
            if state != "passed":
                return subprocess.CompletedProcess(command, 1, "", reason)
            # A legacy receipt can be restored during the quota or final parent query.
            if candidate in ALTERNATIVES and any(
                (runs / f"{prefix(legacy)}.submit.txt").exists() for legacy in LEGACY_EXPERT_ALTERNATIVES
            ):
                return subprocess.CompletedProcess(command, 1, "", "earlier first-slot attempt recorded")
            if candidate in TREE_ALTERNATIVES and any(
                (runs / f"{prefix(legacy)}.submit.txt").exists() for legacy in LEGACY_TREE_ALTERNATIVES
            ):
                return subprocess.CompletedProcess(command, 1, "", "earlier third-slot attempt recorded")
            return response
        if command[1:3] != ["serve", "check"]:
            raise ValueError("unexpected CLI operation")
        return bounded(command, **kwargs)

    evidence = {"shared_jobs": [V68_JOB, prefix(V72a), prefix(V76a)],
                "check_load": check_load(candidate)}
    if candidate in ALTERNATIVES:
        evidence["comparison_receipt"] = prefix(V87)
        evidence["exclusive_alternatives"] = [c.name for c in ALTERNATIVES]
        if candidate == V120b:
            evidence.update(shared_jobs=[], decision_receipt=prefix(V87),
                            comparison_receipt=f"{V64_JOB}.json", comparison_job=V64_JOB)
    else:
        evidence["comparison_receipt"] = prefix(V93)
        evidence["comparison_job"] = V93_JOB
        alternatives = PREFILL_ALTERNATIVES if candidate in PREFILL_ALTERNATIVES else TREE_ALTERNATIVES
        evidence["exclusive_alternatives"] = [c.name for c in alternatives]
        if candidate == V126:
            evidence.update(shared_jobs=[V68_JOB], decision_receipt=prefix(V93), decision_job=V93_JOB,
                            comparison_receipt=f"{V64_JOB}.json", comparison_job=V64_JOB, ancestor_check_job=V68_JOB,
                            comparison_scope="one-user whole-build rung")
        if candidate == V102:
            evidence.update(shared_jobs=[V68_JOB], decision_receipt=prefix(V93), decision_job=V93_JOB,
                            comparison_receipt=f"{V64_JOB}.json", comparison_job=V64_JOB)
    offer = (name, candidate, check_load(candidate), evidence)
    return submit_once(offer, okc=okc, now=clock(), root=root, runs=runs, execute=guarded,
                       start=START, end=END, reset=RESET, reserve_checks=0)


def plan():
    checks = {}
    for candidate in CANDIDATES:
        patch = ROOT / "serving_v0523/candidates" / candidate.directory / "patch.diff"
        if hashlib.sha256(patch.read_bytes()).hexdigest() != candidate.sha256:
            raise ValueError(f"{candidate.name} frozen payload mismatch")
        checks[candidate.name] = {"candidate": candidate.name, "users": check_load(candidate),
                                  "flags": candidate.flags, "sha256": candidate.sha256}
    first = {"alternatives": [checks[c.name] for c in ALTERNATIVES], "decision_receipt": prefix(V87),
             "when_passed": V116b.name, "when_queued_or_running_and_v64_passed": V120b.name,
             "legacy_attempt_receipts": [prefix(c) for c in LEGACY_EXPERT_ALTERNATIVES],
             "v64_scored_receipt": f"{V64_JOB}.json", "when_crashed_and_v64_passed": V120b.name,
             "when_other_failure_or_skipped": V80a.name,
             "when_unknown": "defer this slot"}
    second = {"alternatives": [checks[c.name] for c in PREFILL_ALTERNATIVES], "decision_receipt": prefix(V93), "v93_job": V93_JOB,
              "when_v93_passed_and_shared_parents_eligible": V98.name,
              "when_v93_queued_or_running_and_v64_passed": V102.name,
              "v64_scored_receipt": f"{V64_JOB}.json", "legacy_attempt_receipts": [prefix(V84a)],
              "when_v93_or_v68_failed": V95.name, "when_unknown": "defer this slot"}
    third = {"alternatives": [checks[c.name] for c in TREE_ALTERNATIVES], "decision_receipt": prefix(V93), "v93_job": V93_JOB,
             "when_v93_passed_and_shared_parents_eligible": V101b.name, "legacy_attempt_receipts": [prefix(c) for c in LEGACY_TREE_ALTERNATIVES],
             "when_v93_pending_or_failed_and_exact_v68_and_v64_passed": V126.name,
             "v126_ancestor_check_job": V68_JOB, "v126_one_user_reference_job": V64_JOB,
             "when_v68_pending_and_shared_parents_eligible": V79a.name,
             "when_v68_passed_but_v93_unknown": "defer this slot"}
    return {"start_utc": START.isoformat(), "end_utc": END.isoformat(), "quota_reset": RESET,
            "v68_job": V68_JOB, "optional_parent_receipts": [prefix(V72a), prefix(V76a)],
            "allow_pending_parents": True, "reserve_checks": 0, "maximum_private_checks": 3, "checks": [first, second, third]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    run = parser.parse_args().run
    frozen = plan()
    if not run:
        print(json.dumps(frozen, indent=2))
        return
    okc = os.environ.get("OKC", str(Path.home() / ".okc/bin/okc"))
    with (RUNS / "nov19_decode_runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another Nov19 runner owns the lock.", flush=True)
            return
        print(json.dumps({"armed_at": utc_now().isoformat(), **frozen}), flush=True)
        previous = None
        next_poll = START.timestamp()
        while utc_now() < END:
            until = min(next_poll, END.timestamp()) - time.time()
            if until > 0:
                time.sleep(min(until, 60))
                continue
            results = {}
            for candidate in CANDIDATES:
                try:
                    results[candidate.name] = run_candidate(candidate, okc=okc)
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    results[candidate.name] = f"{type(error).__name__}: {error}"
            if results != previous:
                print(json.dumps({"at": utc_now().isoformat(), "results": results}), flush=True)
                previous = results
            if all(result == "attempt recorded" or result.startswith(("submitted ", "failed "))
                   for result in results.values()):
                break
            next_poll = time.time() + 300
        print(json.dumps({"finished_at": utc_now().isoformat()}), flush=True)


if __name__ == "__main__":
    main()
