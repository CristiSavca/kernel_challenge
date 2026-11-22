#!/usr/bin/env python3
"""Submit only the frozen Nov 17 private experiments when their gates hold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "serving_v0523/runs"
START = datetime(2026, 11, 17, 0, 3, tzinfo=timezone.utc)
END = datetime(2026, 11, 17, 23, 30, tzinfo=timezone.utc)
SELECT_32_BY = datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc)
RESET = "2026-11-18T00:00:00Z"
SWA = {"--swa-full-tokens-ratio": "0.25"}
HUMMING = SWA | {
    "--moe-runner-backend": "humming",
    "--speculative-moe-runner-backend": "marlin",
}
SP = {"--enable-layernorm-sp": ""}


@dataclass(frozen=True)
class Candidate:
    name: str
    directory: str
    sha256: str
    flags: dict[str, str]
    check16: str | None = None


V50 = Candidate("v50", "v50_dense_compact_finalize",
    "7ba1708e049dbfae89eac613efe0bdeb9b7c8ccbcf0c11859ccad49f7f4441ae", SWA,
    "fec66419d5154815")
V53 = Candidate("v53", "v53_humming_indexed",
    "341c37c4d6e8cd7011218597de42870023fe2a33a046c6e2c7249d123107cb48", HUMMING,
    "54654743532e4e50")
V55 = Candidate("v55", "v55_tp_prefill_projections",
    "5ca9d26522ba84bc183537f819811962cac21de546244a48bf86a3a60dc5543e", SWA,
    "8b71561950104724")
V56 = Candidate("v56", "v56_muse_marlin_sequence_parallel",
    "bf9f5590adf2117483ef0a4fcb86e434ff779de334c232dbc78fd2d19bce8126", SWA | SP,
    "2eea52ae2db64189")
V59 = Candidate("v59", "v59_humming_sequence_parallel",
    "ea60c518cb47f98a8a740757f055398c816d1c30aff4573252f53a57c13f0773", HUMMING | SP)
V60 = Candidate("v60", "v60_fa3_paged_tma",
    "8c152846270bd35aa7d58d8cfb6ea46f6cc5b81aa10f03a680bb4487000be79c",
    SWA | {"--page-size": "160", "--chunked-prefill-size": "16320",
        "--max-prefill-tokens": "16320", "--cuda-graph-max-bs-prefill": "16384"})


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def passing_check(candidate, *, max_memory, min_ratio, runs=RUNS):
    """Validate exact identity and all judge gates before using check evidence."""
    data = read_json(runs / f"{candidate.check16}.json")
    if not isinstance(data, dict):
        return None
    result = data.get("result") or {}
    if not isinstance(result, dict):
        return None
    if not (
        data.get("id") == candidate.check16
        and data.get("mode") == "check"
        and data.get("status") == "done"
        and data.get("serving_flags") == candidate.flags
        and result.get("passed") is True
        and (result.get("patch") or {}).get("sha256") == candidate.sha256
        and (result.get("correctness") or {}).get("passed") is True
        and (result.get("later_turns") or {}).get("passed") is True
        and result.get("short_outputs") == 0
    ):
        return None
    memory = (result.get("gpu_memory") or {}).get("over_stock_gb")
    points = (result.get("vs_stock") or {}).get("rungs", [])
    rungs = result.get("rungs", [])
    if len(points) != 1 or len(rungs) != 1:
        return None
    point, rung = points[0], rungs[0]
    ratio = point.get("at_stock_point")
    if not (
        point.get("concurrency") == 16 and rung.get("concurrency") == 16
        and rung.get("truncated") is False
        and type(memory) in (int, float) and math.isfinite(memory)
        and memory <= max_memory
        and type(ratio) in (int, float) and math.isfinite(ratio)
        and ratio >= min_ratio
    ):
        return None
    return {"ratio": ratio, "memory_over_stock_gb": memory, "job": candidate.check16}


def terminal_receipt(candidate, runs=RUNS):
    data = read_json(runs / f"{candidate.check16}.json")
    return isinstance(data, dict) and (
        data.get("id") == candidate.check16
        and data.get("mode") == "check"
        and data.get("serving_flags") == candidate.flags
        and data.get("status") in ("done", "failed", "withdrawn", "cancelled", "canceled")
    )


def decisions(runs=RUNS, now=None):
    now = now or datetime.now(timezone.utc)
    offers = []
    if V60 is not None:
        offers.append(("v60_nov17_check16", V60, 16, {"kind": "new private experiment"}))
    p53 = passing_check(V53, max_memory=2.0, min_ratio=1.13, runs=runs)
    p56 = passing_check(V56, max_memory=2.0, min_ratio=1.18, runs=runs)
    if p53 is not None and p56 is not None:
        offers.append(("v59_nov17_check16", V59, 16, {"parents": [p53, p56]}))
    # Read readiness first: a newly published last receipt must not make an
    # already-built, incomplete candidate list appear ready for selection.
    selection_ready = now >= SELECT_32_BY or all(
        terminal_receipt(candidate, runs) for candidate in (V50, V53, V55, V56)
    )
    qualified = []
    for candidate in (V50, V53, V55, V56):
        evidence = passing_check(candidate, max_memory=1.9, min_ratio=1.17, runs=runs)
        if evidence is not None:
            qualified.append((candidate, evidence))
    if qualified and selection_ready:
        candidate, evidence = max(qualified, key=lambda pair: (pair[1]["ratio"], -pair[1]["memory_over_stock_gb"]))
        offers.append(("best_nov17_check32", candidate, 32, evidence))
    return [offer for offer in offers if not (runs / f"{offer[0]}.submit.txt").exists()]


def submit_once(offer, *, okc, now=None, root=ROOT, runs=RUNS, execute=subprocess.run):
    now = now or datetime.now(timezone.utc)
    if not START <= now < END:
        return "outside submission window"
    receipt_name, candidate, users, evidence = offer
    patch = root / "serving_v0523/candidates" / candidate.directory / "patch.diff"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != candidate.sha256:
        return "patch hash mismatch"
    limits = execute([okc, "serve", "status", "--json"], capture_output=True, text=True, timeout=60)
    if limits.returncode:
        return "quota request failed"
    try:
        data = json.loads(limits.stdout)
        left = data["limits"]["checks"]["left"]
        allowed = data["resets_at"] == RESET and type(left) is int and left > 0
    except (KeyError, TypeError, ValueError):
        allowed = False
    if not allowed:
        return "quota unavailable"
    receipt = runs / f"{receipt_name}.submit.txt"
    try:
        output = receipt.open("x")
    except FileExistsError:
        return "prior attempt exists"
    command = [okc, "serve", "check", str(patch), "--users", str(users)]
    for key, value in sorted(candidate.flags.items()):
        command.extend(["--flag", key.removeprefix("--") + (f"={value}" if value else "")])
    command.append("--no-wait")
    with output:
        output.write(json.dumps({"attempted_at": now.isoformat(), "candidate": candidate.name,
            "sha256": candidate.sha256, "users": users, "flags": candidate.flags, "evidence": evidence}) + "\n")
        output.flush()
        try:
            submitted = execute(command, capture_output=True, text=True, timeout=120)
            output.write(submitted.stdout + submitted.stderr)
            output.write(f"\nCLI exit code: {submitted.returncode}\n")
            job = re.search(r"job ([0-9a-f]{16})\b", submitted.stdout)
            if job and submitted.returncode == 0:
                return f"submitted {job[1]}"
            return "attempt needs manual reconciliation; no retry"
        except subprocess.TimeoutExpired as exc:
            for partial in (exc.stdout, exc.stderr):
                if partial:
                    output.write(partial.decode(errors="replace") if isinstance(partial, bytes) else partial)
            output.write("Submission timed out; reconcile with judge before any retry.\n")
            return "attempt timed out; no retry"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="wait for the UTC window and submit gated private checks")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps([{"receipt": name, "candidate": candidate.name, "users": users, "evidence": evidence}
            for name, candidate, users, evidence in decisions()], indent=2))
        return
    okc = os.environ.get("OKC", str(Path.home() / ".okc/bin/okc"))
    while datetime.now(timezone.utc) < START:
        time.sleep(30)
    last_state = None
    while datetime.now(timezone.utc) < END:
        offers = decisions()
        state = [(name, candidate.name, users) for name, candidate, users, _ in offers]
        if state != last_state:
            print(datetime.now(timezone.utc).isoformat(), "eligible", state, flush=True)
            last_state = state
        for offer in offers:
            try:
                result = submit_once(offer, okc=okc)
                if not result.startswith("quota"):
                    print(datetime.now(timezone.utc).isoformat(), offer[0], result, flush=True)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                print(datetime.now(timezone.utc).isoformat(), offer[0], type(exc).__name__, str(exc), flush=True)
        if all((RUNS / f"{name}.submit.txt").exists() for name in
            ("v60_nov17_check16", "v59_nov17_check16", "best_nov17_check32")):
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
