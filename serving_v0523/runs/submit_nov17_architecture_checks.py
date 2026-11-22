#!/usr/bin/env python3
"""Run the frozen Nov 17 architectural private checks, each at most once."""
from __future__ import annotations
import argparse
import hashlib
import json
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
RESET = "2026-11-18T00:00:00Z"
SWA = {"--swa-full-tokens-ratio": "0.25"}

@dataclass(frozen=True)
class Candidate:
    name: str
    directory: str
    sha256: str
    flags: dict[str, str]

V64 = Candidate("v64", "v64_humming_grouped_sequence_parallel",
    "f6aaf93ce06ccfce9461e06bc3f71dea05dd901c01497c8be31bd2c515da689a",
    SWA | {"--moe-runner-backend": "humming", "--speculative-moe-runner-backend": "marlin", "--enable-layernorm-sp": ""})
V62 = Candidate("v62", "v62_paged_verify_attention",
    "f0595f55c6c13380f67ed033bb4174780276188c44d779dfd9fee00bf517541f", SWA)
V60 = Candidate("v60", "v60_fa3_paged_tma",
    "8c152846270bd35aa7d58d8cfb6ea46f6cc5b81aa10f03a680bb4487000be79c",
    SWA | {"--page-size": "160", "--chunked-prefill-size": "16320",
           "--max-prefill-tokens": "16320", "--cuda-graph-max-bs-prefill": "16384"})
CANDIDATES = (V64, V62, V60)

def decisions(runs=RUNS):
    offers=[]
    for candidate in CANDIDATES:
        receipt = f"{candidate.name}_nov17_check16"
        if not (runs / f"{receipt}.submit.txt").exists():
            offers.append((receipt, candidate, 16,
                {"kind": "new architectural private experiment", "gpu_tested": False}))
    return offers

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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args=parser.parse_args()
    if not args.run:
        print(json.dumps([{"receipt":name,"candidate":c.name,"users":users,"sha256":c.sha256,"flags":c.flags}
            for name,c,users,_ in decisions()],indent=2))
        return
    okc=os.environ.get("OKC",str(Path.home()/".okc/bin/okc"))
    while datetime.now(timezone.utc)<START:
        time.sleep(30)
    last_state=None
    while datetime.now(timezone.utc)<END:
        offers=decisions()
        if not offers:
            break
        state=[name for name,_,_,_ in offers]
        if state!=last_state:
            print(datetime.now(timezone.utc).isoformat(),"eligible",state,flush=True)
            last_state=state
        for offer in offers:
            try:
                result=submit_once(offer,okc=okc)
                if not result.startswith("quota"):
                    print(datetime.now(timezone.utc).isoformat(),offer[0],result,flush=True)
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                print(datetime.now(timezone.utc).isoformat(),offer[0],type(exc).__name__,str(exc),flush=True)
        time.sleep(300)

if __name__=="__main__":
    main()
