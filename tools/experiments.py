#!/usr/bin/env python3
"""Reproducible official okc checks, without handling authentication secrets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "experiments"
PROBLEMS = (
    "muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16",
    "muse_decode_layer", "muse_prefill_layer",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cli(*args: str) -> dict:
    result = subprocess.run(
        ["okc", *args, "--json"], cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"okc returned unexpected output: {result.stdout[:1000]}") from exc


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def snapshot(problem: str, source: Path) -> tuple[str, Path]:
    code = source.read_bytes()
    digest = hashlib.sha256(code).hexdigest()
    target = STORE / "candidates" / problem / f"{digest}.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(code)
    return digest, target


def record(job: str, problem: str, source: Path, label: str) -> dict:
    digest, frozen = snapshot(problem, source)
    result = cli("status", job)
    # Keep the server's identifier separately. It is not necessarily the raw
    # file's SHA256; our immutable local snapshot identifies the uploaded bytes.
    data = {
        "job": job, "problem": problem, "label": label,
        "source_sha256": digest, "source": str(frozen.relative_to(ROOT)),
        "recorded_at": now(), "last_checked_at": now(), "response": result,
    }
    save(STORE / "runs" / f"{job}.json", data)
    return data


def summary(data: dict) -> dict:
    response = data["response"]
    output = {key: data[key] for key in ("job", "problem", "label")}
    output.update(status=response.get("status"), mode=response.get("mode"))
    if response.get("result") is not None:
        result = response["result"]
        output["result"] = {
            key: result[key] for key in
            ("status", "speedup", "speedup_ci", "score_us", "error", "checks")
            if key in result
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "submit", "register"):
        item = sub.add_parser(command)
        item.add_argument("problem", choices=PROBLEMS)
        item.add_argument("--file", type=Path)
        item.add_argument("--label", required=True)
        if command == "register":
            item.add_argument("job")
    status = sub.add_parser("status")
    status.add_argument("jobs", nargs="*")
    args = parser.parse_args()

    if args.command == "status":
        paths = ([STORE / "runs" / f"{job}.json" for job in args.jobs]
                 if args.jobs else sorted((STORE / "runs").glob("*.json")))
        for path in paths:
            data = json.loads(path.read_text())
            if data["response"].get("status") not in ("done", "failed", "rejected", "duplicate"):
                data["response"] = cli("status", data["job"])
                data["last_checked_at"] = now()
                save(path, data)
            print(json.dumps(summary(data)))
        return

    source = (args.file or ROOT / args.problem / "submission.py").resolve()
    if args.command == "register":
        data = record(args.job, args.problem, source, args.label)
    else:
        digest, frozen = snapshot(args.problem, source)
        response = cli(args.command, args.problem, str(frozen), "--no-wait")
        job = response.get("job_id") or response.get("id")
        if not job:
            raise RuntimeError(f"No job id in okc response: {response}")
        # Persist the receipt before querying status: a transient network failure
        # must not cause the already submitted work to be submitted a second time.
        receipt = {
            "job": job, "problem": args.problem, "label": args.label,
            "source_sha256": digest, "source": str(frozen.relative_to(ROOT)),
            "recorded_at": now(), "last_checked_at": now(), "response": response,
        }
        save(STORE / "runs" / f"{job}.json", receipt)
        data = record(job, args.problem, frozen, args.label)
    print(json.dumps(summary(data)))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
