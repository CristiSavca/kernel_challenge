#!/usr/bin/env python3
"""Spending guard for the 8xH200 node work, which the free-credit guard (modal_credit_budget.py) cannot cover.

The cap in modal_paid_budget.json is the user's authorised Modal usage for the cycle, credits included. A run is
allowed when live spend plus every unexpired reservation plus its own worst case (hard timeout at list rates plus a
margin) stays under the cap. Reservations go to an append-only journal; nothing here edits the Nov 4 journal or lock.
"""
import json
import time
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from modal_credit_budget import CreditGuardError, modal_json, provider_spend

ROOT = Path(__file__).resolve().parents[1]
BUDGET = ROOT / "tools" / "modal_paid_budget.json"
JOURNAL = ROOT / "experiments" / "modal_node" / "journal.jsonl"
GRANT = json.loads((ROOT / "tools" / "modal_credit_grant.json").read_text())


def worst_case_usd(timeout_s: int, gpus: int, cpu: float, memory_gib: float) -> Decimal:
    budget = json.loads(BUDGET.read_text(), parse_float=Decimal)
    rate = budget["rates_usd_per_hour"]
    hourly = gpus * Decimal(rate["h200"]) + Decimal(str(cpu)) * Decimal(rate["cpu_core"]) \
        + Decimal(str(memory_gib)) * Decimal(rate["memory_gib"])
    return (hourly * Decimal(budget["margin"]) * timeout_s / 3600).quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def open_reservations(now: float) -> Decimal:
    if not JOURNAL.exists():
        return Decimal(0)
    closed, total = set(), Decimal(0)
    rows = [json.loads(line) for line in JOURNAL.read_text().splitlines() if line.strip()]
    for row in rows:
        if row["event"] == "close":
            closed.add(row["run"])
    for row in rows:
        if row["event"] == "reserve" and row["run"] not in closed and row["expires"] > now:
            total += Decimal(row["usd"])
    return total


def reserve(run: str, timeout_s: int, gpus: int, cpu: float, memory_gib: float, dry_run: bool = False,
            wait_s: int = 0) -> dict:
    """`wait_s` is unbilled time the run may spend waiting for a machine; the reservation stays open through it."""
    cap = Decimal(json.loads(BUDGET.read_text())["cap_usd"])
    summary = modal_json("billing", "summary", "--for", datetime.now(timezone.utc).strftime("%Y-%m"),
                         "--json", "--profile", GRANT["profile"])
    spend = provider_spend(summary)
    now = time.time()
    held = open_reservations(now)
    mine = worst_case_usd(timeout_s, gpus, cpu, memory_gib)
    left = cap - spend - held - mine
    record = {"event": "reserve", "run": run, "usd": str(mine), "expires": now + timeout_s + wait_s + 120, "gpus": gpus,
              "timeout_s": timeout_s, "live_spend_usd": str(spend), "held_usd": str(held), "cap_usd": str(cap),
              "left_after_usd": str(left), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if left < 0:
        raise CreditGuardError(f"Over the authorised cap: spend {spend} + held {held} + this run {mine} > {cap}")
    # Modal stops every workload, running ones included, once charges after credits pass the workspace Spend limit.
    limit = json.loads(BUDGET.read_text()).get("provider_spend_limit_usd")
    charged = Decimal(str(summary.get("billed_cost", "0"))) if isinstance(summary, dict) else Decimal(0)
    record |= {"charged_usd": str(charged), "provider_spend_limit_usd": limit}
    if limit is not None and charged + held + mine > Decimal(limit):
        raise CreditGuardError(f"Modal's Spend limit would stop this run: charged {charged} + held {held} + this run "
                               f"{mine} > {limit}. The user raises it under Settings > Usage & billing.")
    if not dry_run:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with JOURNAL.open("a") as f:
            f.write(json.dumps(record) + "\n")
    return record | {"environment": GRANT["environment"]}


def close(run: str, note: str = "") -> None:
    with JOURNAL.open("a") as f:
        f.write(json.dumps({"event": "close", "run": run, "note": note,
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}) + "\n")
