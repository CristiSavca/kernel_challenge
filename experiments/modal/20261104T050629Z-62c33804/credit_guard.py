"""Fail-closed, local free-credit guard; reservations are estimates, not provider caps.

The reviewed grant must be renewed manually. Reservations are never refunded:
this deliberately double-counts usage that later appears in provider billing.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


class CreditGuardError(RuntimeError):
    pass


def money(value, label: str) -> Decimal:
    if not isinstance(value, (str, int, Decimal)) or isinstance(value, bool):
        raise CreditGuardError(f"Missing or invalid {label}")
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise CreditGuardError(f"Invalid {label}") from exc
    if not amount.is_finite() or amount < 0:
        raise CreditGuardError(f"Invalid {label}")
    return amount


def reservation_usd(timeout: int) -> Decimal:
    if not 60 <= timeout <= 900:
        raise CreditGuardError("Allocation timeout must be 60..900 seconds")
    # Rounded-up H200/CPU/RAM rates, plus 20% margin and $0.50 image-build allowance.
    hourly = Decimal("5.00") + 4 * Decimal("0.15") + 32 * Decimal("0.025")
    return (hourly * Decimal("1.20") * timeout / 3600 + Decimal("0.50")).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING)


def modal_json(*arguments: str):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "modal", *arguments], capture_output=True,
            text=True, check=True, timeout=30)
        return json.loads(result.stdout, parse_float=Decimal)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # Do not echo subprocess output: it may contain account information.
        raise CreditGuardError(f"Modal preflight failed ({' '.join(arguments[:2])}); no allocation") from exc


def provider_spend(summary) -> Decimal:
    if not isinstance(summary, dict):
        raise CreditGuardError("Invalid billing summary")
    breakdown = summary.get("metered_cost_breakdown")
    if not isinstance(breakdown, dict) or not breakdown or "ephemeral_apps" not in breakdown:
        raise CreditGuardError("Billing resource breakdown unavailable")
    # The aggregate can lag its breakdown; take the larger, never subtract credits.
    return max(money(summary.get("metered_cost"), "metered cost"),
               money(summary.get("billed_cost"), "billed cost"),
               sum((money(value, "resource cost") for value in breakdown.values()), Decimal(0)))


def require_stopped_apps(apps) -> None:
    if not isinstance(apps, list):
        raise CreditGuardError("Cannot determine active Modal applications")
    for app in apps:
        if not isinstance(app, dict) or app.get("state") != "stopped" or str(app.get("tasks")) != "0":
            raise CreditGuardError("A Modal application is active or its state is unknown; no overlapping allocation")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(), parse_float=Decimal)
    except (OSError, ValueError) as exc:
        raise CreditGuardError(f"Cannot read valid credit evidence: {path}") from exc


@contextmanager
def allocation_guard(root: Path, run_id: str, timeout: int):
    """Hold one project lock from preflight until the caller finishes termination."""
    folder = root / "experiments" / "modal"
    folder.mkdir(parents=True, exist_ok=True)
    # Never unlink this inode: another process could otherwise lock a replacement.
    with (folder / ".allocation.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CreditGuardError("Another project allocation holds the credit lock") from exc
        try:
            grant_path = root / "tools" / "modal_credit_grant.json"
            grant = read_json(grant_path)
            now = datetime.now(timezone.utc)
            if not isinstance(grant, dict) or grant.get("schema") != 1 or grant.get("cycle") != now.strftime("%Y-%m"):
                raise CreditGuardError("The reviewed credit grant is missing, invalid or expired; manual verification required")
            for field in ("workspace", "profile", "environment", "evidence"):
                if not isinstance(grant.get(field), str) or not grant[field].strip():
                    raise CreditGuardError(f"Credit grant lacks {field}")
            try:
                verified = datetime.fromisoformat(grant["verified_at"].replace("Z", "+00:00"))
                if verified.tzinfo is None or verified > now or verified.strftime("%Y-%m") != grant["cycle"]:
                    raise ValueError("Invalid verification date")
            except (KeyError, TypeError, ValueError) as exc:
                raise CreditGuardError("Credit verification date is invalid") from exc
            ceiling = money(grant.get("grant_usd"), "verified grant")
            headroom = money(grant.get("headroom_usd"), "safety headroom")
            observed = money(grant.get("initial_observed_spend_usd"), "initial observed spend")
            if ceiling <= headroom or headroom < Decimal("5.00"):
                raise CreditGuardError("Credit grant must preserve at least $5 safety headroom")
            if any(os.environ.get(key) for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")):
                raise CreditGuardError("Token environment overrides prevent verifying the reviewed workspace")
            profiles = modal_json("profile", "list", "--json")
            if not isinstance(profiles, list):
                raise CreditGuardError("Cannot verify the active Modal profile")
            active = [p for p in profiles if isinstance(p, dict) and p.get("active") is True]
            if len(active) != 1 or active[0].get("name") != grant["profile"] or active[0].get("workspace") != grant["workspace"]:
                raise CreditGuardError("Active Modal profile does not match the reviewed credit workspace")

            digest = hashlib.sha256(grant_path.read_bytes()).hexdigest()
            journal = folder / f"credit-reservations-{grant['cycle']}.jsonl"
            reserved = Decimal(0)
            records = []
            if journal.exists():
                try:
                    records = [json.loads(line) for line in journal.read_text().splitlines()]
                except (OSError, ValueError) as exc:
                    raise CreditGuardError("Reservation journal is invalid; manual reconciliation required") from exc
            seen = set()
            for record in records:
                if not isinstance(record, dict) or record.get("grant_sha256") != digest:
                    raise CreditGuardError("Credit grant changed after reservations; manual reconciliation required")
                prior_id = record.get("run_id")
                if not isinstance(prior_id, str) or Path(prior_id).name != prior_id or prior_id in seen:
                    raise CreditGuardError("Invalid or duplicate reservation identity")
                seen.add(prior_id)
                reserved += money(record.get("reservation_usd"), "recorded reservation")
                observed = max(observed, money(record.get("provider_observed_usd"), "recorded provider usage"))
                previous = read_json(folder / prior_id / "manifest.json")
                if previous.get("sandbox_terminated") is not True and previous.get("sandbox_created") is not False:
                    raise CreditGuardError(f"Previous allocation {prior_id} has unresolved cleanup; manual review required")
            # Missing journal entries must not silently recreate spent working budget.
            for path in folder.glob("*/manifest.json"):
                previous = read_json(path)
                prior_credit = previous.get("credit_guard")
                if prior_credit and prior_credit.get("grant_sha256") == digest and prior_credit.get("run_id") not in seen:
                    raise CreditGuardError("Reservation evidence is missing from the journal; manual reconciliation required")
            if run_id in seen or Path(run_id).name != run_id:
                raise CreditGuardError("Allocation identity is invalid or already reserved")
            billing = modal_json("billing", "summary", "--for", grant["cycle"], "--json", "--profile", grant["profile"])
            observed = max(observed, provider_spend(billing))
            apps = modal_json("app", "list", "--json", "--limit", "0", "--profile", grant["profile"], "--env", grant["environment"])
            require_stopped_apps(apps)
            reservation = reservation_usd(timeout)
            available = ceiling - headroom - observed - reserved
            if available < reservation:
                raise CreditGuardError(f"Free-credit working budget exhausted: ${available:.2f} available, ${reservation:.2f} required; no allocation")
            receipt = {
                "run_id": run_id, "checked_at": now.isoformat(), "cycle": grant["cycle"],
                "workspace": grant["workspace"], "environment": grant["environment"],
                "grant_sha256": digest, "grant_usd": str(ceiling), "headroom_usd": str(headroom),
                "provider_observed_usd": str(observed), "prior_reservations_usd": str(reserved),
                "reservation_usd": str(reservation), "working_budget_remaining_usd": str(available - reservation),
                "estimate_is_provider_cap": False,
                "billing_summary": json.loads(json.dumps(billing, default=str)),
            }
            # A reservation survives exceptions and process death, before any allocation.
            with journal.open("a") as handle:
                handle.write(json.dumps(receipt, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            yield receipt
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
