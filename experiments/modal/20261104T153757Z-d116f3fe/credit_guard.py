"""Fail-closed, local free-credit guard; reservations are estimates, not provider caps.

The reviewed grant must be renewed manually. Reservations are never refunded.
An optional, separately reviewed certificate can remove proved double counting
of posted provider charges; it never releases an unused reservation.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


class CreditGuardError(RuntimeError):
    pass


# Independently reviewed fixed overlap; activation does not settle charges or
# create a new grant. Every reservation and provider high-water remains counted.
REVIEWED_OVERLAP_SHA256: str | None = "7b93be1da79c69e8e49db9497dd32920631d38c6258ae21d8d9c938a972231ce"
OVERLAP_CERTIFICATE = "tools/modal_credit_overlap_20261104.json"
RESOURCE_H200 = "h200_4cpu_32gib"
RESOURCE_CPU = "cpu_4cpu_32gib"


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


def reservation_usd(timeout: int, resource_kind: str = RESOURCE_H200) -> Decimal:
    if resource_kind not in (RESOURCE_H200, RESOURCE_CPU):
        raise CreditGuardError("Unreviewed allocation resource kind")
    if type(timeout) is not int or not 60 <= timeout <= 900:
        raise CreditGuardError("Allocation timeout must be 60..900 seconds")
    # Rounded-up H200/CPU/RAM rates, plus 20% margin and $0.50 image-build allowance.
    # Both fixed kinds require hard 4-CPU/32-GiB limits in the caller; CPU has no GPU.
    hourly = 4 * Decimal("0.15") + 32 * Decimal("0.025")
    if resource_kind == RESOURCE_H200:
        hourly += Decimal("5.00")
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


def _pinned_json(root: Path, reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise CreditGuardError("Invalid overlap evidence reference")
    relative, expected = reference["path"], reference["sha256"]
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise CreditGuardError("Invalid overlap evidence identity")
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise CreditGuardError("Overlap evidence escapes project")
    try:
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise CreditGuardError("Overlap evidence hash changed")
    except OSError as exc:
        raise CreditGuardError("Missing overlap evidence") from exc
    # Parse exactly the bytes whose hash was checked, not a second file read.
    try:
        return json.loads(contents, parse_float=Decimal)
    except ValueError as exc:
        raise CreditGuardError("Invalid overlap evidence JSON") from exc


def _cost_rows(rows, start: datetime, end: datetime):
    if not isinstance(rows, list) or not rows or len(rows) > 512:
        raise CreditGuardError("Invalid overlap billing report")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CreditGuardError("Invalid overlap billing row")
        fields = tuple(row.get(k) for k in ("object_id", "environment", "interval_start", "resource"))
        if any(not isinstance(v, str) or not v for v in fields):
            raise CreditGuardError("Missing overlap billing identity")
        try:
            moment = datetime.fromisoformat(fields[2])
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            moment = moment.astimezone(timezone.utc)
        except ValueError as exc:
            raise CreditGuardError("Invalid overlap billing interval") from exc
        if not start <= moment < end or any((moment.minute, moment.second, moment.microsecond)):
            raise CreditGuardError("Overlap billing interval is outside certificate")
        key = (fields[0], fields[1], moment.isoformat(), fields[3])
        if key in result:
            raise CreditGuardError("Duplicate or overlapping billing row")
        result[key] = money(row.get("cost"), "overlap row cost")
    return result


def verified_overlap(root: Path, grant, records, journal: Path, billing, observed: Decimal):
    """Validate one immutable certificate; return Pmax and its fixed offset A.

    Called under the original allocation lock. The offset is subtracted once
    from the total debit, never accumulated from earlier reservation receipts.
    """
    pin = REVIEWED_OVERLAP_SHA256
    if pin is None:
        return observed, Decimal(0), None
    try:
        cert = _pinned_json(root, {"path": OVERLAP_CERTIFICATE, "sha256": pin})
        if cert.get("schema") != 1 or cert.get("policy") != "posted-overlap-only-v1":
            raise CreditGuardError("Unsupported overlap certificate")
        for name in ("cycle", "workspace", "profile", "environment"):
            if cert[name] != grant[name]:
                raise CreditGuardError("Overlap certificate does not match credit grant")
        if _pinned_json(root, cert["grant"]) != grant:
            raise CreditGuardError("Overlap grant binding changed")
        prefix = cert["journal_prefix"]
        count = prefix["records"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1 or count > len(records):
            raise CreditGuardError("Missing overlap journal prefix")
        lines = journal.read_bytes().splitlines(keepends=True)
        if hashlib.sha256(b"".join(lines[:count])).hexdigest() != prefix["sha256"]:
            raise CreditGuardError("Overlap journal prefix changed")
        for record in records[count:]:
            if record.get("overlap_certificate_sha256", pin) != pin:
                raise CreditGuardError("Conflicting overlap certificate in journal")
        provider = _pinned_json(root, cert["provider_evidence"])
        metadata = _pinned_json(root, cert["sandbox_evidence"])
        history = _pinned_json(root, cert["highwater_evidence"])
        start = datetime.fromisoformat(cert["report_start"])
        end = datetime.fromisoformat(cert["report_end"])
        if start.tzinfo is not None or end.tzinfo is not None:
            raise CreditGuardError("Certificate report range must use explicit UTC-naive CLI dates")
        start, end = start.replace(tzinfo=timezone.utc), end.replace(tzinfo=timezone.utc)
        if start.strftime("%Y-%m") != grant["cycle"] or any((start.day - 1, start.hour, start.minute, start.second, start.microsecond)):
            raise CreditGuardError("Overlap report must start at the billing cycle")
        if not start < end <= datetime.now(timezone.utc) or any((end.minute, end.second, end.microsecond)):
            raise CreditGuardError("Invalid overlap report range")
        active = [p for p in provider["profile"] if p.get("active") is True]
        if len(active) != 1 or active[0].get("name") != grant["profile"] or active[0].get("workspace") != grant["workspace"]:
            raise CreditGuardError("Overlap provider profile mismatch")
        saved = _cost_rows(provider["report"], start, end)
        tolerance = Decimal("0.00000001") * (len(saved) + 1)
        component = money(provider["summary"]["metered_cost_breakdown"]["ephemeral_apps"], "saved compute cost")
        if abs(sum(saved.values(), Decimal(0)) - component) > tolerance:
            raise CreditGuardError("Overlap report does not reconcile to provider summary")
        highwater = money(cert["provider_highwater_usd"], "overlap high-water")
        if highwater != provider_spend(history["billing"]) or highwater < provider_spend(provider["summary"]):
            raise CreditGuardError("Overlap provider high-water is not proved")
        sandboxes = {s["sandbox_id"]: s for s in metadata["sandboxes"]}
        if len(sandboxes) != len(metadata["sandboxes"]):
            raise CreditGuardError("Duplicate provider sandbox identity")
        if len(cert["runs"]) != count:
            raise CreditGuardError("Incomplete overlap run coverage")
        apps, used_rows, offset = set(), set(), Decimal(0)
        reservations = {}
        for entry, record in zip(cert["runs"], records[:count]):
            manifest = _pinned_json(root, entry["manifest"])
            if entry["run_id"] != record["run_id"] or manifest["credit_guard"] != record:
                raise CreditGuardError("Overlap manifest does not bind reservation")
            if manifest.get("sandbox_terminated") is not True:
                raise CreditGuardError("Overlap sandbox is not terminated")
            sb = sandboxes[manifest["sandbox_id"]]
            app = entry["app_id"]
            if app in apps or sb["app_id"] != app or not 0 < sb["created_at"] < sb["finished_at"]:
                raise CreditGuardError("Missing or duplicate sandbox/app attribution")
            apps.add(app)
            # Bind the original guard and every frozen uploaded source; a later
            # candidate edit cannot silently re-label an old reservation.
            folder = root / "experiments" / "modal" / record["run_id"]
            original_guard = folder / "credit_guard.py"
            if hashlib.sha256(original_guard.read_bytes()).hexdigest() != cert["original_guard_sha256"] or manifest["credit_guard_sha256"] != cert["original_guard_sha256"]:
                raise CreditGuardError("Original overlap guard changed")
            if hashlib.sha256((folder / "worker.py").read_bytes()).hexdigest() != manifest["worker_sha256"]:
                raise CreditGuardError("Frozen overlap worker changed")
            for filename, expected in manifest["file_sha256"].items():
                if Path(filename).name != filename or hashlib.sha256((folder / filename).read_bytes()).hexdigest() != expected:
                    raise CreditGuardError("Frozen overlap source changed")
            selected = {k: v for k, v in saved.items() if k[0] == app}
            if not selected or any(k[1] != grant["environment"] for k in selected) or used_rows.intersection(selected):
                raise CreditGuardError("Missing or reused overlap billing attribution")
            used_rows.update(selected)
            cost = sum(selected.values(), Decimal(0))
            reservation = money(record["reservation_usd"], "original reservation")
            amount = money(entry["offset_usd"], "overlap offset")
            if cost > reservation or amount != cost.quantize(Decimal("0.01"), rounding=ROUND_FLOOR):
                raise CreditGuardError("Overlap offset exceeds proved conservative cost")
            reservations[app] = reservation
            offset += amount
        if offset != money(cert["total_offset_usd"], "total overlap offset") or offset > highwater:
            raise CreditGuardError("Invalid total overlap offset")
        fresh = _cost_rows(modal_json("billing", "report", "--start", cert["report_start"],
            "--end", cert["report_end"], "--resolution", "h", "--show-resources", "--json",
            "--profile", grant["profile"]), start, end)
        if any(k not in fresh or fresh[k] < value for k, value in saved.items()):
            raise CreditGuardError("Provider billing attribution missing or revised downward")
        for app, reservation in reservations.items():
            if sum((v for k, v in fresh.items() if k[0] == app), Decimal(0)) > reservation:
                raise CreditGuardError("Provider cost exceeds its original reservation")
        fresh_component = money(billing["metered_cost_breakdown"]["ephemeral_apps"], "fresh compute cost")
        if sum(fresh.values(), Decimal(0)) > fresh_component + Decimal("0.00000001") * (len(fresh) + 1):
            raise CreditGuardError("Fresh report is not included in current provider summary")
        return max(observed, highwater), offset, pin
    except (OSError, KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise CreditGuardError("Invalid or missing overlap certificate evidence") from exc


@contextmanager
def allocation_guard(root: Path, run_id: str, timeout: int, *, resource_kind: str = RESOURCE_H200):
    """Hold one project lock from preflight until the caller finishes termination."""
    # Reject unreviewed kinds/timeouts before filesystem changes or provider calls.
    reservation = reservation_usd(timeout, resource_kind)
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
                if record.get("resource_kind", RESOURCE_H200) not in (RESOURCE_H200, RESOURCE_CPU):
                    raise CreditGuardError("Unreviewed resource kind in reservation journal")
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
            observed, overlap, overlap_pin = verified_overlap(root, grant, records, journal, billing, observed)
            available = ceiling - headroom - observed - reserved + overlap
            if available < reservation:
                raise CreditGuardError(f"Free-credit working budget exhausted: ${available:.2f} available, ${reservation:.2f} required; no allocation")
            receipt = {
                "run_id": run_id, "checked_at": now.isoformat(), "cycle": grant["cycle"],
                "workspace": grant["workspace"], "environment": grant["environment"],
                "grant_sha256": digest, "grant_usd": str(ceiling), "headroom_usd": str(headroom),
                "provider_observed_usd": str(observed), "prior_reservations_usd": str(reserved),
                "reservation_usd": str(reservation), "working_budget_remaining_usd": str(available - reservation),
                "resource_kind": resource_kind,
                "estimate_is_provider_cap": False,
                "billing_summary": json.loads(json.dumps(billing, default=str)),
            }
            if overlap_pin is not None:
                receipt.update(overlap_certificate_sha256=overlap_pin, proved_overlap_usd=str(overlap))
            # A reservation survives exceptions and process death, before any allocation.
            with journal.open("a") as handle:
                handle.write(json.dumps(receipt, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            yield receipt
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
