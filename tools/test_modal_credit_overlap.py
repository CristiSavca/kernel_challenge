"""CPU-only overlap accounting checks against copied, pinned real evidence.

Every journal write is confined to a TemporaryDirectory. Modal calls are mocked.
"""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import modal_credit_budget as budget


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 4, 15, tzinfo=timezone.utc)


class OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        base = Path(cls.fixture.name)
        cert = json.loads((ROOT / budget.OVERLAP_CERTIFICATE).read_text())
        files = {budget.OVERLAP_CERTIFICATE, "experiments/modal/credit-reservations-2026-11.jsonl"}
        for key in ("grant", "provider_evidence", "sandbox_evidence", "highwater_evidence"):
            files.add(cert[key]["path"])
        for entry in cert["runs"]:
            files.add(entry["manifest"]["path"])
            folder = Path(entry["manifest"]["path"]).parent
            manifest = json.loads((ROOT / entry["manifest"]["path"]).read_text())
            files.add(str(folder / "credit_guard.py"))
            files.add(str(folder / "worker.py"))
            files.update(str(folder / name) for name in manifest["file_sha256"])
        for relative in files:
            target = base / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        # Test the immutable certificate prefix, regardless of later real runs.
        # Only this temporary copy is truncated; the project journal is read-only.
        journal = base / "experiments/modal/credit-reservations-2026-11.jsonl"
        lines = journal.read_bytes().splitlines(keepends=True)
        count = cert["journal_prefix"]["records"]
        prefix = b"".join(lines[:count])
        if len(lines) < count or hashlib.sha256(prefix).hexdigest() != cert["journal_prefix"]["sha256"]:
            raise AssertionError("Saved journal does not contain the certificate-pinned prefix")
        journal.write_bytes(prefix)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "root"
        shutil.copytree(self.fixture.name, self.root)
        self.cert_path = self.root / budget.OVERLAP_CERTIFICATE
        self.cert = json.loads(self.cert_path.read_text())
        self.evidence = json.loads((self.root / self.cert["provider_evidence"]["path"]).read_text())
        self.billing = deepcopy(self.evidence["summary"])
        self.report = deepcopy(self.evidence["report"])
        self.profiles = deepcopy(self.evidence["profile"])
        self.apps = []
        self.calls = []
        self.journal = self.root / "experiments/modal/credit-reservations-2026-11.jsonl"
        self.initial_journal = self.journal.read_bytes()
        self.pin = patch.object(budget, "REVIEWED_OVERLAP_SHA256", sha(self.cert_path))
        self.pin.start()
        # These fixtures isolate the original nine-run overlap policy. The
        # estimate-revision suite separately enables its eleven-record policy.
        self.no_estimate = patch.object(budget, "REVIEWED_ESTIMATE_SHA256", None)
        self.no_estimate.start()
        self.clock = patch.object(budget, "datetime", FrozenDatetime)
        self.clock.start()
        self.cli = patch.object(budget, "modal_json", self.modal_json)
        self.cli.start()

    def tearDown(self):
        self.cli.stop()
        self.clock.stop()
        self.no_estimate.stop()
        self.pin.stop()
        self.temp.cleanup()

    def modal_json(self, *args):
        self.calls.append(args)
        if args[:2] == ("profile", "list"):
            return self.profiles
        if args[:2] == ("billing", "summary"):
            return self.billing
        if args[:2] == ("app", "list"):
            return self.apps
        if args[:2] == ("billing", "report"):
            expected = ("billing", "report", "--start", self.cert["report_start"],
                        "--end", self.cert["report_end"], "--resolution", "h",
                        "--show-resources", "--json", "--profile", "cristisavca44")
            self.assertEqual(args, expected)
            if isinstance(self.report, Exception):
                raise self.report
            return self.report
        raise AssertionError(args)

    def repin(self):
        self.cert_path.write_text(json.dumps(self.cert))
        budget.REVIEWED_OVERLAP_SHA256 = sha(self.cert_path)

    def reserve(self, run_id="new-a", timeout=600):
        with budget.allocation_guard(self.root, run_id, timeout) as receipt:
            folder = self.root / "experiments/modal" / run_id
            folder.mkdir()
            # No actual sandbox is created in these tests.
            (folder / "manifest.json").write_text(json.dumps({
                "sandbox_created": False, "credit_guard": receipt}))
        return receipt

    def rejects(self, pattern):
        with self.assertRaisesRegex(budget.CreditGuardError, pattern):
            self.reserve()
        self.assertEqual(self.journal.read_bytes(), self.initial_journal)

    def test_disabled_pin_retains_old_budget_and_does_not_query_report(self):
        budget.REVIEWED_OVERLAP_SHA256 = None
        self.rejects("budget exhausted")
        self.assertFalse(any(x[:2] == ("billing", "report") for x in self.calls))

    def test_real_evidence_offset_once_future_reservations_full_and_highwater_preserved(self):
        first = self.reserve("new-a")
        self.assertEqual(first["proved_overlap_usd"], "3.74")
        self.assertEqual(first["provider_observed_usd"], "3.90423640")
        self.assertEqual(first["prior_reservations_usd"], "21.14")
        self.assertEqual(first["working_budget_remaining_usd"], "1.91576360")
        second = self.reserve("new-b")
        self.assertEqual(second["prior_reservations_usd"], "22.92")
        self.assertEqual(second["proved_overlap_usd"], "3.74")
        self.assertEqual(second["working_budget_remaining_usd"], "0.13576360")
        with self.assertRaisesRegex(budget.CreditGuardError, "budget exhausted"):
            self.reserve("new-c", 60)
        self.assertEqual(len(self.journal.read_text().splitlines()), 11)
        self.assertTrue(self.journal.read_bytes().startswith(self.initial_journal))

    def test_one_900_second_reservation_fits_without_changing_grant(self):
        receipt = self.reserve(timeout=900)
        self.assertEqual(receipt["working_budget_remaining_usd"], "1.27576360")
        self.assertEqual(receipt["grant_usd"], "30.00")
        self.assertEqual(receipt["headroom_usd"], "5.00")

    def test_certificate_corruption_fails_hash(self):
        self.cert_path.write_text(self.cert_path.read_text() + " ")
        self.rejects("hash changed")

    def test_missing_certificate_fails_closed(self):
        self.cert_path.unlink()
        self.rejects("Missing overlap evidence")

    def test_original_prefix_change_fails(self):
        self.journal.write_bytes(self.initial_journal.replace(b'"2.42"', b'"2.43"', 1))
        with self.assertRaisesRegex(budget.CreditGuardError, "prefix changed"):
            self.reserve()

    def test_missing_manifest_or_uploaded_source_fails(self):
        manifest_path = self.root / self.cert["runs"][0]["manifest"]["path"]
        (manifest_path.parent / "candidate.py").write_text("changed source")
        self.rejects("Frozen overlap source changed")

    def test_saved_provider_file_corruption_fails(self):
        p = self.root / self.cert["provider_evidence"]["path"]
        p.write_text(p.read_text() + " ")
        self.rejects("hash changed")

    def test_frozen_worker_corruption_fails(self):
        folder = (self.root / self.cert["runs"][0]["manifest"]["path"]).parent
        (folder / "worker.py").write_text("changed worker")
        self.rejects("Frozen overlap worker changed")

    def test_duplicate_app_cannot_credit_same_cost_twice(self):
        self.cert["runs"][1]["app_id"] = self.cert["runs"][0]["app_id"]
        self.repin()
        self.rejects("duplicate sandbox/app")

    def test_per_app_floor_overcredit_rejected_even_if_reapproved_hash(self):
        self.cert["runs"][0]["offset_usd"] = "0.84"
        self.cert["total_offset_usd"] = "3.75"
        self.repin()
        self.rejects("exceeds proved conservative cost")

    def test_wrong_cycle_rejected(self):
        self.cert["cycle"] = "2026-10"
        self.repin()
        self.rejects("does not match credit grant")

    def test_wrong_active_profile_rejected_before_offset(self):
        self.profiles[0]["workspace"] = "other"
        self.rejects("profile does not match")

    def test_missing_provider_mapping_rejected(self):
        p = self.root / self.cert["sandbox_evidence"]["path"]
        evidence = json.loads(p.read_text())
        evidence["sandboxes"] = evidence["sandboxes"][1:]
        p.write_text(json.dumps(evidence))
        self.cert["sandbox_evidence"]["sha256"] = sha(p)
        self.repin()
        self.rejects("Invalid or missing overlap")

    def test_downward_provider_revision_rejected_even_below_cent_margin(self):
        self.report[0]["cost"] = str(Decimal(self.report[0]["cost"]) - Decimal("0.00000001"))
        self.rejects("revised downward")

    def test_missing_row_or_duplicate_interval_rejected(self):
        self.report.pop()
        self.rejects("attribution missing")
        self.report = deepcopy(self.evidence["report"])
        self.report.append(deepcopy(self.report[0]))
        self.rejects("Duplicate or overlapping")

    def test_equivalent_timezone_rows_cannot_double_count_interval(self):
        duplicate = deepcopy(self.report[0])
        # 04:00+01:00 identifies the same interval as the original03:00UTC.
        self.assertEqual(duplicate["interval_start"], "2026-11-04T03:00:00")
        duplicate["interval_start"] = "2026-11-04T04:00:00+01:00"
        self.report.append(duplicate)
        self.rejects("Duplicate or overlapping")

    def test_current_summary_must_still_include_report(self):
        self.billing["metered_cost_breakdown"]["ephemeral_apps"] = "3.00"
        self.rejects("not included in current provider summary")

    def test_later_usage_raises_P_without_enlarging_offset(self):
        self.billing["metered_cost"] = "4.00"
        self.billing["metered_cost_breakdown"]["ephemeral_apps"] = "4.00"
        receipt = self.reserve()
        self.assertEqual(receipt["provider_observed_usd"], "4.00")
        self.assertEqual(receipt["proved_overlap_usd"], "3.74")
        self.assertEqual(receipt["working_budget_remaining_usd"], "1.82")

    def test_delayed_app_cost_over_original_R_is_rejected(self):
        app = self.cert["runs"][0]["app_id"]
        next(r for r in self.report if r["object_id"] == app)["cost"] = "3.00"
        self.rejects("exceeds its original reservation")

    def test_report_failure_does_not_append_reservation(self):
        self.report = budget.CreditGuardError("provider report unavailable")
        self.rejects("provider report unavailable")

    def test_crash_after_reservation_keeps_debit_and_blocks_unresolved_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
            with budget.allocation_guard(self.root, "crashed", 600):
                raise RuntimeError("synthetic crash")
        self.assertEqual(len(self.journal.read_text().splitlines()), 10)
        with self.assertRaisesRegex(budget.CreditGuardError, "Cannot read valid credit evidence"):
            self.reserve("after-crash")
        self.assertEqual(len(self.journal.read_text().splitlines()), 10)

    def test_same_lock_excludes_second_process_before_any_provider_call(self):
        code = """import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from modal_credit_budget import allocation_guard, CreditGuardError
try:
    with allocation_guard(Path(sys.argv[2]), 'second-process', 60):
        raise AssertionError('second process acquired allocation lock')
except CreditGuardError as e:
    assert 'holds the credit lock' in str(e), str(e)
"""
        with budget.allocation_guard(self.root, "lock-owner", 600):
            result = subprocess.run([sys.executable, "-c", code, str(Path(budget.__file__).parent),
                                     str(self.root)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
