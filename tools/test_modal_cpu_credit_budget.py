"""Fixed CPU-only reservation tests; provider calls and all journals are local fixtures."""
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import modal_credit_budget as budget
import test_modal_credit_overlap as overlap_fixture


class FixedResourceMathTests(unittest.TestCase):
    def test_default_h200_math_is_unchanged_for_every_allowed_timeout(self):
        for seconds in range(60, 901):
            original = (Decimal("6.40") * Decimal("1.20") * seconds / 3600 + Decimal(".50")).quantize(
                Decimal(".01"), rounding=ROUND_CEILING)
            self.assertEqual(budget.reservation_usd(seconds), original)
            self.assertEqual(budget.reservation_usd(seconds, budget.RESOURCE_H200), original)

    def test_exact_cpu_reservations_keep_build_allowance_and_margin(self):
        self.assertEqual(budget.reservation_usd(60, budget.RESOURCE_CPU), Decimal(".53"))
        self.assertEqual(budget.reservation_usd(600, budget.RESOURCE_CPU), Decimal(".78"))
        self.assertEqual(budget.reservation_usd(900, budget.RESOURCE_CPU), Decimal(".92"))
        self.assertEqual(budget.reservation_usd(600), Decimal("1.78"))
        self.assertEqual(budget.reservation_usd(900), Decimal("2.42"))

    def test_invalid_kind_or_timeout_fails_before_provider_and_filesystem(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "absent"
            with patch.object(budget, "modal_json", side_effect=AssertionError("provider must not be called")) as provider:
                for kind in (None, "cpu", "gpu=None", "h200_8cpu_32gib", 0, {}, []):
                    with self.subTest(kind=kind), self.assertRaises(budget.CreditGuardError):
                        with budget.allocation_guard(root, "bad", 600, resource_kind=kind):
                            self.fail("invalid resource kind accepted")
                for timeout in (True, False, None, 59, 901, 600.0, "600"):
                    with self.subTest(timeout=timeout), self.assertRaises(budget.CreditGuardError):
                        with budget.allocation_guard(root, "bad", timeout, resource_kind=budget.RESOURCE_CPU):
                            self.fail("invalid timeout accepted")
                provider.assert_not_called()
            self.assertFalse(root.exists())


class CpuAccountingTests(unittest.TestCase):
    # Reuse the pinned real nine-run evidence and provider mock, not its test cases.
    @classmethod
    def setUpClass(cls):
        overlap_fixture.OverlapTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    setUp = overlap_fixture.OverlapTests.setUp
    tearDown = overlap_fixture.OverlapTests.tearDown
    modal_json = overlap_fixture.OverlapTests.modal_json

    def reserve(self, run_id, timeout=600, resource_kind=budget.RESOURCE_CPU):
        with budget.allocation_guard(self.root, run_id, timeout, resource_kind=resource_kind) as receipt:
            folder = self.root / "experiments/modal" / run_id
            folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({"sandbox_created": False, "credit_guard": receipt}))
        return receipt

    def test_cpu_receipts_deduct_full_R_without_reapplying_overlap(self):
        first = self.reserve("cpu-a")
        second = self.reserve("cpu-b", 900)
        self.assertEqual(first["resource_kind"], budget.RESOURCE_CPU)
        self.assertEqual(first["reservation_usd"], "0.78")
        self.assertEqual(first["prior_reservations_usd"], "21.14")
        self.assertEqual(first["working_budget_remaining_usd"], "2.91576360")
        self.assertEqual(second["reservation_usd"], "0.92")
        self.assertEqual(second["prior_reservations_usd"], "21.92")
        self.assertEqual(second["working_budget_remaining_usd"], "1.99576360")
        for receipt in (first, second):
            self.assertEqual(receipt["proved_overlap_usd"], "3.74")
            self.assertEqual(receipt["provider_observed_usd"], "3.90423640")
            self.assertEqual(receipt["grant_usd"], "30.00")
            self.assertEqual(receipt["headroom_usd"], "5.00")
        self.assertTrue(self.journal.read_bytes().startswith(self.initial_journal))
        self.assertEqual(len(self.journal.read_text().splitlines()), 11)
        self.assertEqual(sum(x[:2] == ("billing", "report") for x in self.calls), 2)

    def test_omitted_resource_kind_still_reserves_h200(self):
        with budget.allocation_guard(self.root, "default-h200", 900) as receipt:
            self.assertEqual(receipt["resource_kind"], budget.RESOURCE_H200)
            self.assertEqual(receipt["reservation_usd"], "2.42")
            self.assertEqual(receipt["working_budget_remaining_usd"], "1.27576360")

    def test_cpu_and_h200_share_the_same_allocation_lock(self):
        with budget.allocation_guard(self.root, "cpu-lock", 600, resource_kind=budget.RESOURCE_CPU):
            before_calls = list(self.calls)
            with self.assertRaisesRegex(budget.CreditGuardError, "holds the credit lock"):
                with budget.allocation_guard(self.root, "h200-lock", 60):
                    self.fail("H200 acquired lock while CPU allocation was active")
            self.assertEqual(self.calls, before_calls)
        self.assertEqual(len(self.journal.read_text().splitlines()), 10)

    def test_current_like_balance_allows_one_cpu600_and_retains_full_h200_debit(self):
        previous = self.reserve("completed-h200", 900, budget.RESOURCE_H200)
        self.assertEqual(previous["resource_kind"], budget.RESOURCE_H200)
        self.billing["metered_cost"] = "4.16"
        self.billing["metered_cost_breakdown"]["ephemeral_apps"] = "4.16"
        cpu = self.reserve("cpu-current-like")
        self.assertEqual(cpu["prior_reservations_usd"], "23.56")
        self.assertEqual(cpu["provider_observed_usd"], "4.16")
        self.assertEqual(cpu["working_budget_remaining_usd"], "0.24")
        before = self.journal.read_bytes()
        with self.assertRaisesRegex(budget.CreditGuardError, "budget exhausted"):
            self.reserve("cannot-reuse", 60)
        self.assertEqual(self.journal.read_bytes(), before)

    def test_missing_legacy_kind_remains_h200_and_unknown_prior_kind_fails_closed(self):
        legacy = [json.loads(x) for x in self.initial_journal.splitlines()]
        self.assertTrue(all("resource_kind" not in x for x in legacy))
        self.reserve("new-h200", 900, budget.RESOURCE_H200)
        lines = self.journal.read_text().splitlines()
        record = json.loads(lines[-1])
        record["resource_kind"] = "unreviewed-cheaper-kind"
        lines[-1] = json.dumps(record)
        self.journal.write_text("\n".join(lines) + "\n")
        before = self.journal.read_bytes()
        with self.assertRaisesRegex(budget.CreditGuardError, "Unreviewed resource kind"):
            self.reserve("blocked")
        self.assertEqual(self.journal.read_bytes(), before)

    def test_cpu_exception_keeps_full_reservation_and_requires_cleanup_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
            with budget.allocation_guard(self.root, "crashed-cpu", 600, resource_kind=budget.RESOURCE_CPU):
                raise RuntimeError("synthetic crash")
        record = json.loads(self.journal.read_text().splitlines()[-1])
        self.assertEqual(record["resource_kind"], budget.RESOURCE_CPU)
        self.assertEqual(record["reservation_usd"], "0.78")
        with self.assertRaisesRegex(budget.CreditGuardError, "Cannot read valid credit evidence"):
            self.reserve("after-crash")
        self.assertEqual(len(self.journal.read_text().splitlines()), 10)

    def test_cpu_mode_preserves_provider_revision_rejection(self):
        self.report[0]["cost"] = str(Decimal(self.report[0]["cost"]) - Decimal(".00000001"))
        with self.assertRaisesRegex(budget.CreditGuardError, "revised downward"):
            self.reserve("revision-rejected")
        self.assertEqual(self.journal.read_bytes(), self.initial_journal)


if __name__ == "__main__":
    unittest.main()
