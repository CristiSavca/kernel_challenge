"""Monotonic provider-floor checks; real journals/providers are never mutated."""
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import modal_credit_budget as budget
import test_modal_credit_overlap as fixture


class FloorValidationTests(unittest.TestCase):
    def test_invalid_floor_fails_before_filesystem_or_provider(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "absent"
            with patch.object(budget, "modal_json", side_effect=AssertionError("provider call")) as provider:
                for value in (None, True, False, -1, "-0.01", "NaN", "Infinity", "-Infinity", {}, [], 4.16):
                    with self.subTest(value=value), self.assertRaises(budget.CreditGuardError):
                        with budget.allocation_guard(root, "invalid", 600, observed_spend_floor=value):
                            self.fail("invalid floor accepted")
                provider.assert_not_called()
            self.assertFalse(root.exists())


class FloorAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.OverlapTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    setUp = fixture.OverlapTests.setUp
    tearDown = fixture.OverlapTests.tearDown
    modal_json = fixture.OverlapTests.modal_json

    def reserve(self, name, floor="0"):
        with budget.allocation_guard(self.root, name, 60, observed_spend_floor=floor) as receipt:
            folder = self.root / "experiments/modal" / name
            folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({"sandbox_created": False, "credit_guard": receipt}))
        return receipt

    def test_higher_floor_is_debited_and_inherited_after_provider_decrease(self):
        first = self.reserve("higher", "4.16434237")
        self.assertEqual(first["provider_observed_usd"], "4.16434237")
        self.assertEqual(first["caller_observed_spend_floor_usd"], "4.16434237")
        self.assertEqual(first["working_budget_remaining_usd"], "2.80565763")
        self.billing["metered_cost"] = "4.13"
        self.billing["metered_cost_breakdown"]["ephemeral_apps"] = "4.12650025"
        second = self.reserve("inherited")
        self.assertEqual(second["provider_observed_usd"], "4.16434237")
        self.assertEqual(second["working_budget_remaining_usd"], "2.17565763")
        self.assertNotIn("caller_observed_spend_floor_usd", second)
        self.assertTrue(self.journal.read_bytes().startswith(self.initial_journal))

    def test_lower_floor_cannot_reduce_provider_or_certificate_observation(self):
        receipt = self.reserve("lower", "0.01")
        self.assertEqual(receipt["provider_observed_usd"], "3.90423640")
        self.assertEqual(receipt["proved_overlap_usd"], "3.74")
        self.assertEqual(receipt["headroom_usd"], "5.00")
        self.assertEqual(receipt["working_budget_remaining_usd"], "3.06576360")

    def test_high_floor_exhausts_budget_without_journal_write(self):
        with self.assertRaisesRegex(budget.CreditGuardError, "budget exhausted"):
            self.reserve("too-high", "25")
        self.assertEqual(self.journal.read_bytes(), self.initial_journal)


if __name__ == "__main__":
    unittest.main()
