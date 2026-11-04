"""Fixed finished-run estimate tests; all provider data/journal writes are mocked/copied."""
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import modal_credit_budget as budget
import test_modal_credit_overlap as fixture


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 4, 16, tzinfo=timezone.utc)


class FinishedEstimateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.OverlapTests.setUpClass.__func__(cls)
        base = Path(cls.fixture.name)
        cert = json.loads((fixture.ROOT / budget.ESTIMATE_CERTIFICATE).read_text())
        files = {budget.ESTIMATE_CERTIFICATE, cert['journal_prefix']['path']}
        files.update(cert[k]['path'] for k in ('plan', 'plan_evidence', 'highwater_evidence'))
        files.update(f"experiments/modal/{r['run_id']}/manifest.json" for r in cert['excluded_full_reservations'])
        for relative in files:
            target = base / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture.ROOT / relative, target)
        journal = base / cert['journal_prefix']['path']
        prefix = b''.join(journal.read_bytes().splitlines(keepends=True)[:11])
        assert hashlib.sha256(prefix).hexdigest() == cert['journal_prefix']['sha256']
        journal.write_bytes(prefix)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        fixture.OverlapTests.setUp(self)
        self.clock.stop()
        self.clock = patch.object(budget, 'datetime', FrozenDatetime)
        self.clock.start()
        self.estimate_path = self.root / budget.ESTIMATE_CERTIFICATE
        self.estimate = json.loads(self.estimate_path.read_text())
        self.estimate_pin = patch.object(budget, 'REVIEWED_ESTIMATE_SHA256', fixture.sha(self.estimate_path))
        self.estimate_pin.start()
        self.full_report = deepcopy(self.report)
        self.billing['metered_cost'] = '4.10'
        self.billing['metered_cost_breakdown']['ephemeral_apps'] = '4.10'

    def tearDown(self):
        self.estimate_pin.stop()
        fixture.OverlapTests.tearDown(self)

    def modal_json(self, *args):
        if args[:2] == ('billing', 'report') and args[args.index('--end') + 1] == '2026-11-04T16:00:00':
            self.calls.append(args)
            self.assertEqual(args, ('billing', 'report', '--start', '2026-11-01T00:00:00',
                '--end', '2026-11-04T16:00:00', '--resolution', 'h', '--show-resources',
                '--json', '--profile', 'cristisavca44'))
            if isinstance(self.full_report, Exception):
                raise self.full_report
            return self.full_report
        return fixture.OverlapTests.modal_json(self, *args)

    def reserve(self, run_id='new-cpu', seconds=600, kind=budget.RESOURCE_CPU):
        with budget.allocation_guard(self.root, run_id, seconds, resource_kind=kind) as receipt:
            folder = self.root / 'experiments/modal' / run_id
            folder.mkdir()
            (folder / 'manifest.json').write_text(json.dumps({'sandbox_created': False, 'credit_guard': receipt}))
        return receipt

    def repin(self):
        self.estimate_path.write_text(json.dumps(self.estimate))
        budget.REVIEWED_ESTIMATE_SHA256 = fixture.sha(self.estimate_path)

    def reject(self, pattern):
        before = self.journal.read_bytes()
        with self.assertRaisesRegex(budget.CreditGuardError, pattern):
            self.reserve('rejected-new-run')
        self.assertEqual(self.journal.read_bytes(), before)

    def test_exact_fixed_correction_once_and_all_new_R_full(self):
        first = self.reserve()
        self.assertEqual(first['provider_observed_usd'], '4.16')
        self.assertEqual(first['prior_reservations_usd'], '24.34')
        self.assertEqual(first['reservation_usd'], '0.78')
        self.assertEqual(first['working_budget_remaining_usd'], '8.97')
        second = self.reserve('next-h200', 900, budget.RESOURCE_H200)
        self.assertEqual(second['prior_reservations_usd'], '25.12')
        self.assertEqual(second['reservation_usd'], '2.42')
        self.assertEqual(second['working_budget_remaining_usd'], '6.55')
        for receipt in (first, second):
            self.assertEqual(receipt['proved_overlap_usd'], '3.74')
            self.assertEqual(receipt['conservative_estimate_correction_usd'], '9.51')
            self.assertFalse(receipt['estimate_correction_is_final_settlement'])
            self.assertFalse(receipt['estimate_is_provider_cap'])
            self.assertEqual(receipt['headroom_usd'], '5.00')
        self.assertTrue(self.journal.read_bytes().startswith(self.initial_journal))
        self.assertEqual(len(self.journal.read_text().splitlines()), 13)
        self.assertEqual(sum(x[:2] == ('billing', 'report') for x in self.calls), 4)

    def test_disabled_certificate_preserves_existing_budget_and_no_full_report_query(self):
        budget.REVIEWED_ESTIMATE_SHA256 = None
        self.reject('budget exhausted')
        self.assertFalse(any(x[:2] == ('billing', 'report') and '2026-11-04T16:00:00' in x for x in self.calls))

    def test_delayed_cost_in_new_hour_exceeding_U_blocks_before_reservation(self):
        app = self.estimate['runs'][0]['app_id']
        self.full_report.append({'object_id': app, 'environment': 'main', 'interval_start': '2026-11-04T15:00:00',
                                 'resource': 'H200', 'cost': '1.10'})
        self.billing['metered_cost'] = '5.50'
        self.billing['metered_cost_breakdown']['ephemeral_apps'] = '5.50'
        # This late row is outside the old overlap certificate's14:00 endpoint.
        self.reject('exceeds revised estimate')

    def test_higher_provider_usage_stays_high_and_correction_does_not_grow(self):
        self.billing['metered_cost'] = '5.00'
        self.billing['metered_cost_breakdown']['ephemeral_apps'] = '5.00'
        receipt = self.reserve()
        self.assertEqual(receipt['provider_observed_usd'], '5.00')
        self.assertEqual(receipt['working_budget_remaining_usd'], '8.13')
        self.assertEqual(receipt['conservative_estimate_correction_usd'], '9.51')

    def test_full_report_missing_downward_duplicate_or_unreconciled_rows_fail(self):
        original = deepcopy(self.full_report)
        self.full_report.pop()
        self.reject('attribution missing')
        self.full_report = deepcopy(original)
        self.full_report[0]['cost'] = str(Decimal(self.full_report[0]['cost']) - Decimal('.00000001'))
        self.reject('revised downward')
        self.full_report = deepcopy(original) + [deepcopy(original[0])]
        self.reject('Duplicate or overlapping')
        self.full_report = deepcopy(original) + [{'object_id': 'unrelated-app', 'environment': 'main',
            'interval_start': '2026-11-04T15:00:00', 'resource': 'CPU', 'cost': '1.00'}]
        self.reject('not included in current summary')

    def test_fresh_report_failure_retains_full_journal(self):
        self.full_report = budget.CreditGuardError('synthetic report unavailable')
        self.reject('synthetic report unavailable')

    def test_certificate_plan_metadata_or_worker_corruption_fail_hash(self):
        for path in (self.estimate_path, self.root / self.estimate['plan']['path'],
                     self.root / self.estimate['metadata']['path'], self.root / self.estimate['runs'][0]['worker']['path']):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            self.reject('hash changed|Frozen overlap worker changed')
            path.write_bytes(original)

    def test_original_scope_allowance_and_exclusions_cannot_be_reapproved_differently(self):
        original = deepcopy(self.estimate)
        for modify in (
            lambda c: c['parameters'].update(termination_allowance_seconds=0),
            lambda c: c['runs'][0].update(revised_estimate_usd='1.80'),
            lambda c: c['runs'][0].update(lifetime_plus_allowance_seconds=608),
            lambda c: c['excluded_full_reservations'][0].update(reservation_usd='0.50'),
            lambda c: c['runs'][1].update(run_id=c['runs'][0]['run_id']),
            lambda c: c.update(total_estimate_correction_usd='19.02'),
            lambda c: c.update(provider_highwater_floor_usd='4.10'),
        ):
            self.estimate = deepcopy(original)
            modify(self.estimate)
            self.repin()
            self.reject('Estimate|Newer reservations')

    def test_actual_resource_AST_rejects_changed_limits(self):
        # Fault-inject after the byte/hash reader to exercise the separate AST
        # contract. Ordinary on-disk changes are already rejected by hash tests.
        actual = budget._pinned_bytes
        worker_path = self.estimate['runs'][0]['worker']['path']
        def altered(root, reference):
            data = actual(root, reference)
            return data.replace(b'cpu=(4, 4)', b'cpu=(4, 8)') if reference['path'] == worker_path else data
        with patch.object(budget, '_pinned_bytes', altered):
            self.reject('host resource limits changed')

    def test_prefix_change_and_conflicting_future_certificate_fail(self):
        original = self.journal.read_bytes()
        lines = original.splitlines(keepends=True)
        lines[10] = lines[10].replace(b'"0.78"', b'"0.79"', 1)
        self.journal.write_bytes(b''.join(lines))
        self.reject('prefix changed')
        self.journal.write_bytes(original)
        self.reserve()
        lines = self.journal.read_text().splitlines()
        record = json.loads(lines[-1]);record['estimate_certificate_sha256'] = 'other'
        lines[-1] = json.dumps(record);self.journal.write_text('\n'.join(lines) + '\n')
        self.reject('Conflicting estimate certificate')

    def test_crash_retains_new_full_debit_and_global_lock_unchanged(self):
        with self.assertRaisesRegex(RuntimeError, 'synthetic crash'):
            with budget.allocation_guard(self.root, 'crashed', 600, resource_kind=budget.RESOURCE_CPU):
                calls = list(self.calls)
                with self.assertRaisesRegex(budget.CreditGuardError, 'holds the credit lock'):
                    with budget.allocation_guard(self.root, 'concurrent', 60):
                        self.fail('second allocator acquired lock')
                self.assertEqual(self.calls, calls)
                raise RuntimeError('synthetic crash')
        record = json.loads(self.journal.read_text().splitlines()[-1])
        self.assertEqual(record['reservation_usd'], '0.78')
        self.assertEqual(record['conservative_estimate_correction_usd'], '9.51')
        self.reject('Cannot read valid credit evidence')
        self.assertEqual(len(self.journal.read_text().splitlines()), 12)


if __name__ == '__main__':
    unittest.main()
