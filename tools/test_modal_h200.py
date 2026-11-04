"""CPU-only checks of cloud lifecycle and upload boundaries; makes no Modal calls."""
import contextlib
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import modal_h200 as runner
import modal_credit_budget as budget


class Sandbox:
    def __init__(self, fail=False):
        self.object_id = "sb-test"
        self.terminated = False
        self.uploads = {}
        self.fail = fail
        self.filesystem = types.SimpleNamespace(write_text=self.write_text, read_text=self.read_text)

    def write_text(self, text, path):
        self.uploads[path] = text

    def read_text(self, path):
        assert path == "/tmp/okc-result.json"
        return json.dumps({"status": "failed" if self.fail else "passed_subset", "official": False})

    def exec(self, *args, **kwargs):
        self.command = args
        return types.SimpleNamespace(
            stdout=iter(["NVIDIA H200, GPU-test, 590.0, 143771 MiB\n"]),
            stderr=io.StringIO("synthetic worker failure" if self.fail else ""),
            wait=lambda: None, returncode=int(self.fail))

    def get_dashboard_url(self):
        return "https://modal.com/apps/test"

    def terminate(self):
        self.terminated = True


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        # The production certificate binds the real project, not synthetic test
        # roots. Exercise the original conservative guard in these fixtures.
        overlap_pin = patch.object(budget, "REVIEWED_OVERLAP_SHA256", None)
        overlap_pin.start()
        self.addCleanup(overlap_pin.stop)
        estimate_pin = patch.object(budget, "REVIEWED_ESTIMATE_SHA256", None)
        estimate_pin.start()
        self.addCleanup(estimate_pin.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        problem = self.root / "muse_moe_w4a16"
        problem.mkdir()
        self.candidate = problem / "candidate.py"
        self.candidate.write_text("# okc: problem=muse_moe_w4a16\ndef kernel_fn(data, state): return data\n")
        for name in ("baseline.py", "reference.py", "checks.py", "nvfp4.py"):
            (problem / name).write_text("# synthetic public input\n")
        (self.root / ".env").write_text("PRIVATE_TEST_SENTINEL\n")
        (self.root / "tools").mkdir()
        self.grant = {"schema": 1, "workspace": "test", "profile": "test", "environment": "main",
                      "cycle": datetime.now(timezone.utc).strftime("%Y-%m"),
                      "verified_at": datetime.now(timezone.utc).isoformat(), "grant_usd": "30.00",
                      "headroom_usd": "5.00", "initial_observed_spend_usd": "0.00276863", "evidence": "test"}
        self.write_grant()
        self.billing = {"metered_cost": "0", "billed_cost": "0", "metered_cost_breakdown": {"ephemeral_apps": "0.00276863"}}
        self.apps = []
        self.created = []

    def write_grant(self):
        (self.root / "tools/modal_credit_grant.json").write_text(json.dumps(self.grant))

    def modal_json(self, *args):
        if args[:2] == ("profile", "list"):
            return [{"name": "test", "workspace": "test", "active": True}]
        if args[:2] == ("billing", "summary"):
            if isinstance(self.billing, Exception):
                raise self.billing
            return self.billing
        if args[:2] == ("app", "list"):
            return self.apps
        raise AssertionError(args)

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, mode, sandbox=None, create_error=None, dry_run=False, incumbent=None, profile=False, extra_args=()):
        created = []

        def create(*args, **kwargs):
            created.append(kwargs)
            self.created.append(kwargs)
            if create_error:
                raise create_error
            return sandbox

        fake = types.SimpleNamespace(
            __version__="test", enable_output=contextlib.nullcontext,
            App=lambda name: types.SimpleNamespace(run=lambda **kw: contextlib.nullcontext()),
            Image=types.SimpleNamespace(debian_slim=lambda **kw: "small", from_registry=lambda ref: ref),
            Sandbox=types.SimpleNamespace(create=create))
        argv = ["modal_h200.py", mode]
        if mode == "benchmark":
            argv += ["--problem", "muse_moe_w4a16", "--candidate", str(self.candidate)]
        if dry_run:
            argv.append("--dry-run")
        if incumbent:
            argv += ["--incumbent", str(incumbent)]
        if profile:
            argv.append("--profile")
        argv.extend(extra_args)
        with patch.object(runner, "ROOT", self.root), patch.object(sys, "argv", argv), \
             patch.object(budget, "modal_json", self.modal_json), \
             patch.dict(sys.modules, {"modal": fake}), contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            runner.main()
        return created

    def manifest(self):
        return json.loads(next((self.root / "experiments/modal").glob("*/manifest.json")).read_text())

    def test_smoke_stops_one_bounded_gpu(self):
        sandbox = Sandbox()
        created = self.invoke("smoke", sandbox)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["gpu"], "H200")
        self.assertEqual(created[0]["timeout"], 120)
        self.assertTrue(created[0]["block_network"])
        self.assertTrue(sandbox.terminated)
        self.assertEqual(sandbox.uploads, {})
        self.assertEqual(self.manifest()["status"], "smoke_passed")

    def test_failed_worker_stops_gpu_and_preserves_evidence(self):
        sandbox = Sandbox(fail=True)
        with self.assertRaisesRegex(RuntimeError, "Remote process exited"):
            self.invoke("benchmark", sandbox)
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertTrue(self.manifest()["sandbox_terminated"])
        self.assertEqual(set(sandbox.uploads), {"/tmp/okc-worker.py", "/tmp/okc-payload.json"})
        payload = json.loads(sandbox.uploads["/tmp/okc-payload.json"])
        self.assertEqual(set(payload["files"]), {"candidate.py", "baseline.py", "reference.py", "checks.py", "nvfp4.py"})
        self.assertNotIn("PRIVATE_TEST_SENTINEL", str(sandbox.uploads))

    def test_access_denial_records_no_created_sandbox(self):
        with self.assertRaisesRegex(RuntimeError, "payment method"):
            self.invoke("smoke", create_error=RuntimeError("payment method required"))
        self.assertFalse(self.manifest()["sandbox_created"])

    def test_dry_run_never_creates_cloud_resource(self):
        self.billing = AssertionError("Billing call during dry run")
        created = self.invoke("benchmark", create_error=AssertionError("Cloud call"), dry_run=True)
        self.assertEqual(created, [])
        self.assertFalse((self.root / "experiments").exists())

    def test_dry_run_needs_neither_grant_nor_lock(self):
        (self.root / "tools/modal_credit_grant.json").unlink()
        with patch.object(budget, "allocation_guard", side_effect=AssertionError("Dry run acquired lock")):
            self.invoke("benchmark", dry_run=True)

    def test_incumbent_identity_and_upload_preserve_official_source(self):
        incumbent = self.candidate.with_name("incumbent.py")
        incumbent.write_text("# okc: problem=muse_moe_w4a16\ndef kernel_fn(data, state): return data + 1\n")
        official_path = self.candidate.with_name("baseline.py")
        original = official_path.read_text()
        sandbox = Sandbox()
        self.invoke("benchmark", sandbox, incumbent=incumbent)
        payload = json.loads(sandbox.uploads["/tmp/okc-payload.json"])
        self.assertEqual(payload["files"]["baseline.py"], incumbent.read_text())
        self.assertEqual(payload["comparison_baseline"], "incumbent")
        self.assertEqual(payload["incumbent"]["sha256"], runner.sha(incumbent.read_text()))
        self.assertEqual(self.manifest()["worker_sha256"], runner.sha(sandbox.uploads["/tmp/okc-worker.py"]))
        self.assertEqual(self.manifest()["incumbent"], payload["incumbent"])
        self.assertEqual(official_path.read_text(), original)
        self.assertEqual(set(payload["files"]), {"candidate.py", "baseline.py", "reference.py", "checks.py", "nvfp4.py"})

    def test_incumbent_requires_same_problem_even_dry_run(self):
        incumbent = self.candidate.with_name("incumbent.py")
        incumbent.write_text("# okc: problem=muse_bf16_proj\n")
        with self.assertRaises(SystemExit):
            self.invoke("benchmark", dry_run=True, incumbent=incumbent)
        self.assertEqual(self.created, [])

    def test_profile_request_defaults_off_and_can_be_enabled(self):
        for enabled in (False, True):
            sandbox = Sandbox()
            self.invoke("benchmark", sandbox, profile=enabled)
            payload = json.loads(sandbox.uploads["/tmp/okc-payload.json"])
            self.assertIs(payload["profile"], enabled)

    def test_insufficient_credit_stops_before_allocation(self):
        self.billing["metered_cost_breakdown"]["ephemeral_apps"] = "24.50"
        with self.assertRaisesRegex(budget.CreditGuardError, "exhausted"):
            self.invoke("benchmark", Sandbox())
        self.assertEqual(self.created, [])

    def test_billing_failure_stops_before_allocation(self):
        self.billing = budget.CreditGuardError("Billing unavailable")
        with self.assertRaisesRegex(budget.CreditGuardError, "Billing unavailable"):
            self.invoke("smoke", Sandbox())
        self.assertEqual(self.created, [])

    def test_invalid_billing_and_active_apps_stop_before_allocation(self):
        for value in ({}, {"metered_cost": "NaN", "billed_cost": "0", "metered_cost_breakdown": {"ephemeral_apps": "0"}}):
            self.billing = value
            with self.assertRaises(budget.CreditGuardError):
                self.invoke("smoke", Sandbox())
        self.billing = {"metered_cost": "0", "billed_cost": "0", "metered_cost_breakdown": {"ephemeral_apps": "0"}}
        self.apps = [{"state": "ephemeral", "tasks": "1"}]
        with self.assertRaisesRegex(budget.CreditGuardError, "overlapping"):
            self.invoke("smoke", Sandbox())
        self.assertEqual(self.created, [])

    def test_expired_or_changed_grant_cannot_replenish_budget(self):
        self.grant["cycle"] = "2000-01"
        self.write_grant()
        with self.assertRaisesRegex(budget.CreditGuardError, "expired"):
            self.invoke("smoke", Sandbox())
        self.grant["cycle"] = datetime.now(timezone.utc).strftime("%Y-%m")
        self.write_grant()
        self.invoke("smoke", Sandbox())
        self.grant["grant_usd"] = "300"
        self.write_grant()
        with self.assertRaisesRegex(budget.CreditGuardError, "changed"):
            self.invoke("smoke", Sandbox())
        self.assertEqual(len(self.created), 1)

    def test_reservations_survive_failed_creation_and_lagging_billing(self):
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self.invoke("smoke", create_error=RuntimeError("rejected"))
        self.invoke("smoke", Sandbox())
        journal = next((self.root / "experiments/modal").glob("credit-reservations-*.jsonl"))
        receipts = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual(len(receipts), 2)
        self.assertEqual(Decimal(receipts[1]["prior_reservations_usd"]), budget.reservation_usd(120))
        self.assertEqual(Decimal(receipts[1]["working_budget_remaining_usd"]), Decimal("24.99723137") - 2 * budget.reservation_usd(120))

    def test_unresolved_termination_blocks_next_allocation(self):
        sandbox = Sandbox()
        sandbox.terminate = lambda: (_ for _ in ()).throw(RuntimeError("termination failed"))
        with self.assertRaisesRegex(RuntimeError, "termination could not be confirmed"):
            self.invoke("smoke", sandbox)
        with self.assertRaisesRegex(budget.CreditGuardError, "unresolved cleanup"):
            self.invoke("smoke", Sandbox())
        self.assertEqual(len(self.created), 1)

    def test_missing_journal_does_not_reset_spent_budget(self):
        self.invoke("smoke", Sandbox())
        next((self.root / "experiments/modal").glob("credit-reservations-*.jsonl")).unlink()
        with self.assertRaisesRegex(budget.CreditGuardError, "missing from the journal"):
            self.invoke("smoke", Sandbox())

    def test_second_process_cannot_overlap_allocation(self):
        code = """import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from modal_credit_budget import allocation_guard, CreditGuardError
try:
    with allocation_guard(Path(sys.argv[2]), 'second', 120):
        raise AssertionError('second allocation obtained lock')
except CreditGuardError as exc:
    assert 'holds the credit lock' in str(exc), str(exc)
"""
        with patch.object(budget, "modal_json", self.modal_json):
            with budget.allocation_guard(self.root, "first", 120):
                result = subprocess.run([sys.executable, "-c", code, str(Path(budget.__file__).parent), str(self.root)],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_reservation_math_and_breakdown_lag(self):
        self.assertEqual(budget.reservation_usd(300), Decimal("1.14"))
        self.assertEqual(budget.reservation_usd(900), Decimal("2.42"))
        self.assertEqual(budget.provider_spend(self.billing), Decimal("0.00276863"))

    def test_cli_failure_is_fail_closed_without_exposing_output(self):
        with patch.object(budget.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "modal", stderr="private")):
            with self.assertRaisesRegex(budget.CreditGuardError, "preflight failed") as caught:
                budget.modal_json("billing", "summary", "--json")
        self.assertNotIn("private", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
