"""Coverage-mode transport/lifecycle contracts; all provider and CUDA calls mocked."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import modal_h200 as runner
import serving_finalizer_coverage_worker as coverage
import test_modal_h200 as inherited
from test_serving_finalizer_coverage_worker import complete

ROOT = Path(__file__).resolve().parents[1]


class CoverageSandbox(inherited.Sandbox):
    def __init__(self, payload, exit_code=0, status="coverage_complete"):
        super().__init__(fail=bool(exit_code))
        self.exit_code = exit_code
        self.result = complete(payload)
        self.result.update(status=status, coverage_complete=status == "coverage_complete")
        self.executions = []

    def read_text(self, path):
        raise AssertionError("Unbounded result transfer")

    def exec(self, *args, **kwargs):
        self.executions.append((args, kwargs))
        if args == ("python3", "-c", runner.FINALIZER_READ_RESULT):
            return types.SimpleNamespace(stdout=io.BytesIO(json.dumps(self.result).encode()),
                stderr=io.BytesIO(), wait=lambda: None, returncode=0)
        return types.SimpleNamespace(stdout=iter(["coverage fixture\n"]),
            stderr=io.StringIO("coverage failure" if self.exit_code else ""),
            wait=lambda: None, returncode=self.exit_code)


class CoverageRunnerTests(unittest.TestCase):
    setUp = inherited.LifecycleTests.setUp
    tearDown = inherited.LifecycleTests.tearDown
    write_grant = inherited.LifecycleTests.write_grant
    modal_json = inherited.LifecycleTests.modal_json
    invoke = inherited.LifecycleTests.invoke
    manifest = inherited.LifecycleTests.manifest

    def invoke_coverage(self, sandbox, dry_run=False, timeout=300):
        payload = coverage.build_payload(ROOT)
        with patch.object(coverage, "build_payload", return_value=copy.deepcopy(payload)), \
             patch.object(runner, "finalizer_observed_floor", return_value={"usd": "4.17"}):
            return self.invoke("serving-finalizer-coverage", sandbox, dry_run=dry_run,
                               extra_args=("--timeout", str(timeout)))

    def test_complete_uses_one_fixed_gpu_and_preserves_diagnostic_status(self):
        sandbox = CoverageSandbox(coverage.build_payload(ROOT))
        created = self.invoke_coverage(sandbox)
        self.assertEqual(len(created), 1)
        self.assertEqual((created[0]["gpu"], created[0]["timeout"]), ("H200", 300))
        self.assertEqual((created[0]["cpu"], created[0]["memory"]), ((4, 4), (32768, 32768)))
        self.assertTrue(created[0]["block_network"])
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.manifest()["status"], "coverage_complete")
        self.assertEqual(sandbox.result["generic_validation_status"], "failed_at_finite_cancellation")
        payload = json.loads(sandbox.uploads["/tmp/okc-payload.json"])
        self.assertEqual(set(payload["files"]), coverage.FILES)
        reads = [e for e in sandbox.executions if e[0] == ("python3", "-c", runner.FINALIZER_READ_RESULT)]
        self.assertEqual(reads, [(("python3", "-c", runner.FINALIZER_READ_RESULT), {"timeout": 10, "text": False})])

    def test_failed_worker_recovers_but_keeps_primary_failure_and_terminates(self):
        sandbox = CoverageSandbox(coverage.build_payload(ROOT), exit_code=7, status="incomplete")
        with self.assertRaisesRegex(RuntimeError, "Remote process exited 7"):
            self.invoke_coverage(sandbox)
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["remote_failure"]["exit_code"], 7)
        self.assertTrue(manifest["validation_recovery_inside_app_context"])
        self.assertTrue(manifest["validation_checkpoint_recovered"])
        self.assertTrue(sandbox.terminated)
        destination = next((self.root / "experiments/modal").glob("*/manifest.json")).parent
        self.assertTrue((destination / "result-checkpoint.json").exists())
        self.assertFalse((destination / "result.json").exists())

    def test_complete_coverage_with_new_numerical_failure_is_not_generic_pass(self):
        sandbox = CoverageSandbox(coverage.build_payload(ROOT))
        row = sandbox.result["cases"][0]
        metric = row["comparisons"]["candidate_vs_original"]
        metric.update({"pass": False, "equal_values": False, "whole_rel_l2": .02,
                       "worst_row_rel_l2": .02, "max_abs": .02})
        row["pass"] = False
        sandbox.result.update(remaining_case_failures=[row["name"]], remaining_numerics_passed=False)
        self.invoke_coverage(sandbox)
        self.assertEqual(self.manifest()["status"], "coverage_complete")
        self.assertFalse(sandbox.result["remaining_numerics_passed"])
        self.assertEqual(sandbox.result["generic_validation_status"], "failed_at_finite_cancellation")
        self.assertTrue(sandbox.terminated)

    def test_incomplete_zero_exit_never_promoted(self):
        sandbox = CoverageSandbox(coverage.build_payload(ROOT), status="incomplete")
        with self.assertRaisesRegex(RuntimeError, "coverage incomplete or failed"):
            self.invoke_coverage(sandbox)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertTrue(sandbox.terminated)

    def test_wrong_timeout_and_dry_run_allocate_nothing(self):
        self.billing = AssertionError("provider call forbidden")
        with self.assertRaises(SystemExit):
            self.invoke_coverage(None, dry_run=True, timeout=600)
        self.assertEqual(self.invoke_coverage(None, dry_run=True), [])
        self.assertFalse((self.root / "experiments").exists())

    def test_reader_rejects_missing_evidence_and_wrong_identity(self):
        payload = coverage.build_payload(ROOT)
        mutations = [lambda r: r.update(candidate_sha256="bad"),
                     lambda r: r.update(status="pass"),
                     lambda r: r.pop("historical_failure"),
                     lambda r: r["two_graphs"].pop("checks")]
        for mutate in mutations:
            sandbox = CoverageSandbox(payload); mutate(sandbox.result)
            with tempfile.TemporaryDirectory() as temp, self.assertRaises((ValueError, KeyError)):
                runner.retrieve_finalizer_coverage_result(sandbox, Path(temp), payload)

    def test_live_scope_recovers_coverage_before_app_exit(self):
        payload = coverage.build_payload(ROOT); sandbox = CoverageSandbox(payload, status="failed")
        events = []; manifest = {}
        original = sandbox.exec
        def execute(*args, **kw):
            self.assertNotIn("app_exit", events)
            events.append("read")
            return original(*args, **kw)
        sandbox.exec = execute
        @contextlib.contextmanager
        def app():
            try: yield
            finally: events.append("app_exit")
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(RuntimeError, "worker failed"):
            with app(), runner.finalizer_live_recovery(lambda: sandbox, Path(temp), manifest, payload):
                raise RuntimeError("worker failed")
        self.assertEqual(events, ["read", "app_exit"])
        self.assertTrue(manifest["validation_checkpoint_recovered"])


if __name__ == "__main__":
    unittest.main()
