"""Router experiment transport/lifecycle checks; provider and GPU calls mocked."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import types
import unittest
from unittest.mock import patch

import modal_h200 as runner
import serving_router_local_ids_worker as router
import test_modal_h200 as inherited
from test_serving_router_local_ids_worker import complete

ROOT = Path(__file__).resolve().parents[1]


class RouterSandbox(inherited.Sandbox):
    def __init__(self, payload, exit_code=0, status="screen_complete"):
        super().__init__(fail=bool(exit_code))
        self.exit_code = exit_code
        self.result = complete(payload)
        self.result["status"] = status
        self.executions = []
        self.read_error = None

    def read_text(self, path):
        raise AssertionError("Unbounded result transfer")

    def exec(self, *args, **kwargs):
        self.executions.append((args, kwargs))
        if args == ("python3", "-c", runner.FINALIZER_READ_RESULT):
            if self.read_error:
                raise self.read_error
            return types.SimpleNamespace(stdout=io.BytesIO(json.dumps(self.result).encode()),
                stderr=io.BytesIO(), wait=lambda: None, returncode=0)
        return types.SimpleNamespace(stdout=iter(["router fixture\n"]),
            stderr=io.StringIO("router failure" if self.exit_code else ""),
            wait=lambda: None, returncode=self.exit_code)


class RouterRunnerTests(unittest.TestCase):
    setUp = inherited.LifecycleTests.setUp
    tearDown = inherited.LifecycleTests.tearDown
    write_grant = inherited.LifecycleTests.write_grant
    modal_json = inherited.LifecycleTests.modal_json
    invoke = inherited.LifecycleTests.invoke
    manifest = inherited.LifecycleTests.manifest

    def invoke_router(self, sandbox, dry_run=False, timeout=600):
        payload = router.build_payload(ROOT, runner.IMAGE)
        with patch.object(router, "build_payload", return_value=copy.deepcopy(payload)), \
             patch.object(runner, "finalizer_observed_floor", return_value={"usd": "4.21235928"}):
            return self.invoke("serving-router-local-ids", sandbox, dry_run=dry_run,
                               extra_args=("--timeout", str(timeout)))

    def test_complete_uses_one_fixed_gpu_and_exact_frozen_package(self):
        payload = router.build_payload(ROOT, runner.IMAGE)
        sandbox = RouterSandbox(payload)
        created = self.invoke_router(sandbox)
        self.assertEqual(len(created), 1)
        self.assertEqual((created[0]["gpu"], created[0]["timeout"]), ("H200", 600))
        self.assertEqual((created[0]["cpu"], created[0]["memory"]), ((4, 4), (32768, 32768)))
        self.assertTrue(created[0]["block_network"])
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.manifest()["status"], "screen_complete")
        uploaded = json.loads(sandbox.uploads["/tmp/okc-payload.json"])
        self.assertEqual(uploaded["files"], payload["files"])
        self.assertEqual(uploaded["file_sha256"], payload["file_sha256"])
        self.assertEqual(uploaded["provider_observed_floor_evidence"]["usd"], "4.21235928")
        reads = [e for e in sandbox.executions if e[0] == ("python3", "-c", runner.FINALIZER_READ_RESULT)]
        self.assertEqual(reads, [(("python3", "-c", runner.FINALIZER_READ_RESULT), {"timeout": 10, "text": False})])

    def test_failed_worker_recovers_original_failure_and_terminates(self):
        sandbox = RouterSandbox(router.build_payload(ROOT, runner.IMAGE), exit_code=7, status="incomplete")
        with self.assertRaisesRegex(RuntimeError, "Remote process exited 7"):
            self.invoke_router(sandbox)
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["remote_failure"]["exit_code"], 7)
        self.assertTrue(manifest["validation_recovery_inside_app_context"])
        self.assertTrue(manifest["validation_checkpoint_recovered"])
        self.assertTrue(sandbox.terminated)
        self.assertEqual(len(self.created), 1)
        destination = next((self.root / "experiments/modal").glob("*/manifest.json")).parent
        self.assertTrue((destination / "result-checkpoint.json").exists())
        self.assertFalse((destination / "result.json").exists())

    def test_recovery_error_does_not_mask_worker_failure_or_retry(self):
        sandbox = RouterSandbox(router.build_payload(ROOT, runner.IMAGE), exit_code=7, status="incomplete")
        sandbox.read_error = RuntimeError("synthetic checkpoint unavailable")
        with self.assertRaisesRegex(RuntimeError, "Remote process exited 7"):
            self.invoke_router(sandbox)
        manifest = self.manifest()
        self.assertEqual(manifest["remote_failure"]["exit_code"], 7)
        self.assertFalse(manifest["validation_checkpoint_recovered"])
        self.assertTrue(sandbox.terminated)
        self.assertEqual(len(self.created), 1)

    def test_completed_slow_screen_remains_a_rejected_optimization(self):
        sandbox = RouterSandbox(router.build_payload(ROOT, runner.IMAGE))
        ratios = []
        for row in sandbox.result["timings"]:
            for pair in row["pairs"]:
                pair["candidate_us"] = pair["baseline_us"] * 1.03
                pair["latency_ratio"] = pair["candidate_us"] / pair["baseline_us"]
            row["baseline_us"] = statistics.median(p["baseline_us"] for p in row["pairs"])
            row["candidate_us"] = statistics.median(p["candidate_us"] for p in row["pairs"])
            row["latency_ratio"] = row["candidate_us"] / row["baseline_us"]
            ratios.append(row["latency_ratio"])
        sandbox.result["latency_geomean"] = math.exp(sum(map(math.log, ratios)) / len(ratios))
        sandbox.result["performance_gate_passed"] = False
        self.invoke_router(sandbox)
        self.assertEqual(self.manifest()["status"], "screen_complete")
        destination = next((self.root / "experiments/modal").glob("*/manifest.json")).parent
        saved = json.loads((destination / "result.json").read_text())
        self.assertFalse(saved["performance_gate_passed"])
        self.assertFalse(saved["official"])
        self.assertFalse(saved["full_serving_validated"])
        self.assertTrue(sandbox.terminated)

    def test_incomplete_zero_exit_never_promoted(self):
        sandbox = RouterSandbox(router.build_payload(ROOT, runner.IMAGE), status="incomplete")
        with self.assertRaisesRegex(RuntimeError, "Router component screen incomplete or failed"):
            self.invoke_router(sandbox)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertTrue(sandbox.terminated)

    def test_wrong_timeout_and_dry_run_allocate_nothing(self):
        self.billing = AssertionError("provider call forbidden")
        with self.assertRaises(SystemExit):
            self.invoke_router(None, dry_run=True, timeout=300)
        self.assertEqual(self.invoke_router(None, dry_run=True), [])
        self.assertFalse((self.root / "experiments").exists())

    def test_reader_rejects_unknown_status_and_bad_package_identity(self):
        payload = router.build_payload(ROOT, runner.IMAGE)
        mutations = [lambda r: r.update(status="pass"),
                     lambda r: r.update(file_sha256={}),
                     lambda r: r["cases"].pop()]
        for mutate in mutations:
            sandbox = RouterSandbox(payload)
            mutate(sandbox.result)
            with tempfile.TemporaryDirectory() as temp, self.assertRaises((ValueError, KeyError)):
                runner.retrieve_router_local_ids_result(sandbox, Path(temp), payload)

    def test_live_scope_recovers_before_app_exit(self):
        payload = router.build_payload(ROOT, runner.IMAGE)
        sandbox = RouterSandbox(payload, status="failed")
        events = []
        manifest = {}
        original = sandbox.exec

        def execute(*args, **kw):
            self.assertNotIn("app_exit", events)
            events.append("read")
            return original(*args, **kw)

        sandbox.exec = execute

        @contextlib.contextmanager
        def app():
            try:
                yield
            finally:
                events.append("app_exit")

        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(RuntimeError, "worker failed"):
            with app(), runner.finalizer_live_recovery(lambda: sandbox, Path(temp), manifest, payload):
                raise RuntimeError("worker failed")
        self.assertEqual(events, ["read", "app_exit"])
        self.assertTrue(manifest["validation_checkpoint_recovered"])


if __name__ == "__main__":
    unittest.main()
