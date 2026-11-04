"""Fixed result-reader and mocked lifecycle checks; no provider, CUDA or allocation."""
import copy
import contextlib
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
import serving_finalizer_validation_worker as worker
import test_modal_h200 as inherited

ROOT = Path(__file__).resolve().parents[1]


def result_fixture(payload, status="pass"):
    result = {"candidate_sha256": payload["source_sha256"], "baseline_sha256": payload["baseline_sha256"],
              "worker_sha256": payload["file_sha256"]["finalizer_validation.py"],
              "case_manifest_sha256": payload["case_manifest_sha256"], "file_sha256": payload["file_sha256"],
              "image": payload["image"], "status": status, "official": False, "diagnostic_only": True}
    if status == "pass":
        cases = worker.cases()
        result.update(planned_cases=len(cases), completed_cases=len(cases),
            cases=[{"name": c["name"], "seed": c["seed"], "descriptor": c, "pass": True} for c in cases],
            profiles=[{"name": name, "kind": kind, "pass": True, "cuda_positive_duration_events_only": True,
                       "gpu_event_count": 2 if kind == "fast" else 1}
                      for name, kind in [("boundary31", "fast"), ("boundary32", "large"), ("boundary33", "large"),
                          ("fallback_fp32", "eager"), ("fallback_topk33", "eager"), ("fallback_hidden16385", "eager")]],
            two_streams={"pass": True, "calls": 8}, two_graphs={"pass": True, "graphs": 2, "fresh_replays": 8})
    return result


class ValidationSandbox(inherited.Sandbox):
    def __init__(self, payload, worker_exit=0, result_status="pass", retrieval_error=None):
        super().__init__(fail=bool(worker_exit))
        self.worker_exit = worker_exit
        self.result = result_fixture(payload, result_status)
        self.retrieval_error = retrieval_error
        self.executions = []
        self.manifest_observer = None

    def read_text(self, path):
        raise AssertionError("Unbounded filesystem read is forbidden for finalizer validation")

    def exec(self, *args, **kwargs):
        self.executions.append((args, kwargs))
        if args == ("python3", "-c", runner.FINALIZER_READ_RESULT):
            if self.manifest_observer: self.manifest_observer()
            if self.retrieval_error: raise self.retrieval_error
            return types.SimpleNamespace(stdout=io.BytesIO(json.dumps(self.result).encode()), stderr=io.BytesIO(),
                                         wait=lambda: None, returncode=0)
        self.command = args
        return types.SimpleNamespace(stdout=iter(["validation worker fixture\n"]),
            stderr=io.StringIO("synthetic finalizer worker failure" if self.worker_exit else ""),
            wait=lambda: None, returncode=self.worker_exit)


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.payload = worker.build_payload(ROOT)

    def test_success_is_one_fixed_capped_bounded_read(self):
        sandbox = ValidationSandbox(self.payload)
        with tempfile.TemporaryDirectory() as temp:
            got = runner.retrieve_finalizer_validation_result(sandbox, Path(temp), self.payload)
            self.assertEqual(got["status"], "pass")
            self.assertTrue((Path(temp) / "result.json").is_file())
            self.assertFalse((Path(temp) / "result-checkpoint.json").exists())
        self.assertEqual(sandbox.executions, [(("python3", "-c", runner.FINALIZER_READ_RESULT), {"timeout": 10, "text": False})])

    def test_actual_reader_caps_before_transfer(self):
        # Execute the actual reader code with its fixed-path open rebound only
        # inside this isolated subprocess; no real /tmp/okc-result.json write.
        code = """import builtins,io,sys
expected=int(sys.argv[1]); original=builtins.open
def opened(path,mode):
 assert path=='/tmp/okc-result.json' and mode=='rb'
 return io.BytesIO(b'x'*expected)
builtins.open=opened
exec(sys.argv[2])
"""
        for size, success in [(1048576, True), (1048577, False)]:
            result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(size), runner.FINALIZER_READ_RESULT],
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode == 0, success)
            self.assertEqual(len(result.stdout), size if success else 0)

    def test_provenance_mutations_reject_without_result_file(self):
        for key in ("candidate_sha256", "baseline_sha256", "worker_sha256", "case_manifest_sha256", "image", "file_sha256"):
            sandbox = ValidationSandbox(self.payload); sandbox.result[key] = "wrong"
            with tempfile.TemporaryDirectory() as temp, self.subTest(key=key):
                with self.assertRaises(ValueError):
                    runner.retrieve_finalizer_validation_result(sandbox, Path(temp), self.payload)
                self.assertFalse(list(Path(temp).iterdir()))

    def test_incomplete_pass_coverage_rejected(self):
        mutations = [lambda r: r["cases"].pop(), lambda r: r["cases"][0].update(seed=-1),
                     lambda r: r["cases"][0].__setitem__("pass", False),
                     lambda r: r["profiles"].pop(), lambda r: r["profiles"][0].update(cuda_positive_duration_events_only=False),
                     lambda r: r["two_streams"].update(calls=7), lambda r: r["two_graphs"].update(fresh_replays=7)]
        for mutation in mutations:
            sandbox = ValidationSandbox(self.payload); mutation(sandbox.result)
            with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
                runner.retrieve_finalizer_validation_result(sandbox, Path(temp), self.payload)

    def test_partial_and_recovered_pass_stay_checkpoints(self):
        for status, recovery in [("running", False), ("failed", False), ("incomplete", True), ("pass", True)]:
            sandbox = ValidationSandbox(self.payload, result_status=status)
            with tempfile.TemporaryDirectory() as temp:
                runner.retrieve_finalizer_validation_result(sandbox, Path(temp), self.payload, recovery=recovery)
                self.assertTrue((Path(temp) / "result-checkpoint.json").is_file())
                self.assertFalse((Path(temp) / "result.json").exists())

    def test_local_limit_and_remote_exit_rejected(self):
        for data, code in [(b"x" * 1048577, 0), (b"{}", 1), (b"invalid-json", 0)]:
            sandbox = types.SimpleNamespace(exec=lambda *a, **kw: types.SimpleNamespace(
                stdout=io.BytesIO(data), stderr=io.BytesIO(), wait=lambda: None, returncode=code))
            with tempfile.TemporaryDirectory() as temp, self.assertRaises((ValueError, RuntimeError)):
                runner.retrieve_finalizer_validation_result(sandbox, Path(temp), self.payload)


class FinalizerLifecycleTests(unittest.TestCase):
    # Reuse only fixture helpers, without re-running inherited test methods here.
    setUp = inherited.LifecycleTests.setUp
    tearDown = inherited.LifecycleTests.tearDown
    write_grant = inherited.LifecycleTests.write_grant
    modal_json = inherited.LifecycleTests.modal_json
    invoke = inherited.LifecycleTests.invoke
    manifest = inherited.LifecycleTests.manifest

    def invoke_finalizer(self, sandbox, *, dry_run=False):
        payload = worker.build_payload(ROOT)
        with patch.object(worker, "build_payload", return_value=copy.deepcopy(payload)), \
             patch.object(runner, "finalizer_observed_floor", return_value={"usd": "4.16434237", "path": "fixture", "sha256": "fixture"}):
            return self.invoke("serving-finalizer-validation", sandbox, dry_run=dry_run, extra_args=("--timeout", "600"))

    def test_success_resources_provenance_and_termination(self):
        sandbox = ValidationSandbox(worker.build_payload(ROOT))
        created = self.invoke_finalizer(sandbox)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["gpu"], "H200")
        self.assertEqual(created[0]["cpu"], (4, 4))
        self.assertEqual(created[0]["memory"], (32768, 32768))
        self.assertEqual(created[0]["timeout"], 600)
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.manifest()["status"], "pass")
        self.assertTrue(self.manifest()["validation_result_retrieved"])
        self.assertEqual(self.manifest()["exit_code"], 0)
        self.assertEqual(sum(a == ("python3", "-c", runner.FINALIZER_READ_RESULT) for a, _ in sandbox.executions), 1)

    def test_nonzero_exit_saved_before_failed_recovery_and_not_replaced(self):
        sandbox = ValidationSandbox(worker.build_payload(ROOT), worker_exit=7, retrieval_error=RuntimeError("read unavailable"))
        def observe():
            manifest = self.manifest()
            self.assertEqual(manifest["exit_code"], 7)
            self.assertEqual(manifest["remote_failure"], {"exit_code": 7, "stderr_log": "stderr.log"})
        sandbox.manifest_observer = observe
        with self.assertRaisesRegex(RuntimeError, "Remote process exited 7"):
            self.invoke_finalizer(sandbox)
        manifest = self.manifest()
        self.assertTrue(sandbox.terminated)
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(manifest["validation_checkpoint_recovered"])
        self.assertEqual(manifest["validation_retrieval_error_type"], "RuntimeError")
        self.assertNotIn("read unavailable", manifest["error"])
        self.assertEqual(sum(a == ("python3", "-c", runner.FINALIZER_READ_RESULT) for a, _ in sandbox.executions), 1)

    def test_nonzero_exit_with_recoverable_checkpoint_stays_failed(self):
        sandbox = ValidationSandbox(worker.build_payload(ROOT), worker_exit=1, result_status="failed")
        with self.assertRaisesRegex(RuntimeError, "Remote process exited 1"):
            self.invoke_finalizer(sandbox)
        manifest = self.manifest()
        self.assertTrue(sandbox.terminated)
        self.assertTrue(manifest["validation_checkpoint_recovered"])
        self.assertEqual(manifest["status"], "failed")
        folder = next((self.root / "experiments/modal").glob("*/manifest.json")).parent
        self.assertTrue((folder / "result-checkpoint.json").is_file())
        self.assertFalse((folder / "result.json").exists())

    def test_zero_exit_incomplete_result_not_promoted_or_read_twice(self):
        sandbox = ValidationSandbox(worker.build_payload(ROOT), result_status="incomplete")
        with self.assertRaisesRegex(RuntimeError, "incomplete or failed"):
            self.invoke_finalizer(sandbox)
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(sum(a == ("python3", "-c", runner.FINALIZER_READ_RESULT) for a, _ in sandbox.executions), 1)

    def test_zero_exit_retrieval_failure_still_terminates_without_worker_retry(self):
        sandbox = ValidationSandbox(worker.build_payload(ROOT), retrieval_error=RuntimeError("bounded read failed"))
        with self.assertRaisesRegex(RuntimeError, "bounded read failed"):
            self.invoke_finalizer(sandbox)
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertEqual(self.manifest()["exit_code"], 0)
        self.assertFalse(self.manifest()["validation_checkpoint_recovered"])
        readers = [(a, kw) for a, kw in sandbox.executions if a == ("python3", "-c", runner.FINALIZER_READ_RESULT)]
        self.assertEqual(len(readers), 2)  # initial read plus bounded checkpoint recovery, never worker retry
        self.assertTrue(all(kw == {"timeout": 10, "text": False} for _, kw in readers))
        self.assertEqual(len(sandbox.executions) - len(readers), 1)

    def test_dry_run_has_no_provider_calls(self):
        self.billing = AssertionError("provider called")
        self.assertEqual(self.invoke_finalizer(None, dry_run=True), [])
        self.assertFalse((self.root / "experiments").exists())

    def test_checkpoint_recovered_before_app_context_teardown(self):
        payload = worker.build_payload(ROOT)
        sandbox = ValidationSandbox(payload, worker_exit=1, result_status="failed")
        events = []
        alive = {"value": False}
        original_exec = sandbox.exec
        def execute(*args, **kwargs):
            if args == ("python3", "-c", runner.FINALIZER_READ_RESULT):
                events.append("checkpoint_read")
                if not alive["value"]: raise RuntimeError("sandbox disappeared on app context exit")
            return original_exec(*args, **kwargs)
        sandbox.exec = execute
        @contextlib.contextmanager
        def app_run(**kwargs):
            alive["value"] = True
            try: yield
            finally:
                events.append("app_exit")
                alive["value"] = False
        fake = types.SimpleNamespace(__version__="test", enable_output=contextlib.nullcontext,
            App=lambda name: types.SimpleNamespace(run=app_run),
            Image=types.SimpleNamespace(from_registry=lambda ref: ref),
            Sandbox=types.SimpleNamespace(create=lambda *args, **kwargs: sandbox))
        argv = ["modal_h200.py", "serving-finalizer-validation", "--timeout", "600"]
        with patch.object(runner, "ROOT", self.root), patch.object(sys, "argv", argv), \
             patch.object(budget, "modal_json", self.modal_json), patch.dict(sys.modules, {"modal": fake}), \
             patch.object(worker, "build_payload", return_value=copy.deepcopy(payload)), \
             patch.object(runner, "finalizer_observed_floor", return_value={"usd": "4.16434237"}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, "Remote process exited 1"):
            runner.main()
        self.assertEqual(events, ["checkpoint_read", "app_exit"])
        self.assertTrue(self.manifest()["validation_checkpoint_recovered"])
        self.assertEqual(self.manifest()["status"], "failed")
        self.assertTrue(sandbox.terminated)


if __name__ == "__main__": unittest.main()
