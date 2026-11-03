"""CPU-only checks of cloud lifecycle and upload boundaries; makes no Modal calls."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import modal_h200 as runner


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
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        problem = self.root / "muse_moe_w4a16"
        problem.mkdir()
        self.candidate = problem / "candidate.py"
        self.candidate.write_text("# okc: problem=muse_moe_w4a16\ndef kernel_fn(data, state): return data\n")
        for name in ("baseline.py", "reference.py", "checks.py", "nvfp4.py"):
            (problem / name).write_text("# synthetic public input\n")
        (self.root / ".env").write_text("PRIVATE_TEST_SENTINEL\n")

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, mode, sandbox=None, create_error=None, dry_run=False):
        created = []

        def create(*args, **kwargs):
            created.append(kwargs)
            if create_error:
                raise create_error
            return sandbox

        fake = types.SimpleNamespace(
            __version__="test", enable_output=contextlib.nullcontext,
            App=lambda name: types.SimpleNamespace(run=contextlib.nullcontext),
            Image=types.SimpleNamespace(debian_slim=lambda **kw: "small", from_registry=lambda ref: ref),
            Sandbox=types.SimpleNamespace(create=create))
        argv = ["modal_h200.py", mode]
        if mode == "benchmark":
            argv += ["--problem", "muse_moe_w4a16", "--candidate", str(self.candidate)]
        if dry_run:
            argv.append("--dry-run")
        with patch.object(runner, "ROOT", self.root), patch.object(sys, "argv", argv), \
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
        created = self.invoke("benchmark", create_error=AssertionError("Cloud call"), dry_run=True)
        self.assertEqual(created, [])
        self.assertFalse((self.root / "experiments").exists())


if __name__ == "__main__":
    unittest.main()
