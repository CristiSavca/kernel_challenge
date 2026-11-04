"""CPU mocks and native capped-reader checks; no Modal/provider/GPU calls."""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import modal_compile_sass as cs
import modal_h200 as runner
from test_modal_h200 import LifecycleTests, Sandbox

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/candidates/muse_moe_w4a16" / (cs.SOURCE_SHA + ".py")


def payload():
    files = {"baseline.py": SOURCE.read_text(), "compile_sass.py": Path(cs.__file__).read_text(),
             "hardware_metrics.py": (ROOT / "tools/modal_hardware_metrics.py").read_text()}
    return {"compile_sass": True, "gpu": None, "resource_kind": "cpu_4cpu_32gib", "timeout_seconds": 600,
            "compile_budget_seconds": 560, "image": cs.IMAGE, "source_sha256": cs.SOURCE_SHA,
            "files": files, "file_sha256": {n: cs.sha(s.encode()) for n, s in files.items()}}


def symbols():
    return "\n".join("Function : _ZN3okc9moe_fusedILb" + str(b) + "EEEvNS_6ParamsE" for b in (0, 1))


class CompileUnitTests(unittest.TestCase):
    def test_real_isolated_uploaded_payload_import_without_repository_path(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            p = payload()
            for basename, source in p["files"].items(): (folder / basename).write_text(source)
            code = """import importlib.util, pathlib, sys
assert not any('open-kernel-challenge' in p for p in sys.path)
path = pathlib.Path(sys.argv[1]) / 'compile_sass.py'
spec = importlib.util.spec_from_file_location('isolated_compile_sass', path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert 'torch' not in sys.modules
assert helper._bounded_path == path.with_name('hardware_metrics.py')
r = helper.run_bounded([sys.executable, '-I', '-c', 'print(123)'], 3, 1024)
assert helper.command_ok(r) and r['stdout'].strip() == '123'
print('isolated-payload-import-ok')
"""
            env = dict(os.environ); env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(folder)], cwd=folder,
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "isolated-payload-import-ok")
            old_import = subprocess.run([sys.executable, "-I", "-S", "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); from modal_hardware_metrics import run_bounded", str(folder)],
                cwd=folder, env=env, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(old_import.returncode, 0)
            self.assertIn("No module named 'modal_hardware_metrics'", old_import.stderr)

    def test_fixed_source_and_cpu_contract(self):
        p = payload(); cs.validate_payload(p)
        for key, value in (("gpu", "H200"), ("resource_kind", "h200_4cpu_32gib"), ("image", "other"),
                           ("timeout_seconds", 900), ("compile_budget_seconds", 860), ("source_sha256", "other")):
            with self.subTest(key=key), self.assertRaises(ValueError): cs.validate_payload({**p, key: value})
        p["files"]["baseline.py"] += "\n#changed"
        p["file_sha256"]["baseline.py"] = cs.sha(p["files"]["baseline.py"].encode())
        with self.assertRaises(ValueError): cs.validate_payload(p)

    def actual_ext_fixture(self):
        source = SOURCE.read_text()
        module = types.ModuleType("actual_parent_ext")
        module.__dict__.update(cs.source_strings(source), PDL=False, MAIN_DEFINES=cs.DEFINES[:-1],
            DQ7=True, ALT_SRCS={}, MAIN_NS=3, _EXTS={}, hashlib=cs.hashlib,
            prepare=lambda: None, kernel_fn=lambda: None)
        no = lambda *a, **kw: None
        cuda = types.SimpleNamespace(get_device_capability=no, is_available=no, device_count=no,
            _lazy_init=no, init=no, current_device=no, set_device=no, synchronize=no, is_initialized=lambda: False)
        torch = types.SimpleNamespace(cuda=cuda)
        for n in ("empty", "empty_like", "zeros", "ones", "full", "tensor", "as_tensor", "arange", "rand", "randn", "randint"):
            setattr(torch, n, no)
        module.torch = torch
        extension = types.SimpleNamespace(__file__=str(cs.BUILD / cs.EXT_NAME / (cs.EXT_NAME + ".so")), okc_set_pdl=no)
        def load(**kwargs):
            subprocess.run(["ninja", "-v"], check=True, stdout=subprocess.PIPE)
            return extension
        module.load_inline = load
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_ext")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "actual_parent_ext", "exec"), module.__dict__)
        return module, torch

    def test_actual_unchanged_ext_builds_once_with_exact_flags_and_host_only_pdl(self):
        module, torch = self.actual_ext_fixture()
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess(["ninja", "-v"], 0, stdout=b"ptxas fixture\n")) as run, \
             contextlib.redirect_stdout(io.StringIO()) as log:
            got = cs.compile_loaded(module, torch, None)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(got["cuda_flags"], list(cs.CUDA_FLAGS))
        self.assertEqual(got["capability_calls"], 2)
        self.assertEqual(got["pdl_values"], [0])
        self.assertFalse(got["cuda_initialized"])
        self.assertIn("ptxas fixture", log.getvalue())
        self.assertEqual(list(module._EXTS), [cs.EXT_DEFINES])

    def test_device_work_and_second_build_are_rejected(self):
        for action in (lambda m, t: t.ones(1), lambda m, t: m.prepare(), lambda m, t: t.cuda.init()):
            m, t = self.actual_ext_fixture(); m._ext = lambda *a, **kw: action(m, t)
            with self.assertRaisesRegex(RuntimeError, "forbidden"): cs.compile_loaded(m, t, None)
        m, t = self.actual_ext_fixture()
        m.PDL = True
        with self.assertRaises(ValueError): cs.compile_loaded(m, t, None)
        m, t = self.actual_ext_fixture()
        def twice(**kwargs):
            subprocess.run(["ninja", "-v"])
            subprocess.run(["ninja", "-v"])
        m.load_inline = twice
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=b"")), \
             self.assertRaisesRegex(RuntimeError, "second ninja"):
            cs.compile_loaded(m, t, None)

    def run_fixture(self, failure=None):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        paths = {"BUILD": folder / "build", "ARTIFACTS": folder / "artifacts", "CHILD": folder / "child.json",
                 "RESULT": folder / "result.json", "BUNDLE": folder / "artifacts.tar.gz", "FOLDER": folder / "input"}
        calls = []
        def bounded(argv, timeout, cap):
            calls.append((argv, timeout, cap))
            result = {"command": argv, "returncode": 0, "stdout": "version fixture", "stderr": "",
                      "timed_out": False, "output_exceeded": False}
            if "--help" in argv: result["stdout"] = "--print-line-info-inline"
            elif "--child" in argv:
                self.assertLessEqual(timeout, 400)
                build = paths["BUILD"] / cs.EXT_NAME; build.mkdir(parents=True)
                strings = cs.source_strings(SOURCE.read_text())
                for name, text in (("cuda.cu", strings["CUDA_SRC"]), ("main.cpp", strings["CPP_SRC"]), ("build.ninja", "fixed flags")):
                    (build / name).write_text(text)
                owned = build / (cs.EXT_NAME + ".so"); owned.write_bytes(b"owned binary fixture")
                cs.write_json(paths["CHILD"], {"status": "compile_child_complete", "source_sha256": cs.SOURCE_SHA,
                    "native_source_sha256": cs.NATIVE_SHA, "owned_so": str(owned)})
                result["stdout"] = "ptxas fixture"
                if failure == "compile": result.update(returncode=1, stderr="compile failed")
            elif "--dump-resource-usage" in argv: result["stdout"] = symbols()
            elif "--extract-elf" in argv:
                name = "cuda.sm_90.cubin" if failure == "sm90_name" else "cuda.sm_90a.cubin"
                (paths["BUILD"] / "extracted" / name).write_bytes(b"cubin fixture")
                if failure == "multiple_cubins":
                    (paths["BUILD"] / "extracted" / "second.cubin").write_bytes(b"unexpected second cubin")
            elif "--print-line-info-inline" in argv:
                result["stdout"] = symbols() + '\n//## File "cuda.cu", line 723\n'
                if failure == "symbol": result["stdout"] = result["stdout"].replace("ILb0", "ILb2")
                if failure == "line": result["stdout"] = symbols()
                if failure == "output": result["output_exceeded"] = True
            return result
        with contextlib.ExitStack() as stack:
            for n, v in paths.items(): stack.enter_context(patch.object(cs, n, v))
            stack.enter_context(patch.object(cs, "run_bounded", side_effect=bounded))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            cs.run(payload())
        return json.loads(paths["RESULT"].read_text()), paths, calls

    def test_artifacts_both_big_and_source_lines_required_without_retry(self):
        for failure in (None, "sm90_name", "multiple_cubins", "compile", "symbol", "line", "output"):
            result, paths, calls = self.run_fixture(failure)
            success = failure in (None, "sm90_name")
            self.assertEqual(result["status"], "compile_complete" if success else "compile_failed")
            self.assertEqual(sum("--child" in a for a, _, _ in calls), 1)
            self.assertTrue(paths["BUNDLE"].is_file())
            if success:
                self.assertEqual(set(result["artifact_bundle"]["files"]), set(cs.LIMITS))
                self.assertTrue(result["source_line_mapping_present"])
                self.assertEqual(result["extracted_cubin_name"], "cuda.sm_90.cubin" if failure == "sm90_name" else "cuda.sm_90a.cubin")

    def test_retrieval_verifies_ownership_limits_and_keeps_recovery_separate(self):
        result, paths, _ = self.run_fixture()
        result["artifact_bundle"]["path"] = str(cs.BUNDLE)
        bundle = paths["BUNDLE"].read_bytes()
        calls = []
        def execute(*args, **kwargs):
            calls.append((args, kwargs))
            data = json.dumps(result).encode() if args[-1] == str(cs.RESULT) else bundle
            return types.SimpleNamespace(stdout=io.BytesIO(data), stderr=io.BytesIO(), returncode=0, wait=lambda: None)
        sandbox = types.SimpleNamespace(exec=execute)
        out = paths["RESULT"].parent / "local"; out.mkdir()
        cs.retrieve(sandbox, out, recovery=True)
        self.assertTrue((out / "compile-owned.so").is_file())
        self.assertTrue((out / "compile-recovered-result.json").is_file())
        self.assertFalse((out / "result.json").exists())
        self.assertEqual([kw for _, kw in calls], [{"timeout": 5, "text": False}, {"timeout": 15, "text": False}])
        result["artifact_bundle"]["sha256"] = "wrong"
        with self.assertRaises(ValueError): cs.retrieve(sandbox, out)
        self.assertFalse((out / "result.json").exists())


class CompileLifecycleTests(LifecycleTests):
    def setUp(self):
        super().setUp()
        dest = self.root / "experiments/candidates/muse_moe_w4a16" / SOURCE.name
        dest.parent.mkdir(parents=True); dest.write_bytes(SOURCE.read_bytes())
        for name in ("modal_compile_sass.py", "modal_hardware_metrics.py"):
            (self.root / "tools" / name).write_bytes((ROOT / "tools" / name).read_bytes())

    def test_cpu_kind_and_no_gpu_binding_termination(self):
        sandbox = Sandbox()
        with patch.object(cs, "retrieve", return_value={"status": "compile_complete"}):
            created = self.invoke("compile-sass", sandbox, extra_args=("--timeout", "600"))
        self.assertEqual(len(created), 1)
        self.assertIsNone(created[0]["gpu"])
        self.assertEqual(created[0]["cpu"], (4, 4))
        self.assertEqual(created[0]["memory"], (32768, 32768))
        self.assertEqual(created[0]["timeout"], 600)
        self.assertTrue(sandbox.terminated)
        self.assertEqual(self.manifest()["credit_guard"]["resource_kind"], "cpu_4cpu_32gib")
        p = json.loads(sandbox.uploads["/tmp/okc-payload.json"]); cs.validate_payload(p)
        self.assertEqual(set(p["files"]), cs.FILES)
        self.assertEqual(sandbox.command[:2], ("python3", "/tmp/okc-worker.py"))

    def test_compile_failure_and_retrieval_failure_terminate_once(self):
        for fail in (False, True):
            sandbox = Sandbox(fail=fail)
            error = RuntimeError if fail else ValueError
            with patch.object(cs, "retrieve", side_effect=ValueError("retrieval failure")), self.assertRaises(error) as caught:
                self.invoke("compile-sass", sandbox, extra_args=("--timeout", "600"))
            self.assertTrue(sandbox.terminated)
            if fail:
                self.assertIn("Remote process exited 1", str(caught.exception))
                manifests = list((self.root / "experiments/modal").glob("*/manifest.json"))
                manifest_path = next(p for p in manifests if json.loads(p.read_text()).get("exit_code") == 1)
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(manifest["remote_failure"], {"exit_code": 1, "stderr_log": "stderr.log"})
                self.assertEqual(manifest["compile_retrieval_error_type"], "ValueError")
                self.assertFalse(manifest["compile_checkpoint_recovered"])
                self.assertTrue(manifest["sandbox_terminated"])
                self.assertIn("synthetic worker failure", (manifest_path.parent / "stderr.log").read_text())
                self.assertNotIn("retrieval failure", manifest["error"])
        self.assertEqual(len(self.created), 2)  # One allocation per independently requested fixture invocation.

    def test_cpu_dry_run_and_argument_rejection(self):
        self.billing = AssertionError("provider call")
        self.assertEqual(self.invoke("compile-sass", dry_run=True, extra_args=("--timeout", "600")), [])
        for extra in ((), ("--timeout", "900"), ("--timeout", "600", "--gpu", "H200"),
                      ("--timeout", "600", "--hardware-issue"), ("--timeout", "600", "--candidate", str(self.candidate))):
            with self.assertRaises(SystemExit): self.invoke("compile-sass", dry_run=True, extra_args=extra)

    def test_wrong_credit_kind_stops_before_allocation(self):
        @contextlib.contextmanager
        def wrong_guard(*args, **kwargs):
            self.assertEqual(kwargs, {"resource_kind": "cpu_4cpu_32gib"})
            yield {"resource_kind": "h200_4cpu_32gib"}
        with patch("modal_credit_budget.allocation_guard", wrong_guard), self.assertRaisesRegex(RuntimeError, "kind mismatch"):
            self.invoke("compile-sass", Sandbox(), extra_args=("--timeout", "600"))
        self.assertEqual(self.created, [])

    def test_dry_run_never_creates_cloud_resource(self):
        self.assertEqual(self.invoke("compile-sass", dry_run=True, extra_args=("--timeout", "600")), [])
        self.assertFalse((self.root / "experiments/modal").exists())


if __name__ == "__main__": unittest.main()
