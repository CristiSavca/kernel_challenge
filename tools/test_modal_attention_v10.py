"""CPU-only fixed-screen policy, dispatch observation and failure-evidence tests."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import attention_resource_probe as probe
import modal_h200 as runner


class AttentionV10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = runner.build_attention_v10_payload(runner.ROOT, "0123456789abcdef")

    def resources(self):
        return dict(status="ok", template="attn_kernel<8,true,0>", local_bytes=0,
                    threads_per_block=384, smem_struct_bytes=113664, requested_dynamic_shared_bytes=114688,
                    sm_count=132, compute_major=9, compute_minor=0, active_blocks_per_sm=2,
                    max_dynamic_shared_bytes=114688)

    def args(self):
        args = [object() for _ in range(30)]
        for index, value in {13: 16, 14: 8, 15: 16, 16: 2, 17: 1, 18: 0, 19: 0,
                             20: 264, 21: 1, 22: 1, 25: 0, 26: 1, 29: 2 | (1135 << 2)}.items():
            args[index] = value
        return args

    def observe(self, args, *, failure=False):
        outputs = []
        def native(*received):
            self.assertEqual(received, tuple(args))
            if failure:
                raise RuntimeError("original native failure")
            outputs.append(17)
            return 17
        original = lambda: types.SimpleNamespace(run=native)
        module = types.SimpleNamespace(_module=original)
        records = []
        try:
            output = runner.observe_attention_v10_call(module, lambda: module._module().run(*args), records)
            self.assertEqual(output, 17)
        finally:
            self.assertIs(module._module, original)
        self.assertTrue(all(type(value) is int for value in records[0].values()))
        return records

    def test_fixed_payload_and_original_vs_transformed_hashes(self):
        runner.validate_attention_v10_payload(self.payload)
        self.assertEqual(set(self.payload["files"]), runner.V10_FILES)
        self.assertEqual(len(self.payload["cases"]) * 2 * (2 + self.payload["repeats"]), 128)
        for side in ("baseline", "candidate"):
            proof = self.payload["attention_v10_instrumentation"][side]
            self.assertEqual(proof["input_sha256"], self.payload["file_sha256"][side + ".py"])
            self.assertNotEqual(proof["input_sha256"], proof["output_sha256"])
            self.assertTrue(proof["exact_reverse_proof"])

    def test_payload_mutations_rejected_even_with_recomputed_file_hash(self):
        for mutate in (lambda p: p.update(seed=123), lambda p: p.update(repeats=2),
                       lambda p: p["cases"][0].update(batch=8),
                       lambda p: p["files"].update(unapproved="private"),
                       lambda p: p.update(attention_v10_tag="../bad")):
            payload = copy.deepcopy(self.payload)
            mutate(payload)
            with self.assertRaises(ValueError):
                runner.validate_attention_v10_payload(payload)
        for file in runner.V10_SOURCES:
            payload = copy.deepcopy(self.payload)
            payload["files"][file] += "\n# changed\n"
            payload["file_sha256"][file] = runner.sha(payload["files"][file])
            with self.assertRaisesRegex(ValueError, "frozen source changed"):
                runner.validate_attention_v10_payload(payload)

    def test_dry_run_does_not_import_modal_or_acquire_guard(self):
        with patch.object(sys, "argv", ["modal_h200.py", "attention-v10-screen", "--dry-run", "--timeout", "900"]), \
                patch.dict(sys.modules, {"modal": None}), contextlib.redirect_stdout(io.StringIO()) as output:
            runner.main()
        manifest = json.loads(output.getvalue())
        self.assertEqual(manifest["timeout_seconds"], 900)
        self.assertEqual(manifest["file_sha256"], self.payload["file_sha256"])

    def test_cli_rejects_flexible_sources_cases_seeds_and_repeats(self):
        for flag in (("--candidate", "x.py"), ("--cases", "x.json"), ("--seed", "1"),
                     ("--repeats", "2"), ("--incumbent", "x.py"), ("--problem", "muse_attention_decode")):
            with patch.object(sys, "argv", ["modal_h200.py", "attention-v10-screen", "--dry-run", *flag]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()

    def test_observer_forwards_actual_args_and_restores_on_native_error(self):
        self.observe(self.args())
        with self.assertRaisesRegex(RuntimeError, "original native failure"):
            self.observe(self.args(), failure=True)

    def test_observer_rejects_signature_change_and_restores(self):
        args = self.args()
        args[20] = object()
        with self.assertRaisesRegex(RuntimeError, "signature changed"):
            self.observe(args)

    def test_target_occupancy_dispatch_and_controls(self):
        records = self.observe(self.args())
        report = runner.attention_v10_dispatch(records, self.resources(), target=True)
        self.assertTrue(report["activation_gate"])
        self.assertEqual(report["grid"], [264, 1, 1])
        for field, value in (("q_len", 1), ("flat", 0), ("n_splits", 132), ("rope", 1),
                             ("window", 2048), ("tma_flags", 0), ("kv_heads", 4), ("cs", 4)):
            changed = [{**records[0], field: value}]
            self.assertFalse(runner.attention_v10_dispatch(changed, self.resources(), target=True)["activation_gate"])
        self.assertFalse(runner.attention_v10_dispatch(records, {**self.resources(), "active_blocks_per_sm": 1}, target=True)["activation_gate"])
        for records in ([], [{**records[0], "q_len": 1}], [{**records[0], "flat": 0, "window": 2048}]):
            control = runner.attention_v10_dispatch(records, self.resources(), target=False)
            self.assertFalse(control["activation_gate"])
            self.assertFalse(control["target"])

    def test_compiler_command_or_cache_hit_is_not_actual_ptxas(self):
        for text in ("cached module reused", "nvcc -Xptxas=-v source.cu"):
            summary = runner.attention_v10_compiler_summary(probe, {"complete": True, "text": text})
            self.assertFalse(summary["compile_log_gate"])
            with self.assertRaisesRegex(RuntimeError, "actual ptxas"):
                runner.require_attention_v10_resources(self.resources(), summary)
        summary = runner.attention_v10_compiler_summary(probe, {"complete": True, "text": "ptxas info : Used 80 registers\n"})
        runner.require_attention_v10_resources(self.resources(), summary)
        for field, value in (("local_bytes", 8), ("active_blocks_per_sm", 1), ("sm_count", 114),
                             ("requested_dynamic_shared_bytes", 115000), ("max_dynamic_shared_bytes", 0),
                             ("template", "attn_kernel<8,false,0>")):
            with self.assertRaisesRegex(RuntimeError, "resource gate failed"):
                runner.require_attention_v10_resources({**self.resources(), field: value}, summary)
        with self.assertRaisesRegex(RuntimeError, "actual ptxas"):
            runner.require_attention_v10_resources(self.resources(), {**summary, "complete": False})

    def test_changed_or_unattributed_serialization_is_rejected(self):
        summary = {"complete": True, "saw_ptxas": True, "actual_ptxas_diagnostics": True}
        for scope in ("changed_flat", "unattributed", "other_kernel"):
            value = {**summary, "warnings": [{"scope": scope, "line": "ptxas warning : C7515 serializing"}]}
            if scope == "other_kernel":
                runner.require_attention_v10_resources(self.resources(), value)
            else:
                with self.assertRaisesRegex(RuntimeError, "warning requires review"):
                    runner.require_attention_v10_resources(self.resources(), value)

    def test_checkpoint_retrieval_saves_logs_and_never_masks_original_error(self):
        result = {"status": "running", "checkpoint_phase": "candidate_compiled_before_any_attention_launch",
                  "attention_v10": {"compilation": {"candidate": {"capture": {"text": "ptxas info : done\n"}}}}}
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            sandbox = types.SimpleNamespace(filesystem=types.SimpleNamespace(read_text=lambda _: json.dumps(result)))
            manifest = {}
            runner.recover_attention_v10_checkpoint(sandbox, destination, manifest)
            self.assertTrue(manifest["attention_v10_checkpoint_recovered"])
            self.assertEqual((destination / "compile_candidate.log").read_text(), "ptxas info : done\n")
            self.assertEqual(json.loads((destination / "result.json").read_text()), result)
            sandbox.filesystem.read_text = lambda _: (_ for _ in ()).throw(RuntimeError("sandbox already expired"))
            runner.recover_attention_v10_checkpoint(sandbox, destination, manifest)
            self.assertFalse(manifest["attention_v10_checkpoint_recovered"])
            self.assertEqual(manifest["attention_v10_retrieval_error_type"], "RuntimeError")

    def test_outer_timeout_recovers_evidence_before_termination(self):
        import modal_credit_budget
        order = []
        result = {"status": "running", "attention_v10": {"compilation": {
            "candidate": {"capture": {"text": "ptxas info : retained before hang\n"}}}}}
        def read(path):
            order.append("recover")
            return json.dumps(result)
        sandbox = types.SimpleNamespace(object_id="sb-test", get_dashboard_url=lambda: "https://modal.com/apps/test",
            filesystem=types.SimpleNamespace(write_text=lambda *args: None, read_text=read),
            exec=lambda *args, **kw: (_ for _ in ()).throw(TimeoutError("original worker timeout")),
            terminate=lambda: order.append("terminate"))
        create_calls = []
        def create(*args, **kwargs):
            create_calls.append(kwargs)
            return sandbox
        fake_modal = types.SimpleNamespace(__version__="test", enable_output=contextlib.nullcontext,
            App=lambda name: types.SimpleNamespace(run=lambda **kw: contextlib.nullcontext()),
            Image=types.SimpleNamespace(from_registry=lambda image: image),
            Sandbox=types.SimpleNamespace(create=create))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tools").mkdir()
            (root / "tools/modal_credit_grant.json").write_text("{}")
            with patch.object(runner, "ROOT", root), \
                    patch.object(runner, "build_attention_v10_payload", return_value=copy.deepcopy(self.payload)), \
                    patch.object(sys, "argv", ["modal_h200.py", "attention-v10-screen", "--timeout", "900"]), \
                    patch.dict(sys.modules, {"modal": fake_modal}), \
                    patch.object(modal_credit_budget, "allocation_guard", return_value=contextlib.nullcontext({"environment": "main"})) as guard, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(TimeoutError, "original worker timeout"):
                runner.main()
            self.assertEqual(order, ["recover", "terminate"])
            self.assertEqual(guard.call_count, 1)
            self.assertEqual(guard.call_args.args[2], 900)
            self.assertEqual(len(create_calls), 1)
            self.assertTrue(create_calls[0]["block_network"])
            self.assertEqual(create_calls[0]["gpu"], "H200")
            self.assertEqual(create_calls[0]["timeout"], 900)
            manifest_path = next((root / "experiments/modal").glob("*/manifest.json"))
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(manifest["sandbox_terminated"])
            self.assertTrue(manifest["attention_v10_checkpoint_recovered"])
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue((manifest_path.parent / "compile_candidate.log").exists())
            self.assertTrue((manifest_path.parent / "instrumented_candidate.py").exists())

    def test_compile_capture_and_checkpoint_precede_attention_or_synchronize(self):
        # Check the worker's compile branch, including the no-launch builder callable.
        tree = ast.parse(Path(runner.__file__).read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_worker")
        capture = next(node for node in ast.walk(worker) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "capture_compile_call")
        self.assertEqual(ast.unparse(capture.args[0]), "module._module")
        launch = next(node for node in ast.walk(worker) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "observe_attention_v10_call")
        self.assertLess(capture.lineno, launch.lineno)
        text = Path(runner.__file__).read_text()
        self.assertLess(text.index('checkpoint(f"{name}_compiled_before_any_attention_launch")'),
                        text.index('checkpoint(f"case{index}_{name}_before_untimed_launch")'))


if __name__ == "__main__":
    unittest.main()
