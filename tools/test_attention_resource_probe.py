"""CPU-only safety/provenance tests for the local attention resource hook."""
import ast
import hashlib
import os
import unittest
from pathlib import Path

from attention_resource_probe import (CUDA_EXPORT, RESOURCE_FIELDS, CompileCaptureError,
                                     capture_compile_call, collect_attention_resources,
                                     compiler_summary, instrument_source)

ROOT = Path(__file__).resolve().parents[1]


class ResourceProbeTests(unittest.TestCase):
    def test_source_provenance_for_both_wrappers(self):
        for name in ("decoder_parent_v10", "decoder_q8_role_v10"):
            source = (ROOT / "muse_attention_decode/candidates" / (name + ".py")).read_text()
            transformed, proof = instrument_source(source, expected_sha256=hashlib.sha256(source.encode()).hexdigest(), tag=name)
            self.assertTrue(proof["exact_reverse_proof"])
            def native(text):
                return next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n, ast.Assign)
                            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "CUDA_SRC")
            self.assertEqual(native(transformed), native(source) + CUDA_EXPORT)
            self.assertIn('functions=["run", "q8_role_resources_v10"]', transformed)
            self.assertIn('"-Xptxas=-v"', transformed)

    def test_wrong_digest_rejected(self):
        with self.assertRaises(ValueError):
            instrument_source("x=1", expected_sha256="0" * 64, tag="test")

    def test_bounded_scalar_resource_result(self):
        values = [80, 0, 0, 0, 384, 227 * 1024, 90, 80, 2, 384, 114688, 132, 9, 0, 113664, 104, 128, 32, 80]
        class Extension:
            def q8_role_resources_v10(self):
                return values
        result = collect_attention_resources({"_MOD": Extension()})
        self.assertTrue(result["two_cta_resource_gate"])
        self.assertEqual(result["requested_dynamic_shared_bytes"], 114688)
        values[0] = object()
        self.assertEqual(collect_attention_resources({"_ATT7": Extension()})["reason"], "invalid_export_scalar_values")
        self.assertEqual(collect_attention_resources({})["reason"], "extension_not_built")

    def test_capture_preserves_both_streams(self):
        def work():
            os.write(1, b"stdout marker\n")
            os.write(2, b"stderr marker\n")
            return 17
        value, capture = capture_compile_call(work)
        self.assertEqual(value, 17)
        self.assertTrue(capture["complete"])
        self.assertIn("stdout marker", capture["text"])
        self.assertIn("stderr marker", capture["text"])

    def test_capture_overflow_is_explicit_failure(self):
        _, capture = capture_compile_call(lambda: os.write(1, b"x" * 4096), max_bytes=1024)
        self.assertFalse(capture["complete"])
        self.assertTrue(capture["truncated"])
        self.assertEqual(capture["bytes_total"], 4096)
        self.assertEqual(len(capture["text"]), 1024)

    def test_compile_exception_keeps_diagnostics_and_restores_fds(self):
        def work():
            os.write(2, b"compilation failed here\n")
            raise RuntimeError("test failure")
        with self.assertRaises(CompileCaptureError) as result:
            capture_compile_call(work)
        self.assertIn("compilation failed here", result.exception.capture["text"])
        self.assertTrue(result.exception.capture["complete"])
        _, next_capture = capture_compile_call(lambda: os.write(1, b"next call\n"))
        self.assertEqual(next_capture["text"], "next call\n")

    def test_warning_scope_and_unattributed_warning(self):
        capture = {"complete": True, "text": "warning: pre-entry unknown\n"
                   "ptxas info : Compiling entry function '_Zattn_kernelILi8ELb1ELi0EE'\n"
                   "ptxas warning : C7515 serializing\n"
                   "ptxas info : Compiling entry function '_Zattn_kernelILi1ELb0ELi13EE'\n"
                   "ptxas warning : fallback warning\n"}
        result = compiler_summary(capture)
        self.assertEqual([w["scope"] for w in result["warnings"]],
                         ["unattributed", "changed_flat", "other_kernel"])
        self.assertEqual(len(RESOURCE_FIELDS), 19)

    def test_cache_hit_is_not_compiler_evidence(self):
        result = compiler_summary({"complete": True, "text": "cached extension reused\n"})
        self.assertTrue(result["complete"])
        self.assertFalse(result["compile_log_gate"])


if __name__ == "__main__":
    unittest.main()
