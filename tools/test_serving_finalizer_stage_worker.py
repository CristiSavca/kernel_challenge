"""CPU contracts, actual four-file import and staged arithmetic; no CUDA execution."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import typing
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage_worker", ROOT / "tools/serving_finalizer_stage_worker.py")
w = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(w)


def scalar(dtype="torch.float32", shape=None):
    raw = struct.pack("<9f" if dtype.endswith("32") else "<9d", *([1.] * 9))
    width = len(raw) // 9
    return dict(dtype=dtype, shape=shape or [3, 3, 1], bytes=len(raw), values=[1.] * 9,
                bits_little_endian_hex=[raw[i:i + width].hex() for i in range(0, len(raw), width)],
                sha256=w.sha(raw), raw_prefix_hex=raw[:32].hex())


def summary():
    return dict(l2=0., max_abs=0., row_l2=[0., 0., 0.], nonzero_values=0,
                sha256="0" * 64, raw_prefix_hex="0" * 64)


def comparisons():
    equal = dict(pass_=True, equal_values=True, max_abs=0., whole_rel_l2=0., worst_row_rel_l2=0.)
    equal["pass"] = equal.pop("pass_")
    fail = dict(equal, equal_values=False, max_abs=2.384185791015625e-7, whole_rel_l2=1., worst_row_rel_l2=1.)
    fail["pass"] = False
    return {"candidate_vs_fp32_oracle": equal, "candidate_vs_original": fail,
            "original_vs_fp32_oracle": dict(fail, whole_rel_l2=1582823.078587789, worst_row_rel_l2=1582823.078587789)}


def complete(payload):
    section = {"comparisons": comparisons(), "outputs": {n: summary() for n in ("candidate", "original", "oracle")}}
    return dict(status="diagnostic_complete", diagnostic_complete=True, diagnostic_only=True, official=False,
                generic_validation_status="failed_at_finite_cancellation", descriptor=w.CASE,
                candidate_sha256=w.CANDIDATE_SHA, baseline_sha256=w.BASELINE_SHA, reference_sha256=w.REFERENCE_SHA,
                worker_sha256=payload["file_sha256"]["finalizer_stages.py"], image=w.IMAGE,
                case_manifest_sha256=w.manifest_sha(), file_sha256=payload["file_sha256"],
                reproduction=dict(section, matches_retained_failure=True),
                recomposition={n: True for n in ("candidate", "original", "oracle")}, inputs_unchanged=True,
                stages={"original": {"variance": scalar(), "inverse": scalar()},
                        "oracle": {"variance": scalar(), "inverse": scalar()},
                        "candidate": {"inverse": scalar(shape=[3, 3]), "variance_observed": False}},
                layouts=[dict(name=n, logical_values_equal=True, variance=scalar(), inverse=scalar()) for n in
                         ("contiguous_h511", "padded_pitch512_h511", "aligned_row_copies_h511")],
                fp64=[dict(name=n, variance=scalar("torch.float64"), inverse=scalar("torch.float64"), output=summary()) for n in
                      ("bf16_inputs_fp64_products", "rounded_fp32_products_fp64")],
                h512_control=dict(section, changed_denominator=[511, 512]),
                opposing_routes={**{n: True for n in ("input_value_negation", "weighted_value_negation", "squares_bit_equal",
                                 "original_inverse_bit_equal", "oracle_inverse_bit_equal", "candidate_inverse_bit_equal")},
                                 "original_normalized_pair_sum": summary()})


class Tensor:
    def __init__(self, x): self.a = np.asarray(x)
    @property
    def shape(self): return self.a.shape
    @property
    def dtype(self): return self.a.dtype
    device = "cpu-shim"
    def contiguous(self): return Tensor(np.ascontiguousarray(self.a))
    def float(self): return self.to(np.float32)
    def to(self, dtype): return Tensor(self.a.astype(dtype))
    def reshape(self, *shape): return Tensor(self.a.reshape(*shape))
    def unsqueeze(self, dim): return Tensor(np.expand_dims(self.a, dim))
    def pow(self, exponent): return Tensor(self.a ** exponent)
    def mean(self, dim, keepdim=False): return Tensor(self.a.mean(axis=dim, keepdims=keepdim))
    def sum(self, dim, keepdim=False): return Tensor(self.a.sum(axis=dim, keepdims=keepdim))
    def __getitem__(self, item): return Tensor(self.a[item])
    def __mul__(self, other): return Tensor(self.a * (other.a if isinstance(other, Tensor) else other))
    def __add__(self, other): return Tensor(self.a + (other.a if isinstance(other, Tensor) else other))
    def __truediv__(self, other): return Tensor(self.a / other)
    def __ge__(self, other): return Tensor(self.a >= other)


class Torch:
    Tensor = Tensor
    float32 = np.float32
    @staticmethod
    def zeros(shape, dtype, device): return Tensor(np.zeros(shape, dtype))
    @staticmethod
    def empty(shape, dtype, device): return Tensor(np.empty(shape, dtype))
    @staticmethod
    def zeros_like(x): return Tensor(np.zeros_like(x.a))
    @staticmethod
    def where(mask, x, zero): return Tensor(np.where(mask.a, x.a, zero.a))
    @staticmethod
    def rsqrt(x): return Tensor(np.float32(1) / np.sqrt(x.a))
    @staticmethod
    def stack(items, dim): return Tensor(np.stack([x.a for x in items], axis=dim))


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.payload = w.build_payload(ROOT)

    def test_fixed_build_and_pinned_descriptor(self):
        w.validate_payload(self.payload)
        ref = w._load(ROOT / "tools/serving_finalizer_validation_worker.py", "stage_test_reference")
        self.assertEqual(next(c for c in ref.cases() if c["name"] == "finite_cancellation"), w.CASE)
        self.assertNotIn("torch", sys.modules)

    def test_policy_mutations(self):
        for key, value in (("gpu", None), ("timeout_seconds", 301), ("validation_budget_seconds", 241),
                           ("seed", 0), ("source_sha256", "0" * 64), ("reference_sha256", "0" * 64),
                           ("image", "wrong"), ("case_manifest_sha256", "0" * 64), ("serving_finalizer_stages", 1)):
            p = copy.deepcopy(self.payload); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): w.validate_payload(p)

    def test_all_four_source_pins_and_file_set(self):
        for name in w.FILES:
            p = copy.deepcopy(self.payload); p["files"][name] += "\n"
            p["file_sha256"][name] = w.sha(p["files"][name])
            with self.subTest(name=name), self.assertRaises(ValueError): w.validate_payload(p)
        p = copy.deepcopy(self.payload); p["files"].pop("validation_reference.py")
        with self.assertRaises(ValueError): w.validate_payload(p)

    def test_actual_isolated_four_file_package(self):
        with tempfile.TemporaryDirectory(prefix="okc-stage-package-") as folder:
            p = Path(folder)
            for name, source in self.payload["files"].items(): (p / name).write_text(source)
            (p / "payload.json").write_text(json.dumps(self.payload))
            code = """import importlib.util,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('stage',p/'finalizer_stages.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.validate_payload(json.loads((p/'payload.json').read_text()))
r=m._load(p/'validation_reference.py','reference')
assert next(c for c in r.cases() if c['name']=='finite_cancellation')==m.CASE
assert 'torch' not in sys.modules and 'triton' not in sys.modules
assert not any('HASIDIC_PROJECTS' in x for x in sys.path)
print('four-file-package-PASS')
"""
            env = dict(os.environ); env.pop("PYTHONPATH", None)
            result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(p)],
                                    capture_output=True, text=True, check=True, timeout=15, env=env)
            self.assertEqual(result.stdout.strip(), "four-file-package-PASS")
            (p / "validation_reference.py").rename(p / "wrong_reference.py")
            failed = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(p)],
                                    capture_output=True, text=True, timeout=15, env=env)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("FileNotFoundError", failed.stderr)

    def test_terminal_and_partial_status(self):
        r = complete(self.payload); w.validate_result(r, self.payload)
        for status in ("running", "failed", "incomplete"):
            p = {k: v for k, v in r.items() if k not in ("stages", "layouts", "fp64", "h512_control")}
            p.update(status=status, diagnostic_complete=False); w.validate_result(p, self.payload)
        r["status"] = "pass"
        with self.assertRaises(ValueError): w.validate_result(r, self.payload)

    def test_missing_coverage_and_recomposition_fail_closed(self):
        for field in ("reproduction", "recomposition", "inputs_unchanged", "stages", "layouts", "fp64", "h512_control", "opposing_routes"):
            r = complete(self.payload); r.pop(field)
            with self.subTest(field=field), self.assertRaises((ValueError, KeyError)): w.validate_result(r, self.payload)
        for name in ("candidate", "original", "oracle"):
            r = complete(self.payload); r["recomposition"][name] = False
            with self.subTest(name=name), self.assertRaises(ValueError): w.validate_result(r, self.payload)
        r = complete(self.payload); r["h512_control"] = {"changed_denominator": [511, 512]}
        with self.assertRaises(ValueError): w.validate_result(r, self.payload)
        r = complete(self.payload); r["layouts"][1]["logical_values_equal"] = False
        with self.assertRaises(ValueError): w.validate_result(r, self.payload)

    def test_scalar_bits_hash_width_and_values(self):
        mutations = [("dtype", "torch.float64"), ("shape", [9]), ("values", [float("nan")] * 9),
                     ("sha256", "0" * 64), ("bytes", 72), ("bits_little_endian_hex", ["00"] * 9),
                     ("bits_little_endian_hex", ["zzzzzzzz"] * 9), ("values", [2.] * 9)]
        for key, value in mutations:
            r = complete(self.payload); r["stages"]["original"]["inverse"][key] = value
            with self.subTest(key=key, value=str(value)[:20]), self.assertRaises(ValueError): w.validate_result(r, self.payload)

    def test_provenance_and_reproduction_drift(self):
        for key in ("candidate_sha256", "baseline_sha256", "reference_sha256", "worker_sha256", "image", "case_manifest_sha256"):
            r = complete(self.payload); r[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError): w.validate_result(r, self.payload)
        c = comparisons(); self.assertTrue(w.reproduction_matches(c))
        c["candidate_vs_original"]["max_abs"] *= 2
        self.assertFalse(w.reproduction_matches(c)); self.assertFalse(w.reproduction_matches({}))

    def test_atomic_bounded_report_and_fail_before_torch(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(w, "RESULT", Path(folder) / "result.json"):
            w._save({"status": "running"})
            before = w.RESULT.read_bytes()
            with self.assertRaises(RuntimeError): w._save({"bulk": "x" * w.MAX_RESULT_BYTES})
            self.assertEqual(w.RESULT.read_bytes(), before)
            p = copy.deepcopy(self.payload); p["timeout_seconds"] = 0
            with self.assertRaises(ValueError), mock.patch("builtins.print"): w.run(p)
            self.assertEqual(json.loads(w.RESULT.read_text())["status"], "failed")
            self.assertNotIn("torch", sys.modules)

    def test_actual_staged_arithmetic_matches_pinned_cpu_expressions(self):
        source = ast.parse(self.payload["files"]["baseline.py"])
        fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "fused_post_expert_norm_weighted_sum")
        namespace = dict(torch=Torch, Optional=typing.Optional, _MIN_TOKENS_FOR_FUSED=32)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "actual_original", "exec"), namespace)
        ref = w._load(ROOT / "tools/serving_finalizer_validation_worker.py", "stage_arithmetic_reference")
        rng = np.random.default_rng(20268578)
        for cancellation in (False, True):
            x = rng.normal(size=(3, 3, 511)).astype(np.float32)
            if cancellation: x[:, 1] = -x[:, 0]; x[:, 2] = 0
            args = dict(expert_output=Tensor(x), topk_scores=Tensor(np.ones((3, 3), np.float32)),
                        norm_weight=Tensor((1 + .25 * rng.normal(size=511)).astype(np.float32)),
                        slot_expert_ids=Tensor(np.zeros((3, 3), np.int32)), eps=1e-8, apply_norm=True, weight_first=True)
            actual = namespace[fn.name](**args)
            staged, intermediates = w._original_stages(Torch, args)
            np.testing.assert_array_equal(actual.a.view(np.uint32), staged.a.view(np.uint32))
            with mock.patch.object(w, "_tensor_record", return_value={}) as recorder:
                staged_oracle, _ = w._oracle_stages(Torch, args)
            np.testing.assert_array_equal(ref.oracle(Torch, args).a.view(np.uint32), staged_oracle.a.view(np.uint32))
            self.assertEqual(recorder.call_count, 3)
            self.assertEqual(intermediates["variance"].shape, (3, 3, 1))

    def test_existing_candidate_launch_arguments(self):
        calls = []
        class Kernel:
            def __init__(self, name): self.name = name
            def __getitem__(self, grid):
                def call(*args, **kw): calls.append((self.name, grid, args, kw))
                return call
        module = types.SimpleNamespace(_small_post_expert_inverse_kernel=Kernel("inverse"),
                                       _small_post_expert_combine_kernel=Kernel("combine"))
        triton = types.SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda a, b: (a + b - 1) // b)
        for h in (511, 512):
            args = dict(expert_output=Tensor(np.zeros((3, 3, h), np.float32)), norm_weight=Tensor(np.ones(h, np.float32)),
                        topk_scores=Tensor(np.ones((3, 3), np.float32)), slot_expert_ids=Tensor(np.zeros((3, 3), np.int32)), eps=1e-8)
            w._candidate_stages(Torch, triton, module, args)
            inv, comb = calls[-2:]
            self.assertEqual(inv[1], (3, 3)); self.assertEqual(comb[1], (3, 2))
            self.assertIs(inv[2][3], comb[2][4])
            for call in (inv, comb):
                self.assertEqual(call[3]["H"], h); self.assertEqual(call[3]["TOPK"], 3)
                self.assertEqual(call[3]["num_warps"], 4); self.assertEqual(call[3]["num_stages"], 1)
                self.assertIs(call[3]["enable_fp_fusion"], False)
            self.assertEqual(inv[3]["BLOCK_H"], 512); self.assertEqual(comb[3]["BLOCK_H"], 256)


if __name__ == "__main__": unittest.main()
