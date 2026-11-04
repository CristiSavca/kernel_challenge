"""Payload/packaged-import/diagnostic-metric tests only; no Torch or CUDA execution."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("finalizer_validation", ROOT / "tools/serving_finalizer_validation_worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class Tensor:
    def __init__(self, value): self.data = np.asarray(value)
    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    device = "cpu-adapter"
    def to(self, dtype): return Tensor(self.data.astype(dtype))
    def sum(self): return Tensor(self.data.sum())
    def norm(self, dim=None): return Tensor(np.sqrt(np.sum(self.data * self.data, axis=dim)))
    def clamp(self, min): return Tensor(np.maximum(self.data, min))
    def reshape(self, *shape): return Tensor(self.data.reshape(*shape))
    def mean(self): return Tensor(self.data.mean())
    def max(self): return Tensor(self.data.max())
    def abs(self): return Tensor(np.abs(self.data))
    def all(self): return Tensor(self.data.all())
    def numel(self): return self.data.size
    def __float__(self): return float(self.data)
    def __int__(self): return int(self.data)
    def __bool__(self): return bool(self.data)
    def __sub__(self, other): return Tensor(self.data - other.data)
    def __truediv__(self, other): return Tensor(self.data / other.data)
    def __rmul__(self, scalar): return Tensor(scalar * self.data)
    def __eq__(self, other): return Tensor(self.data == other.data)
    def __ne__(self, other): return Tensor(self.data != other.data)


class TorchAdapter:
    float64 = np.float64
    @staticmethod
    def isnan(a): return Tensor(np.isnan(a.data))
    @staticmethod
    def isposinf(a): return Tensor(np.isposinf(a.data))
    @staticmethod
    def isneginf(a): return Tensor(np.isneginf(a.data))
    @staticmethod
    def isfinite(a): return Tensor(np.isfinite(a.data))
    @staticmethod
    def zeros_like(a): return Tensor(np.zeros_like(a.data))
    @staticmethod
    def where(c, a, b): return Tensor(np.where(c.data, a.data, b.data))
    @staticmethod
    def maximum(a, b): return Tensor(np.maximum(a.data, b.data))


class ValidationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.payload = worker.build_payload(ROOT)

    def test_fixed_payload_and_manifest(self):
        worker.validate_payload(self.payload)
        self.assertEqual(len(worker.cases()), 90)
        self.assertEqual(worker.signature_bound()["small_specialization_upper_bound"], 62)
        self.assertEqual({c["t"] for c in worker.cases() if c["name"].startswith("token_")}, set(range(1, 32)))
        self.assertFalse("torch" in sys.modules)

    def test_policy_mutations_rejected(self):
        for field, value in [("gpu", None), ("timeout_seconds", 601), ("validation_budget_seconds", 541),
                             ("seed", 1), ("image", "other"), ("case_manifest_sha256", "0" * 64),
                             ("source_sha256", "0" * 64), ("baseline_sha256", "0" * 64)]:
            p = copy.deepcopy(self.payload); p[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): worker.validate_payload(p)

    def test_source_and_file_mutations_rejected(self):
        for name in worker.FILES:
            p = copy.deepcopy(self.payload); p["files"][name] += "\n"
            with self.subTest(name=name), self.assertRaises(ValueError): worker.validate_payload(p)
            p["file_sha256"][name] = worker.sha(p["files"][name])
            with self.subTest(rehashed=name), self.assertRaises(ValueError): worker.validate_payload(p)
        p = copy.deepcopy(self.payload); p["files"]["other.py"] = ""
        with self.assertRaises(ValueError): worker.validate_payload(p)

    def test_real_isolated_staged_import(self):
        with tempfile.TemporaryDirectory(prefix="okc-finalizer-contract-") as folder:
            path = Path(folder)
            for name, source in self.payload["files"].items(): (path / name).write_text(source)
            (path / "payload.json").write_text(json.dumps(self.payload))
            code = """import importlib.util,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('staged',p/'finalizer_validation.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.validate_payload(json.loads((p/'payload.json').read_text()))
assert 'torch' not in sys.modules and 'triton' not in sys.modules
assert not any('HASIDIC_PROJECTS' in x for x in sys.path)
print('isolated-package-PASS')
"""
            result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(path)],
                                    check=True, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.stdout.strip(), "isolated-package-PASS")

    def test_mask_scenarios_and_profiles_present(self):
        names = {c["name"] for c in worker.cases()}
        self.assertTrue({"zero_epsilon_remote", "remote_infinite_weight", "remote_nan_score", "nan_epsilon",
                         "infinite_epsilon", "negative_epsilon_remote", "finite_subnormal",
                         "no_norm_live_positive_inf", "no_norm_live_negative_inf", "ragged_last_column", "boundary31",
                         "boundary32", "boundary33", "fallback_fp32", "fallback_topk33", "fallback_hidden16385"} <= names)
        source = Path(worker.__file__).read_text()
        self.assertIn('event.time_range.elapsed_us() > 0', source)
        self.assertIn('event.device_type).endswith("CUDA")', source)

    def test_capture_range_checker(self):
        self.assertTrue(worker._disjoint([{"pointer": 100, "bytes": 20}], [{"pointer": 120, "bytes": 40}]))
        self.assertFalse(worker._disjoint([{"pointer": 100, "bytes": 21}], [{"pointer": 120, "bytes": 40}]))


class MetricTests(unittest.TestCase):
    def compare(self, a, b): return worker.compare(TorchAdapter, Tensor(np.asarray(a, np.float32)), Tensor(np.asarray(b, np.float32)))
    def test_equal_and_empty(self):
        self.assertTrue(self.compare([[0, 1, -2]], [[0, 1, -2]])["pass"])
        self.assertTrue(self.compare(np.empty((0, 3)), np.empty((0, 3)))["pass"])
    def test_material_difference(self):
        self.assertFalse(self.compare([[2., 3.]], [[1., 3.]])["pass"])
    def test_matching_exceptional_classes(self):
        r = self.compare([[np.nan, np.inf, -np.inf, 2]], [[np.nan, np.inf, -np.inf, 2]])
        self.assertTrue(r["pass"]); self.assertTrue(r["exceptional"])
    def test_exceptional_mismatch(self):
        self.assertFalse(self.compare([[np.inf]], [[-np.inf]])["pass"])
        self.assertFalse(self.compare([[np.nan]], [[0.]])["pass"])
    def test_large_finite_norm_does_not_overflow(self):
        r = self.compare([[2e30, 3e30]], [[1e30, 3e30]])
        self.assertFalse(r["pass"])
        self.assertAlmostEqual(r["whole_rel_l2"], 1 / np.sqrt(10), places=6)
    def test_zero_reference_not_silent_pass(self):
        self.assertFalse(self.compare([[1e-5]], [[0.]])["pass"])


if __name__ == "__main__": unittest.main()
