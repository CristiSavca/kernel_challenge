"""Focused coverage-policy/lifecycle tests, including actual isolated packaging. No CUDA."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("coverage_worker", ROOT / "tools/serving_finalizer_coverage_worker.py")
w = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(w)


def metric():
    return {"pass": True, "equal_values": True, "exceptional": False, "max_abs": 0.,
            "whole_rel_l2": 0., "worst_row_rel_l2": 0.,
            "classification": {k: {"output": 0, "reference": 0, "mismatch": 0}
                               for k in ("nan", "positive_inf", "negative_inf")}}


def case_record(descriptor, index):
    return {"original_index": index, "descriptor": copy.deepcopy(descriptor),
            "name": descriptor["name"], "seed": descriptor["seed"],
            "shape": [descriptor[k] for k in ("t", "k", "h")], "pass": True,
            "fallback_exact_required": False,
            "comparisons": {k: metric() for k in ("candidate_vs_original", "candidate_vs_fp32_oracle", "original_vs_fp32_oracle")}}


def complete(payload):
    """Synthetic structurally complete report for root's real-reader transport tests."""
    result = dict(status="coverage_complete", coverage_complete=True, diagnostic_only=True, official=False,
                  generic_validation_status="failed_at_finite_cancellation",
                  candidate_sha256=w.CANDIDATE_SHA, baseline_sha256=w.BASELINE_SHA, reference_sha256=w.REFERENCE_SHA,
                  worker_sha256=payload["file_sha256"]["finalizer_coverage.py"], file_sha256=payload["file_sha256"], image=w.IMAGE,
                  case_manifest_sha256=payload["case_manifest_sha256"], original_case_manifest_sha256=w.ORIGINAL_MANIFEST_SHA,
                  historical_failure=w.historical_failure(), numerical_policy=copy.deepcopy(w.POLICY), planned_indices=w.INDICES,
                  completed_cases=15, input_mutation_checks=15, output_alias_checks=15,
                  remaining_case_failures=[], remaining_numerics_passed=True,
                  cases=[case_record(c, i) for c, i in zip(payload["case_descriptors"], w.INDICES)])
    checks = lambda kind: [{"name": f"concurrent_{i}" if kind == "streams" else f"graph_{i % 2}_replay_{i // 2}",
                            "seed": w.SEED + (70000 if kind == "streams" else 90000) + i,
                            "pass": True, "comparisons": [metric(), metric(), metric()]} for i in range(8)]
    result["two_streams"] = dict(pass_=True, calls=8, streams=2, checks=checks("streams"),
                                 distinct_output_storage=True, actual_execution_overlap_measured=False)
    result["two_streams"]["pass"] = result["two_streams"].pop("pass_")
    result["two_graphs"] = {"pass": True, "graphs": 2, "fresh_replays": 8, "checks": checks("graphs"),
                            "cross_graph_ranges_disjoint": True, "graphB_preserves_graphA_output": True,
                            "fresh_values_changed_outputs": True, "scratch_tensor_references_retained": False,
                            "actual_execution_overlap_measured": False,
                            "capture_allocations": [[{"pointer": start, "bytes": 7154, "shape": [7, 511], "dtype": "torch.bfloat16"},
                                                     {"pointer": start + 8192, "bytes": 84, "shape": [7, 3], "dtype": "torch.float32"}]
                                                    for start in (65536, 131072)]}
    return result


def fail_case(row):
    row["pass"] = False
    row["comparisons"]["candidate_vs_original"].update(
        {"pass": False, "equal_values": False, "max_abs": .001, "whole_rel_l2": .02})


class Tensor:
    def __init__(self, raw=b"abc", pointer=100): self.raw, self.pointer = raw, pointer
    def detach(self): return self
    def clone(self): return Tensor(self.raw, self.pointer + 10000)
    def contiguous(self): return self
    def view(self, dtype): return self
    def untyped_storage(self): return self
    def data_ptr(self): return self.pointer


Torch = types.SimpleNamespace(Tensor=Tensor, uint8="uint8", equal=lambda a, b: a.raw == b.raw)


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.payload = w.build_payload(ROOT)

    def test_exact_selection_thresholds_and_historical_evidence(self):
        p = self.payload; w.validate_payload(p)
        reference = w._reference(p["files"]["validation_reference.py"])
        self.assertEqual(p["case_descriptors"], reference.cases()[75:90])
        self.assertEqual(p["remaining_indices"], list(range(75, 90)))
        self.assertEqual([c["name"] for c in p["case_descriptors"][:4]], ["finite_tiny", "finite_subnormal", "finite_large", "finite_zero"])
        self.assertEqual(reference.TOL, p["numerical_policy"]["whole_relative_l2"])
        self.assertEqual(reference.TOL * reference.ROW_FACTOR, p["numerical_policy"]["worst_row_relative_l2"])
        self.assertEqual(reference.ROW_NORM_FLOOR, p["numerical_policy"]["row_norm_floor"])
        self.assertIs(p["historical_failure"]["historical_not_rerun"], True)
        self.assertNotIn("torch", sys.modules)

    def test_source_policy_and_coverage_mutations(self):
        for name in w.FILES:
            p = copy.deepcopy(self.payload); p["files"][name] += "\n"; p["file_sha256"][name] = w.sha(p["files"][name])
            with self.subTest(name=name), self.assertRaises(ValueError): w.validate_payload(p)
        changes = [("timeout_seconds", 600), ("validation_budget_seconds", 241), ("image", "other"),
                   ("gpu", None), ("seed", 1), ("case_manifest_sha256", "0" * 64),
                   ("remaining_indices", list(range(74, 89))), ("historical_failure", {}),
                   ("numerical_policy", {**w.POLICY, "whole_relative_l2": .1})]
        for key, value in changes:
            p = copy.deepcopy(self.payload); p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): w.validate_payload(p)
        p = copy.deepcopy(self.payload); p["case_descriptors"][0]["seed"] += 1
        with self.assertRaises(ValueError): w.validate_payload(p)

    def test_real_isolated_four_file_package(self):
        with tempfile.TemporaryDirectory(prefix="okc-coverage-package-") as folder:
            p = Path(folder)
            for name, source in self.payload["files"].items(): (p / name).write_text(source)
            (p / "payload.json").write_text(json.dumps(self.payload))
            code = """import importlib.util,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('coverage',p/'finalizer_coverage.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.validate_payload(json.loads((p/'payload.json').read_text()))
r=m._load(p/'validation_reference.py','reference')
assert [c['name'] for c in r.cases()[75:]]==[c['name'] for c in json.loads((p/'payload.json').read_text())['case_descriptors']]
assert 'torch' not in sys.modules and 'triton' not in sys.modules
assert not any('HASIDIC_PROJECTS' in x for x in sys.path)
print('isolated-four-file-PASS')
"""
            env = dict(os.environ); env.pop("PYTHONPATH", None)
            cmd = [sys.executable, "-I", "-S", "-c", code, str(p)]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15, env=env)
            self.assertEqual(r.stdout.strip(), "isolated-four-file-PASS")
            (p / "validation_reference.py").rename(p / "wrong.py")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15, env=env)
            self.assertNotEqual(r.returncode, 0); self.assertIn("FileNotFoundError", r.stderr)

    def test_completion_is_not_generic_acceptance(self):
        r = complete(self.payload); w.validate_result(r, self.payload)
        fail_case(r["cases"][0]); r["remaining_numerics_passed"] = False
        r["remaining_case_failures"] = [r["cases"][0]["name"]]
        w.validate_result(r, self.payload)  # complete measured coverage can expose another failure
        self.assertEqual(r["generic_validation_status"], "failed_at_finite_cancellation")
        r["status"] = "pass"
        with self.assertRaises(ValueError): w.validate_result(r, self.payload)

    def test_missing_or_concealed_cases_rejected(self):
        for change in ("missing", "false_flag", "missing_comparison", "bad_seed", "concealed_failure", "policy"):
            r = complete(self.payload)
            if change == "missing": r["cases"].pop(); r["completed_cases"] = 14; r["remaining_numerics_passed"] = None
            elif change == "false_flag": r["cases"][0]["pass"] = False
            elif change == "missing_comparison": r["cases"][0]["comparisons"].pop("original_vs_fp32_oracle")
            elif change == "bad_seed": r["cases"][0]["seed"] += 1
            elif change == "concealed_failure": fail_case(r["cases"][0])
            else: r["numerical_policy"]["row_norm_floor"] = .2
            with self.subTest(change=change), self.assertRaises(ValueError): w.validate_result(r, self.payload)

    def test_partial_checkpoint_retains_failures(self):
        r = complete(self.payload); r.update(status="failed", coverage_complete=False)
        r["cases"] = r["cases"][:2]; r["completed_cases"] = 2; r["remaining_numerics_passed"] = None
        fail_case(r["cases"][0]); r["remaining_case_failures"] = [r["cases"][0]["name"]]
        r.pop("two_streams"); r.pop("two_graphs"); w.validate_result(r, self.payload)
        r["status"] = "coverage_complete"; r["coverage_complete"] = True
        with self.assertRaises(ValueError): w.validate_result(r, self.payload)

    def test_stream_graph_count_identity_comparison_and_ranges(self):
        mutations = [lambda r:r["two_streams"].update(calls=7),
                     lambda r:r["two_streams"]["checks"][0].update(seed=0),
                     lambda r:r["two_graphs"].update(fresh_replays=7),
                     lambda r:r["two_graphs"].update(scratch_tensor_references_retained=True),
                     lambda r:r["two_graphs"].update(graphB_preserves_graphA_output=False),
                     lambda r:r["two_graphs"]["capture_allocations"][1][0].update(pointer=65536),
                     lambda r:r["two_graphs"]["checks"][0]["comparisons"][0].update({"pass":False}),
                     lambda r:r["two_streams"].update(actual_execution_overlap_measured=True)]
        for change in mutations:
            r = complete(self.payload); change(r)
            with self.assertRaises(ValueError): w.validate_result(r, self.payload)

    def test_numerical_and_exceptional_metric_decisions(self):
        r = metric(); self.assertTrue(w._metric_pass(r))
        r.update({"pass":False, "whole_rel_l2":.02, "equal_values":False}); self.assertFalse(w._metric_pass(r))
        r["pass"] = True
        with self.assertRaises(ValueError): w._metric_pass(r)
        r = metric(); r["exceptional"] = True; r["classification"]["nan"].update(output=2, reference=2)
        self.assertTrue(w._metric_pass(r))
        r["classification"]["nan"]["mismatch"] = 1; r.update({"pass":False, "reason":"nonfinite classification mismatch"})
        self.assertFalse(w._metric_pass(r))
        r["pass"] = True
        with self.assertRaises(ValueError): w._metric_pass(r)
        r = metric(); r["whole_rel_l2"] = float("nan")
        with self.assertRaises(ValueError): w._metric_pass(r)
        r = metric(); r["pass"] = False; r.pop("max_abs")
        with self.assertRaises(ValueError): w._metric_pass(r)

    def test_input_mutation_and_output_alias_abort(self):
        args = {"x":Tensor()}; saved = w._snapshots(Torch, args)
        w._assert_input_ownership(Torch, args, saved, Tensor(pointer=500))
        with self.assertRaisesRegex(RuntimeError, "aliases"): w._assert_input_ownership(Torch, args, saved, Tensor(pointer=100))
        args["x"].raw = b"changed"
        with self.assertRaisesRegex(RuntimeError, "mutation"): w._assert_input_ownership(Torch, args, saved, Tensor(pointer=500))

    def test_case_loop_continues_numeric_failure_and_stops_runtime(self):
        selected = self.payload["case_descriptors"]
        def evaluate(torch, new, old, args, descriptor, actual):
            i = 75 + next(i for i, c in enumerate(selected) if c["name"] == descriptor["name"])
            row = case_record(descriptor, i)
            if i == 75: fail_case(row)
            return row
        ref = types.SimpleNamespace(make_inputs=lambda torch,c:{"x":Tensor()}, _evaluate=evaluate)
        report = dict(cases=[],completed_cases=0,input_mutation_checks=0,output_alias_checks=0,
                      remaining_case_failures=[],remaining_numerics_passed=None)
        w._run_cases(Torch, ref, lambda **args:Tensor(pointer=500), None, selected, report, lambda stage:None)
        self.assertEqual(report["completed_cases"], 15); self.assertIs(report["remaining_numerics_passed"], False)
        self.assertEqual(report["remaining_case_failures"], ["finite_tiny"])
        report["cases"] = []; report["completed_cases"] = 0
        with self.assertRaisesRegex(RuntimeError, "aliases"):
            w._run_cases(Torch, ref, lambda **args:Tensor(pointer=100), None, selected, report, lambda stage:None)
        self.assertEqual(report["cases"], [])

    def test_bounded_checkpoint_and_preimport_failure(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(w, "RESULT", Path(folder)/"result.json"):
            w._save({"status":"running"}); before=w.RESULT.read_bytes()
            with self.assertRaises(RuntimeError): w._save({"bulk":"x" * w.MAX_RESULT_BYTES})
            self.assertEqual(w.RESULT.read_bytes(), before)
            p = copy.deepcopy(self.payload); p["timeout_seconds"] = 1
            with self.assertRaises(ValueError), mock.patch("builtins.print"): w.run(p)
            self.assertEqual(json.loads(w.RESULT.read_text())["status"], "failed")
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__": unittest.main()
