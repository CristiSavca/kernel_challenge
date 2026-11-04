"""Use the actual abbreviated official profile format; synthetic scores are parser fixtures only."""
import copy
import json
import unittest

import analyze_native_serving_v26 as gate
from test_analyze_wgs_shared_a_result import rescore


def v25_result():
    return json.loads((gate.ROOT / f"experiments/runs/{gate.V25_JOB}.json").read_text())["response"]["result"]


def fixture(speed=None):
    result = copy.deepcopy(v25_result())
    for row in result["records"]:
        if row["event"] == "profile":
            row["report"] = row["report"].replace("native_serving_v25", "native_serving_v26")
        elif row["event"] == "benchmark" and speed is not None:
            row["speedup"] = speed
            row["median_ns"] = row["baseline_ns"] / speed
    return rescore(result) if speed is not None else result


class SourceAndProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (gate.ROOT / f"experiments/candidates/muse_moe_w4a16/{gate.V25_SHA}.py").read_bytes()
        cls.baseline = (gate.ROOT / "muse_moe_w4a16/baseline.py").read_bytes()

    def test_actual_abbreviated_profiles_have_observable_chain_evidence(self):
        profiles = [r for r in v25_result()["records"] if r["event"] == "profile"]
        self.assertEqual(len(profiles), 6)
        for p in profiles:
            if p["case"]["bs"] < 32:
                self.assertNotIn("MeanOps", p["report"])
                self.assertIn("...", p["report"])
            e = gate.profile_evidence(p["report"], p["case"]["bs"], "native_serving_v25", True)
            self.assertTrue(e["intended_core_in_profile"])
            self.assertTrue(e["stock_finalizer_chain_established"])

    def test_visible_families_do_not_replace_source_binding(self):
        p = next(r for r in v25_result()["records"] if r["event"] == "profile")
        e = gate.profile_evidence(p["report"], 1, "native_serving_v25", False)
        self.assertTrue(e["stock_finalizer_observable_evidence"])
        self.assertFalse(e["stock_finalizer_chain_established"])

    def test_core_only_never_establishes_finalization(self):
        for m in (1, 32):
            e = gate.profile_evidence("native_serving_v26", m, "native_serving_v26", True)
            self.assertTrue(e["intended_core_in_profile"])
            self.assertFalse(e["stock_finalizer_chain_established"])

    def test_exact_common_import_and_reviewed_chain(self):
        result = gate.source_chain(self.source, self.source, self.baseline)
        self.assertEqual(set(result["function_AST_sha256"]), set(gate.CHAIN_FUNCTIONS))

    def test_changed_import_argument_or_binding_fails_closed(self):
        mutations = [
            self.source.replace(gate.FINALIZER_MODULE.encode(), b"unreviewed.finalizer"),
            self.source.replace(b"norm_weight=state['norm_w'],eps=1e-8,", b"norm_weight=state['norm_w'],eps=2e-8,"),
            self.source + b"\nfused_post_expert_norm_weighted_sum = None\n",
        ]
        for mutated in mutations:
            self.assertNotEqual(mutated, self.source)
            with self.assertRaises(ValueError):
                gate.source_chain(mutated, self.source, self.baseline)

    def test_better_profile_reader_does_not_rescue_V25_speed_failure(self):
        report = gate.screen(fixture(), "native_serving_v26", True)
        self.assertTrue(report["gates"]["all_six_stock_finalizer_chains"])
        self.assertFalse(report["gates"]["every_case_normalized_latency_at_most_1p02"])
        self.assertFalse(report["fixed_component_screen_passed"])
        self.assertAlmostEqual(report["score"], 1.2537949061679123)

    def test_fixed_threshold_and_complete_chain_needed_for_synthetic_pass(self):
        self.assertTrue(gate.screen(fixture(1.20), "native_serving_v26", True)["fixed_component_screen_passed"])
        self.assertFalse(gate.screen(fixture(1.149), "native_serving_v26", True)["fixed_component_screen_passed"])
        self.assertFalse(gate.screen(fixture(1.20), "native_serving_v26", False)["fixed_component_screen_passed"])

    def test_flags_and_missing_profile_fail_closed(self):
        for mode in ("flags", "profile"):
            result = fixture(1.20)
            if mode == "flags": result["flags"] = ["unsafe"]
            else: result["records"].remove(next(r for r in result["records"] if r["event"] == "profile"))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                gate.screen(result, "native_serving_v26", True)


if __name__ == "__main__":
    unittest.main()
