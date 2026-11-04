"""Decision fixtures only. Synthetic timings are never GPU evidence."""
import copy
import unittest

import analyze_native_serving_v25 as gate
from test_analyze_wgs_shared_a_result import incumbent, rescore


def fixture(speed=1.2):
    result = copy.deepcopy(incumbent())
    for row in result["records"]:
        if row["event"] == "benchmark":
            row["speedup"] = speed
            row["median_ns"] = row["baseline_ns"] / speed
        elif row["event"] == "profile":
            row["report"] = ("synthetic native_serving_v25 activation fixture " +
                             ("where_kernel_impl MeanOps sum_functor" if row["case"]["bs"] < 32
                              else "_post_expert_norm_weighted_sum_kernel"))
    return rescore(result)


class ComponentGateTests(unittest.TestCase):
    def test_same_run_component_pass_does_not_require_native_speed(self):
        report = gate.screen(fixture(), "native_serving_v25")
        self.assertTrue(report["fixed_component_screen_passed"])
        self.assertLess(report["score"], incumbent()["speedup"])

    def test_below_fixed_gain_closes(self):
        self.assertFalse(gate.screen(fixture(1.149), "native_serving_v25")["fixed_component_screen_passed"])

    def test_one_case_regression_fails_despite_high_aggregate(self):
        result = fixture(2.)
        row = next(r for r in result["records"] if r["event"] == "benchmark")
        row["speedup"] = 1 / 1.021
        row["median_ns"] = row["baseline_ns"] / row["speedup"]
        report = gate.screen(rescore(result), "native_serving_v25")
        self.assertTrue(report["gates"]["all_six_GM_speed_at_least_1p15"])
        self.assertFalse(report["fixed_component_screen_passed"])

    def test_missing_one_activation_fails(self):
        result = fixture()
        next(r for r in result["records"] if r["event"] == "profile")["report"] = "stock only"
        self.assertFalse(gate.screen(result, "native_serving_v25")["fixed_component_screen_passed"])

    def test_core_only_is_not_complete_chain(self):
        for m in (1, 256):
            result = fixture()
            next(r for r in result["records"] if r["event"] == "profile" and r["case"]["bs"] == m)["report"] = "native_serving_v25"
            report = gate.screen(result, "native_serving_v25")
            self.assertTrue(report["gates"]["all_six_intended_core_profiles"])
            self.assertFalse(report["gates"]["all_six_stock_finalizer_profiles"])
            self.assertFalse(report["fixed_component_screen_passed"])

    def test_flags_missing_cases_and_wrong_gpu_fail_closed(self):
        for mode in ("flags", "coverage", "gpu"):
            result = fixture()
            if mode == "flags": result["flags"] = ["unsafe"]
            elif mode == "coverage": result["records"].pop()
            else: result["gpu"] = "H100"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                gate.screen(result, "native_serving_v25")


if __name__ == "__main__":
    unittest.main()
