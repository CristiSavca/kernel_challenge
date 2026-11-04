"""Focused offline negative/positive fixtures for the fixed V23 reader."""
import copy
import math
import unittest
from unittest.mock import patch

import analyze_attention_exact_rescale_v23 as audit


class V23ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = audit.read_run(audit.PARENTS[0][0], audit.PARENT, "run")["result"]

    def candidate(self, affected=1.08, controls=1.0):
        result = copy.deepcopy(self.parent)
        benches = sorted((r for r in result["records"] if r["event"] == "benchmark"), key=lambda r:r["index"])
        for row in benches:
            factor = affected if row["index"] in audit.AFFECTED else controls
            row["speedup"] *= factor
            row["median_ns"] /= factor
            row["median_raw_ns"] /= factor
        loads = []
        for i in range(0,12,2):
            pair=benches[i:i+2]
            loads.append(sum(r["case"]["weight"]*r["baseline_ns"] for r in pair) /
                         sum(r["case"]["weight"]*r["median_ns"] for r in pair))
        result["speedup"] = math.prod(loads)**(1/6)
        result["speedup_ci"] = [result["speedup"]*.95,result["speedup"]*1.05]
        return result

    def test_parent_contract_counts_and_specializations(self):
        audit.pass_gate(self.parent)
        self.assertTrue(audit.activation(self.parent)["pass"])
        self.assertEqual(audit.COUNTS,{"test":16,"benchmark":12,"profile":12})

    def test_synthetic_useful_gain_passes_fixed_comparison(self):
        result=audit.compare(self.candidate(),self.parent)
        self.assertTrue(all(result["gates"].values()))
        self.assertAlmostEqual(result["affected_speedup_ratio_GM"],1.08)
        self.assertAlmostEqual(result["control_normalized_latency_ratio_GM"],1)

    def test_unchanged_parent_cannot_pass(self):
        result=audit.compare(self.parent,self.parent)
        self.assertFalse(result["gates"]["six_load_score_ratio_at_least_1p01"])
        self.assertFalse(result["gates"]["affected_five_case_speed_GM_at_least_1p03"])

    def test_adverse_control_is_not_hidden_by_affected_gain(self):
        result=audit.compare(self.candidate(controls=1/1.03),self.parent)
        self.assertFalse(result["gates"]["seven_control_latency_GM_0p98_to_1p02"])

    def test_bad_score_evaluator_shape_and_coverage_reject(self):
        mutations=[lambda r:r.__setitem__("speedup",r["speedup"]*1.01),
                   lambda r:r["evaluator"].__setitem__("image","wrong"),
                   lambda r:r["records"][0]["case"].__setitem__("ctx",999),
                   lambda r:r["records"].pop()]
        for mutate in mutations:
            value=self.candidate();mutate(value)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                audit.compare(value,self.parent)

    def test_wrong_selected_profile_cannot_pass_activation(self):
        value=self.candidate()
        row=next(r for r in value["records"] if r["event"]=="profile" and r["index"]==1)
        row["report"]=row["report"].replace("<1, false, 11>","<1, false, 13>")
        self.assertFalse(audit.activation(value)["pass"])

    def test_pending_cannot_produce_metrics_or_pass(self):
        with patch.object(audit,"read_run",return_value={"status":"PENDING","saved_status":"running","provenance":{"fixture":True}}) as reader:
            result=audit.analyze()
        self.assertEqual(reader.call_count,1)
        self.assertEqual(result["status"],"PENDING")
        self.assertFalse(result["fixed_gate_passed"])
        self.assertEqual(result["comparisons"],[])
        self.assertNotIn("score",result)

    def test_source_bytes_mismatch_rejects_before_metrics(self):
        original=audit.bounded_file
        def read(path):
            resolved,data=original(path)
            return resolved,(data+b"\n" if str(resolved).endswith(f"/{audit.PARENT}.py") else data)
        with patch.object(audit,"bounded_file",side_effect=read),self.assertRaisesRegex(ValueError,"source bytes mismatch"):
            audit.read_run(audit.PARENTS[0][0],audit.PARENT,"run")


if __name__=="__main__": unittest.main()
