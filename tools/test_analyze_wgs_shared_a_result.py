"""Offline provenance and performance fixtures; synthetic gains are not GPU evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_wgs_shared_a_result as gate


def incumbent(job=gate.PARENTS[0][0]):
    return json.loads((gate.ROOT / f"experiments/runs/{job}.json").read_text())["response"]["result"]


def rescore(result):
    result["speedup"] = math.prod(r["speedup"] for r in result["records"]
                                 if r["event"] == "benchmark") ** (1 / 6)
    result["speedup_ci"] = [result["speedup"] * .985, result["speedup"] * 1.015]
    return result


def changed(affected=1.04, control=1., exceptions=None):
    result = copy.deepcopy(incumbent())
    for row in result["records"]:
        if row["event"] != "benchmark":
            continue
        factor = (exceptions or {}).get(row["index"], affected if row["case"]["bs"] > 16 else control)
        row["median_ns"] /= factor
        row["median_raw_ns"] /= factor
        row["speedup"] *= factor
    return rescore(result)


def fixture(root, result=None, status="done"):
    job = "1" * 16
    source = root / "candidate.py"
    source.write_bytes(b"# synthetic parser fixture; no GPU implementation\n")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    response = {"id": job, "board": gate.PROBLEM, "mode": "check", "status": status, "result": result}
    if result is not None:
        profile = root / "profile.json"
        profile.write_text(json.dumps({"job_id": job, "problem": gate.PROBLEM,
            "submission": str(source),
            "benchmarks": [r for r in result["records"] if r["event"] == "benchmark"],
            "profiles": [r for r in result["records"] if r["event"] == "profile"]}))
        response["profile"] = {"json": str(profile)}
    receipt = {"job": job, "problem": gate.PROBLEM, "source_sha256": sha,
               "source": "candidate.py", "response": response}
    out = root / f"experiments/runs/{job}.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(receipt))
    return job, sha, out


class SharedAPerformanceTests(unittest.TestCase):
    def test_both_real_incumbents_load_and_reproduce(self):
        for job, mode in gate.PARENTS:
            result = gate.read_run(job, gate.PARENT_SHA, mode)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(gate.benchmark_rows(result["result"])), 6)

    def test_identical_run_fails_improvement(self):
        result = gate.compare(incumbent(), incumbent())
        self.assertFalse(result["performance_gate_passed"])
        self.assertEqual(result["aggregate_speedup_ratio"], 1.)

    def test_affected_gain_with_controls_is_half_the_log_aggregate(self):
        result = gate.compare(changed(), incumbent())
        self.assertAlmostEqual(result["affected_speedup_ratio_GM"], 1.04)
        self.assertAlmostEqual(result["aggregate_speedup_ratio"], 1.04 ** .5)
        self.assertEqual(result["control_normalized_latency_ratio_GM"], 1.)
        self.assertTrue(result["performance_gate_passed"])
        self.assertTrue(result["reported_CIs_overlap"])

    def test_one_affected_regression_fails_despite_large_mean_gain(self):
        result = gate.compare(changed(affected=1.20, exceptions={2: 1 / 1.04}), incumbent())
        self.assertTrue(result["gates"]["affected_GM_at_least_3pct_faster"])
        self.assertFalse(result["gates"]["every_affected_case_at_most_3pct_slower"])
        self.assertFalse(result["performance_gate_passed"])

    def test_control_speedups_and_regressions_both_trigger_drift_gate(self):
        for factor in (.95, 1.05):
            result = gate.compare(changed(affected=1.2, control=factor), incumbent())
            self.assertFalse(result["gates"]["control_GM_within_3pct"])
            self.assertFalse(result["performance_gate_passed"])

    def test_three_percent_means_speed_ratio_not_three_percent_latency_cut(self):
        self.assertFalse(gate.compare(changed(affected=1.029), incumbent())["performance_gate_passed"])
        self.assertTrue(gate.compare(changed(affected=1.031), incumbent())["performance_gate_passed"])

    def test_both_incumbents_must_pass_and_no_automatic_promotion(self):
        source = "f" * 64
        candidate = {"status": "PASS", "provenance": {"job": "1" * 16}, "result": changed()}
        public = {"status": "PASS", "provenance": {"job": gate.PARENTS[0][0]}, "result": incumbent()}
        private = {"status": "PASS", "provenance": {"job": gate.PARENTS[1][0]}, "result": changed(affected=1.02)}
        with patch.object(gate, "read_run", side_effect=[candidate, public, private]) as reader:
            result = gate.analyze("1" * 16, source)
        self.assertEqual(reader.call_count, 3)
        self.assertTrue(result["comparisons"][0]["performance_gate_passed"])
        self.assertFalse(result["comparisons"][1]["performance_gate_passed"])
        self.assertFalse(result["performance_gate_passed"])
        self.assertFalse(result["automatic_promotion_allowed"])
        self.assertIn("actual_launch_admission", result["unresolved_gates"])

    def test_local_bytes_and_serialization_diagnostics_do_not_kill_performance(self):
        result = changed()
        result["log_tail"] = "C7511 insufficient registers; SSAR local=128 occ=1\n"
        self.assertTrue(gate.compare(result, incumbent())["performance_gate_passed"])

    def test_case_index_and_field_identity_fail_closed(self):
        for change in ("index", "bs", "weight", "extra"):
            result = changed()
            row = next(r for r in result["records"] if r["event"] == "benchmark")
            if change == "index": row["index"] = 1
            elif change == "extra": row["case"]["new_contract"] = 1
            else: row["case"][change] += 1
            with self.assertRaises(ValueError): gate.compare(result, incumbent())

    def test_complete_evaluator_gpu_and_seed_only_rule(self):
        for change in ("code", "image", "gpu", "file"):
            result = changed()
            if change == "gpu": result["gpu"] = "H100"
            elif change == "code": result["evaluator"]["code_sha256"] = "0" * 64
            elif change == "image": result["evaluator"]["image"] += "other"
            else: result["evaluator"]["files"]["extra"] = "0" * 64
            with self.assertRaises(ValueError): gate.compare(result, incumbent())
        result = changed()
        for row in result["records"]:
            if row["event"] == "benchmark": row["case"]["seed"] += 1
        self.assertTrue(gate.compare(result, incumbent())["performance_gate_passed"])

    def test_absent_machine_does_not_imply_same_node(self):
        a, b = changed(), incumbent()
        a.pop("machine"); b.pop("machine")
        self.assertIsNone(gate.compare(a, b)["same_reported_node"])

    def test_corrupt_scores_and_nonfinite_fields_fail(self):
        for change in ("score", "ratio", "nan", "bool"):
            result = changed()
            row = next(r for r in result["records"] if r["event"] == "benchmark")
            if change == "score": result["speedup"] *= 1.01
            elif change == "ratio": row["speedup"] *= 1.01
            else: row["median_ns"] = float("nan") if change == "nan" else True
            with self.assertRaises(ValueError): gate.compare(result, incumbent())

    def test_explicit_pass_coverage_and_flags(self):
        for change in ("status", "coverage", "flags", "checks"):
            result = changed()
            row = next(r for r in result["records"] if r["event"] == "test")
            if change == "status": row.pop("status")
            elif change == "coverage": result["records"].remove(row)
            elif change == "flags": row["flags"] = ["flag"]
            else: result["checks"]["ok"] = False
            with self.assertRaises(ValueError): gate.pass_gate(result)

    def test_minimal_received_and_source_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job, sha, path = fixture(root, status="received")
            receipt = json.loads(path.read_text())
            receipt["response"] = {"job_id": job, "status": "received"}
            path.write_text(json.dumps(receipt))
            result = gate.analyze(job, sha, root)
            self.assertEqual(result["status"], "PENDING")
            self.assertFalse(result["server_identity_confirmed"])
            self.assertEqual(result["comparisons"], [])
            self.assertNotIn("result", result)
            (root / "candidate.py").write_text("modified")
            with self.assertRaisesRegex(ValueError, "source bytes mismatch"):
                gate.read_run(job, sha, "check", root)

    def test_received_conflicts_and_incomplete_later_states_rejected(self):
        for response in ({"status": "received", "id": "2" * 16},
                         {"status": "received", "board": "wrong"},
                         {"status": "received", "mode": "run"},
                         {"status": "queued"}, {"status": "running"}, {"status": "done"}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); job, sha, path = fixture(root, status="received")
                receipt = json.loads(path.read_text())
                receipt["response"] = {"job_id": job, **response}
                path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError): gate.read_run(job, sha, "check", root)

    def test_failure_has_no_performance_and_unknown_status_rejected(self):
        for status in ("done", "infra_error", "timeout"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); result = changed(); result.update(passed=False, returncode=1)
                job, sha, _ = fixture(root, result, status)
                report = gate.analyze(job, sha, root)
                self.assertEqual(report["status"], "OFFICIAL_GATE_FAILED")
                self.assertNotIn("result", report)
                self.assertEqual(report["comparisons"], [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job, sha, _ = fixture(root, status="unknown")
            with self.assertRaisesRegex(ValueError, "unrecognized job status"):
                gate.read_run(job, sha, "check", root)

    def test_profile_empty_flags_omission_is_only_allowed_record_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job, sha, _ = fixture(root, changed())
            path = root / "profile.json"; profile = json.loads(path.read_text())
            for row in profile["benchmarks"]: row.pop("flags", None)
            path.write_text(json.dumps(profile))
            self.assertEqual(gate.read_run(job, sha, "check", root)["status"], "PASS")
            profile["benchmarks"][0]["median_ns"] += 1
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "records disagree"):
                gate.read_run(job, sha, "check", root)

    def test_profile_case_corruption_even_in_both_archives_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); result = changed()
            next(r for r in result["records"] if r["event"] == "profile")["case"]["seed"] += 1
            job, sha, _ = fixture(root, result)
            with self.assertRaisesRegex(ValueError, "profile/benchmark case mismatch"):
                gate.read_run(job, sha, "check", root)

    def test_nonempty_or_null_external_flags_are_not_normalized_away(self):
        for flags in (["synthetic flag"], None):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); job, sha, _ = fixture(root, changed())
                path = root / "profile.json"; profile = json.loads(path.read_text())
                profile["benchmarks"][0]["flags"] = flags
                path.write_text(json.dumps(profile))
                with self.assertRaisesRegex(ValueError, "records disagree"):
                    gate.read_run(job, sha, "check", root)

    def test_profile_identity_and_mode_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); job, sha, _ = fixture(root, changed())
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                gate.read_run(job, sha, "run", root)
            path = root / "profile.json"; profile = json.loads(path.read_text()); profile["job_id"] = "2" * 16
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "profile identity mismatch"):
                gate.read_run(job, sha, "check", root)


if __name__ == "__main__":
    unittest.main()
