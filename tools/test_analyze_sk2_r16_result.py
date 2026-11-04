"""Meaningful scoring/provenance fixtures. Synthetic changes are never GPU evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from analyze_sk2_r16_result import ROOT, compare, load_scores, pass_gate, read_run


def incumbent(job="7a8c4941741645ee"):
    return json.loads((ROOT / f"experiments/runs/{job}.json").read_text())["response"]["result"]


def rescore(result):
    groups = {}
    for row in result["records"]:
        if row["event"] == "benchmark":
            base, elapsed = groups.setdefault(row["case"]["load"], [0., 0.])
            weight = row["case"]["weight"]
            groups[row["case"]["load"]] = [base + weight * row["baseline_ns"],
                                          elapsed + weight * row["median_ns"]]
    result["speedup"] = math.prod(b / t for b, t in groups.values()) ** (1 / 6)
    result["speedup_ci"] = [result["speedup"] * .985, result["speedup"] * 1.015]
    return result


def changed(affected=1.04, control=1., exceptions=None):
    result = copy.deepcopy(incumbent())
    for row in result["records"]:
        if row["event"] != "benchmark":
            continue
        case = row["case"]
        target = case["batch"] * case.get("q_len", 1) <= 32
        factor = (exceptions or {}).get(row["index"], affected if target else control)
        row["median_ns"] /= factor
        row["median_raw_ns"] /= factor
        row["speedup"] *= factor
    return rescore(result)


def fixture(root, result=None, status="done"):
    job = "1" * 16
    source = root / "candidate.py"
    source.write_bytes(b"# synthetic parser fixture; no GPU implementation\n")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    response = {"id": job, "board": "muse_decode_layer", "mode": "check", "status": status,
                "result": result}
    if result is not None:
        profile = root / "profile.json"
        profile.write_text(json.dumps({"job_id": job, "problem": "muse_decode_layer",
            "submission": str(source),
            "benchmarks": [r for r in result["records"] if r["event"] == "benchmark"],
            "profiles": [r for r in result["records"] if r["event"] == "profile"]}))
        response["profile"] = {"json": str(profile)}
    receipt = {"job": job, "problem": "muse_decode_layer", "source_sha256": sha,
               "source": "candidate.py", "response": response}
    out = root / f"experiments/runs/{job}.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(receipt))
    return job, sha, out


class PerformanceTests(unittest.TestCase):
    def test_real_incumbent_scores_reproduce(self):
        for job in ("7a8c4941741645ee", "b2a773e2d7194de8"):
            benches, loads = load_scores(incumbent(job))
            self.assertEqual((len(benches), len(loads)), (12, 6))

    def test_identical_run_cannot_improve(self):
        report = compare(incumbent(), incumbent())
        self.assertEqual(report["aggregate_speedup_ratio"], 1)
        self.assertFalse(report["performance_gate_passed"])

    def test_four_load_gain_with_unchanged_controls(self):
        report = compare(changed(), incumbent())
        self.assertAlmostEqual(report["aggregate_speedup_ratio"], 1.04 ** (4 / 6))
        self.assertAlmostEqual(report["affected_four_load_speedup_ratio_GM"], 1.04)
        self.assertEqual(report["control_two_load_normalized_latency_ratio_GM"], 1)
        self.assertTrue(report["performance_gate_passed"])
        self.assertTrue(report["reported_CIs_overlap"])

    def test_case_mean_is_not_weighted_load_score(self):
        base = copy.deepcopy(incumbent())
        for row in base["records"]:
            if row["event"] == "benchmark":
                row.update(baseline_ns=100., median_ns=100., median_raw_ns=104., speedup=1.)
        rescore(base)
        candidate = copy.deepcopy(base)
        for row in candidate["records"]:
            if row["event"] == "benchmark":
                factor = 2 if row["case"]["layer"] == "sliding" else .5
                row["median_ns"] /= factor
                row["speedup"] *= factor
        rescore(candidate)
        report = compare(candidate, base)
        self.assertAlmostEqual(report["aggregate_speedup_ratio"], 62 / 55)
        self.assertFalse(report["gates"]["every_affected_case_at_most_3pct_slower"])

    def test_wrong_weight_or_spoofed_score_rejected(self):
        for kind in ("weight", "score"):
            candidate = changed()
            if kind == "weight":
                next(r for r in candidate["records"] if r["event"] == "benchmark")["case"]["weight"] = 1
            else:
                candidate["speedup"] *= 1.01
            with self.assertRaises(ValueError):
                compare(candidate, incumbent())

    def test_control_drift_fails_despite_aggregate_gain(self):
        report = compare(changed(affected=1.2, control=.95), incumbent())
        self.assertTrue(report["gates"]["aggregate_at_least_2pct"])
        self.assertFalse(report["gates"]["two_control_load_drift_within_3pct"])
        self.assertFalse(report["performance_gate_passed"])

    def test_one_bad_affected_case_fails(self):
        report = compare(changed(affected=1.2, exceptions={0: 1 / 1.04}), incumbent())
        self.assertTrue(report["gates"]["aggregate_at_least_2pct"])
        self.assertFalse(report["gates"]["every_affected_case_at_most_3pct_slower"])

    def test_shape_and_evaluator_changes_rejected(self):
        for kind in ("shape", "evaluator"):
            candidate = changed()
            if kind == "shape":
                next(r for r in candidate["records"] if r["event"] == "benchmark")["case"]["ctx"] += 1
            else:
                candidate["evaluator"]["code_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                compare(candidate, incumbent())

    def test_fresh_seeds_are_allowed(self):
        candidate = changed()
        for row in candidate["records"]:
            if row["event"] == "benchmark":
                row["case"]["seed"] += 1
        self.assertTrue(compare(candidate, incumbent())["performance_gate_passed"])

    def test_missing_event_status_and_flags_rejected(self):
        for kind in ("status", "flags", "missing_case"):
            candidate = changed()
            row = next(r for r in candidate["records"] if r["event"] == "test")
            if kind == "status":
                del row["status"]
            elif kind == "flags":
                candidate["flags"] = ["fixture"]
            else:
                candidate["records"].remove(row)
            with self.assertRaises(ValueError):
                pass_gate(candidate)

    def test_nonfinite_times_rejected(self):
        candidate = changed()
        next(r for r in candidate["records"] if r["event"] == "benchmark")["median_ns"] = float("nan")
        with self.assertRaises(ValueError):
            compare(candidate, incumbent())

    def test_pending_receipt_has_no_result_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, sha, _ = fixture(root, status="running")
            report = read_run(job, sha, "check", root)
            self.assertEqual(report["status"], "PENDING")
            self.assertNotIn("result", report)

    def test_actual_minimal_received_acknowledgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, sha, path = fixture(root, status="received")
            receipt = json.loads(path.read_text())
            receipt["response"] = {"job_id": job, "status": "received"}
            path.write_text(json.dumps(receipt))
            report = read_run(job, sha, "check", root)
            self.assertEqual(report["status"], "PENDING")
            self.assertFalse(report["server_identity_confirmed"])
            self.assertNotIn("result", report)
            self.assertNotIn("speedup", report)
            (root / "candidate.py").write_text("modified")
            with self.assertRaisesRegex(ValueError, "frozen source bytes mismatch"):
                read_run(job, sha, "check", root)

    def test_received_ack_cannot_hide_conflicting_identity(self):
        for extra, message in (({"id": "2" * 16}, "receipt job mismatch"),
                               ({"board": "muse_attention_decode"}, "board mismatch"),
                               ({"mode": "run"}, "mode mismatch")):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job, sha, path = fixture(root, status="received")
                receipt = json.loads(path.read_text())
                receipt["response"] = {"job_id": job, "status": "received", **extra}
                path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, message):
                    read_run(job, sha, "check", root)

    def test_other_states_require_full_server_identity(self):
        for status in ("queued", "running", "done", "infra_error"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job, sha, path = fixture(root, status=status)
                receipt = json.loads(path.read_text())
                receipt["response"] = {"job_id": job, "status": status}
                path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "board mismatch"):
                    read_run(job, sha, "check", root)

    def test_infrastructure_failure_is_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, sha, _ = fixture(root, status="infra_error")
            self.assertEqual(read_run(job, sha, "check", root)["status"], "OFFICIAL_GATE_FAILED")

    def test_unknown_status_is_not_assumed_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, sha, _ = fixture(root, status="unrecognized_state")
            with self.assertRaisesRegex(ValueError, "unrecognized job status"):
                read_run(job, sha, "check", root)

    def test_source_and_mode_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, sha, _ = fixture(root, changed())
            with self.assertRaisesRegex(ValueError, "mode mismatch"):
                read_run(job, sha, "run", root)
            (root / "candidate.py").write_text("modified")
            with self.assertRaisesRegex(ValueError, "frozen source bytes mismatch"):
                read_run(job, sha, "check", root)

    def test_profile_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, sha, _ = fixture(root, changed())
            p = root / "profile.json"
            data = json.loads(p.read_text())
            data["benchmarks"][0]["median_ns"] += 1
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "profile and receipt records disagree"):
                read_run(job, sha, "check", root)

    def test_complete_fixture_loads_without_path_alias_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, sha, _ = fixture(root, changed())
            self.assertEqual(read_run(job, sha, "check", root)["status"], "PASS")

    def test_failed_result_cannot_reach_performance_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = changed()
            result.update(passed=False, returncode=1, error="synthetic failure")
            job, sha, _ = fixture(root, result)
            report = read_run(job, sha, "check", root)
            self.assertEqual(report["status"], "OFFICIAL_GATE_FAILED")
            self.assertNotIn("result", report)


if __name__ == "__main__":
    unittest.main()
