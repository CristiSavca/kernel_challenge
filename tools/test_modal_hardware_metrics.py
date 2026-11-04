"""CPU-only validation of the fixed NCU diagnostic. Never allocates a GPU."""
import contextlib
import ast
import csv
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
import types
from unittest.mock import patch

import modal_hardware_metrics as hw
import modal_h200 as runner
from test_modal_h200 import LifecycleTests, Sandbox

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments/candidates/muse_moe_w4a16" / f"{hw.SOURCE_SHA}.py"
WIDE_FIXTURE = ROOT / "tools/fixtures/ncu_2025_3_1_raw_wide.csv"


def payload():
    files = {"baseline.py": FROZEN.read_text(), "hardware_metrics.py": Path(hw.__file__).read_text()}
    files.update({name: "# test helper\n" for name in hw.FILES - files.keys()})
    return {"problem": "muse_moe_w4a16", "hardware_profile": True, "files": files,
            "file_sha256": {name: hw.digest(text) for name, text in files.items()},
            "source_sha256": hw.SOURCE_SHA, "cases": list(hw.CASES), "repeats": 2,
            "hardware_budget_seconds": 60, "hardware_issue": False, "seed": 7202, "weight_seed": 7311,
            "hardware_input_seeds": list(hw.PROFILE_SEEDS)}


def counters(metrics=hw.REQUIRED):
    stream = io.StringIO()
    stream.write('==PROF== Profiling "okc::moe_fused": 0%....100% - 2 passes\n')
    stream.write('==PROF== Profiling "okc::moe_fused": 0%....100% - 3 passes\n')
    writer = csv.writer(stream)
    writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
    units = ("byte", "byte", "nsecond", "%")
    for ident in (4, 9):
        for i, metric in enumerate(metrics):
            writer.writerow([ident, "void okc::moe_fused<true>()", metric,
                             units[i] if i < 4 else "warp" if metric == hw.ISSUE[1] else "%", "1234"])
    return stream.getvalue()


class HardwareUnitTests(unittest.TestCase):
    def test_current_source_and_native_pins_match_frozen_parent(self):
        self.assertEqual(hw.SOURCE_SHA, "247888acca10f50d8c5db283676ffd93334869467f4632e4a320b34cb20c9ef0")
        self.assertEqual(hw.NATIVE_SHA, "85fb417494c6eab5baabad6813275b256f7bae84c0329d0e94cb61a15030445c")
        self.assertEqual(hw.DECODER_SHA, "97af39947d4cab457126d8eea37802130154aa75eb1119601b1f528c781179c2")
        source = FROZEN.read_text()
        native = next(ast.literal_eval(node.value) for node in ast.parse(source).body
                      if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in node.targets))
        self.assertEqual(hw.digest(source), hw.SOURCE_SHA)
        self.assertEqual(hw.digest(native), hw.NATIVE_SHA)

    def test_payload_is_fixed_and_hashed(self):
        p = payload()
        hw.validate_payload(p)
        for key, value in (("source_sha256", "other"), ("cases", list(reversed(hw.CASES))),
                           ("seed", 1), ("weight_seed", 1), ("repeats", 7),
                           ("hardware_input_seeds", [7302, 8302]),
                           ("hardware_budget_seconds", 861), ("hardware_issue", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                hw.validate_payload({**p, key: value})
        p["files"]["../secret"] = "bad"
        with self.assertRaises(ValueError): hw.validate_payload(p)
        p = payload()
        p["files"]["baseline.py"] += "\n# modified\n"
        with self.assertRaises(ValueError): hw.validate_payload(p)

    def test_fixed_commands_no_user_shell_or_general_profile(self):
        cmd = hw.collection_command(list(hw.REQUIRED))
        for flag, value in (("--replay-mode", "kernel"), ("--cache-control", "all"),
                            ("--clock-control", "none"), ("--profile-from-start", "off"),
                            ("--launch-count", "2"), ("--kill", "no"), ("--rename-kernels", "off")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertNotIn("--set", cmd)
        self.assertNotIn("--section", cmd)
        self.assertEqual(cmd[-3:], [str(hw.FOLDER / "hardware_metrics.py"), "--child", str(hw.PAYLOAD)])
        self.assertEqual(hw.query_command(hw.REQUIRED)[-1], ",".join(x.split(".")[0] for x in hw.REQUIRED))
        self.assertEqual(hw.query_command(hw.BACKPRESSURE)[-1], ",".join(x.split(".")[0] for x in hw.BACKPRESSURE))
        for metrics in hw.METRIC_SETS:
            self.assertEqual(hw.collection_command(list(metrics))[hw.collection_command(list(metrics)).index("--metrics") + 1],
                             ",".join(metrics))
        for bad in ([], ["--set full"], ["$(touch bad)"], list(hw.REQUIRED) + ["arbitrary"]):
            with self.assertRaises(ValueError): hw.collection_command(bad)
        for bad in (hw.REQUIRED + hw.BACKPRESSURE[:1], hw.REQUIRED + hw.ISSUE[:-1] + hw.BACKPRESSURE,
                    hw.REQUIRED + tuple(reversed(hw.BACKPRESSURE))):
            with self.assertRaises(ValueError): hw.collection_command(list(bad))
            with self.assertRaises(ValueError): hw.parse_counters("", list(bad))
        with self.assertRaises(ValueError): hw.query_command(("arbitrary",))

    def test_query_suffixes_are_exact_not_prefix_matches(self):
        found = hw.available_metrics("\n".join(hw.REQUIRED) + "\ndram__bytes_read.sum.extra")
        self.assertTrue(set(hw.REQUIRED) <= found)
        self.assertNotIn("dram__bytes", found)

    def test_parser_requires_two_complete_ordered_launches(self):
        result = hw.parse_counters(counters(), list(hw.REQUIRED))
        self.assertEqual([x["M"] for x in result["launches"]], [32, 256])
        self.assertEqual(result["actual_pass_counts"], [2, 3])
        self.assertEqual(result["launches"][0]["read_plus_write_bytes_per_second"], 2e9)
        self.assertFalse(result["launches"][0]["duration_is_benchmark_evidence"])
        lines = counters().splitlines()
        corrupt = ["\n".join(lines[:-1]), "\n".join(lines + [lines[-1]]),
                   counters().replace("1234", "nan", 1), counters().replace("1234", "-1", 1),
                   counters().replace("nsecond,1234", "nsecond,0", 1),
                   counters().replace("byte,1234", "Kbyte,1234", 1),
                   counters().replace("okc::moe_fused", "other_kernel"),
                   counters().replace("2 passes", "unknown passes"),
                   counters().replace("3 passes", "0 passes")]
        for text in corrupt:
            with self.subTest(text=text[:70]), self.assertRaises(ValueError):
                hw.parse_counters(text, list(hw.REQUIRED))
        self.assertEqual(hw.parse_counters("noise\n" + counters() + "more noise\n", list(hw.REQUIRED)), result)

    def test_real_wide_output_fixture_recovers_exact_values(self):
        # Byte-for-byte collection stdout from terminal Modal run ea14816c;
        # contains NCU warnings, a 280-column header, units and two rows.
        self.assertEqual(hashlib.sha256(WIDE_FIXTURE.read_bytes()).hexdigest(),
                         "b471c2c36016f86bfbfc050f82924a925857f26ff4f5b8c7ae89095cced3a598")
        parsed = hw.parse_counters(WIDE_FIXTURE.read_text(), list(hw.REQUIRED + hw.ISSUE))
        self.assertEqual(parsed["csv_layout"], "wide")
        self.assertEqual(parsed["actual_pass_counts"], [4, 4])
        self.assertEqual([x["id"] for x in parsed["launches"]], ["0", "1"])
        self.assertEqual([x["M"] for x in parsed["launches"]], [32, 256])
        expected = [(456964352, 3375872, 128576, 74.43, 59.38, 1.40, 11.22, 7.49, 1.89, 17.97, 8.11),
                    (769691136, 4264192, 214304, 75.06, 59.48, 1.39, 23.71, 4.89, 2.35, 12.81, 9.58)]
        for launch, values in zip(parsed["launches"], expected):
            self.assertEqual([launch["metrics"][m]["value"] for m in hw.REQUIRED + hw.ISSUE], list(values))
            self.assertEqual(launch["metrics"][hw.REQUIRED[2]]["unit"], "ns")
            self.assertFalse(launch["duration_is_benchmark_evidence"])

    def test_optional_metric_combinations_parse_and_reject_partial_new_group(self):
        for metrics in hw.METRIC_SETS:
            with self.subTest(metrics=metrics):
                parsed = hw.parse_counters(counters(metrics), list(metrics))
                self.assertEqual(set(parsed["launches"][0]["metrics"]), set(metrics))
                stream = io.StringIO()
                writer = csv.writer(stream)
                writer.writerow(["ID", "Kernel Name", "profiler__replayer_passes", *metrics])
                writer.writerow(["", "", "pass", *["byte" if m in hw.REQUIRED[:2] else "ns" if m == hw.REQUIRED[2]
                    else "warp" if m == hw.ISSUE[1] else "%" for m in metrics]])
                for i in range(2):
                    writer.writerow([i, "void okc::moe_fused<true>()", 5, *([1234] * len(metrics))])
                parsed = hw.parse_counters(stream.getvalue(), list(metrics))
                self.assertEqual(parsed["actual_pass_counts"], [5, 5])
                self.assertEqual(set(parsed["launches"][1]["metrics"]), set(metrics))
        full = hw.REQUIRED + hw.ISSUE + hw.BACKPRESSURE
        original = counters(full)
        corrupt = [original.replace(hw.BACKPRESSURE[0], "unrequested.metric"),
                   original.replace(hw.BACKPRESSURE[0] + ",%,1234", hw.BACKPRESSURE[0] + ",%,nan", 1),
                   original.replace(hw.BACKPRESSURE[0] + ",%,1234", hw.BACKPRESSURE[0] + ",cycle,1234", 1)]
        for text in corrupt:
            with self.assertRaises(ValueError): hw.parse_counters(text, list(full))

    def test_wide_output_rejects_partial_duplicate_and_invalid_data(self):
        original = list(csv.reader(io.StringIO(WIDE_FIXTURE.read_text())))
        header = original[3]
        tests = []
        # Partial/missing launches, duplicate launch IDs, extra launches, repeated
        # header, and missing units must not silently select a convenient pair.
        tests += [original[:-1], original + [original[-1]], original[:4] + original[5:],
                  original + [header], original[:6] + [original[-1][:-1]]]
        more = copy.deepcopy(original)
        more.append(list(more[-1])); more[-1][0] = "2"; tests.append(more)
        for col in (hw.REQUIRED[0], "profiler__replayer_passes"):
            i = header.index(col)
            tests.append([row[:i] + row[i+1:] if n >= 3 else row for n, row in enumerate(original)])
        duplicate_column = copy.deepcopy(original)
        duplicate_column[3][-1] = hw.REQUIRED[0]
        tests.append(duplicate_column)
        for rownum, col, value in ((4, hw.REQUIRED[0], "Kbyte"), (4, hw.REQUIRED[2], "us"),
                                   (4, hw.ISSUE[1], "%"), (4, "profiler__replayer_passes", "cycle"),
                                   (5, hw.REQUIRED[0], "nan"), (5, hw.REQUIRED[0], "-1"),
                                   (5, hw.REQUIRED[2], "0"), (5, hw.ISSUE[-1], ""),
                                   (5, "profiler__replayer_passes", "0"),
                                   (5, "profiler__replayer_passes", "4.5"),
                                   (5, "profiler__replayer_passes", "nan"),
                                   (6, "ID", "00"), (5, "Kernel Name", "other_kernel")):
            broken = copy.deepcopy(original)
            broken[rownum][header.index(col)] = value
            tests.append(broken)
        for n, rows in enumerate(tests):
            stream = io.StringIO(); csv.writer(stream).writerows(rows)
            with self.subTest(corruption=n), self.assertRaises(ValueError):
                hw.parse_counters(stream.getvalue(), list(hw.REQUIRED + hw.ISSUE))

    def test_route_histogram_counts_partial_jobs_and_nonlocal_ids(self):
        ids = [[32, 33, 34, 35, 36, 37, 38, 39] for _ in range(65)]
        ids[-1] = [32, 33, 64, 65, 66, 67, 68, 69]
        row = hw.route_summary(ids, 32, 32)
        self.assertEqual(row["local_counts"][:8], [65, 65, 64, 64, 64, 64, 64, 64])
        self.assertEqual((row["U"], row["J"], row["padded_job_rows"]), (8, 10, 528))
        self.assertEqual(row["unique_weight_bytes"], 8 * 27 * 2**20)
        self.assertEqual(row["nominal_request_bytes"], 10 * 27 * 2**20)
        with self.assertRaises(ValueError): hw.route_summary([[256] * 8], 0, 32)

    def test_normal_eof_before_process_exit_is_not_killed(self):
        result = hw.run_bounded([sys.executable, "-c", "import os,time; os.close(1); os.close(2); time.sleep(.08)"], 3)
        self.assertTrue(hw.command_ok(result), result)

    def test_timeout_and_output_limit_kill_owned_process_group(self):
        import os
        real_killpg = os.killpg
        for code, timeout, cap, field in (("import time; time.sleep(3)", .1, 1024, "timed_out"),
                                         ("print('x'*100000)", 3, 1024, "output_exceeded")):
            with patch.object(hw.os, "killpg", wraps=real_killpg) as kill:
                result = hw.run_bounded([sys.executable, "-c", code], timeout, cap)
            self.assertTrue(result[field], result)
            self.assertFalse(hw.command_ok(result))
            self.assertGreaterEqual(kill.call_count, 1)
            self.assertLessEqual(len(result["stdout"].encode()) + len(result["stderr"].encode()), cap)

    def profile_fixture(self, issue=False, missing=False, permission=False, issue_available=False,
                        backpressure_available=False, backpressure_query_error=False, child_source=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        binary = folder / "ncu"
        binary.write_text("# exists, never executed\n")
        result, child = folder / "result.json", folder / "child.json"
        p = payload()
        p["hardware_issue"] = issue
        calls = []
        def bounded(cmd, timeout, cap):
            calls.append(cmd)
            response = {"command": cmd, "returncode": 0, "timed_out": False, "output_exceeded": False,
                        "stdout": "", "stderr": ""}
            if "--version" in cmd: response["stdout"] = "Version fixture"
            elif "--query-metrics-mode" in cmd:
                if "dram__" in cmd[-1]: response["stdout"] = "\n".join(hw.REQUIRED[:-1] if missing else hw.REQUIRED)
                elif "math_pipe_throttle" in cmd[-1]:
                    response["stdout"] = "\n".join(hw.BACKPRESSURE if backpressure_available else hw.BACKPRESSURE[:-1])
                    if backpressure_query_error: response["returncode"] = 1
                else: response["stdout"] = "\n".join(hw.ISSUE if issue_available else hw.ISSUE[:-1])
            else:
                hw.write_json(child, {"status": "hardware_child_complete", "profiled_sizes": [32, 256],
                                      "source_sha256": hw.SOURCE_SHA if child_source is None else child_source})
                selected = cmd[cmd.index("--metrics") + 1].split(",")
                response["stdout"] = counters(selected)
                if permission: response.update(returncode=1, stderr="ERR_NVGPUCTRPERM")
            return response
        with patch.object(hw, "NCU", str(binary)), patch.object(hw, "RESULT", result), \
             patch.object(hw, "CHILD_RESULT", child), patch.object(hw, "run_bounded", side_effect=bounded), \
             contextlib.redirect_stdout(io.StringIO()):
            hw.run_profile(p)
        return json.loads(result.read_text()), calls

    def test_optional_group_is_all_or_none_and_result_records_omission(self):
        result, calls = self.profile_fixture(issue=True)
        self.assertEqual(result["status"], "hardware_complete")
        self.assertEqual(result["selected_metrics"], list(hw.REQUIRED))
        self.assertEqual(result["optional_group_omitted"]["missing"], [hw.ISSUE[-1]])
        self.assertEqual(result["backpressure_group_omitted"]["missing"], [hw.BACKPRESSURE[-1]])
        self.assertEqual(sum("--profile-from-start" in c for c in calls), 1)

    def test_optional_groups_are_independent_and_never_retry(self):
        for issue_available in (False, True):
            for backpressure_available in (False, True):
                result, calls = self.profile_fixture(issue=True, issue_available=issue_available,
                                                    backpressure_available=backpressure_available)
                expected = hw.REQUIRED + (hw.ISSUE if issue_available else ()) + (hw.BACKPRESSURE if backpressure_available else ())
                self.assertEqual(result["status"], "hardware_complete")
                self.assertEqual(result["selected_metrics"], list(expected))
                self.assertEqual(sum("--profile-from-start" in c for c in calls), 1)
                self.assertEqual(sum("--query-metrics-mode" in c for c in calls), 3)
        result, calls = self.profile_fixture(issue=True, issue_available=True, backpressure_available=True,
                                            backpressure_query_error=True)
        self.assertEqual(result["selected_metrics"], list(hw.REQUIRED + hw.ISSUE))
        self.assertEqual(result["backpressure_group_omitted"], {"missing": [], "query_ok": False})
        self.assertEqual(sum("--profile-from-start" in c for c in calls), 1)

    def test_wrong_child_source_is_not_a_complete_measurement(self):
        result, calls = self.profile_fixture(child_source="ef7-or-other-source")
        self.assertEqual(result["status"], "hardware_unavailable")
        self.assertNotIn("counters", result)
        self.assertEqual(sum("--profile-from-start" in c for c in calls), 1)

    def test_missing_required_metric_stops_without_collection(self):
        result, calls = self.profile_fixture(missing=True)
        self.assertEqual(result["status"], "hardware_unavailable")
        self.assertEqual(result["missing_required_metrics"], [hw.REQUIRED[-1]])
        self.assertFalse(any("--profile-from-start" in c for c in calls))

    def test_permission_failure_retains_child_and_does_not_retry(self):
        result, calls = self.profile_fixture(permission=True)
        self.assertEqual(result["status"], "hardware_unavailable")
        self.assertTrue(result["permission_denied"])
        self.assertIsNotNone(result["child"])
        self.assertEqual(sum("--profile-from-start" in c for c in calls), 1)


class HardwareRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)

    @staticmethod
    def checkpoints(parent_status="hardware_running", child_status="hardware_child_running"):
        return {
            "/tmp/okc-result.json": {"status": parent_status, "source_sha256": hw.SOURCE_SHA,
                                     "native_source_sha256": hw.NATIVE_SHA},
            "/tmp/okc-hardware-child.json": {"status": child_status, "source_sha256": hw.SOURCE_SHA},
        }

    def sandbox(self, checkpoints):
        calls = []
        def execute(*args, **kwargs):
            calls.append((args, kwargs))
            self.assertEqual(args[:3], ("python3", "-c", runner.HARDWARE_READ_CHECKPOINT))
            self.assertEqual(kwargs, {"timeout": 10})
            self.assertIn(args[3], ("/tmp/okc-result.json", "/tmp/okc-hardware-child.json"))
            value = checkpoints[args[3]]
            if isinstance(value, Exception): raise value
            text = value if type(value) is str else json.dumps(value)
            return types.SimpleNamespace(stdout=io.StringIO(text), stderr=io.StringIO(""),
                                         wait=lambda: None, returncode=0)
        return types.SimpleNamespace(exec=execute), calls

    def test_remote_reader_bounds_before_output(self):
        path = self.folder / "checkpoint.json"
        for data, cap, passes in ((b'{"a":1}', 7, True), (b"x" * 8, 7, False)):
            path.write_bytes(data)
            result = subprocess.run([sys.executable, "-c", runner.HARDWARE_READ_CHECKPOINT, str(path), str(cap)],
                                    capture_output=True, timeout=3)
            self.assertEqual(result.returncode == 0, passes)
            self.assertEqual(result.stdout, data if passes else b"")

    def test_partial_or_complete_checkpoints_never_promote_failure(self):
        for parent, child in (("hardware_running", "hardware_child_running"),
                              ("hardware_complete", "hardware_child_complete")):
            sandbox, calls = self.sandbox(self.checkpoints(parent, child))
            manifest = {"status": "failed", "error": "original disconnect", "source_sha256": hw.SOURCE_SHA}
            runner.recover_hardware_checkpoints(sandbox, self.folder, manifest)
            self.assertEqual(len(calls), 2)
            self.assertEqual([int(c[0][-1]) for c in calls], [4 * 1024 * 1024, 1024 * 1024])
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"], "original disconnect")
            self.assertFalse((self.folder / "result.json").exists())
            for record in manifest["hardware_checkpoint_recovery"]:
                self.assertEqual(record["status"], "recovered_checkpoint")
                self.assertFalse(record["complete_measurement_claimed"])
                self.assertEqual(json.loads((self.folder / record["local_file"]).read_text())["status"], record["checkpoint_status"])

    def test_bad_parent_evidence_does_not_block_valid_child_or_replace_error(self):
        original = self.checkpoints()
        bad_parents = ["{truncated", "[]", "x" * (4 * 1024 * 1024 + 1),
                       {**original["/tmp/okc-result.json"], "source_sha256": "other"},
                       {**original["/tmp/okc-result.json"], "native_source_sha256": "other"},
                       {**original["/tmp/okc-result.json"], "status": "passed_subset"},
                       FileNotFoundError("missing"), TimeoutError("read timeout")]
        for value in bad_parents:
            with self.subTest(type=type(value).__name__):
                sandbox, calls = self.sandbox({**original, "/tmp/okc-result.json": value})
                manifest = {"status": "failed", "error": "original", "source_sha256": hw.SOURCE_SHA}
                runner.recover_hardware_checkpoints(sandbox, self.folder, manifest)
                a, b = manifest["hardware_checkpoint_recovery"]
                self.assertEqual(a["status"], "unavailable")
                self.assertIn("error_type", a)
                self.assertEqual(b["status"], "recovered_checkpoint")
                self.assertEqual(manifest["error"], "original")
                self.assertEqual(manifest["status"], "failed")
                self.assertEqual(len(calls), 2)

    def test_wrong_manifest_makes_no_recovery_call(self):
        sandbox, calls = self.sandbox(self.checkpoints())
        manifest = {"status": "failed", "source_sha256": "other"}
        runner.recover_hardware_checkpoints(sandbox, self.folder, manifest)
        self.assertEqual(calls, [])
        self.assertTrue(all(r["status"] == "unavailable" for r in manifest["hardware_checkpoint_recovery"]))

    def test_nonzero_reader_exit_is_unavailable(self):
        process = types.SimpleNamespace(stdout=io.StringIO("{}"), stderr=io.StringIO("private error"),
                                        wait=lambda: None, returncode=1)
        sandbox = types.SimpleNamespace(exec=lambda *args, **kw: process)
        manifest = {"status": "failed", "source_sha256": hw.SOURCE_SHA}
        runner.recover_hardware_checkpoints(sandbox, self.folder, manifest)
        self.assertTrue(all(r["status"] == "unavailable" for r in manifest["hardware_checkpoint_recovery"]))
        self.assertNotIn("private error", json.dumps(manifest))


class HardwareLifecycleTests(LifecycleTests):
    # Inherit the guard/termination suite and add the new mode's exact boundary.
    def setUp(self):
        super().setUp()
        path = self.root / "experiments/candidates/muse_moe_w4a16" / FROZEN.name
        path.parent.mkdir(parents=True)
        path.write_bytes(FROZEN.read_bytes())
        (self.root / "tools/modal_hardware_metrics.py").write_text(Path(hw.__file__).read_text())

    def test_hardware_payload_uses_guard_one_gpu_and_fixed_files(self):
        sandbox = Sandbox()
        created = self.invoke("hardware-profile", sandbox)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["gpu"], "H200")
        self.assertTrue(created[0]["block_network"])
        self.assertTrue(sandbox.terminated)
        p = json.loads(sandbox.uploads["/tmp/okc-payload.json"])
        hw.validate_payload(p)
        self.assertEqual(p["hardware_input_seeds"], [7302, 9302])
        self.assertNotIn("PRIVATE_TEST_SENTINEL", str(sandbox.uploads))
        self.assertEqual(len(sandbox.uploads), 2)

    def test_hardware_dry_run_has_no_billing_or_allocation(self):
        self.billing = AssertionError("dry run billing")
        created = self.invoke("hardware-profile", dry_run=True, create_error=AssertionError("allocation"))
        self.assertEqual(created, [])
        self.assertFalse((self.root / "experiments/modal").exists())

    def test_exact_900_second_dry_run_records_current_source_and_groups(self):
        output = io.StringIO()
        with patch.object(runner, "ROOT", self.root), \
             patch.object(sys, "argv", ["modal_h200.py", "hardware-profile", "--hardware-issue", "--repeats", "6", "--timeout", "900", "--dry-run"]), \
             patch("modal_credit_budget.allocation_guard", side_effect=AssertionError("dry run allocation")), \
             contextlib.redirect_stdout(output):
            runner.main()
        manifest = json.loads(output.getvalue())
        self.assertEqual(manifest["source_sha256"], hw.SOURCE_SHA)
        self.assertEqual(manifest["timeout_seconds"], 900)
        self.assertEqual(manifest["hardware_budget_seconds"], 860)
        self.assertEqual(manifest["hardware_policy"]["launch_count"], 2)
        self.assertEqual(manifest["hardware_optional_metrics"], list(hw.ISSUE + hw.BACKPRESSURE))
        self.assertEqual(manifest["repeats"], 6)
        self.assertFalse((self.root / "experiments/modal").exists())

    def test_disconnected_worker_recovers_separate_checkpoints_then_terminates(self):
        sandbox = Sandbox()
        calls = []
        checkpoints = HardwareRecoveryTests.checkpoints("hardware_complete", "hardware_child_complete")
        def disconnected():
            yield "partial worker log\n"
            raise ConnectionError("original disconnect")
        def execute(*args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ("python3", "-c"):
                self.assertFalse(sandbox.terminated)
                return types.SimpleNamespace(stdout=io.StringIO(json.dumps(checkpoints[args[3]])),
                                             stderr=io.StringIO(""), wait=lambda: None, returncode=0)
            return types.SimpleNamespace(stdout=disconnected(), stderr=io.StringIO(""), wait=lambda: None, returncode=0)
        sandbox.exec = execute
        with self.assertRaisesRegex(ConnectionError, "original disconnect"):
            self.invoke("hardware-profile", sandbox)
        manifest = self.manifest()
        self.assertEqual(len(self.created), 1)
        self.assertEqual(len(calls), 3)  # One worker, two read-only recovery commands.
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue(manifest["sandbox_terminated"])
        self.assertTrue(sandbox.terminated)
        self.assertTrue(all(r["status"] == "recovered_checkpoint" for r in manifest["hardware_checkpoint_recovery"]))
        destination = next((self.root / "experiments/modal").glob("*/manifest.json")).parent
        self.assertFalse((destination / "result.json").exists())
        self.assertTrue((destination / "hardware-recovered-result.json").exists())

    def test_hardware_failure_terminates_and_retains_result(self):
        sandbox = Sandbox(fail=True)
        with self.assertRaises(RuntimeError): self.invoke("hardware-profile", sandbox)
        self.assertTrue(sandbox.terminated)
        self.assertTrue(self.manifest()["sandbox_terminated"])

    def test_hardware_wrong_frozen_source_fails_before_allocation(self):
        path = self.root / "experiments/candidates/muse_moe_w4a16" / FROZEN.name
        path.write_text("# wrong source\n")
        with self.assertRaises(ValueError): self.invoke("hardware-profile", dry_run=True)
        self.assertEqual(self.created, [])

    def test_hardware_rejects_general_arguments_before_allocation(self):
        for extra in (("--candidate", str(self.candidate)), ("--cases", "other.json"),
                      ("--seed", "7302"), ("--weight-seed", "1"), ("--repeats", "7"),
                      ("--problem", "muse_decode_layer"), ("--profile",), ("--resources",)):
            with self.subTest(extra=extra), patch.object(runner, "ROOT", self.root), \
                 patch.object(sys, "argv", ["modal_h200.py", "hardware-profile", "--dry-run", *extra]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()
        self.assertEqual(self.created, [])

    # Base test assumes no experiments directory; our immutable source fixture
    # intentionally lives under experiments/candidates.
    def test_dry_run_never_creates_cloud_resource(self):
        self.billing = AssertionError("dry run billing")
        self.assertEqual(self.invoke("benchmark", dry_run=True), [])
        self.assertFalse((self.root / "experiments/modal").exists())


if __name__ == "__main__":
    unittest.main()
