"""CPU-only validation of the fixed NCU diagnostic. Never allocates a GPU."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import modal_hardware_metrics as hw
import modal_h200 as runner
from test_modal_h200 import LifecycleTests, Sandbox

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "experiments/candidates/muse_moe_w4a16" / f"{hw.SOURCE_SHA}.py"


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
                             units[i] if i < 4 else "%", "1234"])
    return stream.getvalue()


class HardwareUnitTests(unittest.TestCase):
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
        for bad in ([], ["--set full"], ["$(touch bad)"], list(hw.REQUIRED) + ["arbitrary"]):
            with self.assertRaises(ValueError): hw.collection_command(bad)
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

    def profile_fixture(self, issue=False, missing=False, permission=False):
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
                else: response["stdout"] = "\n".join(hw.ISSUE[:-1])
            else:
                hw.write_json(child, {"status": "hardware_child_complete", "profiled_sizes": [32, 256]})
                response["stdout"] = counters()
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
