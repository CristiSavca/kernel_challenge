"""Exercise dependency and quota gates with simulated judge responses."""

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import submit_nov18_decode_checks as runner


def response(data, code=0):
    return SimpleNamespace(returncode=code, stdout=json.dumps(data), stderr="")


def job(job_id, candidate, *, status="done", passed=True):
    return {
        "id": job_id, "board": "serving", "mode": "check", "status": status,
        "result": {
            "status": "ok", "passed": passed,
            "patch": {"sha256": candidate.sha256}, "serving_flags": candidate.flags,
            "concurrency": [runner.check_load(candidate)],
        } if status == "done" else None,
    }


V72a_JOB = "1111111111111111"
V76a_JOB = "2222222222222222"
V87_JOB = "3333333333333333"
V68 = SimpleNamespace(sha256=runner.V68_SHA, flags=runner.BASE_FLAGS)


class FakeJudge:
    def __init__(self):
        self.left = 3
        self.reset = runner.RESET
        self.jobs = {runner.V68_JOB: job(runner.V68_JOB, V68)}
        self.commands = []
        self.checks = []
        self.behavior = "success"
        self.after_quota = None
        self.status_fail = False

    def __call__(self, command, **kwargs):
        assert command[0] == "okc-sim" and kwargs["capture_output"] and kwargs["text"]
        assert "submit" not in command
        self.commands.append(command)
        if command[1:] == ["serve", "status", "--json"]:
            if self.after_quota:
                self.after_quota()
            return response({"limits": {"checks": {"left": self.left}}, "resets_at": self.reset})
        if command[1:3] == ["serve", "status"]:
            if self.status_fail:
                return response({}, 1)
            return response(self.jobs.get(command[3], {}))
        assert command[1:3] == ["serve", "check"]
        assert command[-1] == "--no-wait"
        self.checks.append(command)
        candidate = next(c for c in (*runner.CANDIDATES, runner.V88) if c.directory in command[3])
        assert command[command.index("--users") + 1] == str(runner.check_load(candidate))
        identifier = {"v72a": V72a_JOB, "v76a": V76a_JOB, "v87": V87_JOB, "v88": "4"*16}[candidate.name]
        if self.behavior == "timeout":
            raise subprocess.TimeoutExpired(command, 120, output=f"job {identifier}\n".encode())
        if self.behavior == "oserror":
            raise OSError("simulated uncertain CLI failure")
        self.jobs[identifier] = job(identifier, candidate, status="queued")
        self.left -= 1
        return SimpleNamespace(returncode=1 if self.behavior == "nonzero" else 0,
                               stdout=f"job {identifier}\n", stderr="")


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="okc-nov18-runner-test-")
        cls.root = Path(cls.scratch.name)
        space = os.statvfs(cls.root)
        cls.free_before = space.f_bavail * space.f_frsize

    @classmethod
    def tearDownClass(cls):
        cls.removed_bytes = sum(p.stat().st_size for p in cls.root.rglob("*") if p.is_file())
        cls.removed_path = str(cls.root)
        cls.scratch.cleanup()
        space = os.statvfs(cls.root.parent)
        cls.free_after = space.f_bavail * space.f_frsize

    def setUp(self):
        self.runs = self.root / self._testMethodName
        self.runs.mkdir()
        self.now = runner.START + timedelta(minutes=1)
        self.judge = FakeJudge()

    def offer(self, candidate):
        return next(offer for offer in runner.decisions(self.runs) if offer[1] is candidate)

    def run_candidate(self, candidate, **kwargs):
        return runner.run_offer(self.offer(candidate), okc="okc-sim", runs=self.runs,
                                execute=self.judge, clock=lambda: self.now, **kwargs)

    def submit_v72a(self):
        self.assertEqual(self.run_candidate(runner.V72a), f"submitted {V72a_JOB}")

    def test_exact_sources_flags_and_order(self):
        self.assertEqual([offer[1].name for offer in runner.decisions(self.runs)], ["v87", "v72a", "v76a"])
        self.assertEqual(len(runner.V72a.flags), 5)
        self.assertEqual(len(runner.V76a.flags), 6)
        self.assertEqual(runner.V76a.flags, runner.V72a.flags | {
            "--speculative-dflash-tp-context-projection": ""})
        self.assertEqual(runner.V87.flags, runner.V76a.flags | {
            "--speculative-dflash-tp-conv-projection": "", "--speculative-dflash-compact-context": "",
            "--speculative-dflash-fused-candidate-head": ""})
        self.assertEqual([runner.check_load(c) for c in runner.CANDIDATES], [4, 16, 16])
        self.assertEqual([runner.receipt_name(c) for c in runner.CANDIDATES], [
            "v87_nov18_check4", "v72a_nov18_check16", "v76a_nov18_check16"])
        for candidate in runner.CANDIDATES:
            path = runner.ROOT / "serving_v0523/candidates" / candidate.directory / "patch.diff"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), candidate.sha256)

    def test_v88_uses_one_spare_check_after_confirmed_parent_failure(self):
        self.judge.jobs[runner.V68_JOB] = job(runner.V68_JOB, V68, passed=False)
        self.assertIn("skipped", self.run_candidate(runner.V72a))
        self.assertIn("submitted", self.run_candidate(runner.V88))
        self.assertEqual(self.judge.left, 2)
        self.assertEqual(self.judge.checks[0][self.judge.checks[0].index("--users")+1], "4")
        self.assertNotIn(runner.V88, [offer[1] for offer in runner.decisions(self.runs)])

    def test_v88_does_not_run_on_changed_status_or_other_skip_reason(self):
        runner.skip(runner.V72a, "judge status identity mismatch", runs=self.runs, now=self.now)
        self.assertNotIn(runner.V88, [offer[1] for offer in runner.decisions(self.runs)])
        path = self.runs / (runner.receipt_name(runner.V72a) + ".skip.json")
        path.write_text(json.dumps({"reason": f"judge job {runner.V68_JOB} is failed"}))
        self.assertTrue(self.run_candidate(runner.V88).startswith("waiting"))
        self.assertEqual(self.judge.checks, [])

    def test_all_three_private_checks_and_flags(self):
        self.assertEqual(self.run_candidate(runner.V87), f"submitted {V87_JOB}")
        self.submit_v72a()
        self.assertEqual(self.run_candidate(runner.V76a), f"submitted {V76a_JOB}")
        self.assertFalse(runner.decisions(self.runs))
        self.assertEqual(len(self.judge.checks), 3)
        self.assertEqual(self.judge.left, 0)
        for command, candidate in zip(self.judge.checks, runner.CANDIDATES):
            flags = [command[i + 1] for i, part in enumerate(command) if part == "--flag"]
            expected = [key[2:] + ("=" + value if value else "")
                        for key, value in sorted(candidate.flags.items())]
            self.assertEqual(flags, expected)

    def test_outside_window_has_no_cli_actions(self):
        for when in (runner.START - timedelta(seconds=1), runner.END):
            self.now = when
            self.assertEqual(self.run_candidate(runner.V72a), "outside submission window")
        self.assertFalse(self.judge.commands)

    def test_quota_reserve_and_types(self):
        for value in (0, -1, True, "3", None):
            self.judge.left = value
            self.assertEqual(self.run_candidate(runner.V72a), "quota unavailable")
        self.assertFalse(self.judge.checks)
        self.assertTrue(self.runs.is_dir() and not list(self.runs.iterdir()))

    def test_wrong_reset_day_waits(self):
        self.judge.reset = "2026-11-18T00:00:00Z"
        self.assertEqual(self.run_candidate(runner.V72a), "quota unavailable")
        self.assertFalse(self.judge.checks)

    def test_v68_pending_and_unavailable_wait(self):
        for state in ("queued", "running"):
            self.judge.jobs[runner.V68_JOB]["status"] = state
            self.assertTrue(self.run_candidate(runner.V72a).startswith("waiting "))
        self.judge.status_fail = True
        self.assertTrue(self.run_candidate(runner.V72a).startswith("waiting "))
        self.assertFalse(self.judge.checks)

    def test_v68_failed_skips_descendants(self):
        self.judge.jobs[runner.V68_JOB]["result"]["passed"] = False
        for candidate in runner.CANDIDATES:
            self.assertTrue(self.run_candidate(candidate).startswith("skipped "))
        self.assertEqual([o[1] for o in runner.decisions(self.runs)], [runner.V88])
        self.assertFalse(self.judge.checks)

    def test_partial_quota_prioritizes_control_pair(self):
        self.judge.left = 2
        self.assertEqual(self.run_candidate(runner.V87), "quota unavailable")
        self.submit_v72a()
        self.assertEqual(self.run_candidate(runner.V76a), f"submitted {V76a_JOB}")
        self.assertEqual(self.judge.left, 0)
        self.assertEqual(self.run_candidate(runner.V87), "quota unavailable")

    def test_v87_pending_parent_can_run_only_with_full_quota(self):
        for state in ("queued", "running"):
            self.judge.jobs[runner.V68_JOB] = job(runner.V68_JOB, V68, status=state)
            for left in (0, 1, 2):
                self.judge.left = left
                self.assertEqual(self.run_candidate(runner.V87), "quota unavailable")
        self.judge.left = 3
        self.assertEqual(self.run_candidate(runner.V87), f"submitted {V87_JOB}")
        for candidate in (runner.V72a, runner.V76a):
            self.assertTrue(self.run_candidate(candidate).startswith("waiting "))
        self.assertEqual(self.judge.left, 2)

    def test_v87_v68_failure_after_quota_blocks_before_receipt(self):
        self.judge.jobs[runner.V68_JOB] = job(runner.V68_JOB, V68, status="queued")
        self.judge.after_quota = lambda: self.judge.jobs.__setitem__(
            runner.V68_JOB, job(runner.V68_JOB, V68, status="failed"))
        self.assertTrue(self.run_candidate(runner.V87).startswith("skipped "))
        self.assertFalse((self.runs / "v87_nov18_check4.submit.txt").exists())
        self.assertEqual(self.judge.left, 3)

    def test_v87_waits_on_unverifiable_v68_identity(self):
        self.judge.jobs[runner.V68_JOB] = job(V72a_JOB, V68, status="queued")
        self.assertTrue(self.run_candidate(runner.V87).startswith("waiting "))
        self.assertFalse(self.judge.checks)

    def test_withdrawn_dependency_skips(self):
        self.judge.jobs[runner.V68_JOB] = job(runner.V68_JOB, V68, status="withdrawn")
        self.assertTrue(self.run_candidate(runner.V72a).startswith("skipped "))
        self.assertFalse(self.judge.checks)

    def test_wrong_dependency_identity_and_payload(self):
        original = copy.deepcopy(self.judge.jobs[runner.V68_JOB])
        for field, value in (("id", V72a_JOB), ("board", "kernel"), ("mode", "run")):
            self.judge.jobs[runner.V68_JOB] = original | {field: value}
            self.assertTrue(self.run_candidate(runner.V72a).startswith("waiting "))
        self.judge.jobs[runner.V68_JOB] = copy.deepcopy(original)
        self.judge.jobs[runner.V68_JOB]["result"]["patch"]["sha256"] = "wrong"
        self.assertTrue(self.run_candidate(runner.V72a).startswith("skipped "))
        self.assertFalse(self.judge.checks)

    def test_v76a_requires_confirmed_v72a_receipt(self):
        self.assertTrue(self.run_candidate(runner.V76a).startswith("waiting "))
        self.assertFalse(self.judge.checks)

    def test_v72a_failure_blocks_v76a(self):
        self.submit_v72a()
        self.judge.jobs[V72a_JOB] = job(V72a_JOB, runner.V72a, status="failed")
        self.assertTrue(self.run_candidate(runner.V76a).startswith("skipped "))
        self.assertEqual(len(self.judge.checks), 1)

    def test_v72a_pass_allows_v76a(self):
        self.submit_v72a()
        self.judge.jobs[V72a_JOB] = job(V72a_JOB, runner.V72a)
        self.assertEqual(self.run_candidate(runner.V76a), f"submitted {V76a_JOB}")

    def test_failure_after_quota_read_blocks_v76a(self):
        self.submit_v72a()
        self.judge.after_quota = lambda: self.judge.jobs.__setitem__(
            V72a_JOB, job(V72a_JOB, runner.V72a, status="failed"))
        self.assertTrue(self.run_candidate(runner.V76a).startswith("skipped "))
        self.assertFalse((self.runs / "v76a_nov18_check16.submit.txt").exists())
        self.assertEqual(len(self.judge.checks), 1)

    def test_deadline_during_preflight_blocks_cli_launch(self):
        self.judge.after_quota = lambda: setattr(self, "now", runner.END)
        self.assertTrue(self.run_candidate(runner.V72a).startswith("skipped "))
        self.assertFalse(self.judge.checks)
        self.assertNotIn(runner.V72a, [offer[1] for offer in runner.decisions(self.runs)])

    def test_ambiguous_attempts_are_never_retried(self):
        for behavior in ("timeout", "nonzero", "oserror"):
            sub = self.runs / behavior
            sub.mkdir()
            judge = FakeJudge()
            judge.behavior = behavior
            offer = next(offer for offer in runner.decisions(sub) if offer[1] is runner.V72a)
            try:
                result = runner.run_offer(offer, okc="okc-sim", runs=sub, execute=judge,
                                          clock=lambda: self.now)
                self.assertIn("no retry", result)
            except OSError:
                self.assertEqual(behavior, "oserror")
            self.assertEqual(len(judge.checks), 1)
            self.assertEqual([offer[1].name for offer in runner.decisions(sub)], ["v87", "v76a"])
            self.assertTrue(runner.run_offer(runner.decisions(sub)[1], okc="okc-sim",
                                            runs=sub, execute=judge, clock=lambda: self.now
                                            ).startswith("waiting "))
            self.assertEqual(len(judge.checks), 1)

    def test_exclusive_attempt_concurrency(self):
        offer = self.offer(runner.V72a)
        def send(_):
            return runner.run_offer(offer, okc="okc-sim", runs=self.runs,
                                    execute=self.judge, clock=lambda: self.now)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(send, range(2)))
        self.assertEqual(sorted(results), ["prior attempt exists", f"submitted {V72a_JOB}"])
        self.assertEqual(len(self.judge.checks), 1)

    def test_patch_mismatch_never_submits(self):
        root = self.runs / "altered_root"
        path = root / "serving_v0523/candidates" / runner.V72a.directory / "patch.diff"
        path.parent.mkdir(parents=True)
        path.write_text("altered bytes")
        self.assertEqual(self.run_candidate(runner.V72a, root=root), "patch hash mismatch")
        self.assertFalse(self.judge.checks)

    def test_v87_shared_control_failure_or_unconfirmed_receipt(self):
        for parent in (runner.V72a, runner.V76a):
            path = self.runs / (runner.receipt_name(parent) + ".submit.txt")
            path.write_text("uncertain CLI")
            self.assertTrue(self.run_candidate(runner.V87).startswith("waiting "))
            path.unlink()
        skip = self.runs / (runner.receipt_name(runner.V76a) + ".skip.json")
        skip.write_text("{}")
        self.assertTrue(self.run_candidate(runner.V87).startswith("skipped "))
        self.assertFalse(self.judge.checks)

    def test_v87_original_receipt_and_source_verified(self):
        self.assertEqual(self.run_candidate(runner.V87), f"submitted {V87_JOB}")
        receipt = self.runs / "v87_nov18_check4.submit.txt"
        meta = json.loads(receipt.read_text().splitlines()[0])
        self.assertEqual((meta["users"], meta["sha256"], meta["flags"]),
                         (4, runner.V87.sha256, runner.V87.flags))
        self.assertNotIn(runner.V87, [o[1] for o in runner.decisions(self.runs)])
        self.assertEqual(len(self.judge.checks), 1)

    def test_original_nov17_suite_preserves_old_receipt(self):
        previous = HERE / "nov17_integrated_runner.cpu.json"
        original = previous.read_bytes()
        directory = self.runs / "legacy"
        directory.mkdir()
        write = Path.write_text
        captured = []
        def intercept(path, content, *args, **kwargs):
            if path == previous:
                captured.append(json.loads(content))
                return len(content)
            return write(path, content, *args, **kwargs)
        with mock.patch.object(Path, "write_text", intercept):
            with mock.patch("tempfile.mkdtemp", return_value=str(directory)):
                runpy.run_path(str(HERE / "test_nov17_integrated_runner.py"),
                               run_name="__legacy_nov17_test__")
        self.assertEqual(previous.read_bytes(), original)
        self.assertEqual(len(captured[0]["cases"]), 17)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "status": "CPU_PASS" if result.wasSuccessful() else "CPU_FAILED",
        "test_cases": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "original_nov17_cases": 17,
        "script_sha256": hashlib.sha256((HERE / "submit_nov18_decode_checks.py").read_bytes()).hexdigest(),
        "shared_helper_sha256": hashlib.sha256((HERE / "submit_nov17_integrated_checks.py").read_bytes()).hexdigest(),
        "cleanup": {"removed_test_directory": RunnerTests.removed_path,
                    "removed_file_bytes": RunnerTests.removed_bytes,
                    "observed_free_before": RunnerTests.free_before,
                    "observed_free_after": RunnerTests.free_after,
                    "observed_free_delta": RunnerTests.free_after - RunnerTests.free_before},
        "judge_requests": 0, "check_submissions": 0,
    }
    (HERE / "nov18_decode_runner.cpu.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
