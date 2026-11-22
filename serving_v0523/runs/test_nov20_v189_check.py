"""Replay the alternative private candidate using temporary receipts and a dummy judge."""
import contextlib
import io
import importlib.util
import json
import multiprocessing
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import submit_nov20_v187_check as runner
import test_nov20_v187_check as parent


def hold_process_lock(runs, selected, pipe):
    # Exercise each real CLI and singleton flock with time held before opening.
    before_open = runner.START - timedelta(days=1)
    now = [before_open]
    def private_wait(_seconds):
        pipe.send("holding")
        if not pipe.poll(20):
            raise RuntimeError("test parent failed to release simulated clock")
        pipe.recv()
        now[0] = runner.END
    def forbidden(*args, **kwargs):
        raise AssertionError("offline process-lock test attempted an allocation poll")
    with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, "RUNS", runs), \
            patch.object(sys, "argv", ["dummy", "--candidate", selected, "--run"]), \
            patch.object(runner.first, "utc_now", lambda: now[0]), \
            patch.object(runner.time, "sleep", private_wait), patch.object(runner, "run_once", forbidden):
        runner.main()
    pipe.send("returned")
    pipe.close()


class Judge(parent.Judge):
    def offer(self, candidate=None):
        return runner.run_once(okc="mock", candidate=candidate or runner.V189,
                               runs=self.runs, clock=lambda: self.now, execute=self.execute)


class Cases(parent.Cases):
    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Judge(temp.name)

    def test_positive_and_exact_command(self):
        for state in ("queued", "running"):
            fake = self.fixture()
            fake.jobs[parent.IDS[2]]["status"] = state
            self.assertEqual(fake.offer(), "submitted " + parent.CHILD)
            accepted = (runner.V189, 16, runner.NAME, runner.START, runner.END)
            self.assertEqual(runner.slots.bound_receipt(fake.runs, accepted), parent.CHILD)
            self.assertIsNone(runner.slots.bound_receipt(fake.runs, runner.OWN))
            self.assertEqual(fake.events[-3:-1], [["serve", "status", "--json"],
                                                ["serve", "status", parent.IDS[2], "--json"]])
            self.assertEqual(len(fake.attempts), 1)
            command = fake.attempts[0]
            self.assertEqual(command[:6], ["mock", "serve", "check", str(
                runner.ROOT/"serving_v0523/candidates"/runner.V189.directory/"patch.diff"), "--users", "16"])
            expected_flags = [key[2:] + ("=" + value if value else "") for key, value in sorted(runner.V187.flags.items())]
            self.assertEqual(command[6:-1:2], ["--flag"] * len(expected_flags))
            self.assertEqual(command[7:-1:2], expected_flags)
            self.assertEqual(command[-1], "--no-wait")
            header = json.loads(fake.receipt.read_text().splitlines()[0])
            self.assertEqual((header["candidate"], header["sha256"], header["flags"], header["users"]),
                             ("v189", runner.V189.sha256, runner.V187.flags, 16))
            self.assertEqual(header["evidence"]["comparison_rung"]["tok_s_per_gpu"], 17512.9)
            self.assertEqual(header["evidence"]["final_quota"]["checks"], runner.slots.THIRD)
            self.assertEqual(header["evidence"]["final_control"]["status"], state)
            self.assertEqual(fake.offer(runner.V187), "attempt recorded")
            self.assertEqual(fake.offer(runner.V189), "attempt recorded")

    def test_rollback_and_cross_candidate_attempts(self):
        for candidate, other in [(runner.V187, runner.V189), (runner.V189, runner.V187)]:
            for timeout in (False, True):
                fake = self.fixture()
                fake.timeout = timeout
                fake.offer(candidate)
                self.assertTrue(fake.receipt.exists())
                self.assertEqual(fake.offer(other), "attempt recorded")
                self.assertEqual(fake.offer(candidate), "attempt recorded")
                self.assertEqual(len(fake.attempts), 1)
        for filename, body in [
                (runner.LEGACY_NAME + ".submit.txt", ""),
                ("unresolved_nov20.submit.txt", "partial"),
                ("old.submit.txt", runner.V187.sha256),
                ("old.submit.txt", runner.V189.sha256)]:
            fake = self.fixture()
            (fake.runs/filename).write_text(body)
            self.held(fake)
            self.assertEqual(fake.events, [])

    def test_frozen_original_and_selected_variant_cannot_claim_each_others_slot(self):
        source = runner.RUNS/"history/nov20_v187_before_v189/submit_nov20_v187_check.py"
        self.assertEqual(runner.first.digest(source), "bdd242ec80a38ecef5c80069e9bdab6af0d2725ff68fd62fabff6ca506f6a7a5")
        spec = importlib.util.spec_from_file_location("archived_private_standby", source)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        self.assertEqual(legacy.NAME, runner.LEGACY_NAME)
        for new_first in (False, True):
            for timeout in (False, True):
                fake = self.fixture()
                fake.timeout = timeout
                def old():
                    return legacy.run_once(okc="mock", runs=fake.runs, clock=lambda:fake.now, execute=fake.execute)
                ordered = [fake.offer, old] if new_first else [old, fake.offer]
                ordered[0]()
                self.assertEqual(len(fake.attempts), 1)
                self.assertFalse(ordered[1]().startswith("submitted "))
                self.assertEqual(len(fake.attempts), 1)
                claims = {p.name for p in fake.runs.glob("*standby_slot_nov20_check16.submit.txt")}
                self.assertEqual(claims, {(runner.NAME if new_first else runner.LEGACY_NAME)+".submit.txt"})

    def test_corrupt_selected_export_or_parent_blocks_only_that_plan(self):
        real = json.loads
        for field, value in [
                ("tree", "0"*40), ("patch_sha256", "0"*64), ("parent", "V182"),
                ("parent_sha256", "0"*64), ("parent_tree", "0"*40),
                ("launch_flags", {}), ("strict_apply_reverse_full_delta", 1)]:
            def damaged(raw, *args, **kwargs):
                value_read = real(raw, *args, **kwargs)
                if isinstance(value_read, dict) and value_read.get("candidate") == "V189":
                    value_read[field] = value
                return value_read
            with self.subTest(field=field), patch.object(runner.json, "loads", damaged):
                self.assertEqual(runner.plan(candidate=runner.V187)["candidate"], "v187")
                with self.assertRaises(ValueError):
                    runner.plan(candidate=runner.V189)

    def test_exclusive_process_lock_in_each_start_order(self):
        ctx = multiprocessing.get_context("spawn")
        for first, second in [("v187", "v189"), ("v189", "v187")]:
            with tempfile.TemporaryDirectory() as directory:
                runs = Path(directory)
                holders = []
                try:
                    for which, expected in [(first, "holding"), (second, "returned")]:
                        local, child = ctx.Pipe()
                        process = ctx.Process(target=hold_process_lock, args=(runs, which, child))
                        process.start()
                        child.close()
                        holders.append((process, local))
                        self.assertTrue(local.poll(10))
                        self.assertEqual(local.recv(), expected)
                    holders[0][1].send("expire")
                    for process, pipe in holders:
                        process.join(10)
                        self.assertEqual(process.exitcode, 0)
                    self.assertEqual(sorted(p.name for p in runs.iterdir()), [runner.PROCESS_LOCK])
                finally:
                    for process, pipe in holders:
                        if process.is_alive():
                            process.terminate()
                            process.join(5)
                        pipe.close()

    def test_reservations_match_in_all_fields_except_candidate_identity(self):
        first = runner.plan(candidate=runner.V187)
        second = runner.plan(candidate=runner.V189)
        self.assertEqual({k for k in first.keys() | second.keys() if first.get(k) != second.get(k)},
                         {"candidate", "sha256", "tree"})
        self.assertEqual(first["receipt"], second["receipt"])
        self.assertEqual(first["process_lock"], "nov20_v187_runner.lock")
        self.assertEqual(second["arbitration_lock"], runner.slots.LOCK)


if __name__ == "__main__":
    unittest.main()
