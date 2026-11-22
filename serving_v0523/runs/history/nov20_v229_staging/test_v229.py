"""Run only against temporary run directories and mock judge results."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

STAGE = Path(__file__).resolve().parent
RUNS = STAGE.parents[1]
sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(STAGE))
import submit_nov20_v187_check as runner
import test_nov20_v198_check as previous
import v229_selection as selection


class Judge(previous.Judge):
    def __init__(self, path):
        super().__init__(path)
        name = selection.V204_RECEIPT + ".submit.txt"
        shutil.copyfile(RUNS/name, self.runs/name)
        reference = copy.deepcopy(self.jobs[runner.v180.nov19.V68_JOB])
        created = runner.v180.nov19.START.timestamp() + 6
        reference.update(id=selection.V204_JOB, created_at=created, finished_at=created+1300,
                         serving_flags=runner.V204.flags, users=16)
        result = reference["result"]
        result["patch"]["sha256"] = runner.V204.sha256
        result["serving_flags"] = runner.V204.flags
        result["rungs"][0]["tok_s_per_gpu"] = 18500.
        result["vs_stock"]["rungs"][0]["at_stock_point"] = 1.53
        self.jobs[selection.V204_JOB] = reference
        self.report = reference

    def offer(self, candidate=None):
        return super().offer(candidate or runner.V229)

    def choice(self):
        self.assert_submitted_once()
        return json.loads(self.receipt.read_text().splitlines()[0])

    def assert_submitted_once(self):
        assert self.offer().startswith("submitted "), self.events
        assert len(self.attempts) == 1


class Cases(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Judge(temp.name)

    def test_real_margin_identity_and_exact_program(self):
        fake = self.fixture()
        receipt = fake.choice()
        self.assertEqual((receipt["candidate"], receipt["sha256"], receipt["users"], receipt["flags"]),
                         ("v229", selection.V229_SHA, 16, runner.V204.flags))
        margin = receipt["evidence"]["v204_comparison"]
        self.assertEqual(margin["noise_control_job"], selection.NOISE_JOB)
        self.assertGreater(margin["raw_gain"], selection.RAW_SPREAD)
        self.assertGreater(margin["normalized_gain"], selection.RAW_SPREAD)
        self.assertEqual(fake.events[-3:-1], [["serve", "status", "--json"], ["serve", "status", previous.parent.IDS[2], "--json"]])
        self.assertEqual(fake.events.count(["serve", "status", selection.V204_JOB, "--json"]), 2)
        self.assertEqual(fake.attempts[0][3], str(runner.ROOT/"serving_v0523/candidates"/runner.V229.directory/"patch.diff"))
        self.assertEqual(fake.offer(), "attempt recorded")
        self.assertEqual(fake.offer(runner.V198), "attempt recorded")
        self.assertEqual(len(fake.attempts), 1)

    def test_pending_failed_and_unqualified_take_original_candidate(self):
        def field(root, keys, value):
            for key in keys[:-1]:
                root = root[key]
            root[keys[-1]] = value
        mutations = [
            ("id", "deadbeef00000000"), ("team", "other"), ("cluster", "uncalibrated"), ("result/node", []),
            ("dedicated", False), ("result/reps", True), ("board", "kernel"), ("mode", "run"),
            ("serving_flags", {}), ("status", "running"), ("duplicate_of", "other"), ("refunded", True),
            ("finished_at", 0), ("finished_at", runner.END.timestamp()+20), ("result/status", "error"),
            ("result/passed", False), ("result/concurrency", [32]), ("result/serving_flags", {}),
            ("result/patch/sha256", "0"*64), ("result/gpus", 4), ("result/image", "changed"),
            ("result/weights", "changed"), ("result/shape/turns", 11), ("result/reps", 3),
            ("result/correctness/passed", False), ("result/later_turns/passed", False), ("result/later_turns/ran", False),
            ("result/later_turns/insufficient", True), ("result/short_outputs", 2),
            ("result/rungs/0/reps", 3), ("result/rungs/0/truncated", True), ("result/rungs/0/concurrency", 4),
            ("result/rungs/0/tok_s_per_gpu", 17512.9*1.008), ("result/rungs/0/tok_s_per_gpu", float("nan")),
            ("result/vs_stock/rungs/0/at_stock_point", 1.424*1.008), ("result/vs_stock/rungs/0/at_stock_point", None),
            ("result/vs_stock/rungs/0/concurrency", 4), ("result/vs_stock/indicative", False),
            ("result/vs_stock/ttft_ratio_limit", 3), ("result/vs_stock/rungs/0/ttft_first_p90", 2.01),
            ("result/rungs/0/ttft_later_p90_s", 1.0), ("result/rungs/0/ttft_p90_s", float("inf")),
            ("result/vs_stock/rungs/0/stock/ttft_first_p90_s", 0),
            ("result/gpu_memory/limit_gb", 3), ("result/gpu_memory/over_stock_gb", 2.01),
            ("result/gpu_memory/over_stock_gb", float("nan")),
        ]
        for path,value in mutations:
            with self.subTest(path=path,value=value):
                f=self.fixture()
                field(f.report, [int(k) if k=="0" else k for k in path.split("/")], value)
                h=f.choice()
                self.assertEqual((h["candidate"],h["sha256"],h["flags"]),("v198",runner.V198.sha256,runner.V198.flags))
                self.assertNotIn("v204_comparison",h["evidence"])
        for state in ("queued","running","failed"):
            f=self.fixture()
            f.report.update(status=state,result=None,finished_at=None)
            self.assertEqual(f.choice()["candidate"],"v198")

    def test_gain_can_be_cross_node_and_later_latency_gate(self):
        f=self.fixture()
        f.report["result"]["node"]="different-H200-node"
        self.assertEqual(f.choice()["candidate"],"v229")
        f=self.fixture()
        f.report["result"]["rungs"][0]["ttft_first_p90_s"]=170.
        self.assertEqual(f.choice()["candidate"],"v198")

    def test_existing_slot_failure_blocks_both_choices(self):
        actions=[
            lambda f: f.quota["limits"]["checks"].update(left=2),
            lambda f: f.quota.update(resets_at="wrong"),
            lambda f: (f.runs/(runner.REQUIRED[0][2]+".submit.txt")).unlink(),
            lambda f: (f.runs/(runner.REQUIRED[1][2]+".submit.txt")).unlink(),
            lambda f: f.jobs[previous.parent.IDS[2]].update(status="done"),
            lambda f: (f.runs/(runner.COMPETITORS[0][2]+".submit.txt")).write_text("uncertain private attempt"),
            lambda f: (f.runs/(runner.LEGACY_NAME+".submit.txt")).touch(),
            lambda f: (f.runs/(runner.NAME+".submit.txt")).touch(),
        ]
        for i, change in enumerate(actions):
            for qualifies in (True,False):
                f=self.fixture()
                if not qualifies:
                    f.report["result"]["passed"]=False
                change(f)
                with self.subTest(action=i,qualifies=qualifies):
                    self.assertFalse(f.offer().startswith("submitted "))
                    self.assertFalse(f.attempts)

    def test_missing_ambiguous_and_moving_receipt(self):
        for mode in ("missing","ambiguous","wrong_flags"):
            f=self.fixture()
            receipt=f.runs/(selection.V204_RECEIPT+".submit.txt")
            if mode=="missing":
                receipt.unlink()
            elif mode=="ambiguous":
                shutil.copyfile(receipt,f.runs/"duplicate_witness.submit.txt")
            else:
                h,body=receipt.read_text().split("\n",1)
                receipt.write_text(json.dumps(json.loads(h)|{"flags":{}})+"\n"+body)
            self.assertEqual(f.choice()["candidate"],"v198")
        for initial in ("v198","v229"):
            f=self.fixture()
            good=f.report["result"]["vs_stock"]["rungs"][0]["at_stock_point"]
            f.report["result"]["vs_stock"]["rungs"][0]["at_stock_point"]=good if initial=="v229" else 1.
            def switch(n):
                if n==1:
                    f.report["result"]["vs_stock"]["rungs"][0]["at_stock_point"]=1. if initial=="v229" else good
            f.on_quota=switch
            self.assertFalse(f.offer().startswith("submitted "))
            self.assertFalse(f.attempts or f.receipt.exists())

    def test_last_quota_and_final_control_recheck_v204_binding(self):
        for point in ("final_quota", "final_control"):
            f=self.fixture()
            receipt = f.runs / (selection.V204_RECEIPT + ".submit.txt")
            original = f.execute
            def execute(cmd, **kwargs):
                response = original(cmd, **kwargs)
                last_quota = f.events.count(["serve", "status", "--json"]) == 2
                if last_quota and ((point == "final_quota" and cmd[1:] == ["serve", "status", "--json"])
                                   or (point == "final_control" and cmd[1:] == ["serve","status",previous.parent.IDS[2],"--json"])):
                    receipt.unlink(missing_ok=True)
                return response
            f.execute = execute
            self.assertFalse(f.offer().startswith("submitted "))
            self.assertFalse(f.attempts)

    def test_old_candidate_does_not_query_v204(self):
        f=self.fixture()
        self.assertTrue(f.offer(runner.V198).startswith("submitted "))
        self.assertFalse(any(selection.V204_JOB in event for event in f.events))
        self.assertEqual(runner.plan(candidate=runner.V198),json.loads((RUNS/"nov20_v198_runner.plan.json").read_text()))

    def test_preview_has_no_side_effects(self):
        with patch.object(subprocess,"run",side_effect=AssertionError("unexpected external command")):
            plan=runner.plan(candidate=runner.V229)
        self.assertEqual(plan["maximum_private_checks"],1)
        self.assertEqual(plan["process_lock"],"nov20_v187_runner.lock")
        self.assertEqual(plan["receipt"],runner.NAME+".submit.txt")
        self.assertEqual(plan["fallback"],"v198")


if __name__ == "__main__":
    unittest.main()
