"""Offline runner scenarios using synthetic CLI responses and temporary receipts."""
import json
import subprocess
import threading
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import submit_nov19_decode_checks as r


class MatchTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="okc-nov19-fixture-")
        self.addCleanup(temp.cleanup)
        self.runs = Path(temp.name)
        self.jobs = {}
        self.commands = []
        self.left = 3
        self.quota = 0
        self.quota_reset = r.RESET
        self.on_quota = lambda: None
        self.at = r.START
        self.submit_result = 0
        self.ids = {c: f"{i:016x}" for i, c in enumerate(("v72a","v76a","v80a","v84a","v79a","v87","v116b","v95","v98","v99","v120b","v101b","v147","v126"), 1)}
        self.jobs[r.V68_JOB] = {"id": r.V68_JOB, "board":"serving", "mode":"check",
                                "status":"queued", "result":None}
        self.ids[r.V87.name] = r.V87_JOB
        self.v64_path = self.runs / f"{r.V64_JOB}.json"
        self.v64_record = {"id": r.V64_JOB, "status": "done", "mode": "run", "board": "serving",
                           "serving_flags": r.BASE_FLAGS, "result": {
                               "status": "ok", "passed": True, "concurrency": [1,4,16,32],
                               "serving_flags": r.BASE_FLAGS, "patch": {"sha256": r.V64_SHA}}}
        self.v64_path.write_text(json.dumps(self.v64_record))
        self.ids[r.V93.name] = r.V93_JOB
        self.job(r.V93, state="queued")
        self.job(r.V87, state="failed")

    def job(self, candidate, state="done", **changes):
        job = self.ids[candidate.name]
        detail = {"status": "ok", "passed": True, "concurrency": [r.check_load(candidate)],
                  "patch": {"sha256": candidate.sha256}, "serving_flags": candidate.flags}
        self.jobs[job] = {"id": job, "board": "serving", "mode": "check", "status": state,
                          "result": detail | changes}
        meta = {"candidate": candidate.name, "sha256": candidate.sha256,
                "flags": candidate.flags, "users": r.check_load(candidate)}
        self.receipt(candidate).write_text(json.dumps(meta) + f"\njob {job}\nCLI exit code: 0\n")
        return job

    def receipt(self, candidate):
        return self.runs / (r.prefix(candidate) + ".submit.txt")

    def execute(self, cmd, **kwargs):
        self.assertGreater(kwargs["timeout"], 0)
        self.commands.append(cmd)
        args = cmd[1:]
        if args == ["serve", "status", "--json"]:
            self.on_quota()
            return subprocess.CompletedProcess(cmd, self.quota, json.dumps({
                "resets_at": self.quota_reset, "limits": {"checks": {"left": self.left}}
            }), "")
        if args[:2] == ["serve", "status"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(self.jobs[args[2]]), "")
        self.assertEqual(args[:2], ["serve", "check"])
        candidate = next(c for c in r.CANDIDATES if c.directory in args[2])
        self.assertEqual(args[args.index("--users")+1], str(r.check_load(candidate)))
        self.assertEqual(args[-1], "--no-wait")
        self.left -= 1
        job = self.ids[candidate.name]
        self.jobs[job] = {"id": job, "board": "serving", "mode": "check", "status": "queued",
                          "users": r.check_load(candidate), "serving_flags": candidate.flags, "result": None}
        return subprocess.CompletedProcess(cmd, self.submit_result, "job " + job, "")

    def offer(self, candidate):
        return r.run_candidate(candidate, runs=self.runs, okc="mock", execute=self.execute,
                               clock=lambda: self.at)

    def test_frozen_plan(self):
        checks = r.plan()["checks"]
        self.assertEqual(len(checks), 3)
        self.assertEqual(r.plan()["maximum_private_checks"], 3)
        self.assertEqual(checks[0]["decision_receipt"], "v87_nov18_check4")
        alternatives = checks[0]["alternatives"]
        self.assertEqual([(v["candidate"], v["users"], len(v["flags"])) for v in alternatives],
                         [("v116b", 4, 9), ("v120b", 32, 11), ("v80a", 16, 7)])
        self.assertEqual(alternatives[0]["flags"], r.V87.flags)
        self.assertEqual(checks[1]["decision_receipt"], "v93_nov18_check16")
        self.assertEqual(checks[1]["v93_job"], r.V93_JOB)
        self.assertEqual([(c["candidate"], c["users"], len(c["flags"])) for c in checks[1]["alternatives"]],
                         [("v98", 16, 10), ("v147", 16, 4), ("v95", 16, 4)])
        self.assertEqual(checks[2]["decision_receipt"], r.prefix(r.V93))
        self.assertEqual(checks[2]["v93_job"], r.V93_JOB)
        self.assertEqual([(v["candidate"], v["users"]) for v in checks[2]["alternatives"]],
                         [("v101b", 16), ("v79a", 16), ("v126", 1)])
        self.assertEqual(checks[2]["alternatives"][0]["flags"],
                         r.V93.flags | {"--speculative-dflash-adaptive-tree": "", "--speculative-dflash-adaptive-spine": ""})
        self.assertEqual(checks[2]["alternatives"][1]["flags"], r.V76a.flags | {"--speculative-num-draft-tokens": "12"})
        self.assertEqual([r.prefix(c) for c in r.CANDIDATES],
                         ["v116b_nov19_check4", "v120b_nov19_check32", "v80a_nov19_check16", "v98_nov19_check16", "v147_nov19_check16", "v95_nov19_check16",
                          "v101b_nov19_check16", "v79a_nov19_check16", "v126_nov19_check1"])
        self.assertEqual(r.START.isoformat(), "2026-11-19T00:03:00+00:00")
        self.assertEqual(r.END.hour,22)

    def test_unattempted_v72a_v76a_and_queued_v68(self):
        self.assertTrue(self.offer(r.V80a).startswith("submitted"))

    def test_known_v68_failure_blocks_pair(self):
        self.jobs[r.V68_JOB]["status"]="failed"
        self.assertTrue(self.offer(r.V80a).startswith("failed"))
        self.assertTrue(self.offer(r.V98).startswith("waiting"))
        self.assertEqual(self.left,3)

    def test_unknown_v68_identity_waits(self):
        self.jobs[r.V68_JOB]["id"]="other"
        self.assertTrue(self.offer(r.V80a).startswith("waiting"))
        self.assertFalse(self.receipt(r.V80a).exists())

    def test_known_v72a_skip_blocks_unattempted_v76a(self):
        (self.runs / (r.prefix(r.V72a)+".skip.json")).write_text("{}")
        self.assertTrue(self.offer(r.V80a).startswith("failed"))

    def test_known_v72a_failure_with_v76a_pending(self):
        self.job(r.V72a,state="done",passed=False)
        self.job(r.V76a,state="queued")
        self.assertTrue(self.offer(r.V80a).startswith("failed"))

    def test_ambiguous_v72a_waits(self):
        self.receipt(r.V72a).write_text("network timeout")
        self.assertTrue(self.offer(r.V80a).startswith("waiting"))

    def test_missing_parent_quota(self):
        self.job(r.V76a)
        for left in [0, True, None]:
            self.left = left
            self.assertEqual(self.offer(r.V80a), "quota unavailable")
        self.assertFalse(self.receipt(r.V80a).exists())

    def test_quota_failure_and_window(self):
        self.job(r.V76a)
        self.quota = 1
        self.assertEqual(self.offer(r.V80a), "quota request failed")
        self.quota, self.quota_reset = 0, "wrong day"
        self.assertEqual(self.offer(r.V80a), "quota unavailable")
        self.at = r.START-timedelta(seconds=1)
        self.assertEqual(self.offer(r.V80a), "outside submission window")
        self.at = r.END
        self.assertEqual(self.offer(r.V80a), "outside submission window")

    def test_unconfirmed_receipt(self):
        self.job(r.V76a)
        path = self.receipt(r.V76a)
        path.write_text(path.read_text().replace("CLI exit code: 0","ambiguous"))
        self.assertIn("unconfirmed", self.offer(r.V80a))
        self.assertTrue(all(cmd[1:3]==["serve","status"] for cmd in self.commands))

    def test_wrong_flags_source_or_load(self):
        for change in ({"serving_flags": {}},{"patch":{"sha256":"different"}},
                       {"concurrency":[4]},{"passed":False}):
            self.job(r.V76a, **change)
            self.assertTrue(self.offer(r.V80a).startswith("failed"))
            (self.runs / (r.prefix(r.V80a)+".skip.json")).unlink()
        self.assertFalse(self.receipt(r.V80a).exists())

    def test_v93_failure_selects_independent_prefill(self):
        self.job(r.V93, state="failed")
        self.assertTrue(self.offer(r.V98).startswith("waiting"))
        self.assertTrue(self.offer(r.V95).startswith("submitted"))
        self.assertEqual(self.offer(r.V98), "attempt recorded")

    def test_parent_still_queued(self):
        self.job(r.V72a,state="running")
        self.job(r.V76a, state="queued")
        self.assertTrue(self.offer(r.V80a).startswith("submitted"))

    def test_v98_independent_when_no_v80a_attempted(self):
        self.job(r.V93)
        self.job(r.V76a)
        self.assertTrue(self.offer(r.V98).startswith("submitted"))
        self.assertTrue(self.receipt(r.V98).exists())

    def test_three_private_submissions_in_order(self):
        self.job(r.V76a)
        self.assertTrue(self.offer(r.V80a).startswith("submitted"))
        self.job(r.V80a,state="running")
        self.assertTrue(self.offer(r.V147).startswith("submitted"))
        self.assertEqual(self.left,1)
        self.job(r.V147, state="queued")
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))
        self.assertEqual(self.left,0)
        self.assertEqual(self.offer(r.V79a), "attempt recorded")
        self.assertEqual(self.offer(r.V80a),"attempt recorded")
        self.assertEqual(self.offer(r.V147),"attempt recorded")
        checks = [cmd for cmd in self.commands if cmd[1:3] == ["serve","check"]]
        self.assertEqual(len(checks),3)
        for cmd,candidate in zip(checks,(r.V80a, r.V147, r.V79a)):
            self.assertEqual(cmd.count("--flag"),len(candidate.flags))
            for k,v in candidate.flags.items():
                self.assertIn(k[2:]+("="+v if v else ""),cmd)

    def test_v79a_independent_of_failed_or_unattempted_pair(self):
        self.job(r.V80a, passed=False)
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))
        self.assertEqual(self.left, 2)

    def test_v79a_shared_failure_and_ambiguous_receipts(self):
        self.receipt(r.V72a).write_text("partial receipt")
        self.assertIn("unconfirmed", self.offer(r.V79a))
        self.receipt(r.V72a).unlink()
        self.job(r.V76a, concurrency=[4])
        self.assertTrue(self.offer(r.V79a).startswith("failed"))
        self.assertEqual(self.left, 3)

    def test_v79a_does_not_repeat_old_day_attempt(self):
        path = self.runs / "v79a_nov18_check16.submit.txt"
        path.write_text("stale runner lost confirmed CLI response")
        self.assertEqual(self.offer(r.V79a), "attempt recorded")
        self.assertFalse(self.commands)

    def test_v79a_queued_v68_and_no_other_receipts(self):
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))
        self.assertTrue(self.receipt(r.V79a).is_file())
        meta = json.loads(self.receipt(r.V79a).read_text().splitlines()[0])
        self.assertEqual((meta["users"], meta["sha256"], meta["flags"]),
                         (16, r.V79a.sha256, r.V79a.flags))

    def test_dynamic_prefill_parent_failure(self):
        job = self.job(r.V93, state="running")
        self.on_quota = lambda: self.jobs[job].update(status="failed")
        self.offer(r.V147)
        self.assertFalse(self.receipt(r.V147).exists())
        self.assertEqual(self.left, 3)
        self.assertTrue(self.offer(r.V95).startswith("submitted"))

    def test_ambiguous_cli_not_retried(self):
        self.job(r.V93)
        self.job(r.V76a)
        self.submit_result=71
        self.assertIn("no retry",self.offer(r.V80a))
        self.assertEqual(self.offer(r.V80a),"attempt recorded")
        self.submit_result = 0
        self.assertTrue(self.offer(r.V98).startswith("submitted"))
        self.assertEqual(self.left,1)

    def test_first_slot_tracks_v87_and_exact_v120b_payload(self):
        for state, selected in (("queued",r.V120b),("running",r.V120b),("done",r.V116b)):
            with self.subTest(state=state):
                self.job(r.V87, state=state)
                for other in r.ALTERNATIVES:
                    if other != selected:
                        self.assertTrue(self.offer(other).startswith("waiting"))
                self.assertTrue(self.offer(selected).startswith("submitted"))
                for other in r.ALTERNATIVES:
                    self.assertEqual(self.offer(other), "attempt recorded")
                meta = json.loads(self.receipt(selected).read_text().splitlines()[0])
                self.assertEqual((meta["candidate"],meta["flags"],meta["users"],meta["sha256"]),
                                 (selected.name,selected.flags,r.check_load(selected),selected.sha256))
                if selected == r.V120b:
                    self.assertEqual(meta["evidence"]["comparison_job"],r.V64_JOB)
                    self.assertEqual(meta["evidence"]["shared_jobs"],[])
                    self.assertEqual(meta["evidence"]["decision_receipt"],r.prefix(r.V87))
                self.receipt(selected).unlink()

    def test_v120b_independent_of_v68_and_optional_regressions(self):
        self.job(r.V87,state="queued")
        self.jobs[r.V68_JOB]["status"]="failed"
        self.job(r.V72a,passed=False)
        self.job(r.V76a,state="failed")
        self.assertTrue(self.offer(r.V120b).startswith("submitted"))
        polled=[cmd[3] for cmd in self.commands if len(cmd)==5 and cmd[1:3]==["serve","status"]]
        self.assertEqual(polled,[r.V87_JOB,r.V87_JOB])

    def test_v120b_rejects_malformed_scored_parent_without_consuming_slot(self):
        from copy import deepcopy
        self.job(r.V87,state="running")
        for field,bad in (("id","different"),("status","queued"),("board","other"),("mode","check"),
                          ("serving_flags",{}),("result",None),("result.passed",1),
                          ("result.status","failed"),("result.concurrency",[32]),
                          ("result.patch",{}),("result.patch",{"sha256":"0"*64}),("result.serving_flags",{})):
            with self.subTest(field=field,bad=bad):
                invalid=deepcopy(self.v64_record)
                if field.startswith("result."):
                    invalid["result"][field[7:]]=bad
                else:
                    invalid[field]=bad
                self.v64_path.write_text(json.dumps(invalid))
                self.assertTrue(self.offer(r.V120b).startswith("waiting"))
        self.v64_path.write_text("invalid")
        self.assertTrue(self.offer(r.V120b).startswith("waiting"))
        self.v64_path.unlink()
        self.assertTrue(self.offer(r.V120b).startswith("waiting"))
        self.assertFalse(self.receipt(r.V120b).exists())
        self.assertEqual(self.left,3)
        self.job(r.V87)
        self.assertTrue(self.offer(r.V116b).startswith("submitted"))

    def test_v120b_revalidates_frozen_parent_after_quota(self):
        self.job(r.V87,state="running")
        self.on_quota=self.v64_path.unlink
        self.assertEqual(self.offer(r.V120b),"quota request failed")
        self.assertFalse(self.receipt(r.V120b).exists())
        self.assertEqual(self.left,3)

    def test_pending_v87_completes_before_independent_check(self):
        self.job(r.V87,state="running")
        self.on_quota=lambda: self.job(r.V87)
        self.assertEqual(self.offer(r.V120b),"quota request failed")
        self.assertFalse(self.receipt(r.V120b).exists())
        self.assertTrue(self.offer(r.V116b).startswith("submitted"))

    def test_v120b_cannot_claim_unknown_or_wrong_v87(self):
        self.job(r.V87,state="queued")
        path=self.receipt(r.V87)
        good=path.read_text()
        for wrong in (good.replace(r.V87_JOB,"1234567890abcdef"),good.replace(r.V87.sha256,"0"*64),
                      good.replace('"users": 4','"users": 16'),"ambiguous"):
            path.write_text(wrong)
            self.assertTrue(self.offer(r.V120b).startswith("waiting"))
        path.unlink()
        self.assertTrue(self.offer(r.V120b).startswith("waiting"))
        self.assertEqual(self.left,3)

    def test_old_first_slot_attempt_blocks_v120b(self):
        self.job(r.V87,state="running")
        for past in (r.V89,r.V116,r.V116b,r.V80a):
            self.receipt(past).write_text("uncertain old attempt")
            self.assertEqual(self.offer(r.V120b),"attempt recorded")
            self.receipt(past).unlink()
        self.assertEqual(self.left,3)

    def test_prior_paged_checks_occupy_first_slot_across_restart(self):
        for legacy in (r.V100c, r.V106):
            self.assertEqual(r.prefix(legacy), f"{legacy.name}_nov19_check32")
            for contents in ("", "prior job status is unknown", "job 123456789abcdef0\nCLI exit code: 71\n"):
                self.receipt(legacy).write_text(contents)
                for state in ("queued", "done", "failed"):
                    self.job(r.V87, state=state)
                    for candidate in r.ALTERNATIVES:
                        self.assertEqual(self.offer(candidate), "attempt recorded")
                self.receipt(legacy).unlink()
            with self.assertRaises(ValueError):
                self.offer(legacy)
        self.assertFalse(self.commands)
        self.assertEqual(self.left, 3)

    def test_any_historical_v89_receipt_occupies_four_user_slot(self):
        self.assertEqual(r.prefix(r.V89), "v89_nov19_check4")
        for old in ("", "orphaned attempt without exit status", "job abcd0123456789ef\nCLI exit code: 71\n"):
            self.receipt(r.V89).write_text(old)
            for state in ("queued", "done", "failed"):
                self.job(r.V87, state=state)
                for candidate in r.ALTERNATIVES:
                    self.assertEqual(self.offer(candidate), "attempt recorded")
            self.receipt(r.V89).unlink()
        self.assertFalse(self.commands)
        self.assertEqual(self.left, 3)
        with self.assertRaises(ValueError):
            self.offer(r.V89)

    def test_previous_v116_at_four_users_blocks_all_first_slot_choices(self):
        self.assertEqual(r.prefix(r.V116), "v116_nov19_check4")
        for data in ("", "previous CLI attempt with no confirmed job", "job 123456789abcdef0\nCLI exit code: 71\n"):
            self.receipt(r.V116).write_text(data)
            for state in ("running", "done", "failed"):
                self.job(r.V87, state=state)
                for candidate in r.ALTERNATIVES:
                    self.assertEqual(self.offer(candidate), "attempt recorded")
            self.receipt(r.V116).unlink()
        with self.assertRaises(ValueError):
            self.offer(r.V116)
        self.assertFalse(self.commands)
        self.assertEqual(self.left, 3)

    def test_late_v89_or_v116_receipt_during_final_v87_pass_blocks_v116b(self):
        self.job(r.V87)
        for old in (r.V89, r.V116):
            with self.subTest(old=old.name):
                calls = []
                def recover(command, **kwargs):
                    if command[1:] == ["serve", "status", r.V87_JOB, "--json"]:
                        calls.append(command)
                        if len(calls) == 2:
                            self.receipt(old).write_text("previous four-user attempt")
                    return self.execute(command, **kwargs)
                result = r.run_candidate(r.V116b, runs=self.runs, okc="mock", execute=recover,
                                         clock=lambda: self.at)
                self.assertEqual(result, "quota request failed")
                self.assertEqual(len(calls), 2)
                self.assertEqual(self.offer(r.V116b), "attempt recorded")
                self.assertFalse(self.receipt(r.V116b).exists())
                self.assertEqual(self.left, 3)
                self.receipt(old).unlink()

    def test_v116b_frozen_patch_has_matched_v87_flags_and_four_users(self):
        import hashlib
        directory = r.ROOT / "serving_v0523/candidates" / r.V116b.directory
        receipt = json.loads((directory / "validation.json").read_text())
        self.assertEqual(r.V116b.sha256, receipt["patch_sha256"])
        self.assertEqual(r.V116b.flags, receipt["launch_flags"])
        self.assertNotEqual(r.V116b.sha256, r.V116.sha256)
        self.assertEqual(r.V116b.flags, r.V116.flags)
        self.assertEqual(r.V116b.flags, r.V89.flags)
        self.assertEqual(r.V116b.flags, r.V87.flags)
        self.assertEqual(r.V116b.sha256, hashlib.sha256((directory / "patch.diff").read_bytes()).hexdigest())
        self.assertEqual(r.plan()["checks"][0]["when_passed"], "v116b")
        self.job(r.V87)
        self.assertTrue(self.offer(r.V116b).startswith("submitted"))
        command = self.commands[-1]
        flags = [item for name, value in sorted(r.V87.flags.items())
                 for item in ("--flag", name[2:] + ("=" + value if value else ""))]
        self.assertEqual(command, ["mock", "serve", "check", str(directory / "patch.diff"),
                                   "--users", "4", *flags, "--no-wait"])

    def test_recovered_legacy_first_slot_attempt_during_quota(self):
        for legacy in (r.V100c, r.V106, r.V89, r.V116, r.V111, r.V120):
            for state, selected in (("running", r.V120b), ("done", r.V116b), ("failed", r.V80a)):
                self.job(r.V87, state=state)
                self.on_quota = lambda: self.receipt(legacy).write_text("recovered uncertain attempt")
                self.assertEqual(self.offer(selected), "quota request failed")
                self.assertFalse(self.receipt(selected).exists())
                self.assertEqual(self.offer(r.V120b), "attempt recorded")
                self.receipt(legacy).unlink()
        self.assertEqual(self.left, 3)

    def test_legacy_v106_receipt_appears_during_final_v87_recheck(self):
        self.job(r.V87, state="queued")
        calls = []
        def recover(cmd, **kwargs):
            if cmd[1:] == ["serve", "status", r.V87_JOB, "--json"]:
                calls.append(cmd)
                if len(calls) == 2:
                    self.receipt(r.V106).write_text("previous check32 submit attempt")
            return self.execute(cmd, **kwargs)
        result = r.run_candidate(r.V120b, runs=self.runs, okc="mock", execute=recover,
                                 clock=lambda: self.at)
        self.assertEqual(result, "quota request failed")
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.receipt(r.V120b).exists())
        self.assertEqual(self.left, 3)

    def test_recovered_v111_first_slot_attempt_survives_migration_and_restart(self):
        self.assertEqual(r.prefix(r.V111), "v111_nov19_check32")
        self.assertNotIn(r.V111, r.CANDIDATES)
        with self.assertRaises(ValueError):
            self.offer(r.V111)
        for content in ("", "ambiguous attempt", "job 123456789abcdef0\nCLI exit code: 0\n"):
            self.receipt(r.V111).write_text(content)
            for state in ("running", "done", "failed"):
                self.job(r.V87, state=state)
                for candidate in r.ALTERNATIVES:
                    self.assertEqual(self.offer(candidate), "attempt recorded")
            self.assertFalse(self.commands)
        self.assertEqual(self.left, 3)
        self.assertTrue(self.offer(r.V147).startswith("submitted"))
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))
        self.assertEqual(self.left, 1)

    def test_v120_attempt_prevents_a_second_private_check_after_correction(self):
        self.assertEqual(r.prefix(r.V120), "v120_nov19_check32")
        self.assertNotIn(r.V120, r.CANDIDATES)
        self.assertEqual(r.V120b.flags, r.V120.flags)
        self.receipt(r.V120).write_text("original check was attempted")
        for candidate in r.ALTERNATIVES:
            self.assertEqual(self.offer(candidate), "attempt recorded")
        self.assertFalse(self.commands)
        self.assertEqual(self.left, 3)

    def test_v120b_exact_frozen_payload_and_cli_flags(self):
        import hashlib
        root = r.ROOT / "serving_v0523/candidates"
        self.assertEqual(r.V111.flags, r.V106.flags | {"--speculative-dflash-sp-prefill-context": ""})
        self.assertEqual(r.V120b.flags, r.V111.flags | {"--speculative-dflash-direct-prefix-kv": "",
                                                       "--speculative-dflash-parallel-verify-context": ""})
        manifest = json.loads((root / r.V120b.directory / "validation.json").read_text())
        self.assertEqual(manifest["launch_flags"], r.V120b.flags)
        self.assertEqual(manifest["patch_sha256"], r.V120b.sha256)
        self.assertEqual(r.V106.flags, r.V100c.flags)
        self.assertEqual(r.plan()["checks"][0]["legacy_attempt_receipts"],
                         [r.prefix(r.V100c), r.prefix(r.V106), r.prefix(r.V89), r.prefix(r.V116), r.prefix(r.V111), r.prefix(r.V120)])
        for candidate in (r.V100c, r.V106, r.V111, r.V120, r.V120b):
            self.assertEqual(hashlib.sha256((root / candidate.directory / "patch.diff").read_bytes()).hexdigest(),
                             candidate.sha256)
        self.assertEqual(len({c.sha256 for c in (r.V100c, r.V106, r.V111, r.V120, r.V120b)}), 5)
        self.job(r.V87, state="running")
        self.assertTrue(self.offer(r.V120b).startswith("submitted"))
        command = self.commands[-1]
        expected_flags = [
            "chunked-prefill-size=16320", "cuda-graph-max-bs-prefill=16384", "enable-layernorm-sp",
            "max-prefill-tokens=16320", "moe-runner-backend=humming", "page-size=160",
            "speculative-dflash-direct-prefix-kv", "speculative-dflash-parallel-verify-context",
            "speculative-dflash-sp-prefill-context", "speculative-moe-runner-backend=marlin",
            "swa-full-tokens-ratio=0.25",
        ]
        self.assertEqual(command, ["mock", "serve", "check", str(root/r.V120b.directory/"patch.diff"), "--users", "32",
                                   *(item for flag in expected_flags for item in ("--flag", flag)), "--no-wait"])

    def test_exact_v87_crash_selects_v120b(self):
        self.job(r.V87, status="crashed", passed=False)
        self.assertEqual(r.plan()["checks"][0]["when_crashed_and_v64_passed"], "v120b")
        self.assertTrue(self.offer(r.V116b).startswith("waiting"))
        self.assertTrue(self.offer(r.V80a).startswith("waiting"))
        self.assertTrue(self.offer(r.V120b).startswith("submitted"))
        self.assertEqual(self.left, 2)

    def test_v87_crash_requires_provenance_and_valid_v64(self):
        job = self.job(r.V87, status="crashed", passed=False)
        result = self.jobs[job]["result"].copy()
        for key in ("patch", "serving_flags", "concurrency", "passed"):
            self.jobs[job]["result"] = {k: v for k, v in result.items() if k != key}
            self.assertTrue(self.offer(r.V120b).startswith("waiting"))
        self.jobs[job]["result"] = result
        self.v64_path.unlink()
        self.assertTrue(self.offer(r.V120b).startswith("waiting"))
        self.assertEqual(self.left, 3)

    def test_v80a_for_exact_failure_or_skip_and_no_v116b(self):
        self.job(r.V87, passed=False)
        self.assertTrue(self.offer(r.V116b).startswith("waiting"))
        self.assertTrue(self.offer(r.V80a).startswith("submitted"))
        self.receipt(r.V80a).unlink()
        self.receipt(r.V87).unlink()
        skip = self.runs / (r.prefix(r.V87) + ".skip.json")
        skip.write_text(json.dumps({"candidate":"v87","reason":"shared dependency skip"}))
        self.assertTrue(self.offer(r.V116b).startswith("waiting"))
        self.assertTrue(self.offer(r.V80a).startswith("submitted"))
        self.assertFalse(self.receipt(r.V116b).exists())

    def test_v87_mismatch_never_guesses_but_unrelated_jobs_run(self):
        for changes in ({"patch":{"sha256":"wrong"}},{"serving_flags":{}},{"concurrency":[16]}):
            self.job(r.V87, passed=False, **changes)
            self.assertTrue(self.offer(r.V116b).startswith("waiting"))
            self.assertTrue(self.offer(r.V80a).startswith("waiting"))
        self.receipt(r.V87).write_text("ambiguous CLI")
        self.assertTrue(self.offer(r.V116b).startswith("waiting"))
        self.assertTrue(self.offer(r.V80a).startswith("waiting"))
        self.assertTrue(self.offer(r.V147).startswith("submitted"))
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))
        self.assertEqual(self.left,1)
        self.job(r.V87)
        self.assertTrue(self.offer(r.V116b).startswith("submitted"))
        self.assertEqual(self.left,0)
        self.assertFalse(self.receipt(r.V80a).exists())

    def test_alternative_checked_again_after_quota(self):
        job=self.job(r.V87, state="running")
        self.on_quota=lambda: self.jobs[job].update(status="failed")
        self.offer(r.V120b)
        self.assertFalse(self.receipt(r.V120b).exists())
        self.assertEqual(self.left,3)
        self.assertTrue(self.offer(r.V80a).startswith("submitted"))

    def test_ambiguous_alternative_attempt_permanently_occupies_slot(self):
        self.job(r.V87)
        self.submit_result=71
        self.assertIn("no retry",self.offer(r.V116b))
        self.job(r.V87,state="failed")
        self.assertEqual(self.offer(r.V80a), "attempt recorded")
        self.assertEqual(self.offer(r.V116b), "attempt recorded")
        self.assertEqual(self.left,2)

    def test_conflicting_or_unverifiable_v87_provenance(self):
        job=self.job(r.V87,state="queued")
        self.jobs[job]["serving_flags"]={}
        self.assertTrue(self.offer(r.V116b).startswith("waiting"))
        self.jobs[job]["serving_flags"]=r.V87.flags
        self.jobs[job]["users"]=16
        self.assertTrue(self.offer(r.V80a).startswith("waiting"))
        self.receipt(r.V87).unlink()
        self.assertTrue(self.offer(r.V116b).startswith("waiting"))
        skip=self.runs/(r.prefix(r.V87)+".skip.json")
        skip.write_text("{")
        self.assertTrue(self.offer(r.V80a).startswith("waiting"))
        self.job(r.V87,state="failed")
        skip.write_text(json.dumps({"candidate":"v87","reason":"skip"}))
        self.assertTrue(self.offer(r.V80a).startswith("waiting"))
        self.assertEqual(self.left,3)

    def test_two_process_alternatives_cannot_submit_together(self):
        self.job(r.V87,state="running")
        entered, release = threading.Event(), threading.Event()
        def hold_quota():
            entered.set()
            self.assertTrue(release.wait(3))
        self.on_quota=hold_quota
        results=[]
        thread=threading.Thread(target=lambda: results.append(self.offer(r.V120b)))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertIn("lock",self.offer(r.V80a))
        finally:
            release.set()
            thread.join(3)
        self.assertTrue(results[0].startswith("submitted"))
        self.job(r.V87,state="failed")
        self.assertEqual(self.offer(r.V80a),"attempt recorded")
        self.assertEqual(self.left,2)


    def test_prefill_rejects_legacy_v84a_attempt_and_unknown_candidate(self):
        self.receipt(r.V84a).write_text("ambiguous old runner attempt")
        for c in r.PREFILL_ALTERNATIVES:
            self.assertEqual(self.offer(c), "attempt recorded")
        with self.assertRaises(ValueError):
            self.offer(r.V84a)
        self.assertFalse(self.commands)

    def test_v68_failure_admits_v95_and_blocks_v98(self):
        self.jobs[r.V68_JOB]["status"] = "failed"
        self.assertTrue(self.offer(r.V98).startswith("waiting"))
        self.assertTrue(self.offer(r.V95).startswith("submitted"))
        self.assertEqual(self.offer(r.V98), "attempt recorded")
        self.assertEqual(self.left, 2)

    def test_v93_pending_or_pass_selects_its_matched_prefill_control(self):
        for state, selected in (("queued", r.V147), ("running", r.V147), ("done", r.V98)):
            with self.subTest(state=state):
                self.job(r.V93, state=state)
                for other in r.PREFILL_ALTERNATIVES:
                    if other != selected:
                        self.assertTrue(self.offer(other).startswith("waiting"))
                self.assertTrue(self.offer(selected).startswith("submitted"))
                self.assertEqual(self.offer(r.V95), "attempt recorded")
                self.receipt(selected).unlink()

    def test_v93_unknown_evidence_defers_prefill_only(self):
        path = self.receipt(r.V93)
        orig = path.read_text()
        for content in ("", "{", orig.replace(r.V93_JOB,"ffffffffffffffff"),
                        orig.replace(r.V93.sha256, "0"*64), orig.replace("CLI exit code: 0","timeout"),
                        orig.replace('"users": 16','"users": 4'), orig.replace('"candidate": "v93"','"candidate": "v98"')):
            path.write_text(content)
            for c in r.PREFILL_ALTERNATIVES:
                self.assertTrue(self.offer(c).startswith("waiting"))
        path.unlink()
        self.assertTrue(self.offer(r.V98).startswith("waiting"))
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))
        self.assertEqual(self.left, 2)

    def test_ambiguous_v93_or_v68_live_status_blocks_both_routes(self):
        for c,job in ((r.V93,r.V93_JOB),(r.V68,r.V68_JOB)):
            saved=self.jobs[job].copy()
            for changes in ({"id":"other"},{"serving_flags":{}},{"users":4},{"mode":"run"},
                            {"result":{"passed":False,"patch":{"sha256":"wrong"}}}):
                self.jobs[job]=saved | changes
                self.assertTrue(self.offer(r.V95).startswith("waiting"))
                self.assertTrue(self.offer(r.V98).startswith("waiting"))
            self.jobs[job]=saved
        self.assertEqual(self.left,3)

    def test_unknown_v68_incomplete_result_does_not_select_fallback(self):
        self.jobs[r.V68_JOB].update(status="done",result={"status":"unknown"})
        self.assertTrue(self.offer(r.V95).startswith("waiting"))
        self.assertTrue(self.offer(r.V98).startswith("waiting"))

    def test_prefill_attempt_persists_across_flip_and_restarts(self):
        self.job(r.V93)
        self.submit_result=71
        self.assertIn("no retry", self.offer(r.V98))
        self.jobs[r.V93_JOB]["status"]="failed"
        self.assertEqual(self.offer(r.V95),"attempt recorded")
        self.assertEqual(self.offer(r.V98),"attempt recorded")
        self.assertEqual(self.left,2)

    def test_prefill_shared_control_failure_without_known_v68_v93_failure(self):
        self.job(r.V93)
        self.job(r.V76a, passed=False)
        self.assertTrue(self.offer(r.V98).startswith("waiting"))
        self.assertTrue(self.offer(r.V95).startswith("waiting"))
        self.job(r.V93, passed=False)
        self.assertTrue(self.offer(r.V95).startswith("submitted"))

    def test_prefill_concurrent_offers_are_exclusive(self):
        self.job(r.V93)
        entered, release=threading.Event(), threading.Event()
        def pause():
            entered.set()
            self.assertTrue(release.wait(3))
        self.on_quota=pause
        result=[]
        worker=threading.Thread(target=lambda:result.append(self.offer(r.V98)))
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertIn("lock",self.offer(r.V95))
        finally:
            release.set()
            worker.join(3)
        self.assertTrue(result[0].startswith("submitted"))
        self.assertEqual(self.left,2)

    def test_shared_failure_during_quota_flips_second_slot(self):
        self.job(r.V93)
        self.on_quota=lambda:self.jobs[r.V68_JOB].update(status="failed")
        self.assertEqual(self.offer(r.V98), "quota request failed")
        self.assertFalse(self.receipt(r.V98).exists())
        self.assertTrue(self.offer(r.V95).startswith("submitted"))

    def test_quota_and_deadline_bound_independent_choice(self):
        self.job(r.V93, passed=False)
        self.left=0
        self.assertEqual(self.offer(r.V95), "quota unavailable")
        self.assertFalse(self.receipt(r.V95).exists())
        self.at=r.END
        self.assertEqual(self.offer(r.V95), "outside submission window")

    def exact_v68(self):
        self.jobs[r.V68_JOB].update(status="done", result={
            "status": "ok", "passed": True, "concurrency": [16],
            "serving_flags": r.V68.flags, "patch": {"sha256": r.V68.sha256},
        })

    def test_v126_with_measured_v68_and_unresolved_v93(self):
        self.exact_v68()
        for state in ("queued", "running", "failed"):
            self.job(r.V93, state=state)
            self.assertTrue(self.offer(r.V79a).startswith("waiting"))
            self.assertTrue(self.offer(r.V126).startswith("submitted"))
            meta = json.loads(self.receipt(r.V126).read_text().splitlines()[0])
            self.assertEqual((meta["users"], meta["flags"], meta["sha256"]), (1, r.V68.flags, r.V126.sha256))
            self.assertEqual(meta["evidence"]["comparison_job"], r.V64_JOB)
            self.assertEqual(meta["evidence"]["ancestor_check_job"], r.V68_JOB)
            self.assertEqual(meta["evidence"]["shared_jobs"], [r.V68_JOB])
            self.assertEqual(self.offer(r.V101b), "attempt recorded")
            self.receipt(r.V126).unlink()

    def test_v126_independent_of_optional_tree_parent(self):
        self.exact_v68()
        self.job(r.V76a, passed=False)
        self.assertTrue(self.offer(r.V126).startswith("submitted"))

    def test_v126_requires_matched_v68_and_identifiable_v93(self):
        self.assertTrue(self.offer(r.V126).startswith("waiting"))
        self.exact_v68()
        self.receipt(r.V93).write_text("ambiguous CLI")
        for candidate in r.TREE_ALTERNATIVES:
            self.assertTrue(self.offer(candidate).startswith("waiting"))
        self.job(r.V93, state="queued")
        self.jobs[r.V68_JOB]["result"]["patch"]["sha256"] = "incorrect"
        self.assertTrue(self.offer(r.V126).startswith("failed"))
        self.assertEqual(self.left, 3)

    def test_v126_requires_scored_one_user_reference(self):
        self.exact_v68()
        self.v64_path.unlink()
        self.assertTrue(self.offer(r.V126).startswith("waiting"))
        self.assertFalse(self.receipt(r.V126).exists())

    def test_legacy_v125_in_third_slot_including_late_receipt(self):
        self.exact_v68()
        old = self.receipt(r.V125)
        self.assertEqual(old.name, "v125_nov19_check16.submit.txt")
        self.on_quota = lambda: old.write_text("earlier third-slot attempt")
        self.assertEqual(self.offer(r.V126), "quota request failed")
        self.assertFalse(self.receipt(r.V126).exists())
        self.assertEqual(self.offer(r.V126), "attempt recorded")
        self.assertEqual(self.left, 3)

    def test_v126_switch_after_quota(self):
        self.exact_v68()
        self.on_quota = lambda: self.job(r.V93)
        self.assertEqual(self.offer(r.V126), "quota request failed")
        self.assertFalse(self.receipt(r.V126).exists())
        self.assertTrue(self.offer(r.V101b).startswith("submitted"))
        self.assertEqual(self.offer(r.V126), "attempt recorded")
        self.assertEqual(self.left, 2)

    def test_exact_v93_pass_selects_v101b_for_third_and_last_check(self):
        self.job(r.V93)
        self.assertTrue(self.offer(r.V80a).startswith("submitted"))
        self.assertTrue(self.offer(r.V98).startswith("submitted"))
        self.assertTrue(self.offer(r.V79a).startswith("waiting"))
        self.assertTrue(self.offer(r.V101b).startswith("submitted"))
        self.assertEqual(self.offer(r.V79a), "attempt recorded")
        self.assertEqual(self.left, 0)
        meta = json.loads(self.receipt(r.V101b).read_text().splitlines()[0])
        self.assertEqual((meta["candidate"], meta["users"], meta["flags"], meta["sha256"]),
                         ("v101b", 16, r.V93.flags | {"--speculative-dflash-adaptive-tree": "", "--speculative-dflash-adaptive-spine": ""}, r.V101b.sha256))
        self.assertEqual(meta["evidence"]["comparison_job"], r.V93_JOB)
        self.assertEqual(meta["evidence"]["exclusive_alternatives"], ["v101b", "v79a", "v126"])

    def test_unpassed_or_wrong_v93_keeps_wide_tree(self):
        for state, fields in (
            ("queued", {}), ("running", {}), ("failed", {}), ("cancelled", {}),
            ("done", {"passed": False}), ("done", {"status": "timeout"}),
            ("done", {"patch": {"sha256": "different"}}), ("done", {"serving_flags": {}}),
            ("done", {"concurrency": [4]}), ("done", {"passed": 1}),
        ):
            with self.subTest(state=state, fields=fields):
                self.job(r.V93, state=state, **fields)
                self.assertTrue(self.offer(r.V101b).startswith("waiting"))
                self.assertTrue(self.offer(r.V79a).startswith("submitted"))
                self.receipt(r.V79a).unlink()
                self.left = 3

    def test_missing_or_conflicting_v93_receipt_keeps_wide_tree(self):
        self.job(r.V93)
        receipt = self.receipt(r.V93)
        valid = receipt.read_text()
        for invalid in (None, "", "partial CLI", valid.replace(r.V93_JOB, "f"*16),
                        valid.replace(r.V93.sha256, "0"*64), valid.replace('"users": 16', '"users": 4'),
                        valid.replace("CLI exit code: 0", "CLI exit code: 1")):
            with self.subTest(invalid=invalid):
                if invalid is None:
                    receipt.unlink()
                else:
                    receipt.write_text(invalid)
                self.assertTrue(self.offer(r.V101b).startswith("waiting"))
                self.assertTrue(self.offer(r.V79a).startswith("submitted"))
                self.receipt(r.V79a).unlink()
                self.left = 3
        receipt.write_text(valid)
        (self.runs / f"{r.prefix(r.V93)}.skip.json").write_text("{}")
        self.assertTrue(self.offer(r.V101b).startswith("waiting"))
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))

    def test_v99_pass_on_ineligible_shared_stack_is_blocked(self):
        self.job(r.V93)
        self.job(r.V76a, passed=False)
        for candidate in (r.V101b, r.V79a):
            self.assertTrue(self.offer(candidate).startswith("failed"))
        self.assertTrue(self.offer(r.V126).startswith("waiting"))
        self.assertEqual(self.left, 3)

    def test_v93_status_failure_does_not_require_wide_tree_to_wait(self):
        self.job(r.V93)
        execute = self.execute
        def v93_timeout(cmd, **kwargs):
            if cmd[1:] == ["serve", "status", r.V93_JOB, "--json"]:
                raise subprocess.TimeoutExpired(cmd, timeout=60)
            return execute(cmd, **kwargs)
        self.execute = v93_timeout
        self.assertTrue(self.offer(r.V101b).startswith("waiting"))
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))

    def test_v93_result_after_quota_promotes_native_tree_before_attempt(self):
        self.on_quota = lambda: self.job(r.V93)
        self.offer(r.V79a)
        self.assertFalse(self.receipt(r.V79a).exists())
        self.assertEqual(self.left, 3)
        self.assertTrue(self.offer(r.V101b).startswith("submitted"))

    def test_v93_rejected_after_quota_reverts_to_wide_tree_before_attempt(self):
        self.job(r.V93)
        self.on_quota = lambda: self.job(r.V93, passed=False)
        self.offer(r.V101b)
        self.assertFalse(self.receipt(r.V101b).exists())
        self.assertEqual(self.left, 3)
        self.assertTrue(self.offer(r.V79a).startswith("submitted"))

    def test_legacy_nov19_wide_tree_attempt_and_ambiguous_native_attempt_occupy_slot(self):
        self.job(r.V93)
        self.receipt(r.V79a).write_text("ambiguous Nov19 attempt")
        self.assertEqual(self.offer(r.V101b), "attempt recorded")
        self.receipt(r.V79a).unlink()
        self.submit_result = 71
        self.assertIn("no retry", self.offer(r.V101b))
        self.job(r.V93, state="failed")
        self.assertEqual(self.offer(r.V79a), "attempt recorded")
        self.assertEqual(self.left, 2)

    def test_tree_alternatives_cannot_race_during_quota_recheck(self):
        self.job(r.V93)
        entered, release = threading.Event(), threading.Event()
        def pause():
            entered.set()
            self.assertTrue(release.wait(3))
        self.on_quota = pause
        results = []
        thread = threading.Thread(target=lambda: results.append(self.offer(r.V101b)))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertIn("lock", self.offer(r.V79a))
        finally:
            release.set()
            thread.join(3)
        self.assertTrue(results[0].startswith("submitted"))
        self.assertEqual(self.offer(r.V79a), "attempt recorded")
        self.assertEqual(self.left, 2)

    def test_retired_v102_blocks_every_second_slot_candidate(self):
        self.receipt(r.V102).write_text("previous attempted submission")
        for candidate in r.PREFILL_ALTERNATIVES:
            self.assertEqual(self.offer(candidate), "attempt recorded")
        self.assertEqual(self.left, 3)
        self.assertEqual(self.commands, [])
        with self.assertRaises(ValueError):
            self.offer(r.V102)

    def test_retired_v102_receipt_appearing_at_quota_blocks_v147(self):
        self.on_quota = lambda: self.receipt(r.V102).write_text("late receipt")
        self.assertEqual(self.offer(r.V147), "quota request failed")
        self.assertFalse(self.receipt(r.V147).exists())
        self.assertEqual(self.left, 3)

    def test_v147_pending_promotes_to_v98_after_quota(self):
        self.on_quota = lambda: self.job(r.V93)
        self.assertEqual(self.offer(r.V147), "quota request failed")
        self.assertFalse(self.receipt(r.V147).exists())
        self.assertEqual(self.left, 3)
        self.assertTrue(self.offer(r.V98).startswith("submitted"))
        self.assertEqual(self.offer(r.V147), "attempt recorded")

    def test_v147_receipt_blocks_new_choices_on_restart(self):
        for exit_code in (0, 71):
            with self.subTest(exit_code=exit_code):
                self.job(r.V93, state="running")
                self.submit_result = exit_code
                self.offer(r.V147)
                self.assertTrue(self.receipt(r.V147).is_file())
                self.job(r.V93)
                self.assertEqual(self.offer(r.V98), "attempt recorded")
                self.job(r.V93, passed=False)
                self.assertEqual(self.offer(r.V95), "attempt recorded")
                self.receipt(r.V147).unlink()
        self.assertEqual(self.left, 1)

    def test_v147_v64_gate_and_post_quota_recheck(self):
        self.v64_path.write_text(self.v64_path.read_text().replace(r.V64_SHA, "0"*64))
        self.assertTrue(self.offer(r.V147).startswith("waiting"))
        self.v64_path.write_text(json.dumps(self.v64_record))
        self.on_quota = self.v64_path.unlink
        self.assertEqual(self.offer(r.V147), "quota request failed")
        self.assertFalse(self.receipt(r.V147).exists())
        self.assertEqual(self.left, 3)
        self.job(r.V93)
        self.on_quota = lambda: None
        self.assertTrue(self.offer(r.V98).startswith("submitted"))

    def test_v147_uses_scored_v64_and_remains_independent_of_optional_checks(self):
        self.job(r.V76a, passed=False)
        self.assertTrue(self.offer(r.V147).startswith("submitted"))
        meta = json.loads(self.receipt(r.V147).read_text().splitlines()[0])
        self.assertEqual((meta["users"], meta["flags"], meta["sha256"]), (16, r.BASE_FLAGS, r.V147.sha256))
        self.assertEqual(meta["evidence"]["comparison_receipt"], self.v64_path.name)
        self.assertEqual(meta["evidence"]["comparison_job"], r.V64_JOB)
        self.assertEqual(meta["evidence"]["decision_job"], r.V93_JOB)

    def test_old_v99_attempt_excludes_both_third_slot_choices(self):
        self.receipt(r.V99).write_text("ambiguous previous native-width attempt")
        for state in ("queued", "done", "failed"):
            self.job(r.V93, state=state)
            for candidate in r.TREE_ALTERNATIVES:
                self.assertEqual(self.offer(candidate), "attempt recorded")
        self.assertEqual(self.left, 3)
        self.assertFalse(self.commands)
        with self.assertRaises(ValueError):
            self.offer(r.V99)

    def test_frozen_v101b_and_v147_flags_and_payloads(self):
        expected = ((r.V101b, "launch_flags"), (r.V147, "launch_flags"))
        for candidate, key in expected:
            root = r.ROOT / "serving_v0523/candidates" / candidate.directory
            receipt = json.loads((root / "validation.json").read_text())
            self.assertEqual(candidate.flags, receipt[key])
            self.assertEqual(candidate.sha256, receipt.get("patch_sha256", receipt.get("sha256")))
        self.assertEqual(r.plan()["checks"][2]["legacy_attempt_receipts"], [r.prefix(r.V99), r.prefix(r.V125)])

if __name__ == "__main__":
    unittest.main()
