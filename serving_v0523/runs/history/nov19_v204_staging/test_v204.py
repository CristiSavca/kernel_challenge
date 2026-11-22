"""No network. Replay the real sources and official receipt format against isolated histories."""
import fcntl
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import submit_nov19_decode_checks as r
import submit_nov19_integrated_verify as i
import test_nov19_decode_checks as baseline

STAGE = Path(__file__).resolve().parent
JUDGE_ID = "abc01234fa098abc"


class Selection(unittest.TestCase):
    def setUp(self):
        self.h = baseline.MatchTest()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.runs = self.h.runs
        self.report = self.runs / (r.V68_JOB + ".json")
        self.report.write_bytes((r.RUNS / self.report.name).read_bytes())
        self.h.jobs[r.V68_JOB] = json.loads(self.report.read_text())
        self.claim = self.runs / (r.prefix(r.V204) + ".submit.txt")
        self.checks = []
        self.extra = None
        self.result = None
        self.code = 0

    def dispatch(self, cmd, **kw):
        if self.extra:
            self.extra(cmd)
        if cmd[1:3] != ["serve", "check"]:
            return self.h.execute(cmd, **kw)
        self.checks.append(cmd)
        self.assertTrue(self.claim.is_file())
        self.assertIn("timeout", kw)
        text = self.result if self.result is not None else f"✓ submitted {cmd[3]} as a check run at 16 users  42 files changed · job {JUDGE_ID}"
        self.h.left -= 1
        return subprocess.CompletedProcess(cmd, self.code, text, "")

    def offer(self, **kw):
        return r.run_candidate(r.V204, runs=self.runs, okc="mock", clock=lambda:self.h.at,
                               execute=self.dispatch, **kw)

    def test_plain_plan_and_two_invariant_allocations(self):
        saved=json.loads((r.RUNS/"nov19_integrated_verify_runner.launch.json").read_text())["plan"]
        self.assertEqual(i.checked_plan(),saved)
        new=i.checked_plan(pending_choice="v204")
        self.assertEqual([new["checks"][j] for j in (0,2)], [saved["checks"][j] for j in (0,2)])
        self.assertEqual(new["maximum_private_checks"],3)
        a,b=saved["checks"][1],new["checks"][1]
        self.assertEqual(a["alternatives"][::2],b["alternatives"][::2])
        self.assertEqual(b["legacy_attempt_receipts"],a["legacy_attempt_receipts"]+[r.prefix(r.V168)])
        self.assertEqual(b["alternatives"][1],dict(candidate="v204",users=16,flags=r.V204.flags,sha256=r.V204.sha256))
        for key in ("when_v93_passed_and_shared_parents_eligible","when_v93_or_v68_failed","when_unknown"):
            self.assertEqual(a[key],b[key])
        self.assertEqual(b["shared_receipt"],r.prefix(r.V204))
        self.assertEqual(b["pending_ancestor_sha256"],r.v204_source.V68_REPORT)
        with self.assertRaises(ValueError): i.checked_plan(pending_choice="v999")

    def test_confirmation_and_exact_expected_command_with_restart(self):
        self.assertEqual(self.offer(),"submitted "+JUDGE_ID)
        self.assertEqual(len(self.checks),1)
        self.assertEqual(self.checks[0],["mock","serve","check",str(r.ROOT/"serving_v0523/candidates"/r.V204.directory/"patch.diff"),
            "--users","16","--flag","enable-layernorm-sp","--flag","moe-runner-backend=humming",
            "--flag","speculative-dflash-mixed-prefill","--flag","speculative-dflash-sp-prefill-context",
            "--flag","speculative-moe-runner-backend=marlin","--flag","swa-full-tokens-ratio=0.25","--no-wait"])
        self.assertEqual(self.h.commands[-1][3],r.V93_JOB)
        header=json.loads(self.claim.read_text().splitlines()[0])
        self.assertEqual((header["candidate"],header["sha256"],header["flags"]),("v204",r.V204.sha256,r.V204.flags))
        self.assertEqual(header["evidence"]["v68_comparison_job"],r.V68_JOB)
        self.assertEqual(header["evidence"]["comparison_job"],r.V64_PRIVATE_JOB)
        self.assertEqual(header["evidence"]["exclusive_alternatives"],["v98","v204","v95"])
        self.assertEqual(self.offer(),"attempt recorded")
        for c in r.PREFILL_ALTERNATIVES:
            self.assertEqual(self.h.offer(c),"attempt recorded")
        # Execute the actual unmodified old worker module against the new claim.
        spec=importlib.util.spec_from_file_location("old_nov19_for_rollback", r.RUNS/"submit_nov19_decode_checks.py")
        old=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        self.assertTrue(old.prefill_attempt(self.runs))
        self.assertFalse(self.h.receipt(r.V168).exists())
        from submit_nov20_v173_check import slots, PARENT
        self.assertIsNone(slots.bound_receipt(self.runs,PARENT))

    def test_competing_named_receipts_and_renamed_prior_sources(self):
        archive=self.runs/"historical"
        archive.mkdir()
        for c in (*r.PREFILL_ALTERNATIVES,*r.LEGACY_PREFILL_ALTERNATIVES,r.V204):
            for path,raw in ((self.runs / (r.prefix(c)+".submit.txt"),"partial"),
                             (archive / "recovered.submit.json",json.dumps(dict(candidate=c.name,sha256=c.sha256))),
                             (archive / "beef1234beefaaaa.json",json.dumps(dict(
                                result=dict(patch=dict(sha256=c.sha256)))))):
                with self.subTest(c=c.name,p=path.name):
                    path.write_text(raw)
                    self.offer()
                    self.assertEqual(self.checks,[])
                    path.unlink()
        self.assertFalse(self.claim.exists())

    def test_v93_branches_and_ambiguous_ids_no_attempt(self):
        for state in ("done","failed","canceled","unrecognized"):
            self.h.job(r.V93,state=state)
            self.assertTrue(self.offer().startswith("waiting"),state)
        self.h.job(r.V93,state="queued")
        native=self.h.jobs[r.V93_JOB].copy()
        for change in ({"mode":"run"},{"users":16.0},{"users":4},{"duplicate_of":JUDGE_ID},{"result":{}},
                       {"serving_flags":r.V204.flags},{"created_at":r.START.timestamp()},{"id":JUDGE_ID}):
            self.h.jobs[r.V93_JOB]=native|change
            self.assertTrue(self.offer().startswith("waiting"),change)
        self.assertEqual(self.checks,[])
        self.assertFalse(self.claim.exists())

    def test_authentic_v93_pending_optional_integer_load(self):
        for load in (None,16):
            self.h.jobs[r.V93_JOB]["users"]=load
            self.assertEqual(self.offer(),"submitted "+JUDGE_ID)
            self.claim.unlink()
        self.assertEqual(len(self.checks),2)

    def test_parent_or_legacy_attempt_changes_during_quota(self):
        def pass_v93():
            self.h.job(r.V93,state="done")
        def fail_v68():
            self.h.jobs[r.V68_JOB]["status"]="failed"
        def restore_prior():
            (self.runs/(r.prefix(r.V102)+".submit.txt")).write_text("partially restored")
        for cb in (pass_v93,fail_v68,restore_prior):
            self.h.on_quota=cb
            self.assertEqual(self.offer(),"quota request failed")
            self.assertFalse(self.claim.exists())
            self.h.job(r.V93,state="queued")
            self.h.jobs[r.V68_JOB]=json.loads(self.report.read_text())
        self.assertEqual(self.checks,[])

    def test_final_v93_observation_after_v68_and_quota(self):
        seen=0
        def late(cmd):
            nonlocal seen
            if cmd[1:]==["serve","status",r.V68_JOB,"--json"]:
                seen+=1
                if seen==2: self.h.job(r.V93,state="done")
        self.extra=late
        self.assertEqual(self.offer(),"quota request failed")
        self.assertFalse(self.claim.exists())
        self.assertTrue(self.h.offer(r.V98).startswith("submitted"))
        self.assertEqual(self.checks,[])

    def test_archived_and_live_v68_must_agree_both_before_and_after_quota(self):
        for corrupt in (lambda d:d.update(status="queued",result=None),lambda d:d.update(duplicate_of=JUDGE_ID),
                        lambda d:d.update(users=16.0),lambda d:d.update(created_at=0),
                        lambda d:d["result"]["rungs"][0].update(tok_s_per_gpu=0),
                        lambda d:d["result"].update(passed=False),lambda d:d.update(refunded=True),
                        lambda d:d["result"]["rungs"][0].update(truncated=True)):
            for late in (False,True):
                self.h.jobs[r.V68_JOB]=json.loads(self.report.read_text())
                flip=lambda:corrupt(self.h.jobs[r.V68_JOB])
                self.h.on_quota=flip if late else lambda:None
                if not late: flip()
                self.assertTrue(self.offer() in ("quota request failed","waiting V68 status unavailable"))
                self.assertEqual(self.checks,[])
                self.assertFalse(self.claim.exists())

    def test_three_controls_are_checked_after_quota(self):
        for path in (self.report,self.h.private,self.h.v64_path):
            raw=path.read_bytes()
            self.h.on_quota=lambda:path.write_text("{}")
            self.assertEqual(self.offer(),"quota request failed")
            self.assertFalse(self.claim.exists())
            path.write_bytes(raw)
        self.assertEqual(self.checks,[])

    def test_quota_window_and_shared_lock(self):
        self.h.left=0
        self.assertEqual(self.offer(),"quota request failed")
        self.h.left=2
        self.h.quota_reset="2026-11-21T00:00:00Z"
        self.assertEqual(self.offer(),"quota request failed")
        self.h.quota_reset=r.RESET
        for when in (r.START-timedelta(microseconds=1),r.END):
            self.h.at=when
            self.assertEqual(self.offer(),"outside submission window")
        self.h.at=r.START
        with (self.runs/"nov19_prefill_alternative.lock").open("a") as fp:
            fcntl.flock(fp,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertEqual(self.offer(),"waiting for alternative lock")
        self.h.on_quota=lambda:setattr(self.h,"at",r.END)
        with self.assertRaises(TimeoutError): self.offer()
        self.assertFalse(self.claim.exists())
        self.assertEqual(self.checks,[])

    def test_donor_hash_tree_and_byte_validation_before_spend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/"serving_v0523/candidates"/r.V204.directory
            shutil.copytree(r.ROOT/"serving_v0523/candidates"/r.V204.directory,folder,
                            ignore=lambda _,names:[n for n in names if n not in {"patch.diff","delta.diff","validation.json"}])
            self.assertTrue(r.v204_source.verified(root,self.runs,r.V68_JOB))
            path=folder/"validation.json"
            manifest=json.loads(path.read_text())
            for key in ("launch_flags","patch_sha256","tree","parent_sha256","donor_sha256",
                        "strict_both_donor_orders","strict_apply_reverse_full_delta"):
                path.write_text(json.dumps(manifest|{key:None}))
                self.assertTrue(self.offer(root=root).startswith("waiting"),key)
            path.write_text(json.dumps(manifest))
            (folder/"delta.diff").write_bytes(b"other")
            self.assertTrue(self.offer(root=root).startswith("waiting"))
        self.assertEqual(self.checks,[])
        self.assertFalse(self.claim.exists())

    def test_ambiguous_or_conflicting_cli_claim_persists(self):
        for body,code in [("timeout",1),("job "+JUDGE_ID,0),
                          (f"submitted /fake/{r.V204.directory}/patch.diff as a check run at 16 users · job {r.V93_JOB}",0),
                          (f"submitted /fake/{r.V168.directory}/patch.diff as a check run at 16 users · job {JUDGE_ID}",0)]:
            self.result=body
            self.code=code
            self.h.left=2
            self.assertTrue(self.offer().startswith("attempt needs"))
            self.assertTrue(self.claim.exists())
            self.assertEqual(self.offer(),"attempt recorded")
            self.claim.unlink()


if __name__ == "__main__":
    unittest.main()
