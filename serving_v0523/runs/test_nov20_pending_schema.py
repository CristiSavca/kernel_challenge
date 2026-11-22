"""Replay the public shape of the authenticated queued V93 status seen November 18."""
import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import nov20_check_slots as slots
import submit_nov20_v154c_check as fallback
import test_nov20_slot_arbitration as arbitration
from test_nov20_v154c_check import IDS, Judge


class PendingSchema(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        runs=Path(tmp.name)
        previous=slots.previous
        req=(previous.V93,16,previous.prefix(previous.V93),
             previous.START-timedelta(days=1),previous.END-timedelta(days=1))
        receipt=req[2]+".submit.txt"
        shutil.copyfile(fallback.RUNS/receipt,runs/receipt)
        job=previous.V93_JOB
        # All 11 top-level keys match the live response. Opaque team and storage keys are sanitized.
        data=dict(id=job,mode="check",board="serving",status="queued",created_at=1794986590.9584968,
                  duplicate_of=None,result=None,serving_flags=previous.V93.flags,team="redacted",
                  payload_ref="redacted",content_hash="dd914a8050ce08c034d9eaa6cf8680c8c834bc1ebe3572d233c69a953afa3690")
        def state():
            return slots.official_state(req,job,"fake",
                lambda *a,**kw:subprocess.CompletedProcess(a,0,json.dumps(data),""),runs=runs)
        return runs,runs/receipt,req,data,state

    def test_sanitized_real_queued_and_running_status(self):
        for status in ("queued","running"):
            for user_field in ({},{"users":None},{"users":16}):
                runs,doc,req,data,state=self.fixture()
                data.update(status=status,**user_field)
                self.assertEqual(state(),"pending")

    def test_missing_or_null_cannot_mask_wrong_identity_or_payload(self):
        for key,value in (("id","f"*16),("mode","run"),("board","kernels"),("duplicate_of","f"*16),
                          ("created_at",0),("serving_flags",{}),("result",{}),
                          ("result",{"passed":True}),("users",True),("users",16.0),("users","16"),("users",4)):
            with self.subTest(key=key,value=value):
                runs,doc,req,data,state=self.fixture()
                data[key]=value
                self.assertIsNone(state())

    def test_omitted_load_requires_unique_dated_same_line_official_confirmation(self):
        for fault in ("absent","header_load","header_flags","header_sha","undated","wrong_cli_users","wrong_job",
                      "detached_job","ambiguous_job","another_candidate_claims_same_job","skip"):
            with self.subTest(fault=fault):
                runs,doc,req,data,state=self.fixture()
                raw=doc.read_text();header,tail=raw.split("\n",1);meta=json.loads(header)
                if fault=="absent":doc.unlink()
                elif fault in ("header_load","header_flags","header_sha","undated"):
                    meta.update({"header_load":{"users":32},"header_flags":{"flags":{}},"header_sha":{"sha256":"x"},
                                 "undated":{"attempted_at":req[4].isoformat()}}[fault])
                    doc.write_text(json.dumps(meta)+"\n"+tail)
                elif fault=="wrong_cli_users":doc.write_text(raw.replace("16 users","4 users"))
                elif fault=="wrong_job":doc.write_text(raw.replace(data["id"],"e"*16))
                elif fault=="detached_job":doc.write_text(raw.replace("· job "+data["id"],"\njob "+data["id"]))
                elif fault=="ambiguous_job":doc.write_text(raw.replace("CLI exit code: 0","job "+"e"*16+"\nCLI exit code: 0"))
                elif fault=="another_candidate_claims_same_job":
                    (runs/"other.submit.txt").write_text(json.dumps({"candidate":"other","sha256":"f"*64})
                         +"\nsubmitted /x/other/patch.diff as a check run at 4 users · job "+data["id"])
                else:(runs/(req[2]+".skip.json")).write_text("{}")
                self.assertIsNone(state())

    def test_distinct_official_job_may_reuse_patch_and_cite_pending_parent(self):
        for load,mode in ((1,"check"),(32,"check"),(16,"run")):
            runs,doc,req,data,state=self.fixture()
            other="a"*16
            meta=dict(candidate=req[0].name,sha256=req[0].sha256,flags=req[0].flags,users=load,
                      attempted_at=(req[3]-timedelta(days=1)).isoformat(),evidence={"related_job":data["id"]})
            (runs/"historical.submit.txt").write_text(json.dumps(meta)+
                    f"\n✓ submitted /archive/{req[0].directory}/patch.diff as a {mode} run at {load} users · job {other}\nCLI exit code: 0\n")
            (runs/(other+".json")).write_text(json.dumps(dict(id=other,mode=mode,patch=req[0].sha256)))
            self.assertEqual(state(),"pending")

    def test_non_pending_failure_with_no_load_or_result_defers(self):
        for status in ("failed","withdrawn","canceled"):
            runs,doc,req,data,state=self.fixture()
            data["status"]=status
            self.assertIsNone(state())

    def test_both_v154c_allocations_survive_observed_pending_shape(self):
        for branch in (0,1):
            for null in (False,True):
                with tempfile.TemporaryDirectory() as tmp:
                    f=Judge(Path(tmp),branch)
                    for job in IDS[:2]:
                        f.jobs[job].pop("users")
                        if null:f.jobs[job]["users"]=None
                    self.assertTrue(f.offer().startswith("submitted "))
                    self.assertEqual(len(f.checks),1)

    def test_quota_race_with_duplicate_sparse_pending_receipt_defers(self):
        import submit_nov20_v132_check as v132
        import submit_nov20_v141_check as v141
        for runner in (v132,v141):
            case=arbitration.Arbitration("runTest")
            self.addCleanup(case.doCleanups)
            f=case.setup_batch("done" if runner==v132 else "queued")
            first_doc=f.docs[0]
            first_job=next(j for j in f.jobs if j in first_doc.read_text())
            f.jobs[first_job].pop("users")
            def race():
                (f.runs/"ambiguous.submit.txt").write_text("submitted /x/other/patch.diff as a check run at 32 users · job "+first_job)
            f.when_quota=race
            self.assertFalse(case.offer(f,runner).startswith("submitted "))
            self.assertFalse(f.checks)
