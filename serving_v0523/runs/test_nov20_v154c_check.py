"""Offline third-slot arbitration. All status and check operations use an in-process fake."""
import json
from datetime import timedelta
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import submit_nov20_v154c_check as runner

IDS=("1111111111111111","2222222222222222","3333333333333333")
OFFER="4444444444444444"


class Judge:
    def __init__(self,runs,branch):
        self.runs=runs
        self.now=runner.slots.GRACE+timedelta(minutes=2)
        self.taken,self.missing,self.control=runner.BRANCHES[branch]
        self.quota={"resets_at":runner.RESET,"limits":{"checks":runner.slots.THIRD.copy()}}
        self.on_quota=lambda n:None
        self.reads=0
        self.checks=[]
        self.status_jobs=[]
        self.jobs={}
        self.cli_exit=0
        self.check_output=None
        for name in (runner.REFERENCE+".json","v64_nov17_check16.submit.txt"):
            shutil.copyfile(runner.RUNS/name,runs/name)
        for req,job in zip((runner.FIRST,self.taken,self.control),IDS,strict=True):
            self.write(req,job)
            c,u,_,start,_=req
            self.jobs[job]=dict(id=job,board="serving",mode="check",users=u,status="queued",
                 serving_flags=c.flags,created_at=(start+timedelta(seconds=10 if req==self.taken else 3)).timestamp(),
                 result=None,duplicate_of=None)
        self.failure()
    def failure(self):
        c,u,*_=self.control
        self.jobs[IDS[2]].update(status="done",result=dict(passed=False,status="crashed",
                   concurrency=[u],serving_flags=c.flags,patch={"sha256":c.sha256}))
    def write(self,req,job):
        c,u,name,start,_=req
        text=json.dumps(dict(candidate=c.name,sha256=c.sha256,flags=c.flags,users=u,attempted_at=start.isoformat()))
        (self.runs/(name+".submit.txt")).write_text(text+f"\n✓ submitted /x/{c.directory}/patch.diff as a check run at {u} users · job {job}\nCLI exit code: 0\n")
    def execute(self,cmd,**kwargs):
        if cmd[1:]==["serve","status","--json"]:
            self.reads+=1
            self.on_quota(self.reads)
            return subprocess.CompletedProcess(cmd,0,json.dumps(self.quota),"")
        if cmd[1:3]==["serve","status"] and len(cmd)==5:
            self.status_jobs.append(cmd[3])
            j=self.jobs.get(cmd[3])
            return subprocess.CompletedProcess(cmd,0 if j else 1,json.dumps(j),"")
        assert cmd[1:3]==["serve","check"]
        self.checks.append(cmd)
        if self.cli_exit=="timeout":
            raise subprocess.TimeoutExpired(cmd,kwargs["timeout"])
        return subprocess.CompletedProcess(cmd,self.cli_exit,self.check_output or
                   f"✓ submitted {cmd[3]} as a check run at 16 users · job {OFFER}","")
    def offer(self):
        return runner.run_once(okc="mock",runs=self.runs,clock=lambda:self.now,execute=self.execute)


class Cases(unittest.TestCase):
    def make(self,branch=0):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Judge(Path(tmp.name),branch)
    def denied(self,f):
        self.assertFalse(f.offer().startswith("submitted "))
        self.assertFalse(f.checks)
    def test_both_allocation_branches_and_idempotence(self):
        for branch in range(2):
            f=self.make(branch)
            with patch.object(runner.measured,"submit_once",wraps=runner.measured.submit_once) as call:
                self.assertEqual(f.offer(),"submitted "+OFFER)
                self.assertEqual(call.call_args.kwargs["reserve_checks"],0)
            self.assertFalse(f.offer().startswith("submitted "))
            target=runner.ROOT/"serving_v0523/candidates"/runner.V154C.directory/"patch.diff"
            self.assertEqual(f.checks,[["mock","serve","check",str(target),"--users","16",*runner.first.FLAGS,"--no-wait"]])
            h=json.loads((f.runs/(runner.NAME+".submit.txt")).read_text().splitlines()[0])
            self.assertEqual(h["evidence"]["comparison_job"],runner.REFERENCE)
            self.assertEqual((h["sha256"],h["flags"],h["users"]),(runner.V154C.sha256,runner.V64.flags,16))
    def test_carry_consumed_slots_if_live_result_failed(self):
        for i in (0,1):
            f=self.make();f.jobs[IDS[i]].update(status="failed",result=None)
            self.assertEqual(f.offer(),"submitted "+OFFER)
    def test_positive_control_top_level_explicit_failure(self):
        f=self.make(1);f.jobs[IDS[2]].update(status="failed",result=None)
        self.assertEqual(f.offer(),"submitted "+OFFER)
    def test_competing_results_must_be_terminal_failures(self):
        for branch in range(2):
            for status in ("queued","running","unknown","canceled","cancelled","withdrawn","passed","noresult","unavailable"):
                with self.subTest(branch=branch,status=status):
                    f=self.make(branch);j=f.jobs[IDS[2]]
                    if status=="passed":j["result"].update(passed=True,status="ok")
                    elif status=="noresult":j.update(status="done",result=None)
                    elif status=="unavailable":del f.jobs[IDS[2]]
                    else:j.update(status=status,result=None)
                    self.denied(f)
    def test_every_slot_identity_checked(self):
        for branch in range(2):
            for i in range(3):
                for field,value in (("id","0"*16),("created_at",0),("users",True),("serving_flags",{}),
                                    ("duplicate_of",OFFER),("board","kernels"),("mode","run")):
                    with self.subTest(branch=branch,i=i,field=field):
                        f=self.make(branch);f.jobs[IDS[i]][field]=value;self.denied(f)
    def test_complete_payload_required_and_no_conflicting_terminal(self):
        for key,val in (("passed",1),("passed",True),("status","unreviewed"),("serving_flags",{}),
                        ("patch",{}),("concurrency",[False])):
            with self.subTest(key=key):
                f=self.make();f.jobs[IDS[2]]["result"][key]=val;self.denied(f)
    def test_top_level_failure_rejects_contradictory_or_incomplete_payload(self):
        for branch in range(2):
            for kind in ("passed","no_patch","unknown","ambiguous_ok","valid_failure"):
                with self.subTest(branch=branch,kind=kind):
                    f=self.make(branch);j=f.jobs[IDS[2]];j["status"]="failed"
                    if kind=="passed":j["result"].update(status="ok",passed=True,correctness={"passed":True})
                    elif kind=="no_patch":j["result"]["patch"]=None
                    elif kind=="unknown":j["result"]["status"]="running"
                    elif kind=="ambiguous_ok":j["result"].update(status="ok",passed=False,correctness={"passed":True})
                    if kind=="valid_failure":self.assertEqual(f.offer(),"submitted "+OFFER)
                    else:self.denied(f)
    def test_fresh_receipts_and_prior_slot_position(self):
        for branch in range(2):
            for i in range(3):
                for fault in ("missing","ambiguous","misdated","summary","job_alias","skip"):
                    with self.subTest(branch=branch,i=i,fault=fault):
                        f=self.make(branch);req=(runner.FIRST,f.taken,f.control)[i];doc=f.runs/(req[2]+".submit.txt")
                        s=doc.read_text()
                        if fault=="missing":doc.unlink()
                        elif fault=="ambiguous":doc.write_text(s+f"job {OFFER}\n")
                        elif fault=="misdated":doc.write_text(s.replace(req[3].isoformat(),req[4].isoformat()))
                        elif fault=="summary":doc.write_text(s.replace("as a check run","as a scored run"))
                        elif fault=="job_alias":doc.write_text(s.replace(IDS[i],IDS[(i+1)%3]))
                        else:(f.runs/(req[2]+".skip.json")).write_text("{}")
                        self.denied(f)
    def test_no_ambiguous_rival_no_prior_own_attempt(self):
        for branch in range(2):
            for fault in ("rival_own","rival_orphan","rival_result","both_pass","own_prior","own_orphan"):
                f=self.make(branch);c=f.missing[0]
                if fault=="rival_own":(f.runs/(f.missing[2]+".submit.txt")).write_text("{")
                elif fault=="rival_orphan":(f.runs/"unknown.submit.txt").write_text(c.sha256)
                elif fault=="rival_result":(f.runs/("1"*16+".json")).write_text(json.dumps({"patch":c.sha256}))
                elif fault=="both_pass":f.write(f.missing,"7"*16)
                elif fault=="own_prior":(f.runs/(runner.NAME+".submit.txt")).write_text("")
                else:(f.runs/"orphan.submit.json").write_text(runner.V154C.sha256)
                self.denied(f)
    def test_no_conflicting_jobs_or_secondary_receipts_for_any_required_identity(self):
        for branch in range(2):
            for i in range(3):
                for kind in ("old_receipt","other_official_id","unrelated_citation","matching_official_id"):
                    with self.subTest(branch=branch,i=i,kind=kind):
                        f=self.make(branch);req=(runner.FIRST,f.taken,f.control)[i];c=req[0]
                        if kind=="old_receipt":
                            (f.runs/"legacy.submit.txt").write_text("unidentified job "+c.sha256)
                        elif kind=="other_official_id":
                            (f.runs/("9"*16+".json")).write_text(json.dumps({"patch":c.sha256}))
                        elif kind=="matching_official_id":
                            (f.runs/(IDS[i]+".json")).write_text(json.dumps({"patch":c.sha256}))
                        else:
                            (f.runs/"other.submit.txt").write_text(json.dumps({"candidate":"other","sha256":"b"*64,
                                                   "evidence":{"parent":c.sha256}})+"\nsubmitted other")
                        if kind in ("matching_official_id","unrelated_citation"):
                            self.assertEqual(f.offer(),"submitted "+OFFER)
                        else:self.denied(f)
    def test_quota_exact_and_reset(self):
        for key,val in (("per_day",4),("used",1),("left",True),("used",3),("left",0)):
            f=self.make();f.quota["limits"]["checks"][key]=val;self.denied(f)
        f=self.make();f.quota["resets_at"]=runner.START.isoformat();self.denied(f)
    def test_postquota_changes_at_either_read(self):
        for n in (1,2):
            for fault in ("rival_pending","first_disappears","receipt_torn","reserve_changed","deadline"):
                with self.subTest(n=n,fault=fault):
                    f=self.make(1)
                    def change(read):
                        if read!=n:return
                        if fault=="rival_pending":f.jobs[IDS[2]].update(status="running",result=None)
                        elif fault=="first_disappears":f.jobs.pop(IDS[0])
                        elif fault=="receipt_torn":(f.runs/(f.taken[2]+".submit.txt")).write_text("")
                        elif fault=="reserve_changed":f.quota["limits"]["checks"]["left"]=0
                        else:f.now=runner.END
                    f.on_quota=change
                    self.denied(f)
                    self.assertFalse((f.runs/(runner.NAME+".submit.txt")).exists())
    def test_verified_peer_after_first_and_cutoff(self):
        for n in (runner.START-timedelta(microseconds=1),runner.END):
            f=self.make();f.now=n;self.denied(f);self.assertEqual(f.reads,0)
        f=self.make(1);f.jobs[IDS[1]]["created_at"]=f.jobs[IDS[0]]["created_at"]-1;self.denied(f)
        f=self.make();f.jobs[IDS[1]]["created_at"]=runner.START.timestamp();self.denied(f)
    def test_shared_lock_defers_all_remote_calls(self):
        f=self.make()
        with runner.slots.arbitrate(f.runs):
            self.assertEqual(f.offer(),"another allocation poll is active");self.assertFalse(f.checks);self.assertEqual(f.reads,0)
    def test_transient_final_control_retry_precedes_own_exclusive_receipt(self):
        for branch in range(2):
            f=self.make(branch)
            def glitch(n):
                if n==2:f.jobs[IDS[2]].update(status="running",result=None)
            f.on_quota=glitch
            self.denied(f)
            self.assertFalse((f.runs/(runner.NAME+".submit.txt")).exists())
            f.failure();f.on_quota=lambda n:None
            self.assertEqual(f.offer(),"submitted "+OFFER)
    def test_pending_competitor_polled_first(self):
        f=self.make(1);f.jobs[IDS[2]].update(status="running",result=None)
        self.denied(f);self.assertEqual(f.status_jobs,[IDS[2]]);self.assertEqual(f.reads,0)
    def test_private_reference_formatting_and_irrelevant_metadata_do_not_block(self):
        f=self.make();doc=f.runs/(runner.REFERENCE+".json")
        d=json.loads(doc.read_text());d["annotation"]="archived";doc.write_text(json.dumps(d,indent=3))
        self.assertEqual(f.offer(),"submitted "+OFFER)
    def test_anchor_control_is_exact_and_measured(self):
        for filename in ("v64_nov17_check16.submit.txt",runner.REFERENCE+".json"):
            f=self.make();p=f.runs/filename
            p.write_text(p.read_text().replace("17349.9","17349.8") if filename.endswith(".json") else p.read_text().replace("at 16 users","at 32 users"))
            with self.assertRaises(ValueError):f.offer()
            self.assertFalse(f.checks)
    def test_patch_immutability(self):
        f=self.make()
        with patch.object(runner.first,"digest",return_value="0"*64),self.assertRaises(ValueError):
            f.offer()
        self.assertFalse(f.checks)
    def test_incomplete_and_timed_out_attempts_remain_exclusive(self):
        for case in ("timeout","err","misreported"):
            f=self.make()
            if case=="timeout":f.cli_exit="timeout"
            if case=="err":f.cli_exit=1
            if case=="misreported":f.check_output=f"job {OFFER}"
            self.assertFalse(f.offer().startswith("submitted "))
            self.assertEqual(f.offer(),"attempt recorded")
            self.assertEqual(len(f.checks),1)


if __name__=="__main__":
    unittest.main()
