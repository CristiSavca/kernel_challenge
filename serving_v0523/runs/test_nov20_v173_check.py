"""Offline replay of the actual V163 and V173 private-slot transitions."""
import copy
import json
import subprocess
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import submit_nov20_v173_check as r
import test_nov20_v163_check as previous

V168_JOB = "5555555555555555"
CHILD = "6666666666666666"
IDS = (*previous.IDS,previous.ACCEPTED,V168_JOB)


class Judge(previous.Fake):
    def __init__(self,directory):
        super().__init__(directory)
        self.now += timedelta(minutes=3)
        result = super().offer()
        assert result == "submitted " + previous.ACCEPTED
        self.set_state(r.fallback.OWN,previous.ACCEPTED,"pending")
        self.jobs[previous.ACCEPTED]["created_at"] = self.now.timestamp()
        self.write(r.PARENT,V168_JOB)
        self.set_state(r.PARENT,V168_JOB,"pass")
        self.now += timedelta(minutes=1)
        self.quota["limits"]["checks"] = r.slots.THIRD.copy()
        self.child_job=CHILD
        self.events=[]
        self.attempts.clear()
        self.queries.clear()
        self.reads = 0

    def execute(self,cmd,**kw):
        if hasattr(self,"events"):
            self.events.append(cmd[1:] if cmd[1:]==["serve","status","--json"] else cmd[1:3])
        response = super().execute(cmd,**kw)
        if cmd[1:3]==["serve","check"] and r.V173.directory in cmd[3]:
            response.stdout = response.stdout.replace(previous.ACCEPTED,self.child_job)
        return response

    def offer(self):
        return r.run_once(okc="mock",runs=self.runs,clock=lambda:self.now,execute=self.execute)

    @property
    def receipt(self):
        return self.runs/(r.NAME+".submit.txt")


class Standby(unittest.TestCase):
    def fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Judge(directory.name)

    def held(self,f):
        self.assertFalse(f.offer().startswith("submitted "))
        self.assertFalse(f.attempts)
        self.assertFalse(f.receipt.exists())

    def test_exact_disjoint_pair_and_matched_metrics(self):
        f=self.fixture()
        for lane in (r.fallback.previous.second,r.fallback.previous.third):
            with patch.object(lane,"plan",return_value={}):
                self.assertFalse(lane.run_once(okc="mock",runs=f.runs,clock=lambda:f.now,
                                               execute=f.execute).startswith("submitted "))
        self.assertIsNone(r.fallback.previous.local_position(f.runs))
        with patch.object(r.measured,"submit_once",wraps=r.measured.submit_once) as submit:
            self.assertEqual(f.offer(),"submitted "+CHILD)
            self.assertEqual(submit.call_args.kwargs["reserve_checks"],0)
        target=r.ROOT/"serving_v0523/candidates"/r.V173.directory/"patch.diff"
        self.assertEqual(f.events[-2:],[["serve","status","--json"],["serve","check"]])
        self.assertEqual(f.attempts,[["mock","serve","check",str(target),"--users","16","--flag","enable-layernorm-sp",
            "--flag","moe-runner-backend=humming","--flag","muse-sparse-prompt-exchange","--flag",
            "speculative-moe-runner-backend=marlin","--flag","swa-full-tokens-ratio=0.25","--no-wait"]])
        self.assertEqual(r.slots.dated_receipt(f.runs,r.OWN),CHILD)
        meta=json.loads(f.receipt.read_text().splitlines()[0]); proof=meta["evidence"]
        self.assertEqual(proof["comparison_sha256"],r.PARENT_SHA)
        self.assertEqual(proof["v163_job"],previous.ACCEPTED)
        self.assertEqual(proof["comparison_rung"],previous.REAL_PASS["result"]["rungs"][0])
        self.assertEqual(proof["comparison_vs_stock"],previous.REAL_PASS["result"]["vs_stock"])
        self.assertEqual(proof["comparison_node"],previous.REAL_PASS["result"]["node"])
        self.assertEqual(f.offer(),"attempt recorded")
        self.assertEqual(len(f.attempts),1)

    def test_parent_metrics_must_be_conclusive(self):
        specimen=copy.deepcopy(previous.REAL_PASS["result"]["rungs"][0])
        for field,value in (("truncated",True),("truncated",0),("concurrency",True),("concurrency",32),
                    ("reps",False),("reps",0),("tok_s_per_gpu",float("nan")),("tok_s_per_gpu",True),
                    ("tok_s_per_user_e2e",0),("tok_s_per_user_p50",float("inf")),
                    ("tok_s_per_user_e2e_p50",float("inf")),("ttft_first_p90_s",float("nan"))):
            with self.subTest(field=field,value=value):
                f=self.fixture();f.jobs[V168_JOB]["result"]["rungs"]=[specimen|{field:value}];self.held(f)
        for field,value in (("status","crashed"),("passed",False),("rungs",[]),("rungs",[specimen,specimen]),
                    ("correctness",{"passed":False}),("correctness",{}),("patch",{"sha256":"0"*64}),
                    ("concurrency",[32]),("serving_flags",{}),("later_turns",{"ran":True,"passed":False}),
                    ("later_turns",{"ran":True,"passed":True,"insufficient":True}),
                    ("later_turns",{"ran":True,"passed":True,"incomplete":True}),("short_outputs",1),("short_outputs",False)):
            f=self.fixture();f.jobs[V168_JOB]["result"][field]=value;self.held(f)
        f=self.fixture();f.jobs[V168_JOB]["result"].pop("node")
        f.jobs[V168_JOB]["result"].pop("vs_stock");self.assertEqual(f.offer(),"submitted "+CHILD)
        for s in ("pending","crash","timeout"):
            f=self.fixture();f.set_state(r.PARENT,V168_JOB,s);self.held(f)

    def test_every_job_identity_and_precedence(self):
        for i,job in enumerate(IDS):
            for field,value in (("id","9"*16),("mode","run"),("board","other"),("users",True),("created_at",0),
                                ("serving_flags",{}),("duplicate_of","9"*16)):
                with self.subTest(job=job,field=field):
                    f=self.fixture();f.jobs[job][field]=value;self.held(f)
            f=self.fixture();p=f.runs/(r.REQUIRED[i][2]+".submit.txt");p.write_text(p.read_text().replace("CLI exit code: 0","CLI exit code: 1"));self.held(f)
            f=self.fixture();(f.runs/"other.submit.txt").write_text("job "+job);self.held(f)
        for job in (IDS[0],IDS[3],IDS[4]):
            f=self.fixture();f.jobs[job]["refunded"]=True;self.held(f)
        f=self.fixture();f.jobs[IDS[0]]["created_at"]=f.jobs[IDS[3]]["created_at"]+1;self.held(f)
        for i in (1,2):
            for state in ("pending","pass"):
                f=self.fixture();f.set_state(r.REQUIRED[i],IDS[i],state);self.held(f)
            f=self.fixture();f.set_state(r.REQUIRED[i],IDS[i],"timeout")
            self.assertEqual(f.offer(),"submitted "+CHILD)

    def test_slot_two_receipt_binds_exact_branch(self):
        for field,value in (("slot",1),("slot",2.0),("v129_job","9"*16),
                             ("failed_controls",{}),("failed_controls",{"v120b":IDS[2],"v139":IDS[1]})):
            f=self.fixture();p=f.runs/(r.fallback.NAME+".submit.txt");line,_,tail=p.read_text().partition("\n")
            header=json.loads(line);header["evidence"][field]=value;p.write_text(json.dumps(header)+"\n"+tail);self.held(f)
        for req in r.REQUIRED:
            f=self.fixture();(f.runs/(req[2]+".submit.txt")).unlink();self.held(f)
        for req in r.fallback.PEERS:
            for path,body in ((req[2]+".submit.txt",""),(req[2]+".skip.json","{}"),("older.submit.txt",req[0].sha256)):
                f=self.fixture();(f.runs/path).write_text(body);self.held(f)
        for path,body in (("other.submit.txt",r.V173.sha256),("1"*16+".json",r.V173.sha256)):
            f=self.fixture();(f.runs/path).write_text(body);self.held(f)

    def test_quota_window_and_evidence_changes_defer_without_receipt(self):
        for stream in (r.START-timedelta(microseconds=1),r.END):
            f=self.fixture();f.now=stream;self.held(f);self.assertEqual(f.queries,[])
        for n in (1,2):
            for change in ("used","left","cap","reset","cutoff","refunded","comparator","peer","lost"):
                if n==2 and change in ("refunded","comparator","lost"):
                    continue  # The final remote read is deliberately quota.
                f=self.fixture()
                def alter(read):
                    if read!=n:return
                    if change=="used":f.quota["limits"]["checks"]["used"]=True
                    if change=="left":f.quota["limits"]["checks"]["left"]=0
                    if change=="cap":f.quota["limits"]["checks"]["per_day"]=4
                    if change=="reset":f.quota["resets_at"]="wrong day"
                    if change=="cutoff":f.now=r.END
                    if change=="refunded":f.jobs[IDS[3]]["refunded"]=True
                    if change=="comparator":f.jobs[V168_JOB]["result"]["rungs"][0]["truncated"]=True
                    if change=="peer":(f.runs/(r.fallback.PEERS[2][2]+".skip.json")).write_text("{}")
                    if change=="lost":f.jobs.pop(V168_JOB)
                f.on_quota=alter;self.held(f)
        for malformed in (None,[],False):
            f=self.fixture();f.quota["limits"]["checks"]=malformed;self.held(f)
        for quota in ({"used":0,"left":3},{"used":1,"left":2},{"used":3,"left":0}):
            f=self.fixture();f.quota["limits"]["checks"].update(quota);self.held(f)

    def test_shared_lock_and_ambiguous_attempt_are_once_only(self):
        f=self.fixture()
        with r.slots.arbitrate(f.runs):
            self.held(f);self.assertEqual((f.queries,f.reads),([],0))
        arrived=threading.Event();release=threading.Event();answers=[]
        def hold(read):
            if read==1:arrived.set();self.assertTrue(release.wait(5))
        f.on_quota=hold
        t=threading.Thread(target=lambda:answers.append(f.offer()));t.start();self.assertTrue(arrived.wait(5))
        self.assertEqual(f.offer(),"another allocation poll is active")
        release.set();t.join(6);self.assertEqual(answers,["submitted "+CHILD]);self.assertEqual(len(f.attempts),1)
        for duplicate in IDS:
            f=self.fixture();f.child_job=duplicate
            self.assertTrue(f.offer().startswith("attempt needs manual reconciliation"))
            self.assertEqual(f.offer(),"attempt recorded");self.assertEqual(len(f.attempts),1)
        for mode in ("timeout","bad_acceptance"):
            f=self.fixture();setattr(f,mode,True)
            self.assertTrue(f.offer().startswith("attempt "))
            self.assertEqual(f.offer(),"attempt recorded");self.assertEqual(len(f.attempts),1)

    def test_plan_matches_frozen_source(self):
        actual=previous.REAL_PASS["result"]
        self.assertEqual(actual["short_outputs"],0)
        self.assertEqual([actual["later_turns"][k] for k in ("ran","passed","incomplete","insufficient")],[True,True,False,False])
        plan=r.plan()
        self.assertEqual((plan["required_quota"],plan["maximum_private_checks"],plan["mode"]),(r.slots.THIRD,1,"check"))
        self.assertEqual(plan["measured_parent"]["sha256"],r.PARENT_SHA)
        self.assertEqual(plan["flags"],r.V168.flags)


if __name__=="__main__":
    unittest.main()
