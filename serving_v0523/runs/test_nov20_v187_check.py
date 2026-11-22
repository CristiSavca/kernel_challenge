"""Exercise only offline responses and temporary local receipts."""
import json
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import submit_nov20_v187_check as r
import test_nov20_v163_check as base

IDS = ("1111111111111111", "2222222222222222", "3333333333333333", r.v180.nov19.V68_JOB)
CHILD = "7777777777777777"


class Judge(base.Fake):
    def __init__(self,directory):
        self.runs=Path(directory); self.now=r.START; self.jobs={}; self.attempts=[]; self.queries=[]
        self.reads=0; self.on_quota=lambda n:None; self.timeout=False; self.bad_acceptance=False
        self.quota=dict(resets_at=r.RESET,limits=dict(checks=r.slots.THIRD.copy()))
        self.events=[];self.child=CHILD
        for req,job in zip(r.REQUIRED[:3],IDS[:3]):
            self.write(req,job)
            self.set_state(req,job,"pending")
        self.meta(1, evidence=dict(slot=2,v129_job=IDS[0],v132_job=None))
        ref=r.REQUIRED[-1][2]+".submit.txt"
        shutil.copyfile(r.RUNS/ref,self.runs/ref)
        self.jobs[IDS[-1]]=json.loads((r.RUNS/(IDS[-1]+".json")).read_text())

    def meta(self,i,**changes):
        p=self.runs/(r.REQUIRED[i][2]+".submit.txt")
        header,_,body=p.read_text().partition("\n")
        p.write_text(json.dumps(json.loads(header)|changes)+"\n"+body)

    def execute(self,cmd,**kwargs):
        self.events.append(cmd[1:])
        response=super().execute(cmd,**kwargs)
        if cmd[1:3]==["serve","check"]:
            response.stdout=response.stdout.replace(base.ACCEPTED,self.child)
        return response

    def offer(self):
        return r.run_once(okc="mock",runs=self.runs,clock=lambda:self.now,execute=self.execute)

    @property
    def receipt(self):
        return self.runs/(r.NAME+".submit.txt")


class Cases(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);return Judge(tmp.name)
    def held(self,f):
        self.assertFalse(f.offer().startswith("submitted "))
        self.assertFalse(f.attempts or f.receipt.exists())

    def test_positive_and_exact_command(self):
        for state in ("queued","running"):
            f=self.fixture();f.jobs[IDS[2]]["status"]=state
            self.assertEqual(f.offer(),"submitted "+CHILD)
            self.assertEqual(f.events[-3],["serve","status","--json"])
            self.assertEqual(f.events[-2],["serve","status",IDS[2],"--json"])
            self.assertEqual(f.events.count(["serve","status","--json"]),2)
            cmd=["mock","serve","check",str(r.ROOT/"serving_v0523/candidates"/r.V187.directory/"patch.diff"),"--users","16"]
            for k,v in sorted(r.V187.flags.items()):cmd.extend(("--flag",k[2:]+("="+v if v else "")))
            self.assertEqual(f.attempts,[cmd+["--no-wait"]])
            self.assertEqual(r.slots.bound_receipt(f.runs,r.OWN),CHILD)
            head=json.loads(f.receipt.read_text().splitlines()[0])
            self.assertEqual((head["candidate"],head["users"],head["sha256"],head["flags"]),("v187",16,r.V187.sha256,r.V187.flags))
            self.assertEqual(head["evidence"]["comparison_rung"]["tok_s_per_gpu"],17512.9)
            self.assertEqual(head["evidence"]["final_quota"]["checks"],r.slots.THIRD)
            self.assertEqual(head["evidence"]["final_control"]["status"],state)
            self.assertEqual(f.offer(),"attempt recorded")
            self.assertIsNotNone(r.sibling.second.prior_attempt(f.runs))
            self.assertIsNone(r.sibling.local_position(f.runs))

    def test_delayed_gate_and_cutoff_and_lock(self):
        for ts in (r.START-timedelta(microseconds=1),r.END,r.END+timedelta(seconds=1)):
            f=self.fixture();f.now=ts;self.held(f);self.assertEqual(f.events,[])
        f=self.fixture();f.now=r.END-timedelta(milliseconds=50)
        f.on_quota=lambda n:setattr(f,"now",r.END) if n==2 else None
        self.held(f)
        f=self.fixture()
        with r.slots.arbitrate(f.runs) as held:
            self.assertTrue(held);self.assertEqual(f.offer(),"another allocation poll is active")
            self.assertEqual(f.events,[])

    def test_every_competitor_and_false_or_duplicate_evidence(self):
        for req in r.COMPETITORS:
            f=self.fixture();(f.runs/(req[2]+".submit.txt")).write_text("unreconciled")
            self.held(f);self.assertEqual(f.events,[])
        for name,body in (("other_nov20.submit.txt","partial"),("renamed.submit.txt",json.dumps({"attempted_at":r.START.isoformat()}))):
            f=self.fixture();(f.runs/name).write_text(body);self.held(f)
        for i in range(4):
            for damage in ("missing","ambiguous","wrongflags","wrongload","wrongday","wrongpatch"):
                f=self.fixture();p=f.runs/(r.REQUIRED[i][2]+".submit.txt")
                if damage=="missing":p.unlink()
                elif damage=="ambiguous":(f.runs/"orphan.submit.txt").write_text("job "+IDS[i])
                else:f.meta(i,**{"wrongflags":{"flags":{}},"wrongload":{"users":True},"wrongday":{"attempted_at":r.END.isoformat()},"wrongpatch":{"sha256":"0"*64}}[damage])
                self.held(f)
        for evidence in (None,{},{"slot":True},{"slot":2,"v129_job":IDS[2],"v132_job":None},
                         {"slot":3,"v129_job":IDS[0],"v132_job":None},{"slot":2,"v129_job":IDS[0]}):
            f=self.fixture();f.meta(1,evidence=evidence);self.held(f)
        f=self.fixture();f.jobs[IDS[1]]["created_at"]=f.jobs[IDS[0]]["created_at"]-1;self.held(f)
        f=self.fixture();f.meta(0,attempted_at=(r.REQUIRED[1][3]+timedelta(minutes=2)).isoformat());self.held(f)

    def test_each_official_identity_and_comparator(self):
        for i in range(4):
            for field,val in (("refunded",True),("duplicate_of",CHILD),("mode","submit"),("board","kernel"),
                              ("id",CHILD),("created_at",r.END.timestamp()),("serving_flags",{}),("users",True)):
                f=self.fixture();f.jobs[IDS[i]][field]=val;self.held(f)
        for state in ("pass","incorrect","crash","timeout"):
            f=self.fixture();f.set_state(r.REQUIRED[2],IDS[2],state);self.held(f)
        for state in ("canceled","unknown","failed"):
            f=self.fixture();f.jobs[IDS[2]]["status"]=state;self.held(f)
        for val in ([],{},False):
            f=self.fixture();f.jobs[IDS[2]]["result"]=val;self.held(f)
        for mutate in (
            lambda q:q.update(passed=False),
            lambda q:q["rungs"][0].update(truncated=True),
            lambda q:q["rungs"][0].update(reps=0),
            lambda q:q["rungs"][0].update(tok_s_per_gpu=17510.0),
            lambda q:q["vs_stock"].update(rungs=[]),
            lambda q:q.update(patch={"sha256":"0"*64}),
        ):
            f=self.fixture();mutate(f.jobs[IDS[-1]]["result"]);self.held(f)

    def test_quota_remote_changes_and_local_race(self):
        for field,val in (("used",1),("left",2),("per_day",8),("left",True),("used",2.0)):
            f=self.fixture();f.quota["limits"]["checks"][field]=val;self.held(f)
        f=self.fixture();f.quota["resets_at"]="2026-11-20T00:00:00Z";self.held(f)
        f=self.fixture()
        def new_state(n):
            if n==1:f.set_state(r.REQUIRED[2],IDS[2],"pass")
        f.on_quota=new_state;self.held(f)
        for last in ("pass","crash"):
            f=self.fixture();f.on_quota=lambda n:f.set_state(r.REQUIRED[2],IDS[2],last) if n==2 else None
            self.held(f)
        f=self.fixture();f.on_quota=lambda n:setattr(f,"now",f.now+timedelta(seconds=35)) if n==2 else None
        self.held(f)
        f=self.fixture();f.on_quota=lambda n:f.quota["limits"]["checks"].update(left=0) if n==2 else None
        self.held(f)
        f=self.fixture();f.on_quota=lambda n:(f.runs/(r.COMPETITORS[0][2]+".submit.txt")).write_text("new") if n==2 else None
        self.held(f)

    def test_ambiguous_slot_is_reserved_across_peer_states(self):
        for failure in ("timeout","old_job","unconfirmed"):
            f=self.fixture()
            if failure=="timeout":f.timeout=True
            if failure=="old_job":f.child=IDS[1]
            if failure=="unconfirmed":f.bad_acceptance=True
            self.assertTrue(f.offer().startswith("attempt "))
            self.assertTrue(f.receipt.exists())
            self.assertEqual(f.offer(),"attempt recorded")
            f.set_state(r.REQUIRED[2],IDS[2],"pass")
            self.assertIsNotNone(r.sibling.second.prior_attempt(f.runs))
            f.set_state(r.REQUIRED[2],IDS[2],"crash")
            self.assertIsNone(r.sibling.local_position(f.runs))
            self.assertEqual(len(f.attempts),1)

    def test_no_receipt_when_plan_is_damaged(self):
        f=self.fixture()
        with patch.object(r,"plan",side_effect=ValueError("digest")):
            with self.assertRaises(ValueError):f.offer()
        self.assertFalse(f.receipt.exists() or f.attempts)
        self.assertIsNone(r.sibling.second.prior_attempt(f.runs))


if __name__ == "__main__":
    unittest.main()
