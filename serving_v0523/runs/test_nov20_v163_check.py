"""Exercise second-slot fallback with real result shapes and fake in-process judge calls."""
import copy
import json
import subprocess
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import submit_nov20_v163_check as r
import test_nov20_v154c_check as previous_tests

IDS = ("1111111111111111", "2222222222222222", "3333333333333333")
ACCEPTED = "4444444444444444"
REAL_CRASH = json.loads((r.RUNS/"00c134c978104d65.json").read_text())
REAL_TIMEOUT = json.loads((r.RUNS/"8ff48fb6618b4188.json").read_text())
REAL_PASS = json.loads((r.RUNS/"ac89fddc4ee147ab.json").read_text())

class Fake:
    write = previous_tests.Judge.write
    def __init__(self,directory):
        self.runs = Path(directory)
        self.now = r.START
        self.jobs = {}
        self.attempts = []
        self.queries = []
        self.reads = 0
        self.quota = dict(resets_at=r.RESET, limits=dict(checks=r.slots.SECOND.copy()))
        self.on_quota = lambda n: None
        self.timeout = False
        self.bad_acceptance = False
        for item,job in zip(r.REQUIRED,IDS):
            self.write(item,job)
            self.set_state(item,job,"pending" if job==IDS[0] else "crash")

    def set_state(self,item,job,state):
        c,u,_,begin,_=item
        if state=="pending":
            d=dict(status="queued",result=None,content_hash="b"*64,payload_ref="serving/"+"b"*64+".diff")
        elif state=="timeout":
            d=copy.deepcopy(REAL_TIMEOUT)
            d.pop("users",None)
        else:
            d=copy.deepcopy(REAL_CRASH if state=="crash" else REAL_PASS)
            d["result"].update(concurrency=[u],serving_flags=c.flags,patch={"sha256":c.sha256})
            d["users"]=u
            if state=="incorrect":
                d["result"].update(status="ok",passed=False,correctness={"passed":False})
        self.jobs[job]=d|dict(id=job,board="serving",mode="check",duplicate_of=None,
                             serving_flags=c.flags,created_at=begin.timestamp()+0.1)
    def execute(self,cmd,**kw):
        assert 0<kw["timeout"]<=120
        if cmd[1:]==["serve","status","--json"]:
            self.reads+=1;self.on_quota(self.reads)
            return subprocess.CompletedProcess(cmd,0,json.dumps(self.quota),"")
        if len(cmd)==5 and cmd[1:3]==["serve","status"] and cmd[-1]=="--json":
            self.queries.append(cmd[3]);job=self.jobs.get(cmd[3])
            return subprocess.CompletedProcess(cmd,0 if job else 1,json.dumps(job),"")
        assert cmd[1:3]==["serve","check"] and cmd[-1]=="--no-wait"
        self.attempts.append(cmd)
        if self.timeout: raise subprocess.TimeoutExpired(cmd,10)
        line=f"✓ submitted {cmd[3]} as a check run at 16 users · job {ACCEPTED}"
        if self.bad_acceptance:line=line.replace("16 users","32 users")
        return subprocess.CompletedProcess(cmd,0,line,"")
    def offer(self):
        return r.run_once(okc="mock",runs=self.runs,clock=lambda:self.now,execute=self.execute)
    @property
    def receipt(self):
        return self.runs/(r.NAME+".submit.txt")

class Checks(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);return Fake(tmp.name)
    def withheld(self,f):
        self.assertFalse(f.offer().startswith("submitted "))
        self.assertFalse(f.attempts or f.receipt.exists())

    def test_full_failure_schema_matrix_and_exact_private_request(self):
        for a in ("crash","timeout","incorrect"):
            for b in ("crash","timeout","incorrect"):
                for first_state in ("pending","pass","crash"):
                    with self.subTest(a=a,b=b,first_state=first_state):
                        f=self.fixture()
                        for req,j,state in zip(r.REQUIRED,IDS,(first_state,a,b)): f.set_state(req,j,state)
                        with patch.object(r.measured,"submit_once",wraps=r.measured.submit_once) as submit:
                            self.assertEqual(f.offer(),"submitted "+ACCEPTED)
                            self.assertEqual(submit.call_args.kwargs["reserve_checks"],1)
                        target=r.ROOT/"serving_v0523/candidates"/r.V163.directory/"patch.diff"
                        self.assertEqual(f.attempts,[["mock","serve","check",str(target),"--users","16","--flag","enable-layernorm-sp",
                            "--flag","moe-runner-backend=humming","--flag","muse-sparse-prompt-exchange","--flag",
                            "speculative-moe-runner-backend=marlin","--flag","swa-full-tokens-ratio=0.25","--no-wait"]])
                        self.assertEqual(r.slots.dated_receipt(f.runs,r.OWN),ACCEPTED)
                        self.assertEqual(f.offer(),"attempt recorded")
                        self.assertEqual(len(f.attempts),1)
    def test_reference_identity_fields(self):
        for i in range(3):
            for key,val in (("id","9"*16),("board","another"),("mode","run"),("users",True),("users",0),
                            ("created_at",float("nan")),("created_at",True),("created_at",0),("duplicate_of","another"),
                            ("serving_flags",{})):
                with self.subTest(i=i,k=key,val=val):
                    f=self.fixture();f.jobs[IDS[i]][key]=val;self.withheld(f)
        f=self.fixture();f.jobs[IDS[0]]["refunded"]=True;self.withheld(f)
    def test_terminal_results_and_local_corroboration(self):
        for i in (1,2):
            for state in ("pass","pending","running","cancelled","withdrawn","unknown"):
                f=self.fixture();f.set_state(r.REQUIRED[i],IDS[i],state if state in ("pass","pending") else "pending")
                if state not in ("pass","pending"):f.jobs[IDS[i]]["status"]=state
                self.withheld(f)
            for key,value in (("patch",{"sha256":"0"*64}),("passed",True),("concurrency",[True]),("serving_flags",{}),
                              ("status","unrecognized")):
                f=self.fixture();f.jobs[IDS[i]]["result"][key]=value;self.withheld(f)
            for key,value in (("error",""),("rungs",[{"reps":1}]),("passed",True),("concurrency",[999]),("patch",{})):
                f=self.fixture();f.set_state(r.REQUIRED[i],IDS[i],"timeout");f.jobs[IDS[i]]["result"][key]=value;self.withheld(f)
        for i,req in enumerate(r.REQUIRED):
            for broken in ("missing","shared-job","skip","unconfirmed","wrong-load","wrong-sha","wrong-flags","wrong-day","ambiguous"):
                f=self.fixture();p=f.runs/(req[2]+".submit.txt");s=p.read_text()
                if broken=="missing":p.unlink()
                elif broken=="skip":(f.runs/(req[2]+".skip.json")).write_text("{}")
                elif broken=="ambiguous":(f.runs/"other.submit.txt").write_text("job "+IDS[i])
                elif broken=="shared-job":p.write_text(s.replace(IDS[i],IDS[(i+1)%3]))
                elif broken=="unconfirmed":p.write_text(s.replace("CLI exit code: 0","HTTP 422 refused\nCLI exit code: 1"))
                else:
                    h,_,body=s.partition("\n");meta=json.loads(h)
                    key,value={"wrong-load":("users",0),"wrong-sha":("sha256","0"*64),"wrong-flags":("flags",{}),
                               "wrong-day":("attempted_at",r.END.isoformat())}[broken]
                    meta[key]=value;p.write_text(json.dumps(meta)+"\n"+body)
                self.withheld(f)
    def test_competing_attempts_and_double_failure_wins_only_second(self):
        for peer in r.PEERS:
            for path,body in [(peer[2]+".submit.txt",""),(peer[2]+".skip.json","{}"),
                              ("legacy.submit.txt",peer[0].sha256),("9"*16+".json",peer[0].sha256)]:
                f=self.fixture();(f.runs/path).write_text(body);self.withheld(f)
        for left in (0,1,3):
            f=self.fixture();f.quota["limits"]["checks"].update(used=3-left,left=left);self.withheld(f)
        for key,val in (("used",True),("left",2.0),("per_day",4)):
            f=self.fixture();f.quota["limits"]["checks"][key]=val;self.withheld(f)
        f=self.fixture();f.quota["resets_at"]="yesterday";self.withheld(f)
        f=self.fixture(); self.assertEqual(f.offer(),"submitted "+ACCEPTED)
        # All other November 20 waiters decline on this exact control state and one
        # consumed fallback. No peer is allowed to spend the leftover third check.
        f.quota["limits"]["checks"]=r.slots.THIRD.copy()
        with patch.object(r.previous.second,"plan",return_value={}),patch.object(r.previous.third,"plan",return_value={}):
            for lane in (r.previous.second,r.previous.third):
                self.assertFalse(lane.run_once(okc="mock",runs=f.runs,clock=lambda:f.now,
                                   execute=f.execute).startswith("submitted "))
        self.assertIsNone(r.previous.local_position(f.runs))
        self.assertEqual(len(f.attempts),1)
    def test_postquota_change_and_liveness(self):
        for n in (1,2):
            for change in ("peer","quota","deadline","control","disappeared","torn-first"):
                f=self.fixture()
                def flip(read):
                    if n!=read:return
                    if change=="peer":(f.runs/(r.PEERS[0][2]+".submit.txt")).write_text("")
                    if change=="quota":f.quota["limits"]["checks"]=r.slots.THIRD
                    if change=="deadline":f.now=r.END
                    if change=="control":f.set_state(r.REQUIRED[2],IDS[2],"pending")
                    if change=="disappeared":f.jobs.pop(IDS[1])
                    if change=="torn-first":(f.runs/(r.REQUIRED[0][2]+".submit.txt")).write_text("")
                f.on_quota=flip;self.withheld(f)
        f=self.fixture()
        f.set_state(r.REQUIRED[1],IDS[1],"pending");self.withheld(f)
        f.set_state(r.REQUIRED[1],IDS[1],"crash");self.assertEqual(f.offer(),"submitted "+ACCEPTED)
    def test_shared_lock_process_local_file_and_concurrent_submission(self):
        f=self.fixture()
        with r.slots.arbitrate(f.runs):
            self.withheld(f);self.assertEqual((f.reads,f.queries),(0,[]))
        accepted=[];started=threading.Event();release=threading.Event()
        def coordinate(n):
            if n==1:started.set();self.assertTrue(release.wait(5))
        f.on_quota=coordinate
        t=threading.Thread(target=lambda:accepted.append(f.offer()));t.start();self.assertTrue(started.wait(5))
        self.assertEqual(f.offer(),"another allocation poll is active")
        self.assertFalse(f.attempts)
        release.set();t.join(8);self.assertFalse(t.is_alive())
        self.assertEqual(accepted,["submitted "+ACCEPTED]);self.assertEqual(len(f.attempts),1)
    def test_cutoff_outer_timeout_and_accepted_receipt_manual_reconciliation(self):
        for now in (r.START-timedelta(microseconds=1),r.END):
            f=self.fixture();f.now=now;self.withheld(f);self.assertEqual(f.queries,[])
        for switch in ("bad_acceptance","timeout"):
            f=self.fixture();setattr(f,switch,True);self.assertTrue(f.offer().startswith("attempt "))
            self.assertEqual(f.offer(),"attempt recorded");self.assertEqual(len(f.attempts),1)
        f=self.fixture();f.receipt.touch();self.assertEqual(f.offer(),"attempt recorded");self.assertFalse(f.attempts)
    def test_trusted_actual_archived_records_and_manifest(self):
        self.assertEqual(REAL_CRASH["mode"],"check");self.assertEqual(REAL_CRASH["result"]["status"],"crashed")
        self.assertEqual(set(REAL_TIMEOUT["result"]),{"status","rungs","error"});self.assertTrue(REAL_TIMEOUT["refunded"])
        self.assertEqual(REAL_PASS["result"]["rungs"][0]["tok_s_per_gpu"],17349.9)
        plan=r.plan()
        self.assertEqual((plan["required_quota"],plan["reserve_checks"],plan["users"]),(r.slots.SECOND,1,16))
        self.assertEqual((plan["comparison_job"],plan["comparison_tokens_s_gpu"],plan["comparison_sha256"]),
                         ("ac89fddc4ee147ab",17349.9,r.V64.sha256))
        self.assertEqual(plan["frozen_comparison_report_sha256"],plan["current_comparison_report_sha256"])

if __name__=="__main__":
    unittest.main()
