"""Exercise the disjoint, post-V168-hardware-failure V180 private standby without a network."""
import copy
import json
import shutil
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import submit_nov20_v180_check as r
import test_nov20_v163_check as base

V168_JOB = "5555555555555555"
CHILD = "7777777777777777"
IDS = (*base.IDS, base.ACCEPTED, V168_JOB, r.nov19.V68_JOB)


class Judge(base.Fake):
    def __init__(self,name):
        super().__init__(name)
        self.now=r.nov19.START
        self.quota["resets_at"]=r.nov19.RESET
        self.child=CHILD
        self.events=[]
        with patch.object(r.nov19,"dependency",return_value=("passed","emulated positive controls with V93 queued")):
            marker=r.nov19._offer(r.peer.V168,self.runs,r.ROOT,"mock",lambda:self.now,self.execute)
        assert marker=="submitted "+V168_JOB,marker
        self.set_state(r.peer.PARENT,V168_JOB,"crash")
        self.now=r.START+timedelta(minutes=3)
        self.quota["resets_at"]=r.RESET
        assert base.Fake.offer(self)=="submitted "+base.ACCEPTED
        self.set_state(r.fallback.OWN,base.ACCEPTED,"pending")
        self.jobs[base.ACCEPTED]["created_at"]=self.now.timestamp()
        self.now+=timedelta(minutes=1)
        self.quota["limits"]["checks"]=r.slots.THIRD.copy()
        ref=r.REFERENCE[2]+".submit.txt"
        shutil.copyfile(r.RUNS/ref,self.runs/ref)
        self.jobs[r.nov19.V68_JOB]=json.loads((r.RUNS/(r.nov19.V68_JOB+".json")).read_text())
        self.attempts.clear();self.events.clear();self.queries.clear();self.reads=0

    def set_state(self,item,job,state):
        super().set_state(item,job,state)
        if item == r.peer.PARENT and state == "incorrect":
            quality = copy.deepcopy(base.REAL_PASS["result"]["correctness"])
            quality.update(passed=False, prefill_kl_mean=0.5)
            self.jobs[job]["result"]["correctness"] = quality

    def execute(self,cmd,**kw):
        self.events.append(cmd[1:])
        response=super().execute(cmd,**kw)
        if cmd[1:3]==["serve","check"]:
            job={r.V180.directory:self.child,r.peer.V168.directory:V168_JOB}.get(Path(cmd[3]).parent.name,base.ACCEPTED)
            response.stdout=response.stdout.replace(base.ACCEPTED,job)
        return response

    def offer(self):
        return r.run_once(okc="mock",runs=self.runs,clock=lambda:self.now,execute=self.execute)

    @property
    def receipt(self):
        return self.runs/(r.NAME+".submit.txt")


class Checks(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);return Judge(tmp.name)

    def held(self,f):
        self.assertFalse(f.offer().startswith("submitted "))
        self.assertFalse(f.attempts or f.receipt.exists())

    def test_measured_parent_and_private_slot_only(self):
        for state in ("crash","incorrect"):
            f=self.fixture();f.set_state(r.peer.PARENT,V168_JOB,state)
            self.assertIsNone(r.earlier.local_position(f.runs))
            self.assertFalse(r.peer.run_once(okc="mock",runs=f.runs,clock=lambda:f.now,execute=f.execute).startswith("submitted "))
            with patch.object(r.measured,"submit_once",wraps=r.measured.submit_once) as submit:
                self.assertEqual(f.offer(),"submitted "+CHILD)
                self.assertEqual(submit.call_args.kwargs["reserve_checks"],0)
            proof=json.loads(f.receipt.read_text().splitlines()[0])["evidence"]
            self.assertEqual(proof["v168_hardware_failure"]["job"],V168_JOB)
            self.assertEqual((proof["comparison_job"],proof["comparison_rung"]["tok_s_per_gpu"]),(r.nov19.V68_JOB,17512.9))
            dest=str(r.ROOT/"serving_v0523/candidates"/r.V180.directory/"patch.diff")
            cmd=["mock","serve","check",dest,"--users","16"]
            for k,v in sorted(r.nov19.V68.flags.items()):cmd.extend(["--flag",k[2:]+("="+v if v else "")])
            self.assertEqual(f.attempts,[cmd+["--no-wait"]]);self.assertEqual(f.events[-2],["serve","status","--json"])
            self.assertEqual(r.slots.bound_receipt(f.runs,r.OWN),CHILD);self.assertEqual(f.offer(),"attempt recorded")

    def test_v168_requires_completed_physical_failure(self):
        for state in ("pending","pass","timeout"):
            f=self.fixture();f.set_state(r.peer.PARENT,V168_JOB,state);self.held(f);self.assertEqual(f.queries,[V168_JOB])
        for top in ("failed","canceled","withdrawn","running","queued"):
            f=self.fixture();f.jobs[V168_JOB]["status"]=top;self.held(f)
        for key,value in (("gpus",None),("gpus",True),("gpus",7),("node",""),("pod_s",0),("pod_s",float("nan")),
                ("startup_s",True),("startup_s",float("inf")),("startup_log",{}),("startup_log",{"lines":0}),
                ("crash",{}),("crash",{"events":[]}),("crash",{"events":[None]}),("crash",{"events":[" "]}),("status","timeout"),("status","error"),("passed",True),
                ("serving_flags",{}),("patch",{}),("concurrency",[True])):
            f=self.fixture();f.jobs[V168_JOB]["result"][key]=value;self.held(f)
        for flag in (True,None,0,1):
            f=self.fixture();f.jobs[V168_JOB]["refunded"]=flag;self.held(f)
        f=self.fixture();f.set_state(r.peer.PARENT,V168_JOB,"incorrect")
        f.jobs[V168_JOB]["result"]["correctness"]={"passed":True};self.held(f)

    def test_traceback_event_and_real_metric_evidence(self):
        recorded=json.loads((r.RUNS/"30bb36feb528416f.json").read_text())["result"]
        for drift in ("none","missing_frame","missing_type","foreign","empty_message","capture_unknown","infra"):
            f=self.fixture();result=f.jobs[V168_JOB]["result"]
            result.update(crash=copy.deepcopy(recorded["crash"]),startup_log=recorded["startup_log"],error=recorded["error"])
            result.pop("startup_s",None)
            if drift=="none":
                self.assertEqual(f.offer(),"submitted "+CHILD);continue
            if drift=="missing_frame":result["crash"]["frames"]=[]
            if drift=="missing_type":result["crash"]["type"]=None
            if drift=="foreign":result["crash"]["frames"]=[{"file":"cloud/prepare.py","line":9,"function":"setup"}]
            if drift=="empty_message":result["crash"]["message"]=" "
            if drift=="capture_unknown":result["crash"]["captured"]=None
            if drift=="infra":f.jobs[V168_JOB]["refunded"]=True
            self.held(f)
        for fault in ("unattributed","unknown_event","wrong_cut"):
            f=self.fixture();result=f.jobs[V168_JOB]["result"]
            if fault=="unattributed":result["error"]="lost node"
            if fault=="unknown_event":result["crash"]["events"]=["network timeout"]
            if fault=="wrong_cut":result["startup_log"]["cut"]="container startup"
            self.held(f)
        for fault in ("memory","memory_limit","memory_unknown","memory_nan","threshold","ttft","ttft_stock",
                "ttft_ratio","ttft_load","ttft_limit","ttft_truncated","ttft_drift","ttft_zero","ttft_strings",
                "quality_flag","quality_references","quality_insufficient","quality_bounds","quality_nan"):
            f=self.fixture();f.set_state(r.peer.PARENT,V168_JOB,"pass");result=f.jobs[V168_JOB]["result"]
            result["passed"]=False
            if fault.startswith("memory") or fault=="threshold":
                result["gpu_memory"]={"limit_gb":2,"over_stock_gb":2.1}
                if fault=="memory_limit":result["gpu_memory"]["limit_gb"]=1
                if fault=="memory_unknown":result["gpu_memory"].pop("limit_gb")
                if fault=="memory_nan":result["gpu_memory"]["over_stock_gb"]=float("inf")
                if fault=="threshold":result["gpu_memory"]["over_stock_gb"]=2.0
            elif fault.startswith("ttft"):
                rung=result["rungs"][0];paired=result["vs_stock"]["rungs"][0]
                rung["ttft_first_p90_s"]=30;paired["ttft_first_p90"]=3;paired["stock"]["ttft_first_p90_s"]=10
                if fault=="ttft_stock":paired["stock"]={}
                if fault=="ttft_ratio":paired["ttft_first_p90"]=2
                if fault=="ttft_load":paired["concurrency"]=32
                if fault=="ttft_limit":result["vs_stock"]["ttft_ratio_limit"]=1
                if fault=="ttft_truncated":rung["truncated"]=True
                if fault=="ttft_drift":paired["ttft_first_p90"]=2.5
                if fault=="ttft_zero":paired["stock"]["ttft_first_p90_s"]=0
                if fault=="ttft_strings":rung["ttft_first_p90_s"]="30"
            else:
                result["correctness"]["passed"]=False
                if fault!="quality_flag":result["correctness"]["decode_kl_mean"]=2.3
                if fault=="quality_references":result["correctness"]["references"]=0
                if fault=="quality_insufficient":result["correctness"]["references"]=23
                if fault=="quality_bounds":result["correctness"]["bounds"]={}
                if fault=="quality_nan":result["correctness"]["decode_kl_mean"]=float("nan")
            if fault in ("memory","ttft"):self.assertEqual(f.offer(),"submitted "+CHILD)
            else:self.held(f)

    def test_exact_official_ids_controls_and_nonrefunded_quota(self):
        for i,job in enumerate(IDS):
            for field,value in (("id","8"*16),("board","other"),("mode","run"),("users",True),("duplicate_of","8"*16),
                                ("created_at",0),("serving_flags",{})):
                f=self.fixture();f.jobs[job][field]=value;self.held(f)
            f=self.fixture();(f.runs/"other.submit.txt").write_text("job "+job);self.held(f)
            f=self.fixture();(f.runs/(r.CHECKS[i][2]+".submit.txt")).unlink();self.held(f)
        for i in (1,2):
            for state in ("pending","pass"):
                f=self.fixture();f.set_state(r.CHECKS[i],IDS[i],state);self.held(f)
        for job in (IDS[0],IDS[3],IDS[4],IDS[5]):
            f=self.fixture();f.jobs[job]["refunded"]=True;self.held(f)
        f=self.fixture();f.jobs[IDS[0]]["created_at"]=f.jobs[IDS[3]]["created_at"]+1;self.held(f)

    def test_reference_must_still_be_the_measured_private_rung(self):
        for key,value in (("truncated",True),("reps",2),("reps",True),("tok_s_per_gpu",17512.8),
                          ("tok_s_per_user_p50",0),("ttft_first_p90_s",float("nan"))):
            f=self.fixture();f.jobs[IDS[5]]["result"]["rungs"][0][key]=value;self.held(f)
        for key,value in (("acceptance",float("inf")),("cache_hit",-10),("spread",-1)):
            f=self.fixture();f.jobs[IDS[5]]["result"]["rungs"][0][key]=value;self.held(f)
        f=self.fixture();f.jobs[IDS[5]]["result"]["rungs"][0].pop("ttft_first_p90_s");self.held(f)
        for k,v in (("short_outputs",1),("later_turns",{"ran":True,"passed":False}),("correctness",{"passed":False})):
            f=self.fixture();f.jobs[IDS[5]]["result"][k]=v;self.held(f)

    def test_disjoint_receipts_including_restores_and_conflicting_decisions(self):
        for req in (*r.earlier.SLOT_HISTORY,*r.fallback.PEERS,r.peer.OWN,r.earlier.OWN,r.OWN):
            for file,txt in ((req[2]+".submit.txt",""),(req[2]+".skip.json","{}"),("archived.submit.json",req[0].sha256),
                             ("9"*16+".json",req[0].sha256)):
                f=self.fixture();(f.runs/file).write_text(txt)
                if f.runs/file == f.receipt:
                    self.assertEqual(f.offer(), "attempt recorded")
                    self.assertEqual(f.receipt.read_text(), txt)
                    self.assertFalse(f.attempts)
                else:
                    self.held(f)
        for req,k,v in [(r.peer.PARENT,"decision_job",r.nov19.V68_JOB),(r.peer.PARENT,"comparison_job","8"*16),
                (r.peer.PARENT,"comparison_report_sha256","invalid"),(r.peer.PARENT,"exclusive_alternatives",[]),
                (r.fallback.OWN,"slot",True),(r.fallback.OWN,"failed_controls",{}),(r.fallback.OWN,"v129_job",V168_JOB)]:
            f=self.fixture();p=f.runs/(req[2]+".submit.txt");s,_,tail=p.read_text().partition("\n")
            h=json.loads(s);h["evidence"][k]=v;p.write_text(json.dumps(h)+"\n"+tail);self.held(f)
        for req in (r.peer.PARENT,r.fallback.OWN):
            f=self.fixture();p=f.runs/(req[2]+".submit.txt");p.write_text("[]\n");self.held(f)
        f=self.fixture();f.write((r.peer.V168,16,"restored",r.nov19.START,r.nov19.END),"8"*16);self.held(f)

    def test_final_preflight_and_time_bounds(self):
        for t in (r.START-timedelta(microseconds=1),r.END):
            f=self.fixture();f.now=t;self.held(f);self.assertFalse(f.queries)
        for target in (1,2):
            for event in ("allocation","bool","cap","reset","deadline","peer","lost","recovery","refund"):
                if target==2 and event in ("lost","recovery","refund"):continue
                f=self.fixture()
                def alter(n):
                    if n!=target:return
                    if event=="allocation":f.quota["limits"]["checks"].update(used=1,left=2)
                    if event=="bool":f.quota["limits"]["checks"]["left"]=True
                    if event=="cap":f.quota["limits"]["checks"]["per_day"]=4
                    if event=="reset":f.quota["resets_at"]="other"
                    if event=="deadline":f.now=r.END
                    if event=="peer":(f.runs/(r.peer.NAME+".submit.txt")).touch()
                    if event=="lost":f.jobs.pop(IDS[5])
                    if event=="recovery":f.set_state(r.peer.PARENT,V168_JOB,"pass")
                    if event=="refund":f.jobs[V168_JOB]["refunded"]=True
                f.on_quota=alter;self.held(f)

    def test_concurrency_idempotency_and_ambiguous_acceptance(self):
        f=self.fixture()
        with r.slots.arbitrate(f.runs):
            self.held(f);self.assertFalse(f.queries)
        arrived=threading.Event();release=threading.Event();results=[]
        def hold(n):
            if n==1:arrived.set();self.assertTrue(release.wait(15))
        f.on_quota=hold
        t=threading.Thread(target=lambda:results.append(f.offer()));t.start();self.assertTrue(arrived.wait(15))
        self.assertEqual(f.offer(),"another allocation poll is active");release.set();t.join(15)
        self.assertEqual(results,["submitted "+CHILD])
        for job in IDS:
            f=self.fixture();f.child=job;self.assertIn("manual reconciliation",f.offer())
            self.assertEqual(f.offer(),"attempt recorded");self.assertEqual(len(f.attempts),1)
        for m in ("bad_acceptance","timeout"):
            f=self.fixture();setattr(f,m,True);self.assertTrue(f.offer().startswith("attempt "))
            self.assertEqual(f.offer(),"attempt recorded")

    def test_release_plan(self):
        p=r.plan();self.assertEqual((p["tree"],p["users"],p["mode"]),(r.TREE,16,"check"))
        self.assertEqual(p["flags"],r.nov19.V68.flags)
        self.assertEqual(p["comparison"]["comparison_rung"]["tok_s_per_gpu"],17512.9)
        self.assertEqual(p["required_quota"],{"used":2,"left":1,"per_day":3})


if __name__=="__main__":
    unittest.main()
