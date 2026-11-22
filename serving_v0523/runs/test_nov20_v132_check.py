"""Exercise the guarded second check using fake CLI responses and isolated receipts."""
import concurrent.futures
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path

import submit_nov20_v132_check as m

JOBS=("1111111111111111","2222222222222222")
ACCEPTED="3333333333333333"


class Fixture:
    def __init__(self,root):
        self.root=root; self.runs=root/"runs";self.runs.mkdir()
        self.now=m.START+timedelta(seconds=5)
        self.calls=[];self.jobs={};self.docs=[]
        self.quota={"resets_at":m.RESET,"limits":{"checks":{"per_day":3,"used":1,"left":2}}}
        self.after_quota=lambda:None
        self.check_rc=0;self.status_rc=0;self.timeout=False
        self.check_text=f"submitted /fixture/{m.V132.directory}/patch.diff as a check run at 32 users · job {ACCEPTED}"
        for c in (m.first.V129,m.V120b,m.V132):
            dest=root/"serving_v0523/candidates"/c.directory
            dest.mkdir(parents=True)
            origin=m.ROOT/"serving_v0523/candidates"/c.directory
            for name in ("patch.diff","validation.json"):shutil.copyfile(origin/name,dest/name)
        self.patch=root/"serving_v0523/candidates"/m.V132.directory/"patch.diff"
        # Reuse the official passing comparator receipt schema, replacing fixture IDs/geometry.
        template=json.loads((m.RUNS/"04b01f958aa5487c.json").read_text())
        for index,((c,users,name),job) in enumerate(zip(m.PARENTS,JOBS)):
            at=m.START if index==0 else m.START-timedelta(days=1)
            h={"attempted_at":at.isoformat(),"candidate":c.name,"sha256":c.sha256,"users":users,"flags":c.flags}
            p=self.runs/(name+".submit.txt");self.docs.append(p)
            p.write_text(json.dumps(h)+"\nsubmitted /fixture/"+c.directory+f"/patch.diff as a check run at {users} users · job {job}\nCLI exit code: 0\n")
            d={"board":"serving","mode":"check","id":job,"users":users,"serving_flags":c.flags,
               "created_at":at.timestamp()+1,"status":"queued" if index==0 else "done","result":None}
            if index:
                r=template["result"]; r["patch"]["sha256"]=c.sha256;r["serving_flags"]=c.flags;r["concurrency"]=[32]
                r["rungs"][0]["concurrency"]=32;d["result"]=r
            self.jobs[job]=d

    def execute(self,cmd,**kwargs):
        self.calls.append(cmd)
        if cmd[1:]==["serve","status","--json"]:
            self.after_quota()
            return subprocess.CompletedProcess(cmd,self.status_rc,json.dumps(self.quota),"")
        if cmd[1:3]==["serve","status"]:
            return subprocess.CompletedProcess(cmd,self.status_rc,json.dumps(self.jobs[cmd[3]]),"")
        assert cmd[1:3]==["serve","check"]
        if self.timeout:raise subprocess.TimeoutExpired(cmd,kwargs["timeout"])
        return subprocess.CompletedProcess(cmd,self.check_rc,self.check_text,"")

    def run(self):
        return m.run_once(okc="dummy",root=self.root,runs=self.runs,clock=lambda:self.now,execute=self.execute)

    @property
    def checks(self):return [cmd for cmd in self.calls if cmd[1:3]==["serve","check"]]


class Scenarios(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Fixture(Path(tmp.name))

    def denied(self,f):
        self.assertFalse(f.run().startswith("submitted"))
        self.assertFalse(f.checks)

    def test_exact_once_and_private_flags(self):
        f=self.fixture()
        self.assertEqual(f.run(),"submitted "+ACCEPTED)
        command=f.checks[0]
        self.assertEqual(command[:6],["dummy","serve","check",str(f.patch),"--users","32"])
        self.assertEqual(command[-1],"--no-wait")
        self.assertEqual(command[6:-1:2],["--flag"]*12)
        self.assertEqual(set(command[7:-1:2]),{k[2:]+("="+v if v else "") for k,v in m.V132.flags.items()})
        self.assertEqual(f.run(),"attempt recorded")
        header=json.loads((f.runs/(m.NAME+".submit.txt")).read_text().splitlines()[0])
        self.assertEqual(header["evidence"],{"first_check_job":JOBS[0],"comparison_job":JOBS[1],"comparison_patch":m.V120b.sha256})

    def test_dates_and_first_gate(self):
        for when in (m.START-timedelta(microseconds=1),m.END,m.END+timedelta(days=1)):
            with self.subTest(when=when):
                f=self.fixture();f.now=when;self.denied(f);self.assertFalse(f.calls)
        for mod in ("missing","wrong_flags","wrong_sha","bad_load","prior_day","two_ids","ambiguous_return","live_id","live_load",
                    "live_flags","unknown_state","aliased_job","wrong_board","old_created","no_created","duplicate","wrong_summary"):
            with self.subTest(mod=mod):
                f=self.fixture(); path=f.docs[0]; raw=path.read_text();h=json.loads(raw.splitlines()[0]); d=f.jobs[JOBS[0]]
                if mod=="missing":path.unlink()
                elif mod=="wrong_flags":h["flags"]={}
                elif mod=="wrong_sha":h["sha256"]="-"
                elif mod=="bad_load":h["users"]=True
                elif mod=="prior_day":h["attempted_at"]=(m.START-timedelta(days=1)).isoformat()
                elif mod=="two_ids":raw=raw.replace(" · job "," · job 4444444444444444 · job ")
                elif mod=="ambiguous_return":raw=raw.replace("exit code: 0","exit code: 1")
                elif mod=="live_id":d["id"]=JOBS[1]
                elif mod=="live_load":d["users"]=32
                elif mod=="live_flags":d["serving_flags"]={}
                elif mod=="unknown_state":d["status"]="pending_unknown"
                elif mod=="aliased_job":raw=raw.replace(JOBS[0],JOBS[1])
                elif mod=="wrong_board":d["mode"]="run"
                elif mod=="old_created":d["created_at"]-=86400
                elif mod=="no_created":d.pop("created_at")
                elif mod=="duplicate":d["duplicate_of"]=JOBS[1]
                elif mod=="wrong_summary":raw=raw.replace("at 1 users","at 16 users")
                if mod!="missing":path.write_text(json.dumps(h)+"\n"+raw.partition("\n")[2])
                self.denied(f)
        for state in ("running","failed","cancelled"):
            with self.subTest(accepted_state=state):
                f=self.fixture();f.jobs[JOBS[0]]["status"]=state;self.assertTrue(f.run().startswith("submitted"))

    def test_full_measured_parent_required(self):
        for mod in ("no_receipt","skip","pending","failed","passed_false","no_patch","wrong_sha","wrong_flags","wrong_users",
                    "empty_rungs","extra_rung","wrong_rung","truncated","missing_reps","nonpositive_reps","nan","infinite","boolean",
                    "negative","incorrect","bad_correctness","job_collision","empty_identity","rc_failure"):
            with self.subTest(mod=mod):
                f=self.fixture();d=f.jobs[JOBS[1]];r=d["result"];u=r["rungs"][0]
                if mod=="no_receipt":f.docs[1].unlink()
                elif mod=="skip":(f.runs/(m.prefix(m.V120b)+".skip.json")).write_text("{}")
                elif mod=="pending":d["status"]="queued";d["result"]=None
                elif mod=="failed":d["status"]="failed"
                elif mod=="passed_false":r["passed"]=False
                elif mod=="no_patch":r.pop("patch")
                elif mod=="wrong_sha":r["patch"]["sha256"]=m.V132.sha256
                elif mod=="wrong_flags":r["serving_flags"]=m.V132.flags
                elif mod=="wrong_users":r["concurrency"]=[16]
                elif mod=="empty_rungs":r["rungs"]=[]
                elif mod=="extra_rung":r["rungs"].append(dict(u))
                elif mod=="wrong_rung":u["concurrency"]=16
                elif mod=="truncated":u["truncated"]=True
                elif mod=="missing_reps":u.pop("reps")
                elif mod=="nonpositive_reps":u["reps"]=0
                elif mod=="nan":u["tok_s_per_gpu"]=float("nan")
                elif mod=="infinite":u["tok_s_per_user_p50"]=float("inf")
                elif mod=="boolean":u["tok_s_per_user_e2e"]=True
                elif mod=="negative":u["tok_s_per_gpu"]=-1
                elif mod=="incorrect":r["correctness"]["passed"]=False
                elif mod=="bad_correctness":r["correctness"]=None
                elif mod=="job_collision":f.docs[1].write_text(f.docs[1].read_text().replace(JOBS[1],JOBS[0]))
                elif mod=="empty_identity":d["id"]=""
                elif mod=="rc_failure":f.status_rc=1
                self.denied(f)

    def test_quota_reset_bounds_and_duplicate_histories(self):
        for key,val in (("used",0),("used",2),("used",True),("left",1),("left",3),("left","2"),("per_day",2),("per_day",None),
                        ("reset","2026-11-20T00:00:00Z"),("reset",None)):
            with self.subTest(key=key,val=val):
                f=self.fixture()
                if key=="reset":f.quota["resets_at"]=val
                else:f.quota["limits"]["checks"][key]=val
                self.denied(f)
        for name,raw in (("v132_previous_check32.submit.txt",""),("another.submit.txt",'{"candidate":"V132"}'),
                         ("alias.submit.txt",m.V132.sha256),("0123456789abcdef.json",m.V132.sha256),
                         ("historical/v132_nov20_check32.submit.txt",""),("odd.submit.txt",'{"users":1}\nunknown V132 attempt')):
            with self.subTest(name=name):
                f=self.fixture();target=f.runs/name;target.parent.mkdir(exist_ok=True);target.write_text(raw);self.denied(f);self.assertFalse(f.calls)

    def test_late_changes_and_timeouts(self):
        for change in ("deadline","parent_receipt","prior","patch"):
            with self.subTest(change=change):
                f=self.fixture()
                def callback():
                    if change=="deadline":f.now=m.END
                    elif change=="parent_receipt":f.docs[1].unlink()
                    elif change=="prior":(f.runs/"v132_old.submit.txt").write_text("")
                    else:f.patch.write_text("changed")
                f.after_quota=callback;self.denied(f)
        for mode in ("timeout","no_id","error","two_ids"):
            with self.subTest(mode=mode):
                f=self.fixture()
                if mode=="timeout":f.timeout=True
                elif mode=="no_id":f.check_text="partial"
                elif mode=="error":f.check_rc=1
                else:f.check_text+=" job 4444444444444444"
                self.assertTrue(f.run().startswith("attempt "))
                self.assertEqual(f.run(),"attempt recorded");self.assertEqual(len(f.checks),1)

    def test_frozen_inputs_and_concurrent_offers(self):
        for c in (m.first.V129,m.V120b,m.V132):
            with self.subTest(c=c.name):
                f=self.fixture(); (f.patch.parent.parent/c.directory/"patch.diff").write_text("bad")
                with self.assertRaises(ValueError):f.run()
                self.assertFalse(f.calls)
        f=self.fixture(); entered=threading.Event(); release=threading.Event()
        def pause():
            entered.set()
            self.assertTrue(release.wait(5))
        f.after_quota=pause
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending=pool.submit(f.run)
            self.assertTrue(entered.wait(5))
            try:
                self.assertEqual(f.run(),"another allocation poll is active")
            finally:
                release.set()
            self.assertEqual(pending.result(),"submitted "+ACCEPTED)
        self.assertEqual(len(f.checks),1)


if __name__=="__main__":unittest.main(verbosity=2)
