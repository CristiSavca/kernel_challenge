"""No network. Reproduce the final slot, predecessor evidence, closure races and receipts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import submit_nov20_v141_check as runner


JOBS=["1111111111111111","2222222222222222","3333333333333333"]
JOB="4444444444444444"


class Fixture:
    def __init__(self, root):
        self.root=root; self.runs=root/"runs"; self.runs.mkdir()
        self.now=runner.START+timedelta(seconds=10)
        self.calls=[]; self.docs=[]; self.jobs={}; self.quota_code=0
        self.when_quota=lambda:None; self.check_code=0; self.timeout=False
        self.stdout=f"submitted /fixture/{runner.V141.directory}/patch.diff as a check run at 32 users · job {JOB}"
        self.quota={"resets_at":runner.RESET,"limits":{"checks":{"per_day":3,"used":2,"left":1},
                    "runs":{"left":0}}}
        for c in (runner.V64,*[req[0] for req in runner.REQUIRED],runner.V141):
            d=root/"serving_v0523/candidates"/c.directory;d.mkdir(parents=True)
            real=runner.ROOT/"serving_v0523/candidates"/c.directory
            for n in ("validation.json","patch.diff"):
                shutil.copyfile(real/n,d/n)
        shutil.copyfile(runner.RUNS/(runner.first.PARENT_JOB+".json"), self.runs/(runner.first.PARENT_JOB+".json"))
        self.patch=root/"serving_v0523/candidates"/runner.V141.directory/"patch.diff"
        for (c,u,n,start,end),job in zip(runner.REQUIRED,JOBS):
            header=dict(attempted_at=(start+timedelta(seconds=1)).isoformat(),candidate=c.name,sha256=c.sha256,users=u,flags=c.flags)
            path=self.runs/(n+".submit.txt")
            path.write_text(json.dumps(header)+f"\n✓ submitted /test/{c.directory}/patch.diff as a check run at {u} users · job {job}\nCLI exit code: 0\n")
            self.docs.append(path)
            self.jobs[job]=dict(id=job,board="serving",mode="check",status="queued",users=u,
                                serving_flags=c.flags,created_at=(start+timedelta(seconds=2)).timestamp(),
                                duplicate_of=None,result=None)
    def execute(self,cmd,**kwargs):
        self.calls.append((cmd,kwargs))
        if cmd[1:]==["serve","status","--json"]:
            self.when_quota()
            return subprocess.CompletedProcess(cmd,self.quota_code,json.dumps(self.quota),"")
        if cmd[1:3]==["serve","status"]:
            return subprocess.CompletedProcess(cmd,0,json.dumps(self.jobs[cmd[3]]),"")
        assert cmd[1:3]==["serve","check"]
        if self.timeout:raise subprocess.TimeoutExpired(cmd,kwargs["timeout"])
        return subprocess.CompletedProcess(cmd,self.check_code,self.stdout,"")
    def offer(self):
        return runner.run_once(okc="test-cli",clock=lambda:self.now,runs=self.runs,root=self.root,execute=self.execute)
    def change_header(self,i,**changes):
        h,_,tail=self.docs[i].read_text().partition("\n")
        self.docs[i].write_text(json.dumps(json.loads(h)|changes)+"\n"+tail)
    @property
    def checks(self):
        return [c for c,_ in self.calls if c[1:3]==["serve","check"]]


class Cases(unittest.TestCase):
    def make(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        return Fixture(Path(temp.name))
    def denied(self,f):
        self.assertFalse(f.offer().startswith("submitted")); self.assertFalse(f.checks)
    def test_exact_private_once(self):
        f=self.make(); self.assertEqual(f.offer(),"submitted "+JOB)
        self.assertEqual(f.checks, [["test-cli","serve","check",str(f.patch),"--users","32",*runner.first.FLAGS,"--no-wait"]])
        self.assertEqual(f.offer(),"attempt recorded");self.assertEqual(len(f.checks),1)
        h=json.loads((f.runs/(runner.NAME+".submit.txt")).read_text().splitlines()[0])
        self.assertEqual((h["sha256"],h["flags"],h["users"]),(runner.V141.sha256,runner.V64.flags,32))
        self.assertEqual((h["evidence"]["v129_job"],h["evidence"]["v132_job"],h["evidence"]["v139_job"]),tuple(JOBS))
    def test_all_exact_receipts_required(self):
        for i in range(3):
            for mod in ("missing","sha","flags","date","wrong_users","alias","skip","ambiguous","wrong_cli_dir","absent_cli_dir"):
                with self.subTest(i=i,mod=mod):
                    f=self.make();d=f.docs[i]
                    if mod=="missing":d.unlink()
                    elif mod in ("sha","flags","date","wrong_users"):
                        change={"sha":{"sha256":"bad"},"flags":{"flags":{}},"date":{"attempted_at":runner.END.isoformat()},"wrong_users":{"users":True}}
                        f.change_header(i,**change[mod])
                    elif mod=="skip":(f.runs/(runner.REQUIRED[i][2]+".skip.json")).write_text("{}")
                    elif mod=="alias":d.write_text(d.read_text().replace(JOBS[i],JOBS[(i+1)%3]))
                    elif mod=="ambiguous": d.write_text(d.read_text()+f"job {JOB}\n")
                    elif mod=="wrong_cli_dir":d.write_text(d.read_text().replace("/test/","/other/wrong-"))
                    else: d.write_text(d.read_text().replace("/patch.diff as a check run","/unrelated as a run"))
                    self.denied(f);self.assertFalse(f.calls)
    def test_all_live_identities(self):
        for i in range(3):
            for mod in ("id","user","flags","date","duplicate","board","mode","unknown","missing_patch_on_pass"):
                with self.subTest(i=i,mod=mod):
                    f=self.make();j=f.jobs[JOBS[i]]
                    j.update({"id":{"id":JOB},"user":{"users":True},"flags":{"serving_flags":{}},"date":{"created_at":0},
                              "duplicate":{"duplicate_of":JOB},"board":{"board":"other"},"mode":{"mode":"run"},
                              "unknown":{"status":"mystery"},"missing_patch_on_pass":{"status":"done","result":{"status":"ok","passed":True}}}[mod])
                    self.denied(f)
    def test_parent_terminal_and_pending(self):
        for state in ("running","queued","failed","canceled","withdrawn","crashed","passed"):
            with self.subTest(state=state):
                f=self.make();j=f.jobs[JOBS[2]]
                if state in ("crashed","passed"):
                    j.update(status="done",result=dict(status="ok" if state=="passed" else "crashed",
                             passed=state=="passed",patch={"sha256":runner.parent.V139.sha256},
                             concurrency=[4],serving_flags=runner.parent.V139.flags))
                else:j["status"]=state
                if state in ("running","queued","passed"):self.assertTrue(f.offer().startswith("submitted"))
                else:self.assertEqual(f.offer(),"terminal V139 failure");self.assertFalse(f.checks)
    def test_verified_prior_slots_may_finish_independently(self):
        for i in (0,1):
            f=self.make();f.jobs[JOBS[i]]["status"]="failed";self.assertTrue(f.offer().startswith("submitted"))
    def test_reserve_quota(self):
        for vals in ((3,0,3),(3,1,2),(3,2,True),(3,3,0),(4,3,1),(3,-1,1),(3,2,0)):
            f=self.make();f.quota["limits"]["checks"]=dict(zip(("per_day","used","left"),vals));self.denied(f)
        f=self.make();f.quota["resets_at"]="2026-11-20T00:00:00Z";self.denied(f)
        f=self.make();f.quota_code=1;self.denied(f)
    def test_outside_window(self):
        for when in (runner.START-timedelta(microseconds=1),runner.END,runner.END+timedelta(days=1)):
            f=self.make();f.now=when;self.denied(f);self.assertFalse(f.calls)
        f=self.make();f.now=runner.END-timedelta(milliseconds=5);self.assertTrue(f.offer().startswith("submitted"))
        self.assertLessEqual(f.calls[-1][1]["timeout"],.005)
    def test_quota_races(self):
        for mode in ("time","receipt","parent","new_attempt","patch"):
            f=self.make()
            def mutate():
                if mode=="time":f.now=runner.END
                elif mode=="receipt":f.docs[1].unlink()
                elif mode=="parent":f.jobs[JOBS[2]]["status"]="failed"
                elif mode=="new_attempt":(f.runs/"v141_older_check32.submit.txt").write_text("ambiguous")
                else:f.patch.write_text("different")
            f.when_quota=mutate;self.denied(f)
    def test_duplicate_receipts_and_legacy(self):
        for name,txt in (("v141_nov20_check32.submit.txt",""),("old/v141_check32.skip.json",""),
                         ("renamed.submit.txt",runner.V141.sha256),("9999999999999999.json",runner.V141.sha256)):
            f=self.make();d=f.runs/name;d.parent.mkdir(exist_ok=True);d.write_text(txt);self.denied(f);self.assertFalse(f.calls)
        f=self.make();(f.runs/"different.submit.txt").write_text(json.dumps(dict(candidate="v999",sha256="a"*64,evidence={"comparison":runner.V141.sha256}))+"\n")
        self.assertTrue(f.offer().startswith("submitted"))
    def test_ambiguous_attempt_no_retry(self):
        for outcome in ("timeout","two_ids","error","no_id"):
            f=self.make();f.timeout=outcome=="timeout";f.check_code=int(outcome=="error")
            if outcome=="two_ids":f.stdout+=f" and job {JOBS[0]}"
            if outcome=="no_id":f.stdout="?"
            self.assertTrue(f.offer().startswith("attempt "));self.assertEqual(f.offer(),"attempt recorded");self.assertEqual(len(f.checks),1)
    def test_concurrent_launch(self):
        f=self.make()
        with ThreadPoolExecutor(3) as ex: results=list(ex.map(lambda _:f.offer(),range(3)))
        self.assertEqual(len(f.checks),1);self.assertEqual(results.count("submitted "+JOB),1)
    def test_frozen_sources(self):
        for mode in ("candidate","parent","v64","flags"):
            f=self.make();d=f.root/"serving_v0523/candidates"
            if mode=="flags":
                p=f.patch.with_name("validation.json");v=json.loads(p.read_text());v["launch_flags"]={};p.write_text(json.dumps(v))
            else:(d/{"candidate":runner.V141,"parent":runner.parent.V139,"v64":runner.V64}[mode].directory/"patch.diff").write_text("drift")
            with self.assertRaises(ValueError):f.offer()
            self.assertFalse(f.calls)


if __name__=="__main__":unittest.main()
