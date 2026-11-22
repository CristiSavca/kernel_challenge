"""Differential, offline third-slot replay. All claims and locks live in private temporary directories."""
import argparse
from collections import Counter
import contextlib
from datetime import timedelta
from hashlib import sha256
import importlib.util
import io
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
RUNS=HERE.parents[1]/"runs"
OLD_STAGE=RUNS/"history/nov20_v215_staging"
OLD_WORKER=OLD_STAGE/"submit_nov20_v187_check.py"
EXISTING=HERE.parent/"v215_decode_reduction_paired_context/independent_standby.py"
V204="7fcdf9abf39b486a"
CONTROL="3333333333333333"
QUOTA=["serve","status","--json"]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


old_cases=load("independent_v215_reference",EXISTING)
old=old_cases.worker
assert Path(old.__file__).resolve()==OLD_WORKER and digest(OLD_WORKER)=="ee80f0bd9e785b326cd0eaad948c6626a26dcbc5443e4c71f03d875ba801e8f5"


def collected_inputs():
    entries=[]
    def collect(label,**kwargs):
        entries.append((label,kwargs))
        return kwargs["expect"]
    with patch.object(old_cases,"run_case",side_effect=collect),contextlib.redirect_stdout(io.StringIO()):
        old_cases.main()
    return entries


def fixture(directory,kwargs):
    f=old_cases.incumbent.Judge(directory)
    shutil.copyfile(RUNS/old_cases.receipt,f.runs/old_cases.receipt)
    f.jobs[V204]=old_cases.improved()
    if kwargs.get("mutate"):
        old_cases.change(f.jobs[V204],*kwargs["mutate"])
    if kwargs.get("prepare"):
        kwargs["prepare"](f)
    return f


def poll(w,chosen,f,**clocks):
    try:
        return w.run_once(okc="mock",candidate=chosen,runs=f.runs,clock=lambda:f.now,execute=f.execute,**clocks)
    except (ValueError,KeyError,TypeError,OSError,subprocess.SubprocessError) as err:
        # Same boundary as the real polling main. Record the error class and the
        # receipt so that a post-send exception cannot silently become a retry.
        return "recoverable exception "+type(err).__name__


def collect(w,chosen,f,**kwargs):
    result=poll(w,chosen,f)
    header=None
    if f.receipt.exists() and f.receipt.stat().st_size:
        try: header=json.loads(f.receipt.read_text().splitlines()[0])
        except (ValueError,IndexError): pass
    calls=[x.copy() for x in f.attempts]
    events=[x.copy() for x in f.events]
    if calls:
        assert len(calls)==1 and header is not None
        candidate=w.CANDIDATES[header["candidate"]]
        cmd=["mock","serve","check",str(w.ROOT/"serving_v0523/candidates"/candidate.directory/"patch.diff"),"--users","16"]
        for k,v in sorted(candidate.flags.items()):
            cmd.extend(("--flag",k.removeprefix("--")+("="+v if v else "")))
        assert calls[0]==cmd+["--no-wait"] and header["sha256"]==candidate.sha256
        assert header["flags"]==candidate.flags and header["users"]==16 and header["evidence"]["final_quota"]["checks"]==old.slots.THIRD
        assert events[-3:-1]==[QUOTA,["serve","status",CONTROL,"--json"]]
        assert poll(w,chosen,f)=="attempt recorded" and f.attempts==calls
    return dict(result=result,claim=header["candidate"] if header else ("existing" if f.receipt.exists() else None),
                send=len(calls),queries=[e for e in events if e[1]=="status"],proof=(header or {}).get("evidence"))


def err(j, target, phase, kind):
    original=j.execute
    n=0
    def mocked(cmd,**kw):
        nonlocal n
        assert cmd[0]=="mock"
        matching=cmd[1:]==target if isinstance(target,list) else cmd[1:3]==["serve","check"]
        if matching:
            n+=1
        if not matching or n!=phase:
            return original(cmd,**kw)
        response=original(cmd,**kw)
        if kind=="timeout": raise subprocess.TimeoutExpired(cmd,kw["timeout"])
        if kind=="network": raise OSError("mock transport loss")
        if kind=="malformed": response.stdout="{"
        if kind=="quota_changed": response.stdout=json.dumps(dict(resets_at=old.RESET,limits=dict(checks=dict(used=3,per_day=3,left=0))))
        if kind=="failure": response.returncode=1;response.stderr="mock unavailable"
        if kind=="unknown": response.stdout=json.dumps({"status":"unknown"})
        if kind=="ambiguous": response.stdout="accepted. No job ID returned"
        if kind=="wrong_id": response.stdout=response.stdout.replace("7777777777777777",CONTROL)
        if kind=="bad_shape": response.stdout=response.stdout.replace("16 users","32 users")
        if kind=="stale_final": j.now+=timedelta(seconds=30)
        if kind=="stale_preflight": j.now+=timedelta(seconds=120)
        if kind=="back_before_start": j.now=old.START-timedelta(microseconds=1)
        if kind=="receipt_lost": (j.runs/old_cases.receipt).unlink(missing_ok=True)
        if kind=="choice_flipped": j.jobs[V204]["result"]["passed"]=False
        return response
    j.execute=mocked


def extras():
    entries=[]
    def scenario(label,prepare,expect=None):
        entries.append((label,dict(prepare=prepare,expect=expect)))
    for target,name in [(QUOTA,"quota"),(["serve","status",CONTROL,"--json"],"control"),
                        (["serve","status",V204,"--json"],"selection")]:
        rounds=(1,2,3) if name=="control" else (1,2)
        for phase in rounds:
            for kind in ("failure","malformed","timeout","network","unknown"):
                expected="v198" if name=="selection" and phase==1 else None
                # A transient V204 outage on only the first pass can change the
                # choice on recheck; in that case the original worker also holds.
                if name=="selection": expected=None
                scenario(f"{name}_{phase}_{kind}",lambda j,t=target,p=phase,k=kind:err(j,t,p,k),expected)
    for kind in ("timeout","network","failure","ambiguous","wrong_id","bad_shape"):
        scenario("dispatch_"+kind,lambda j,k=kind:err(j,"submit",1,k),"v215")
    for target,phase,kind in [(QUOTA,2,"quota_changed"),(QUOTA,2,"stale_final"),
                              (["serve","status",CONTROL,"--json"],3,"stale_final"),
                              (["serve","status",CONTROL,"--json"],3,"receipt_lost"),(QUOTA,1,"stale_preflight"),
                              (QUOTA,2,"back_before_start")]:
        scenario("bound_"+"_".join([str(target[-2]),str(phase),kind]),lambda j,t=target,p=phase,k=kind:err(j,t,p,k))
    for when in (old.START-timedelta(microseconds=1),old.END,old.END+timedelta(microseconds=1)):
        scenario("outside_"+when.isoformat(),lambda j,t=when:setattr(j,"now",t))
    for req in [*old.COMPETITORS]:
        scenario("claim_"+req[2],lambda j,name=req[2]:(j.runs/(name+".submit.txt")).write_text("ambiguous prior send"))
    for req in old.REQUIRED[:2]:
        scenario("missing_"+req[2],lambda j,name=req[2]:(j.runs/(name+".submit.txt")).unlink())
    scenario("malformed_claim_reused",lambda j:(j.runs/(old.NAME+".submit.txt")).touch())
    scenario("renamed_old_v215_claim",lambda j:(j.runs/"legacy_alias.submit.txt").write_text(
        json.dumps(dict(candidate="unknown",sha256=old.V215.sha256,attempted_at=j.now.isoformat()))+"\nunknown dispatch"))
    scenario("known_v215_partial_history",lambda j:(j.runs/"v215_nov20_check16.skip.txt").write_text("unknown prior attempt"))
    scenario("legacy_claim_reused",lambda j:(j.runs/(old.LEGACY_NAME+".submit.txt")).touch())
    now=json.loads((RUNS/"nov20_v215_runner.launch.json").read_text())["private_and_scored_quota_after"]
    for kind in ("network","malformed","failure"):
        def persistent(j,k=kind):
            orig=j.execute
            def send(cmd,**kw):
                res=orig(cmd,**kw)
                if cmd[1:]==["serve","status",V204,"--json"]:
                    if k=="network": raise OSError("mock parent transport loss")
                    if k=="malformed": res.stdout="{"
                    if k=="failure": res.returncode=1
                return res
            j.execute=send
        scenario("persistent_v204_"+kind,persistent,"v198")
    scenario("archived_nov19_quota",lambda j:setattr(j,"quota",now))
    scenario("observed_current_nov19_clock",lambda j:setattr(j,"now",old.START-timedelta(hours=25)))
    return entries


def race(first,first_choice,second,second_choice):
    with tempfile.TemporaryDirectory() as tmp:
        f=fixture(tmp,{})
        entered,release=threading.Event(),threading.Event()
        expected=[]
        def pause(n):
            if n==1:
                entered.set()
                assert release.wait(10)
        f.on_quota=pause
        t=threading.Thread(target=lambda:expected.append(poll(first,first_choice,f)))
        t.start()
        try:
            assert entered.wait(10)
            assert poll(second,second_choice,f)=="another allocation poll is active"
        finally:
            release.set()
            t.join(10)
        assert not t.is_alive() and len(expected)==len(f.attempts)==1 and expected[0].startswith("submitted ")
        assert poll(second,second_choice,f)=="attempt recorded" and len(f.attempts)==1
        return json.loads(f.receipt.read_text().splitlines()[0])["candidate"]


def timed_case(worker,choice,phase,wall_step,elapsed_step):
    """Advance the mock clocks after a completed observation or right after O_EXCL."""
    with tempfile.TemporaryDirectory() as tmp:
        f=fixture(tmp,{})
        f.now=worker.START+timedelta(minutes=10)
        elapsed=9000.0
        changed=False
        def advance():
            nonlocal changed,elapsed
            assert not changed
            changed=True
            f.now+=timedelta(seconds=wall_step)
            elapsed+=elapsed_step
        original_execute=f.execute
        def execute(cmd,**kw):
            response=original_execute(cmd,**kw)
            if not changed:
                if phase=="preflight" and cmd[1:]==QUOTA and f.reads==1:
                    advance()
                if phase=="final_quota" and cmd[1:]==QUOTA and f.reads==2:
                    advance()
                if phase=="final_control" and f.reads==2 and cmd[1:]==["serve","status",CONTROL,"--json"]:
                    advance()
            return response
        f.execute=execute
        real_digest=worker.first.digest
        def hashing(path):
            value=real_digest(path)
            if phase=="claimed" and f.receipt.exists() and not changed:
                advance()
            return value
        clocks={name:(lambda:elapsed) for name in inspect.signature(worker.run_once).parameters
                if name in {"monotonic","mono","monotonic_clock"}}
        with patch.object(worker.first,"digest",side_effect=hashing):
            result=poll(worker,choice,f,**clocks)
        assert changed,(phase,result)
        report=dict(result=result,attempts=len(f.attempts),claim=f.receipt.exists())
        if f.receipt.exists():
            evidence=json.loads(f.receipt.read_text().splitlines()[0])["evidence"]
            report.update(quota_read=evidence["final_quota"]["observed_at"],
                          control_read=evidence["final_control"]["observed_at"])
            calls=len(f.attempts)
            assert poll(worker,choice,f,**clocks)=="attempt recorded" and len(f.attempts)==calls
        return report


def timing_regressions(new):
    specs=[
        ("rollback_final_control","final_control",-300,1,False),
        ("rollback_final_quota","final_quota",-0.000001,0.01,False),
        ("rollback_after_claim","claimed",-1,0.5,False),
        ("elapsed_preflight_boundary","preflight",0,120,False),
        ("elapsed_final_quota_boundary","final_quota",0,30,False),
        ("elapsed_final_control_boundary","final_control",0,30,False),
        ("elapsed_after_claim_boundary","claimed",0,30,False),
        ("elapsed_just_inside_preflight","preflight",0,119.999,True),
        ("elapsed_just_inside_final","final_control",0,29.999,True),
    ]
    out={}
    for label,phase,wall,elapsed,expected_new_send in specs:
        before=timed_case(old,old.V215,phase,wall,elapsed)
        after=timed_case(new,new.V229,phase,wall,elapsed)
        assert before["attempts"]==1 and before["claim"],(label,before)
        assert after["attempts"]==int(expected_new_send),(label,after)
        assert after["claim"]==(expected_new_send or phase=="claimed"),(label,after)
        out[label]=dict(parent=before,staged=after)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",type=Path,nargs="?",default=RUNS/"history/nov20_v229_staging")
    args=ap.parse_args()
    sys.path.insert(0,str(args.stage))
    path=args.stage/"submit_nov20_v187_check.py"
    new=load("standby_v229_independent",path)
    assert (new.NAME,new.PROCESS_LOCK,new.slots.LOCK,new.START,new.END,new.RESET)==(
        old.NAME,old.PROCESS_LOCK,old.slots.LOCK,old.START,old.END,old.RESET)
    rows={}
    for label,kwargs in collected_inputs()+extras():
        outputs=[]
        for worker,chosen in [(old,old.V215),(new,new.V229)]:
            with tempfile.TemporaryDirectory() as tmp:
                outputs.append(collect(worker,chosen,fixture(tmp,kwargs)))
        a,b=outputs
        want="v229" if a["claim"]=="v215" else a["claim"]
        assert b["claim"]==want and a["send"]==b["send"] and a["queries"]==b["queries"],(label,a,b)
        assert a["result"]==b["result"],(label,a,b)
        if a["claim"] in ("v215","v198"):
            assert a["proof"]==b["proof"],(label,a["proof"],b["proof"])
        if a["send"]==1:
            assert kwargs["expect"]==a["claim"],(label,kwargs["expect"],a)
        else:
            assert kwargs["expect"] is None,(label,kwargs["expect"],a)
        rows[label]=dict(parent=a["claim"],staged=b["claim"],attempts=b["send"],outcome=b["result"])
    races={}
    for first,fc,second,sc in [(old,old.V215,new,new.V229),(new,new.V229,old,old.V215)]:
        races[fc.name+"_first"]=race(first,fc,second,sc)
    plans={}
    assert "v215" not in new.CANDIDATES and {c.name for c in new.PRIOR_CHOICES}==set(old.CANDIDATES)|{"v229"}
    for c in ("v187","v189","v198"):
        a=old.plan(candidate=old.CANDIDATES[c]);b=new.plan(candidate=new.CANDIDATES[c])
        assert a==b or {k:v for k,v in a.items() if k!="alternative_candidates"}=={k:v for k,v in b.items() if k!="alternative_candidates"},c
        plans[c]=True
    manifest=json.loads((HERE/"validation.json").read_text())
    assert new.V229.sha256==manifest["patch_sha256"]==digest(HERE/"patch.diff")
    print(json.dumps(dict(baseline_worker_sha256=digest(OLD_WORKER),selector_sha256=digest(OLD_STAGE/"v215_selection.py"),
        staged_worker_sha256=digest(path),staged_selector_sha256=digest(args.stage/"v229_selection.py"),candidate_patch_sha256=new.V229.sha256,total=len(rows),
        comparison="identical outcome, status query order, claim retention, evidence and attempt count; positive V215 choice selects V229",
        races=races,previous_plans=plans,
        clock_boundaries=timing_regressions(new),outcomes=dict(Counter(v["staged"] or "held" for v in rows.values())),cases=rows,
        network_or_real_submissions=False),indent=2))


if __name__=="__main__": main()
