"""Offline adversarial offers. Canonical archive and submitted files are read only."""
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from datetime import timedelta

ROOT=Path(__file__).resolve().parents[3];RUNS=ROOT/'serving_v0523/runs'
STAGE=RUNS/'history/nov19_v204_staging'
sys.path[:0]=[str(STAGE),str(RUNS)]
import submit_nov19_decode_checks as r
import submit_nov19_integrated_verify as wrapper
import test_nov19_decode_checks as fixtures

def load(name, path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
old=load('independent_legacy_204',RUNS/'submit_nov19_decode_checks.py')


def history(mark=None,origin='complete',at='start',dup=None):
    f=fixtures.MatchTest();f.setUp()
    try:
        v68=f.runs/(r.V68_JOB+'.json')
        raw=(r.RUNS/v68.name).read_bytes();v68.write_bytes(raw)
        f.jobs[r.V68_JOB]=json.loads(raw)
        f.ids['v204']='aaddee3311225599'
        job=lambda candidate: f.jobs[f.ids[candidate.name]]
        if origin=='v68pending':f.jobs[r.V68_JOB].update(status='queued',result=None)
        if origin=='v68wrong':f.jobs[r.V68_JOB]['result']['patch']['sha256']='ffff'
        if origin=='v68contradictmetric':f.jobs[r.V68_JOB]['result']['rungs'][0]['tok_s_per_gpu']=0
        if origin=='v68duplicate':f.jobs[r.V68_JOB]['duplicate_of']='feed'*4
        if origin=='v93pass':f.job(r.V93)
        if origin=='v93fail':f.job(r.V93,passed=False,status='error')
        if origin=='v93emptypending':job(r.V93)['serving_flags']={}
        if origin=='v93floatusers':job(r.V93)['users']=16.0
        if origin=='v93duplicate':job(r.V93)['duplicate_of']='dead'*4
        if origin=='v93falsenull':job(r.V93)['result']={}
        if origin=='v64private':f.private.write_text('{}')
        if origin=='v68archive':v68.write_text('{}')
        if at=='deadline':f.at=r.END
        if at=='before':f.at=r.START-timedelta(seconds=1)
        if mark=='v93pass':f.on_quota=lambda:f.job(r.V93)
        if mark=='v93fail':f.on_quota=lambda:f.job(r.V93,passed=False,status='error')
        if mark=='v68wrong':f.on_quota=lambda:f.jobs[r.V68_JOB]['result']['patch'].update(sha256='bad')
        if mark=='v68pending':f.on_quota=lambda:f.jobs[r.V68_JOB].update(status='queued',result=None)
        if mark=='v64private':f.on_quota=lambda:f.private.write_text('{}')
        if mark=='deadline':f.on_quota=lambda:setattr(f,'at',r.END)
        if mark=='v168appears':f.on_quota=lambda:(f.runs/(r.prefix(r.V168)+'.submit.txt')).write_text('uncertain')
        if mark=='selfappears':f.on_quota=lambda:(f.runs/(r.prefix(r.V204)+'.submit.txt')).write_text('uncertain')
        if dup:
            cand,form=dup;q=f.runs/'restored';q.mkdir()
            raw_cli=f'submitted /private/{cand.directory}/patch.diff as a check run at 16 users\njob eeeeffffaaaabbbb\nCLI exit code: 0'
            shapes=dict(name=(f.runs/(r.prefix(cand)+'.submit.txt'),''),
                        metaraw=(q/'restored.submit.txt',json.dumps({'candidate':cand.name,'sha256':cand.sha256})+'\n'+raw_cli),
                        result=(q/'eeeeffffaaaabbbb.json',json.dumps({'id':'eeeeffffaaaabbbb','mode':'check','result':{'patch':{'sha256':cand.sha256}}})),
                        clipath=(q/'restored.submit.txt',raw_cli))
            loc,data=shapes[form];loc.write_text(data)
        calls=[]
        def execute(command,**kw):
            if command[1:3]==['serve','check']:
                calls.append(command.copy())
                assert command[1:3]==['serve','check'] and command[-1]=='--no-wait'
                return subprocess.CompletedProcess(command,0,f'submitted {command[3]} as a check run at 16 users. job aaddee3311225599\n','')
            return f.execute(command,**kw)
        try:result=r.run_candidate(r.V204,runs=f.runs,okc='offline',execute=execute,clock=lambda:f.at)
        except TimeoutError:result='scope expired; main handles timeout'
        if calls:
            receipt=f.runs/(r.prefix(r.V204)+'.submit.txt')
            assert json.loads(receipt.read_text().splitlines()[0])['sha256']==r.V204.sha256
            assert old.prefill_attempt(f.runs) and r.prefill_attempt(f.runs)
            for cand in (r.V168,r.V98,r.V95,r.V204):
                assert r.run_candidate(cand,runs=f.runs,okc='offline',execute=execute,clock=lambda:f.at)=='attempt recorded'
            assert len(calls)==1
        return len(calls),result
    finally:f.doCleanups()
def main():
    cases={}
    for name in ('complete','v68pending','v68wrong','v68contradictmetric','v68duplicate','v93pass','v93fail','v93emptypending','v93floatusers','v93duplicate','v93falsenull','v64private','v68archive'):
        cases['parent_'+name]=history(origin=name)
    for name in ('v93pass','v93fail','v68wrong','v68pending','v64private','deadline','v168appears','selfappears'):
        cases['after_quota_'+name]=history(mark=name)
    for c in (*r.PREFILL_ALTERNATIVES,*r.LEGACY_PREFILL_ALTERNATIVES,r.V204):
        for f in ('name','metaraw','result','clipath'):
            cases['prior_'+c.name+'_'+f]=history(dup=(c,f))
    for time in ('before','deadline'):cases['time_'+time]=history(at=time)
    assert cases['parent_complete']==(1,'submitted aaddee3311225599')
    assert all(n==0 for key,(n,_) in cases.items() if key!='parent_complete')
    return dict(isolated_offline=True,sources={n:sha256((STAGE/n).read_bytes()).hexdigest() for n in
                ('submit_nov19_decode_checks.py','submit_nov19_integrated_verify.py','v204_source.py')},checked=cases)
if __name__=='__main__':print(json.dumps(main(),indent=2))
