"""Exercise the actual released runners, mocking every CLI call into temporary state."""
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import timedelta
import copy
import importlib.util
import json
import sys
import unittest

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"nov20_v229_staging"
sys.path.insert(0,str(HERE))
import submit_nov20_v187_check as runner
sys.path.insert(0,str(OLD))
import test_v229 as fixtures
spec=importlib.util.spec_from_file_location("frozen_unsplit_v229",OLD/"submit_nov20_v187_check.py")
prior=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
SCORE="b01dface00001234"


class ReceiptMode(unittest.TestCase):
    def fixture(self):
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return fixtures.Judge(tmp.name)

    def score(self,f,name="v204_nov20_scored.submit.txt",job=SCORE):
        candidate=runner.V204
        now=f.now-timedelta(minutes=1)
        meta=dict(utc=now.isoformat(),sha256=candidate.sha256,flags=candidate.flags)
        patch=Path("serving_v0523/candidates")/candidate.directory/"patch.diff"
        body=f"✓ submitted {patch} to serving  42 files changed · job {job}\nnext  okc serve status {job} --wait\nCLI exit code: 0\n"
        dst=f.runs/name
        dst.write_text(json.dumps(meta)+"\n"+body)
        f.jobs[job]=dict(id=job,board="serving",mode="run",team=runner.scored_receipt.TEAM,
                         created_at=now.timestamp()+1,serving_flags=candidate.flags,status="queued",result=None)
        return dst

    def held(self,f):
        outcome=f.offer()
        self.assertFalse(outcome.startswith('submitted '),outcome)
        self.assertFalse(f.attempts)

    def test_ablation(self):
        for impl,allow in ((prior,False),(runner,True)):
            f=self.fixture(); self.score(f)
            outcome=impl.run_once(okc='mock',candidate=impl.V229,runs=f.runs,clock=lambda:f.now,execute=f.execute)
            self.assertEqual(outcome.startswith('submitted '),allow)
            self.assertEqual(len(f.attempts),int(allow))
            if allow:
                self.assertEqual(f.events[-3:-1],[['serve','status','--json'],['serve','status',fixtures.previous.parent.IDS[2],'--json']])
                self.assertEqual(f.events.count(['serve','status',SCORE,'--json']),2)
                proof=json.loads(f.receipt.read_text().splitlines()[0])['evidence']
                self.assertEqual(proof['independent_scored_run']['job'],SCORE)
                self.assertEqual(f.offer(),'attempt recorded')

    def test_official_identity_and_working_prior_status(self):
        for status in ('queued','running','done'):
            f=self.fixture(); self.score(f,'arbitrary.submit.txt')
            j=f.jobs[SCORE];j['status']=status
            if status=='done':j['result']=dict(serving_flags=runner.V204.flags,patch=dict(sha256=runner.V204.sha256))
            self.assertTrue(f.offer().startswith('submitted '))
        for prop,val in [('id','ab'*8),('mode','check'),('team','foreign'),('board','kernel'),('status',''),
                         ('duplicate_of','ab'*8),('refunded',True),('created_at',float('nan')),('created_at',0),
                         ('created_at',runner.END.timestamp()+1),('serving_flags',{}),('users',16),('result',{})]:
            with self.subTest(prop=prop):
                f=self.fixture(); self.score(f); f.jobs[SCORE][prop]=val; self.held(f)
        f=self.fixture(); self.score(f);f.jobs[SCORE].update(status='done',result=dict(patch=dict(sha256='0'*64),serving_flags=runner.V204.flags));self.held(f)
        f=self.fixture(); self.score(f);del f.jobs[SCORE]
        original=f.execute
        def unknown(cmd,**kw):
            if cmd[1:]==['serve','status',SCORE,'--json']:
                return runner.subprocess.CompletedProcess(cmd,1,'','unavailable')
            return original(cmd,**kw)
        f.execute=unknown;self.held(f)

    def test_ambiguous_or_relabelled_receipts(self):
        for kind in ('pending','rc','ascheck','dup_cli','id','extra_lookup','sha','flags','time','duplicate','two_scores','local_mode'):
            with self.subTest(kind=kind):
                f=self.fixture();p=self.score(f);meta,body=p.read_text().split('\n',1);h=json.loads(meta)
                if kind=='pending':body=''
                elif kind=='rc':body=body.replace('CLI exit code: 0','CLI exit code: 1')
                elif kind=='ascheck':body=body.replace('to serving','as a check run at 16 users')
                elif kind=='dup_cli':body=body.split('\n')[0]+'\n'+body
                elif kind=='id':h['job']='cc'*8
                elif kind=='extra_lookup':body=body.replace('status '+SCORE,'status '+'de'*8)
                elif kind=='sha':h['sha256']='0'*64
                elif kind=='flags':h['flags']={}
                elif kind=='time':h['utc']='2026-11-20'
                elif kind=='local_mode':h['mode']='check'
                elif kind=='duplicate':(f.runs/'renamed.submit.txt').write_text(p.read_text())
                elif kind=='two_scores':self.score(f,'another_nov20.submit.txt','abcd'*4)
                p.write_text(json.dumps(h)+'\n'+body)
                self.held(f)

    def test_moving_evidence_before_quota_and_delivery(self):
        for point in ('score1','score2','quota2','control','after_claim'):
            for mutation in ('delete','change','new'):
                with self.subTest(point=point,mutation=mutation):
                    f=self.fixture();p=self.score(f);original=f.execute
                    def change():
                        if mutation=='delete':p.unlink()
                        elif mutation=='change':p.write_text(p.read_text()+'\n')
                        else:(f.runs/'unplanned_nov20.submit.txt').touch()
                    if point=='after_claim':
                        saved=runner.measured.submit_once
                        from unittest.mock import patch
                        def inject(offer,**kw):
                            guarded=kw['execute']
                            def intercept(cmd,**extra):
                                if cmd[1:3]==['serve','check']:change()
                                return guarded(cmd,**extra)
                            return saved(offer,**(kw|{'execute':intercept}))
                        with patch.object(runner.measured,'submit_once',inject):self.held(f)
                        self.assertTrue(f.receipt.exists())
                        continue
                    def execute(cmd,**kw):
                        got=original(cmd,**kw);tags=f.events
                        sc=['serve','status',SCORE,'--json'];quota=['serve','status','--json']
                        ctl=['serve','status',fixtures.previous.parent.IDS[2],'--json']
                        if ((point=='score1' and cmd[1:]==sc and tags.count(sc)==1)
                            or (point=='score2' and cmd[1:]==sc and tags.count(sc)==2)
                            or (point=='quota2' and cmd[1:]==quota and tags.count(quota)==2)
                            or (point=='control' and cmd[1:]==ctl and tags.count(quota)==2)):
                            change()
                        return got
                    f.execute=execute;self.held(f)

    def test_scored_run_preserves_allocations_and_fallback(self):
        f=self.fixture();self.score(f);f.report["result"]["passed"]=False
        self.assertTrue(f.offer().startswith("submitted "))
        h=json.loads(f.receipt.read_text().splitlines()[0]);self.assertEqual(h['candidate'],"v198")
        for case in ("quota","first","second","parent","deadline","competing","same_candidate"):
            f=self.fixture();p=self.score(f)
            if case=="quota":f.quota["limits"]["checks"].update(used=1,left=2)
            elif case in ("first","second"):
                index=0 if case=="first" else 1
                (f.runs/(runner.REQUIRED[index][2]+".submit.txt")).unlink()
            elif case=="parent":f.jobs[fixtures.previous.parent.IDS[2]].update(status="done")
            elif case=="deadline":f.now=runner.END
            elif case=="competing":(f.runs/(runner.COMPETITORS[0][2]+".submit.txt")).touch()
            else:
                p.rename(f.runs/"v229_nov20_scored.submit.txt")
            with self.subTest(case=case):self.held(f)

    def test_same_score_id_as_private_or_reused_child(self):
        f=self.fixture();self.score(f,job=fixtures.previous.parent.IDS[0]);self.held(f)
        f=self.fixture();self.score(f);f.child=SCORE
        self.assertIn('reconciliation',f.offer())
        self.assertTrue(f.receipt.exists());self.assertEqual(len(f.attempts),1);self.assertEqual(f.offer(),'attempt recorded')
