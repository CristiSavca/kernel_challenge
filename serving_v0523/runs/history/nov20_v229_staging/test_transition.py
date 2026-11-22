"""Differential replay of the frozen V215 worker and its V229 replacement; mock transport only."""
import copy
from datetime import timedelta
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_v229 import Judge, STAGE, previous, runner, selection

OLD_STAGE = STAGE.parent / "nov20_v215_staging"
OLD_SOURCE = OLD_STAGE / "submit_nov20_v187_check.py"
assert sha256(OLD_SOURCE.read_bytes()).hexdigest() == "ee80f0bd9e785b326cd0eaad948c6626a26dcbc5443e4c71f03d875ba801e8f5"
assert sha256((OLD_STAGE / "v215_selection.py").read_bytes()).hexdigest() == "f137c898ad37eb5ed687d2d4a9bac90b744244e420fda0a4fc3b18bad57b322c"
sys.path.append(str(OLD_STAGE))
spec=importlib.util.spec_from_file_location("archived_v215_standby",OLD_SOURCE)
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


class Transition(unittest.TestCase):
    def fixture(self):
        sandbox=tempfile.TemporaryDirectory()
        self.addCleanup(sandbox.cleanup)
        return Judge(sandbox.name)

    def run_worker(self, impl, f):
        choice=old.V215 if impl is old else runner.V229
        with patch('subprocess.run',side_effect=AssertionError('unmocked process')):
            return impl.run_once(okc="mock",candidate=choice,runs=f.runs,clock=lambda:f.now,execute=f.execute)

    def test_real_workers_agree_on_both_branches_and_holds(self):
        def price(f,value):
            f.report['result']['rungs'][0]['tok_s_per_gpu']=value
        def stock(f,value):
            f.report['result']['vs_stock']['rungs'][0]['at_stock_point']=value
        def cancel(f):
            f.jobs[previous.parent.IDS[2]]['status']='done'
        def status(f,s):
            f.report.update(status=s, result=None)
        def late(f):
            f.now=runner.END
        def missing(f):
            (f.runs/(selection.V204_RECEIPT+'.submit.txt')).unlink()
        def refunded(f):
            f.report['refunded']=True
        def other_weights(f):
            f.report['result']['weights']='unmatched revision'
        cases=[
            ('qualifies',lambda f:None,'positive'),
            ('just below raw threshold',lambda f:price(f,17798.36),'fallback'),
            ('just above raw threshold',lambda f:price(f,17798.37),'positive'),
            ('below normalized threshold',lambda f:stock(f,1.4472111),'fallback'),
            ('above normalized threshold',lambda f:stock(f,1.4472113),'positive'),
            ('nan raw',lambda f:price(f,float('nan')),'fallback'),
            ('infinite normalized',lambda f:stock(f,float('inf')),'fallback'),
            ('queued',lambda f:status(f,'queued'),'fallback'),('running',lambda f:status(f,'running'),'fallback'),
            ('refunded',refunded,'fallback'),('different weights',other_weights,'fallback'),
            ('unbound private result',missing,'fallback'),('final parent resolved',cancel,'hold'),
            ('08:00 UTC',late,'hold')]
        for label,edit,expected in cases:
            with self.subTest(label=label):
                pairs=[]
                for impl in (old,runner):
                    f=self.fixture(); edit(f)
                    state=self.run_worker(impl,f)
                    header=json.loads(f.receipt.read_text().splitlines()[0]) if f.receipt.exists() else None
                    pairs.append((state,f.events,header,f.attempts))
                (s1,e1,r1,c1),(s2,e2,r2,c2)=pairs
                self.assertEqual(s1,s2)
                if expected=='hold':
                    self.assertIsNone(r1);self.assertIsNone(r2)
                    self.assertEqual((c1,c2),([],[]))
                    self.assertEqual(e1,e2)
                    continue
                self.assertEqual(len(c1),1);self.assertEqual(len(c2),1)
                self.assertEqual((c1[0][:3],c1[0][4:]),(c2[0][:3],c2[0][4:]))
                self.assertEqual(e1[:-1],e2[:-1])
                self.assertEqual(r1['evidence'],r2['evidence'])
                self.assertEqual((r1['flags'],r1['users']),(r2['flags'],r2['users']))
                if expected=='fallback':
                    self.assertEqual(r1,r2);self.assertEqual(c1,c2)
                else:
                    self.assertEqual((r1['candidate'],r1['sha256']),('v215',old.V215.sha256))
                    self.assertEqual((r2['candidate'],r2['sha256']),('v229',runner.V229.sha256))
                    self.assertEqual(Path(c1[0][3]).parent.name,old.V215.directory)
                    self.assertEqual(Path(c2[0][3]).parent.name,runner.V229.directory)

    def test_first_claim_fences_either_worker(self):
        for prior,following in ((old,runner),(runner,old)):
            for timeout in (False,True):
                f=self.fixture();f.timeout=timeout
                self.run_worker(prior,f)
                self.assertEqual(self.run_worker(following,f),'attempt recorded')
                self.assertEqual(len(f.attempts),1)

    def test_renamed_legacy_v215_claim_blocks_query(self):
        for contents in (json.dumps({'candidate':'v215','sha256':old.V215.sha256}),
                         'timeout '+old.V215.sha256):
            f=self.fixture()
            (f.runs/'other.submit.txt').write_text(contents+'\n')
            self.assertEqual(self.run_worker(runner,f),'attempt recorded')
            self.assertEqual(f.events,[])
            self.assertFalse(f.receipt.exists())

    def test_clock_rollback_and_monotonic_expiry_before_claim(self):
        # A valid UTC instant can move backwards while remaining inside the 04:00–08:00 slot.
        for event,change in (('first_read','elapsed_120'),('final_quota','wall'),('last_parent','wall'),
                             ('last_parent','elapsed_30'),('last_parent','elapsed_120')):
            with self.subTest(event=event,change=change):
                f=self.fixture()
                f.now=runner.START+timedelta(minutes=10)
                ticks=[1000.0]; original=f.execute
                parent_id=previous.parent.IDS[2]
                def execute(cmd,**kw):
                    result=original(cmd,**kw)
                    quotas=f.events.count(['serve','status','--json'])
                    if (event=='first_read' and len(f.events)==1
                            or event=='final_quota' and quotas==2 and cmd[1:]==['serve','status','--json']
                            or event=='last_parent' and quotas==2 and cmd[1:]==['serve','status',parent_id,'--json']):
                        if change=='wall': f.now-=timedelta(minutes=5)
                        else: ticks[0]+=30.001 if change=='elapsed_30' else 120.001
                    return result
                state=runner.run_once(okc='mock',candidate=runner.V229,runs=f.runs,clock=lambda:f.now,
                                      monotonic=lambda:ticks[0],execute=execute)
                self.assertFalse(state.startswith(('submitted ','attempt ','prior attempt','outside submission window')))
                self.assertEqual(f.attempts,[]);self.assertFalse(f.receipt.exists())
                # A new poll deliberately samples both clocks afresh and revalidates all the controls.
                self.assertTrue(f.offer().startswith('submitted '))

    def test_post_claim_clock_anomaly_preserves_exclusive_receipt(self):
        for changed in ('wall','monotonic'):
            f=self.fixture(); f.now=runner.START+timedelta(minutes=10)
            def utc():
                return f.now - timedelta(minutes=5) if changed=='wall' and f.receipt.exists() else f.now
            def ticks():
                return 1040. if changed=='monotonic' and f.receipt.exists() else 1000.
            state=runner.run_once(okc='mock',candidate=runner.V229,runs=f.runs,clock=utc,
                                  monotonic=ticks,execute=f.execute)
            self.assertTrue(state.startswith('attempt '))
            self.assertEqual(f.attempts,[])
            self.assertTrue(f.receipt.exists())
            self.assertEqual(f.offer(),'attempt recorded')
            self.assertEqual(f.attempts,[])

    def test_four_source_and_manifest_lineage(self):
        with tempfile.TemporaryDirectory() as stage:
            base=Path(stage)
            for folder,_,_,_ in selection.LINEAGE.values():
                path=runner.ROOT/'serving_v0523/candidates'/folder
                (base/folder).mkdir()
                for name in ['patch.diff','delta.diff','validation.json']:
                    shutil.copyfile(path/name,base/folder/name)
            for name,(folder,_,_,_) in selection.LINEAGE.items():
                directory=base/folder
                for filename in ('patch.diff','delta.diff'):
                    path=directory/filename
                    source=path.read_bytes()
                    path.write_bytes(source+b'\n')
                    with self.assertRaises(ValueError): selection.verify_lineage(base,runner.V229)
                    path.write_bytes(source)
                path=directory/'validation.json';source=path.read_bytes()
                info=json.loads(source)
                changes=[{'parent_tree':'bad'},{'launch_flags':{}},{'strict_apply_reverse_full_delta':1}]
                if name=='V224': changes.append({'all_six_orders_forward_and_reverse':1})
                if name=='V229': changes.extend([{'both_orders_forward_and_reverse':1},{'donors':{'V225':'foreign'}}])
                for delta in changes:
                    path.write_text(json.dumps(info|delta))
                    with self.assertRaises(ValueError): selection.verify_lineage(base,runner.V229)
                path.write_bytes(source)
            selection.verify_lineage(base,runner.V229)

    def test_preview_matches_current_plan_outside_selected_artifact(self):
        new=runner.plan(candidate=runner.V229)
        original=old.plan(candidate=old.V215)
        protected={'candidate','flags','tree','sha256','alternative_candidates','candidate_lineage'}
        self.assertEqual({k:v for k,v in new.items() if k not in protected},
                         {k:v for k,v in original.items() if k not in protected})
        self.assertEqual(new['flags'],original['flags'])
        self.assertEqual(new['alternative_candidates'],original['alternative_candidates']+['v229'])
        self.assertEqual(new['candidate_lineage'],selection.plan_lineage())
        self.assertEqual((old.PROCESS_LOCK,old.NAME,old.slots.LOCK),(runner.PROCESS_LOCK,runner.NAME,runner.slots.LOCK))

if __name__=='__main__': unittest.main()
