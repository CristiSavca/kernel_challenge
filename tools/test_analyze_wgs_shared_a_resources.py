#!/usr/bin/env python3
"""Offline synthetic SSAR fixtures; no reported fixture value is GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import analyze_wgs_shared_a_resources as audit


def marker(big,**changes):
    r=dict(v=1,big=big,ntl=576,dyn=194304,regs=96,local=0,static=0,maxdyn=232448,occ=1,set=0,attr=0,cap=0)
    r.update(changes)
    return 'SSAR '+' '.join(f'{k}={r[k]}' for k in audit.FIELDS)


def complete(**changes):return marker(0,**changes)+'\n'+marker(1,**changes)+'\n'


def terminal_result(log):
    parent=json.loads((audit.ROOT/'experiments/runs/d24dd50196cb4158.json').read_text())['response']['result']
    r=copy.deepcopy(parent);r['log_tail']=log
    return r


def fixture(root,result=None,status='done'):
    source=root/'candidate.py'
    source.write_bytes((audit.ROOT/f'experiments/candidates/muse_moe_w4a16/{audit.SOURCE}.py').read_bytes())
    response={'id':audit.JOB,'board':'muse_moe_w4a16','mode':'check','status':status,'result':result}
    if result is not None:
        p=root/'profile.json';p.write_text(json.dumps({'job_id':audit.JOB,'problem':'muse_moe_w4a16','submission':str(source),
            'benchmarks':[r for r in result['records'] if r['event']=='benchmark'],
            'profiles':[r for r in result['records'] if r['event']=='profile']}))
        response['profile']={'json':str(p)}
    receipt={'job':audit.JOB,'problem':'muse_moe_w4a16','source_sha256':audit.SOURCE,'source':'candidate.py','response':response}
    path=root/f'experiments/runs/{audit.JOB}.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(receipt))
    return path,receipt


class Resources(unittest.TestCase):
    def test_complete_pair(self):
        r=audit.parse_log(complete());self.assertEqual(r['gate'],'PASS_RETAINED_PAIRS')
        self.assertEqual(r['missing_BIG'],[])

    def test_nonzero_local_and_C7511_are_not_kills(self):
        log='ptxas info : (C7511) wgmma instructions are serialized in _ZN3okc9moe_fusedILb1EEEvNS_6ParamsE\n'+complete(local=32,regs=88)
        r=audit.parse_log(log);self.assertEqual(r['gate'],'PASS_RETAINED_PAIRS')
        self.assertEqual(len(r['compiler_diagnostics']['positive_warning_records']),1)
        self.assertEqual(r['compiler_diagnostics']['positive_warning_records'][0]['BIG_if_symbol_present'],1)

    def test_each_API_failure(self):
        for field in ('set','attr','cap'):
            with self.subTest(field=field):
                r=audit.parse_log(complete(**{field:1}));self.assertEqual(r['gate'],'FAIL')

    def test_required_geometry_and_admission(self):
        for kwargs in (dict(ntl=640),dict(dyn=194176),dict(maxdyn=190000),dict(occ=0),dict(regs=0),dict(local=-1)):
            with self.subTest(kwargs=kwargs):self.assertEqual(audit.parse_log(complete(**kwargs))['gate'],'FAIL')

    def test_invalid_attributes_not_usable(self):
        r=audit.parse_log(complete(regs=0))
        self.assertTrue(all(not row['attributes_usable'] for row in r['records']))

    def test_quoted_source_warning_not_compiler_fact(self):
        r=audit.parse_log('// C7511 in an inherited source comment\n'+complete())
        self.assertEqual(r['compiler_diagnostics']['positive_warning_records'],[])

    def test_missing_BIG(self):
        r=audit.parse_log(marker(0));self.assertEqual(r['gate'],'UNKNOWN_INCOMPLETE_OR_CONFLICTING')
        self.assertEqual(r['missing_BIG'],[1])

    def test_truncated_marker(self):
        r=audit.parse_log(complete()+'SSAR v=1 big=0 ntl=576 dyn=194304')
        self.assertEqual(r['gate'],'UNKNOWN_INCOMPLETE_OR_CONFLICTING')
        self.assertEqual(len(r['malformed']),1)

    def test_wrong_version_unknown(self):
        self.assertEqual(audit.parse_log(complete(v=2))['gate'],'UNKNOWN_INCOMPLETE_OR_CONFLICTING')

    def test_repeated_identical_pairs(self):
        r=audit.parse_log(complete()*2);self.assertEqual(r['gate'],'PASS_RETAINED_PAIRS')
        self.assertEqual(len(r['paired_record_ids']),2)

    def test_conflict_quarantines_same_identity(self):
        r=audit.parse_log(complete()+complete(regs=88))
        self.assertEqual(r['gate'],'UNKNOWN_INCOMPLETE_OR_CONFLICTING')
        self.assertTrue(all(not row['eligible_for_clean_pair'] for row in r['records']))

    def test_out_of_order_and_orphan(self):
        for text in (marker(1)+'\n'+marker(0),complete()+marker(0)):
            self.assertEqual(audit.parse_log(text)['gate'],'UNKNOWN_INCOMPLETE_OR_CONFLICTING')

    def test_warning_absence_always_qualified(self):
        r=audit.parse_log('[1/3] c++\n[2/3] nvcc\n[3/3] link\n'+complete())
        self.assertEqual(r['compiler_diagnostics']['positive_warning_records'],[])
        self.assertTrue(r['compiler_diagnostics']['absence_of_other_warnings'].startswith('UNKNOWN'))

    def test_log_bound(self):
        with self.assertRaises(ValueError):audit.parse_log('x'*(512*1024+1))

    def test_profile_conditional_MAIN_identity_unknown(self):
        r=audit.profile_activation(terminal_result(''))
        self.assertEqual(r['gate'],'VISIBLE_DISPATCH_CONSISTENT_WGS_IDENTITY_UNKNOWN')
        self.assertTrue(all(p['SS_A_activation']=='UNKNOWN' for p in r['captures'] if p['affected']))

    def test_wrong_BIG_is_visible_failure(self):
        r=terminal_result('')
        next(p for p in r['records'] if p['event']=='profile' and p['case']['bs']==32)['report']='void okc::moe_fused<false>(okc::Params) 12.0 us 100.0% 1'
        self.assertEqual(audit.profile_activation(r)['gate'],'FAIL_DISPATCH')

    def test_truncated_or_multiple_symbols_unknown(self):
        for symbol in ('void okc::moe_fused<true>(okc::Par…)',
                       'void okc::moe_fused<true>(okc::Params)'):
            r=terminal_result('')
            p=next(p for p in r['records'] if p['event']=='profile')
            p['report']=symbol+' 12.0 us 100.0% 2'
            self.assertEqual(audit.profile_activation(r)['gate'],'UNKNOWN_PROFILE_SHAPE')

    def test_pending_has_no_invented_records(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root,status='queued');r=audit.analyze(root=root)
        self.assertEqual(r['status'],'PENDING');self.assertEqual(r['metrics'],{})
        self.assertNotIn('resources',r)

    def test_source_bound_terminal_no_promotion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root,terminal_result(complete(local=16)));r=audit.analyze(root=root)
        self.assertEqual(r['status'],'PASS');self.assertEqual(r['resource_gate'],'PASS_RETAINED_PAIRS')
        self.assertEqual(r['metrics'],{});self.assertFalse(r['automatic_promotion_allowed'])

    def test_failed_official_result_retains_resources_no_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);result={'passed':False,'returncode':112,'records':[],
                'log_tail':marker(0,occ=0),'error':'fixture failure'}
            fixture(root,result);r=audit.analyze(root=root)
        self.assertEqual(r['status'],'OFFICIAL_GATE_FAILED');self.assertEqual(r['resource_gate'],'FAIL')
        self.assertEqual(r['activation_gate'],'UNAVAILABLE_OFFICIAL_GATE_FAILED');self.assertEqual(r['metrics'],{})

    def test_mismatched_source_and_job_rejected(self):
        for args in (dict(job='0'*16),dict(source_sha='0'*64)):
            with self.assertRaises(ValueError):audit.analyze(**args)

    def test_changed_frozen_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root,status='queued');(root/'candidate.py').write_text('# different')
            with self.assertRaisesRegex(ValueError,'source bytes'):audit.analyze(root=root)

    def test_bad_evaluator_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r=terminal_result(complete());r['evaluator']['code_sha256']='0'*64
            fixture(root,r)
            with self.assertRaisesRegex(ValueError,'evaluator'):audit.analyze(root=root)

    def test_evaluator_image_or_inventory_change_rejected(self):
        for kind in ('image','files'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                root=Path(td);r=terminal_result(complete())
                if kind=='image':r['evaluator']['image']='different-image'
                else:r['evaluator']['files']['challenge/nvfp4.py']='0'*64
                fixture(root,r)
                with self.assertRaisesRegex(ValueError,'inventory/image'):audit.analyze(root=root)

    def test_profile_receipt_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fixture(root,terminal_result(complete()))
            p=root/'profile.json';d=json.loads(p.read_text());d['profiles'][0]['report']='changed';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'records disagree'):audit.analyze(root=root)

    def test_conflicting_logs_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path,r=fixture(root,terminal_result(complete()))
            p=root/'saved.log';p.write_text(complete(regs=80))
            r['response']['output']={'log':str(p)};path.write_text(json.dumps(r))
            parsed=audit.analyze(root=root)
        self.assertEqual(parsed['resource_gate'],'UNKNOWN_ARTIFACT_CONFLICT')


if __name__=='__main__':unittest.main(verbosity=2)
