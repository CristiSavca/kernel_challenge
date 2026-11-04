"""Bounded offline V30 fixtures. Modified timings/profiles/logs are synthetic."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_moe_v30 as a


def parent():
    return json.loads((a.ROOT / 'experiments/runs/d24dd50196cb4158.json').read_text())['response']['result']


def resource_log():
    return '\n'.join(f'V30R name={name} nt={nt} dyn={sm} regs=80 local=32 static=0 maxdyn={sm} '
                     f'occ={need} min={need} api=0,0,0' for name, (nt, sm, need) in a.RESOURCE_GEOMETRIES.items())


def main_profile():
    names = ['v30_route_gather(okc::Params, okc::Smem*)',
             'v30_tiles<false, true>(okc::Params, const okc::Smem*)',
             'v30_tiles<false, false>(okc::Params, const okc::Smem*)',
             'v30_tiles<true, true>(okc::Params, const okc::Smem*)',
             'v30_tiles<true, false>(okc::Params, const okc::Smem*)',
             'v30_final(okc::Params, const okc::Smem*)']
    return '\n'.join('void okc::' + n + ' 2.0 µs 16.7% 1' for n in names)


def synthetic(ratios=None):
    result = copy.deepcopy(parent()); ratios = ratios or {2: 1.06, 4: 1.06, 5: 1.06}
    for row in result['records']:
        if row['event'] == 'benchmark':
            factor = ratios.get(row['index'], 1.)
            row['median_ns'] /= factor; row['median_raw_ns'] /= factor; row['speedup'] *= factor
        elif row['event'] == 'profile' and row['case']['bs'] > 16:
            row['report'] = main_profile()
    result['speedup'] = a.gm(
        r['speedup'] for r in result['records'] if r['event'] == 'benchmark')
    result['speedup_ci'] = [result['speedup']*.985, result['speedup']*1.015]
    result['log_tail'] = resource_log()
    return result


def fixture(root, status='done', result=None):
    job = '0123456789abcdef'; source = b'# synthetic V30 fixture\n'; digest = hashlib.sha256(source).hexdigest()
    sp = root/'source.py'; sp.write_bytes(source)
    response = {'job_id': job, 'status': status}
    if status != 'received': response.update(board='muse_moe_w4a16', mode='check')
    receipt = {'job': job, 'problem': 'muse_moe_w4a16', 'source_sha256': digest,
               'source': 'source.py', 'response': response}
    (root/'experiments/runs').mkdir(parents=True)
    if result is not None:
        response['result'] = result
        profile = {'job_id': job, 'problem': 'muse_moe_w4a16', 'submission': 'source.py',
                   'benchmarks': [r for r in result['records'] if r['event'] == 'benchmark'],
                   'profiles': [r for r in result['records'] if r['event'] == 'profile']}
        (root/'profile.json').write_text(json.dumps(profile))
        response['profile'] = {'json': 'profile.json'}
        (root/'saved.log').write_text(result.get('log_tail', ''))
        response['output'] = {'log': 'saved.log'}
    rp = root/'experiments/runs'/f'{job}.json'; rp.write_text(json.dumps(receipt))
    return job, digest, rp


class V30Tests(unittest.TestCase):
    def test_real_parent_provenance_and_no_gain(self):
        for jid, mode in a.PARENTS:
            p = a.read_parent(jid, mode)
            self.assertEqual(p['status'], 'PASS')
            self.assertFalse(a.compare(p['result'], p['result'])['performance_gate_passed'])
        self.assertFalse(a.activation(parent(), parent())['passed'])

    def test_clear_synthetic_gain_and_six_launches(self):
        result = synthetic()
        self.assertTrue(a.compare(result, parent())['performance_gate_passed'])
        self.assertAlmostEqual(a.compare(result, parent())['aggregate_speedup_ratio'], 1.06**.5)
        self.assertTrue(a.activation(result, parent())['passed'])
        self.assertTrue(a.resource_records(resource_log())['passed'])  # nonzero local is informational

    def test_whole_and_per_case_caps_cannot_hide(self):
        self.assertFalse(a.compare(synthetic({2:1.04, 4:1.04, 5:1.04}), parent())['performance_gate_passed'])
        for bad in (2, 0):
            ratios = {2:1.3, 4:1.3, 5:1.3}; ratios[bad] = 1/1.04
            self.assertFalse(a.compare(synthetic(ratios), parent())['performance_gate_passed'])
        self.assertFalse(a.compare(synthetic({0:1.05, 1:1.05, 3:1.05, 2:1.2, 4:1.2, 5:1.2}), parent())['performance_gate_passed'])

    def test_resources_missing_conflicting_malformed_failed_api(self):
        base = resource_log()
        self.assertFalse(a.resource_records('')['passed'])
        for bad in (base.replace('name=coldA', 'name=unknown'),
                    base.replace('occ=2 min=2', 'occ=1 min=2'),
                    base.replace('api=0,0,0', 'api=0,1,0', 1),
                    base.replace('dyn=89856', 'dyn=89000', 1),
                    base.replace('maxdyn=89856', 'maxdyn=89000', 1),
                    base.replace('regs=80', 'regs=-1', 1), base+'\nV30R name=coldA',
                    base+'\n'+base.splitlines()[1].replace('regs=80', 'regs=81')):
            self.assertFalse(a.resource_records(bad)['passed'])
        self.assertTrue(a.resource_records(base+'\n'+base)['passed'])

    def test_profile_activation_requires_all_real_positive_markers(self):
        for change in ('missing', 'wrong_boolean', 'zero_time', 'residual', 'control'):
            r = synthetic()
            row = next(x for x in r['records'] if x['event']=='profile' and x['index']==(0 if change=='control' else 2))
            if change == 'missing': row['report'] = row['report'].replace('okc::v30_final(', 'other::final(')
            elif change == 'wrong_boolean': row['report'] = row['report'].replace('<true, true>', '<false, true>')
            elif change == 'zero_time': row['report'] = row['report'].replace('2.0 µs', '0.0 µs', 1)
            elif change == 'residual': row['report'] += '\nvoid okc::moe_fused<true>(okc::Params) 1.0 µs 1.0% 1'
            else: row['report'] = main_profile()
            self.assertFalse(a.activation(r, parent())['passed'], change)

    def test_full_evaluator_image_flags_and_records(self):
        for change in ('image', 'inventory', 'flags', 'test_status', 'duplicate_test', 'coverage', 'shape'):
            r = synthetic()
            if change == 'image': r['evaluator']['image'] += 'x'
            elif change == 'inventory': r['evaluator']['files']['challenge/checks.py']='0'*64
            elif change == 'flags': r['flags']=['flag']
            elif change == 'test_status': next(x for x in r['records'] if x['event']=='test').pop('status')
            elif change == 'duplicate_test': [x for x in r['records'] if x['event']=='test'][1]['index']=0
            elif change == 'coverage': r['records'].pop()
            else: next(x for x in r['records'] if x['event']=='benchmark')['case']['bs']=17
            with self.assertRaises((ValueError, KeyError)): a.validate_result(r)

    def test_actual_reader_pending_failed_and_source_profile_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); jid, digest, rp = fixture(root, 'received')
            r=a.read_run(jid,digest,'check',root)
            self.assertEqual(r['status'],'PENDING'); self.assertNotIn('result',r)
            data=json.loads(rp.read_text());data['response']['mode']='run';rp.write_text(json.dumps(data))
            with self.assertRaises(ValueError): a.read_run(jid,digest,'check',root)
        for change in ('source','profile','failed'):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td); result=synthetic()
                if change=='failed': result['passed']=False
                jid,digest,rp=fixture(root,result=result)
                if change=='source': (root/'source.py').write_text('changed')
                elif change=='profile':
                    p=json.loads((root/'profile.json').read_text());p['profiles'][0]['report']='changed';(root/'profile.json').write_text(json.dumps(p))
                if change=='failed': self.assertEqual(a.read_run(jid,digest,'check',root)['status'],'OFFICIAL_GATE_FAILED')
                else:
                    with self.assertRaises(ValueError): a.read_run(jid,digest,'check',root)

    def test_saved_log_prefix_normal_and_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); r=synthetic(); r['log_tail']+='\nUSDT: trailing profiler line\n'
            jid,digest,_=fixture(root,result=r)
            (root/'saved.log').write_text(resource_log())
            c=a.read_run(jid,digest,'check',root)
            self.assertTrue(a.read_resources(c,root)['passed'])
            (root/'saved.log').write_text('unrelated')
            self.assertFalse(a.read_resources(c,root)['passed'])

    def test_full_analyzer_orchestration_and_missing_resources(self):
        # Real parent reads are independently pinned above; redirect only their
        # already-validated objects while the candidate lives in a temporary root.
        parents = {jid: a.read_parent(jid, mode) for jid, mode in a.PARENTS}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); result = synthetic({2:1.15, 4:1.15, 5:1.15})
            jid, digest, rp = fixture(root, result=result)
            pins = {'source.py': digest}
            for name in ('analyze_moe_v30.py', 'test_analyze_moe_v30.py', 'analyze_wgs_shared_a_result.py'):
                path = 'tools/' + name; p = root/path; p.parent.mkdir(exist_ok=True)
                p.write_bytes((a.ROOT/path).read_bytes()); pins[path] = a.digest(p.read_bytes())
            plan = {'candidate_sha256': digest, 'candidate_path':'source.py',
                    'parent_sha256':a.PARENT_SHA, 'parents':[{'job':j,'mode':m} for j,m in a.PARENTS],
                    'fixed_gate':a.GATE, 'source_review_status':'GO_FOR_OFFICIAL_CHECK', 'evidence_sha256':pins}
            (root/a.PLAN).parent.mkdir(parents=True, exist_ok=True)
            (root/a.PLAN).write_text(json.dumps(plan))
            with patch.object(a, 'read_parent', side_effect=lambda j,m,r: parents[j]):
                report = a.analyze(jid, digest, root)
                self.assertTrue(report['fixed_screen_passed'])
                self.assertFalse(report['automatic_promotion'])
                rec = json.loads(rp.read_text()); rec['response']['result']['log_tail'] = ''
                rp.write_text(json.dumps(rec)); (root/'saved.log').write_text('')
                self.assertEqual(a.analyze(jid, digest, root)['status'], 'CLOSE_FIXED_ATTEMPT')

    def test_plan_pin_and_gate_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'experiments/iterations').mkdir(parents=True)
            paths=('source.py','tools/analyze_moe_v30.py','tools/test_analyze_moe_v30.py','tools/analyze_wgs_shared_a_result.py')
            pins={}
            for path in paths:
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(path);pins[path]=a.digest(p.read_bytes())
            plan={'candidate_sha256':pins['source.py'],'candidate_path':'source.py','parent_sha256':a.PARENT_SHA,
                  'parents':[{'job':j,'mode':m} for j,m in a.PARENTS],'fixed_gate':a.GATE,
                  'source_review_status':'GO_FOR_OFFICIAL_CHECK','evidence_sha256':pins}
            p=root/a.PLAN;p.write_text(json.dumps(plan));a.load_plan(pins['source.py'],root)
            (root/'source.py').write_text('changed')
            with self.assertRaises(ValueError):a.load_plan(pins['source.py'],root)
            (root/'source.py').write_text('source.py');plan['fixed_gate']={};p.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):a.load_plan(pins['source.py'],root)


if __name__ == '__main__':
    unittest.main()
