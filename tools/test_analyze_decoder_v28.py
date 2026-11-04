"""Raw-parent format fixtures and explicitly synthetic gate counterexamples."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import analyze_decoder_v28 as a


def parent():
    return json.loads((a.ROOT / 'experiments/runs/7a8c4941741645ee.json').read_text())['response']['result']


def synthetic(ratios):
    result = copy.deepcopy(parent())
    benches = {}
    for row in result['records']:
        if row['event'] == 'benchmark':
            ratio = ratios.get(row['index'], 1.0)
            row['median_ns'] /= ratio
            row['median_raw_ns'] /= ratio
            row['speedup'] *= ratio
            benches[row['index']] = row
        elif row['event'] == 'profile' and row['case']['load'] in a.AFFECTED:
            row['report'] = row['report'].replace('okc::moe_fused<true>', 'okc::private_v28<true>')
    speeds = []
    for i in range(6):
        rr = [benches[2 * i], benches[2 * i + 1]]
        speeds.append(sum(r['baseline_ns'] * r['case']['weight'] for r in rr)
                      / sum(r['median_ns'] * r['case']['weight'] for r in rr))
    result['speedup'] = a.gm(speeds)
    result['speedup_ci'] = [result['speedup'] * .985, result['speedup'] * 1.015]
    return result


MAIN = (4, 5, 8, 9, 10, 11)


class Screen(unittest.TestCase):
    def test_real_parent_receipts_and_weighted_aggregate(self):
        for job, mode in a.PARENTS:
            self.assertEqual(a.read_run(job, a.PARENT_SHA, mode)['status'], 'PASS')
        rows, loads, _ = a.benchmarks(parent())
        self.assertAlmostEqual(a.gm(x['speedup'] for x in loads.values()), parent()['speedup'])
        self.assertFalse(math.isclose(a.gm(x['speedup'] for x in rows.values()), parent()['speedup'], rel_tol=1e-8))

    def test_unchanged_parent_is_not_gain(self):
        r = a.compare(parent(), parent())
        self.assertFalse(r['performance_gate_passed'])
        self.assertFalse(r['gates']['affected_weighted_load_GM_at_least_1p03'])

    def test_synthetic_positive_threshold_and_activation(self):
        r = synthetic({i: 1.04 for i in MAIN})
        self.assertTrue(a.compare(r, parent())['performance_gate_passed'])
        self.assertTrue(a.activation(r, 'okc::private_v28<true>')['passed'])
        r['records'] = [x for x in r['records'] if not (x['event'] == 'profile' and x['index'] == 5)]
        with self.assertRaises(ValueError):
            a.activation(r, 'okc::private_v28<true>')

    def test_each_affected_regression_cannot_hide_in_aggregate(self):
        factors = {i: 1.15 for i in MAIN}
        factors[5] = .96
        r = a.compare(synthetic(factors), parent())
        self.assertTrue(r['gates']['affected_weighted_load_GM_at_least_1p03'])
        self.assertFalse(r['gates']['each_affected_latency_at_most_1p03'])

    def test_control_regression_and_false_activation_rejected(self):
        factors = {i: 1.10 for i in MAIN}
        factors[1] = .96
        r = synthetic(factors)
        self.assertFalse(a.compare(r, parent())['gates']['each_control_latency_at_most_1p03'])
        for row in r['records']:
            if row['event'] == 'profile' and row['index'] == 1:
                row['report'] += '\nokc::private_v28<true>\n'
        self.assertFalse(a.activation(r, 'okc::private_v28<true>')['passed'])

    def test_weighted_loads_not_unweighted_case_mean(self):
        factors = {i: 1.04 if i % 2 == 0 else .99 for i in MAIN}
        r = a.compare(synthetic(factors), parent())
        unweighted = a.gm(x['speedup_ratio'] for x in r['cases'] if x['affected'])
        self.assertNotAlmostEqual(r['affected_weighted_load_GM_speedup_ratio'], unweighted, places=6)

    def test_failing_flags_missing_cases_and_metadata(self):
        for mutation in ('flags', 'coverage', 'weight', 'score', 'dtype'):
            with self.subTest(mutation=mutation):
                r = synthetic({i: 1.04 for i in MAIN})
                if mutation == 'flags':
                    r['flags'] = ['unexpected']
                elif mutation == 'coverage':
                    r['records'].pop(1)
                elif mutation == 'weight':
                    next(x for x in r['records'] if x['event'] == 'benchmark')['case']['weight'] = 16
                elif mutation == 'score':
                    r['speedup'] *= 1.1
                else:
                    next(x for x in r['records'] if x['event'] == 'benchmark')['median_ns'] = True
                with self.assertRaises(ValueError):
                    a.benchmarks(r)

    def test_source_and_evaluator_identity_are_required(self):
        with self.assertRaises(ValueError):
            a.read_run(a.PARENTS[0][0], '0' * 64, 'run')
        r = synthetic({i: 1.04 for i in MAIN})
        r['evaluator']['image'] = 'changed'
        with self.assertRaises(ValueError):
            a.compare(r, parent())

    def test_received_minimal_ack_is_pending_but_conflicting_identity_fails(self):
        with tempfile.TemporaryDirectory(prefix='v28-reader-') as tmp:
            root = Path(tmp)
            source = root / 'candidate.py'
            source.write_bytes(b'# synthetic receipt fixture, not GPU evidence\n')
            digest = a.sha(source.read_bytes())
            job = '0' * 16
            receipt = root / 'experiments/runs' / (job + '.json')
            receipt.parent.mkdir(parents=True)
            r = {'job': job, 'problem': a.PROBLEM, 'source_sha256': digest,
                 'source': 'candidate.py', 'response': {'job_id': job, 'status': 'received'}}
            receipt.write_text(json.dumps(r))
            x = a.read_run(job, digest, 'check', root)
            self.assertEqual(x['status'], 'PENDING')
            self.assertFalse(x['server_identity_confirmed'])
            r['response']['board'] = 'wrong_problem'
            receipt.write_text(json.dumps(r))
            with self.assertRaises(ValueError):
                a.read_run(job, digest, 'check', root)


if __name__ == '__main__':
    unittest.main()
