"""V29 fixed-gate and activation counterexamples; all changed timings are synthetic."""
import copy
import json
import unittest
import analyze_decoder_v29 as a


def parent():
    return json.loads((a.ROOT / 'experiments/runs/7a8c4941741645ee.json').read_text())['response']['result']


def synthetic(ratios):
    r = copy.deepcopy(parent()); rows = {}
    for x in r['records']:
        if x['event'] == 'benchmark':
            factor = ratios.get(x['index'], 1.)
            x['median_ns'] /= factor; x['median_raw_ns'] /= factor; x['speedup'] *= factor
            rows[x['index']] = x
        elif x['event'] == 'profile':
            x['report'] = x['report'].replace('sk2_kernel', 'sk2_exact_patch_v29')
    speeds = []
    for i in range(6):
        pair = [rows[2*i], rows[2*i+1]]
        speeds.append(sum(x['baseline_ns'] * x['case']['weight'] for x in pair)
                      / sum(x['median_ns'] * x['case']['weight'] for x in pair))
    r['speedup'] = a.gm(speeds)
    r['speedup_ci'] = [r['speedup']*.985, r['speedup']*1.015]
    return r


class GateTests(unittest.TestCase):
    def test_parent_no_gain_and_real_profile_domain(self):
        self.assertFalse(a.compare(parent(), parent())['performance_gate_passed'])
        self.assertFalse(a.activation(parent(), parent())['passed'])
        profiles = [x for x in parent()['records'] if x['event'] == 'profile']
        self.assertEqual([sum(a.compressed_instances(x['report'], 'sk2_kernel').values()) for x in profiles],
                         [3, 3, 4, 4, 4, 4, 3, 3, 0, 0, 0, 0])

    def test_synthetic_clear_gain_passes_weighted_gate_and_activation(self):
        r = synthetic({i:1.05 for i in range(8)})
        c = a.compare(r, parent())
        self.assertTrue(c['performance_gate_passed'])
        self.assertAlmostEqual(c['whole_score_ratio'], 1.05**(4/6))
        self.assertTrue(a.activation(r, parent())['passed'])

    def test_regressions_cannot_hide_and_small_gain_does_not_pass(self):
        self.assertFalse(a.compare(synthetic({i:1.02 for i in range(8)}), parent())['performance_gate_passed'])
        for bad in (3, 9):
            factors = {i:1.15 for i in range(8)}; factors[bad] = .96
            self.assertFalse(a.compare(synthetic(factors), parent())['performance_gate_passed'])

    def test_unchanged_router_name_or_partial_activation_cannot_pass(self):
        r = synthetic({i:1.05 for i in range(8)})
        row = next(x for x in r['records'] if x['event'] == 'profile' and x['index'] == 4)
        # All true COMP instances disappear, while the unchanged COMP0 router remains.
        row['report'] = row['report'].replace(', true, false,', ', false, false,')
        self.assertFalse(a.activation(r, parent())['passed'])
        r = synthetic({i:1.05 for i in range(8)})
        row = next(x for x in r['records'] if x['event'] == 'profile' and x['index'] == 0)
        row['report'] = row['report'].replace('sk2_exact_patch_v29', 'sk2_kernel', 1)
        self.assertFalse(a.activation(r, parent())['passed'])


if __name__ == '__main__':
    unittest.main()
