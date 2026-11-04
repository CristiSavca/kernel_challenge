import copy
import json
import re
import unittest
from analyze_decoder_v33 import ROOT, resources, activation, compare


class V33ScreenTests(unittest.TestCase):
    def setUp(self):
        self.parent = json.loads((ROOT / 'experiments/runs/7a8c4941741645ee.json').read_text())['response']['result']
        self.log = json.loads((ROOT / 'experiments/runs/5f48e0da4dcd4595.json').read_text())['response']['result']['log_tail'].replace('V31', 'V33')
        self.zero_log = re.sub(r'attr=(\d+),\d+,', r'attr=\1,0,', self.log)
        self.candidate = copy.deepcopy(self.parent)
        for row in self.candidate['records']:
            if row['event'] == 'profile':
                row['report'] = row['report'].replace('sk2_kernel<', 'sk2_nibble_sidecar_v33<')

    def test_observed_array_storage_fails(self):
        result, signatures = resources(self.log)
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['rows']), 19)
        self.assertTrue(signatures)
        self.assertTrue(all(r['local_bytes'] == r['template'][2] * 32
                            for r in result['rows'] if r['compressed']))

    def test_zero_storage_and_matching_activation(self):
        result, signatures = resources(self.zero_log)
        self.assertTrue(result['passed'])
        self.assertTrue(activation(self.candidate, self.parent, signatures)['passed'])
        self.assertFalse(compare(self.parent, self.parent)['performance_gate_passed'])

    def test_missing_stopped_failed_api_or_bad_capacity(self):
        for log in ('', self.zero_log + '\nV33R_STOP coverage=incomplete',
                    self.zero_log.replace('api=0,0', 'api=7,0'),
                    self.zero_log.replace('occ=1', 'occ=0'),
                    self.zero_log + '\nV33R truncated'):
            with self.subTest(log=log[-60:]):
                self.assertFalse(resources(log)[0]['passed'])

    def test_missing_active_signature_and_no_compressed_activation(self):
        self.assertFalse(activation(self.candidate, self.parent, set())['passed'])
        self.assertFalse(activation(self.parent, self.parent, resources(self.zero_log)[1])['passed'])

    def test_previous_repair_marker_is_rejected(self):
        candidate = copy.deepcopy(self.candidate)
        for row in candidate['records']:
            if row['event'] == 'profile':
                row['report'] = row['report'].replace('sk2_nibble_sidecar_v33<', 'sk2_exact_patch_v32<')
        self.assertFalse(activation(candidate, self.parent, resources(self.zero_log)[1])['passed'])

    def test_conflicting_pointer_observation(self):
        row = next(x for x in self.zero_log.splitlines() if x.startswith('V33R '))
        self.assertFalse(resources(self.zero_log + '\n' + row.replace('occ=1', 'occ=2'))[0]['passed'])


if __name__ == '__main__':
    unittest.main()
