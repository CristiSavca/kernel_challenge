#!/usr/bin/env python3
"""Bounded offline fixtures; synthetic attributes are never measurements."""
import copy
import json
import unittest
from unittest.mock import patch

import analyze_sk2_r16_resources as audit


SOURCE, INVENTORY = audit.load_contract()


def fixture_rows(pooler_pmax=32, qkv_pmax=48):
    rows=[]
    for entry in INVENTORY:
        t=copy.deepcopy(entry['template'])
        x={k:entry[k] for k in ('M','N','K')}
        x['pmax']=pooler_pmax if entry['index']<9 else qkv_pmax
        x.update({k:v for k,v in audit.geometry(t,x).items() if k in audit.XF})
        rows.append(dict(t=t,x=x,a=dict(zip(audit.AF,[64,0,0,227328,1024])),
                         occ=2,minimum=entry['minimum'],err=dict.fromkeys(audit.EF,0)))
    return rows


def row_text(r):
    csv=lambda value,keys:','.join(str(value[k]) for k in keys)
    return (f"R16R t={csv(r['t'],audit.TF)} x={csv(r['x'],audit.XF)} "
            f"a={csv(r['a'],audit.AF)} occ={r['occ']} min={r['minimum']} "
            f"err={csv(r['err'],audit.EF)}")


def log(rows,header=True):
    return ('\n'.join(([audit.SCHEMA] if header else [])+[row_text(r) for r in rows]))+'\n'


class Fixtures(unittest.TestCase):
    def parse(self,rows=None,**kw):
        return audit.parse_log(log(rows if rows is not None else fixture_rows(),**kw),INVENTORY)

    def test_inventory_pins_and_identity(self):
        self.assertEqual(len(INVENTORY),14)
        self.assertEqual(len({tuple(r['template'].values()) for r in INVENTORY}),12)
        self.assertNotEqual(INVENTORY[7]['N'],INVENTORY[11]['N'])
        self.assertNotIn('pmax',INVENTORY[7])
        self.assertNotIn('attributes',INVENTORY[7])

    def test_complete_synthetic_case_inventory(self):
        r=self.parse();self.assertEqual(r['gate'],'PASS_RETAINED_PREFLIGHT_CASES')
        self.assertEqual(r['unobserved_case_indices'],[])
        self.assertEqual(r['records'][7]['launch']['pmax'],32)
        self.assertEqual(r['records'][11]['launch']['pmax'],48)

    def test_actual_failure_geometry_fixture(self):
        rows=fixture_rows()[:3];rows[0]['a']['registers_per_thread']=78
        rows[1]['a']['registers_per_thread']=78
        rows[2]['a'].update(registers_per_thread=74,static_shared_bytes=384)
        rows[2]['occ']=1
        r=self.parse(rows)
        self.assertEqual(r['gate'],'FAIL')
        self.assertEqual(r['failed_record_ids'],[2])
        self.assertEqual(r['unobserved_case_indices'],list(range(3,14)))
        self.assertEqual(r['records'][0]['launch']['dynamic_shared_bytes'],90768)
        self.assertEqual(r['records'][2]['launch']['dynamic_shared_bytes'],104160)
        self.assertEqual(r['records'][2]['launch']['threads'],416)

    def test_every_API_failure_rejects(self):
        for name in audit.EF:
            with self.subTest(name=name):
                rows=fixture_rows();rows[0]['err'][name]=1
                self.assertEqual(self.parse(rows)['gate'],'FAIL')

    def test_local_allocation_rejects(self):
        rows=fixture_rows();rows[0]['a']['local_bytes_per_thread']=4
        self.assertEqual(self.parse(rows)['gate'],'FAIL')

    def test_wrong_minimum_rejects(self):
        rows=fixture_rows();rows[2]['minimum']=1
        self.assertEqual(self.parse(rows)['gate'],'FAIL')

    def test_wrong_dynamic_rejects(self):
        rows=fixture_rows();rows[2]['x']['dynamic_shared_bytes']-=1024
        self.assertEqual(self.parse(rows)['gate'],'FAIL')

    def test_multi_preparation_variable_pmax(self):
        r=self.parse(fixture_rows()+fixture_rows(64,80))
        self.assertEqual(r['gate'],'PASS_RETAINED_PREFLIGHT_CASES')
        self.assertEqual(len(r['preparation_sequence_groups']),2)

    def test_same_table_pmax_conflict_unknown(self):
        for index in (7,11,13):
            with self.subTest(index=index):
                rows=fixture_rows();rows[index]['x']['pmax']+=2
                rows[index]['x']['dynamic_shared_bytes']=audit.geometry(rows[index]['t'],rows[index]['x'])['dynamic_shared_bytes']
                r=self.parse(rows)
                self.assertEqual(r['gate'],'UNKNOWN_INCOMPLETE_EVIDENCE')
                self.assertTrue(r['preparation_sequence_groups'][0]['issues'])

    def test_same_pointer_conflict_unknown(self):
        rows=fixture_rows();rows[11]['a']['registers_per_thread']+=1
        r=self.parse(rows);self.assertEqual(r['gate'],'UNKNOWN_INCOMPLETE_EVIDENCE')
        self.assertTrue(r['conflicts'])

    def test_missing_schema_unknown(self):
        self.assertEqual(self.parse(header=False)['gate'],'UNKNOWN_INCOMPLETE_EVIDENCE')

    def test_stop_unknown(self):
        r=audit.parse_log(log(fixture_rows())+audit.STOP+'\n',INVENTORY)
        self.assertEqual(r['gate'],'UNKNOWN_INCOMPLETE_EVIDENCE')

    def test_partial_and_out_of_order_unknown(self):
        for rows in (fixture_rows()[:2],fixture_rows()[1:],fixture_rows()+fixture_rows()[:3]):
            self.assertEqual(self.parse(rows)['gate'],'UNKNOWN_INCOMPLETE_EVIDENCE')

    def test_malformed_metadata_unknown(self):
        bad=log(fixture_rows()).replace('occ=2 min=1','occ=2',1)
        self.assertEqual(audit.parse_log(bad,INVENTORY)['gate'],'UNKNOWN_INCOMPLETE_EVIDENCE')

    def test_unobserved_empty_log(self):
        r=audit.parse_log('',INVENTORY)
        self.assertEqual(r['gate'],'UNKNOWN_INCOMPLETE_EVIDENCE')
        self.assertEqual(r['unobserved_case_indices'],list(range(14)))

    def test_pinned_catalog_rejects_modified_source(self):
        with self.assertRaisesRegex(ValueError,'exact source'):
            audit.generated_catalog(SOURCE+'\n')

    def test_pending_has_no_resources_or_metrics(self):
        with patch.object(audit.performance,'read_run',return_value={'status':'PENDING','provenance':{}}):
            r=audit.analyze()
        self.assertEqual(r['activation_gate'],'PENDING')
        self.assertEqual(r['metrics'],{})
        self.assertNotIn('resources',r)

    def test_actual_source_bound_terminal(self):
        r=audit.analyze()
        self.assertEqual(r['status'],'OFFICIAL_GATE_FAILED')
        self.assertEqual(r['decision'],'REJECT_MECHANISM')
        self.assertEqual(r['resource_gate'],'FAIL')
        self.assertEqual(r['activation_gate'],'UNAVAILABLE_OFFICIAL_GATE_FAILED')
        self.assertEqual(r['resources']['failed_record_ids'],[2])
        self.assertEqual(r['resources']['unobserved_case_indices'],list(range(3,14)))
        self.assertEqual(r['metrics'],{})


if __name__=='__main__':
    unittest.main(verbosity=2)
