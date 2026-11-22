"""Run the dry comparison and peer suites. All simulated submission commands use temporary directories."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

STAGE=Path(__file__).resolve().parent
sys.path[:0]=[str(STAGE),str(STAGE.parents[1])]
from test_transition import old, runner

MODULES=('test_v229','test_transition','test_nov20_v198_check','test_nov20_v189_check','test_nov20_v187_check',
         'test_nov20_slot_arbitration','test_nov20_pending_schema','test_nov20_v154c_check','test_nov20_v163_check',
         'test_nov20_v173_check','test_nov20_v176_check','test_nov20_v180_check','test_nov20_v141_check',
         'test_nov20_v132_check','test_nov20_v129_check')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    records=[]
    for name in MODULES:
        output=io.StringIO()
        suite=unittest.defaultTestLoader.loadTestsFromName(name)
        result=unittest.TextTestRunner(stream=output,verbosity=1).run(suite)
        records.append(dict(module=name,tests=result.testsRun,errors=len(result.errors),
                            failures=len(result.failures),skips=len(result.skipped)))
        if not result.wasSuccessful():
            sys.stderr.write(output.getvalue());return 1
    print(json.dumps(dict(live_actions=False,judge_calls=False,
                scope='offline temporary receipt directories and injected mock judge transport',
                frozen_parent_worker_sha256=digest(Path(old.__file__)),
                combined_worker_sha256=digest(Path(runner.__file__)),selection_sha256=digest(STAGE/'v229_selection.py'),
                test_sha256={n:digest(STAGE/n) for n in ('test_v229.py','test_transition.py','offline_suite.py')},
                plan_sha256=digest(STAGE/'nov20_v229.plan.json'),cases=sum(r['tests'] for r in records),modules=records,
                transition_paired_outcomes=14,old_new_claim_order_and_timeout_variants=4,lineage_negative_cases=23,
                clock_anomaly_probes=dict(preclaim_retryable=5,postclaim_preserved=2),
                preserved=dict(shared_claim=runner.NAME,worker_lock=runner.PROCESS_LOCK,arbitration_lock=runner.slots.LOCK)),indent=2))
    return 0

if __name__=='__main__':sys.exit(main())
