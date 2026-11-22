"""Run the frozen 145-case suite against this release, plus the quota-mode ablations."""
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

here=Path(__file__).resolve().parent
old=here.parent/'nov20_v229_staging'
sys.path.insert(0,str(here))
import submit_nov20_v187_check as runner
import test_receipt_mode
sys.path.insert(0,str(old))
import offline_suite

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    records=[]
    for name in (*offline_suite.MODULES,'test_receipt_mode'):
        stream=io.StringIO()
        suite=unittest.defaultTestLoader.loadTestsFromName(name)
        result=unittest.TextTestRunner(stream=stream).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(name+'\n'+stream.getvalue())
        records.append(dict(module=name,tests=result.testsRun))
    assert runner is sys.modules['submit_nov20_v187_check']
    assert digest(old/'submit_nov20_v187_check.py')=='72e4d0b86709452534a65560669a111b8b78bcf2d318c28625d44be7039162e0'
    assert digest(old/'v229_selection.py')=='344ed160ca81eeb248af3ba8d2eac25e72f0b64b5c7951db03c99aba12898595'
    assert (old/'nov20_v229.plan.json').read_bytes()==(here/'nov20_v229.plan.json').read_bytes()
    print(json.dumps(dict(cases=sum(row['tests'] for row in records),modules=records,
        isolate_sha256={p.name:digest(p) for p in sorted(here.glob('*.py'))},
        source=runner.__file__,plan_sha256=digest(here/'nov20_v229.plan.json'),
        read_only_frozen_parent=digest(old/'submit_nov20_v187_check.py'),shared_claim=runner.NAME,
        shared_locks=[runner.slots.LOCK,runner.PROCESS_LOCK],external_requests=0,live_worker_changes=0),indent=2))

if __name__=="__main__":
    main()
