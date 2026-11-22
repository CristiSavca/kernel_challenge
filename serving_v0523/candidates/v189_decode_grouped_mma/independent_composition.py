"""Run V187's seven integration fixtures on the V189 exported tree, comparing every receipt field."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
PARENT = next(HERE.parent.glob("v187_*"))
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
suites = {
    "transport":"v185_*/check_transport.py",
    "worker":"v185_*/worker_check.py",
    "model":"v185_*/check_model.py",
    "producer":"v182_*/model_check.py",
    "prompt_verify":"v180_*/check_cycle.py",
    "paged":"v182_*/independent_paged.py",
    "dispatch":"v136_*/dispatch_check.py",
}
with tempfile.TemporaryDirectory() as scratch:
    env = {**os.environ, "GIT_INDEX_FILE":str(Path(scratch)/"index")}
    def git(*args, data=None):
        return subprocess.check_output(["git", "-C", str(STAGE), *args], env=env, input=data)
    # The inherited fixtures also load selected sources by their earlier frozen tree IDs.
    for version in (64,68,180):
        patch = next(HERE.parent.glob("v"+str(version)+"_*/patch.diff"))
        git("read-tree", PIN)
        git("apply","--cached","--whitespace=error-all","-",data=patch.read_bytes())
        git("write-tree")
    parent = json.loads((PARENT/"validation.json").read_bytes())
    results = {}
    for name, pattern in suites.items():
        test = next(HERE.parent.glob(pattern))
        baseline = (PARENT/(name+".json")).read_bytes()
        assert sha256(baseline).hexdigest() == parent["suites"][name]
        replay = subprocess.check_output([sys.executable,str(test),str(STAGE)],
                                         env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
        assert json.loads(replay) == json.loads(baseline), name
        results[name] = dict(source=str(test.relative_to(HERE.parent)), caller_sha256=sha256(test.read_bytes()).hexdigest(),
                             parent_result_sha256=sha256(baseline).hexdigest(), result_sha256=sha256(replay).hexdigest())
print(json.dumps(dict(parent_tree=parent["tree"], tree=subprocess.check_output(["git","-C",str(STAGE),"write-tree"],text=True).strip(),
                      parent_checks=results, identical_all_fields=True, physical_H200=False),indent=2))
