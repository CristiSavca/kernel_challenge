"""Run the V189 integration manifest on V190 and account for the modified FA backend source."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
V189 = next(HERE.parent.glob("v189_*/independent_composition.json"))
V187 = next(HERE.parent.glob("v187_*"))
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
manifest = json.loads(V189.read_bytes())
backend = STAGE / "python/sglang/srt/layers/attention/flashattention_backend.py"
backend_digest = sha256(backend.read_bytes()).hexdigest()
results = {}
with tempfile.TemporaryDirectory() as folder:
    env = {**os.environ,"GIT_INDEX_FILE":str(Path(folder)/"index")}
    for version in (64,68,180):
        patch = next(HERE.parent.glob("v"+str(version)+"_*/patch.diff"))
        for args,inp in [(["read-tree",PIN],None),(["apply","--cached","--whitespace=error-all","-"],patch.read_bytes()),
                         (["write-tree"],None)]:
            subprocess.run(["git","-C",str(STAGE),*args],env=env,check=True,input=inp,stdout=subprocess.DEVNULL)
    for name, prior in manifest["parent_checks"].items():
        test = HERE.parent / prior["source"]
        baseline = (V187/(name+".json")).read_bytes()
        assert sha256(test.read_bytes()).hexdigest()==prior["caller_sha256"]
        assert sha256(baseline).hexdigest()==prior["parent_result_sha256"]
        expected=json.loads(baseline)
        result=json.loads(subprocess.check_output([sys.executable,str(test),str(STAGE)],
                          env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"}))
        intentional={}
        if name=="dispatch":
            assert expected["release_source_sha256"] != backend_digest
            intentional["release_source_sha256"]=backend_digest
        assert result==expected|intentional, name
        results[name] = dict(source=prior["source"], inherited_reference=prior["parent_result_sha256"],
                             updated_source_fields=intentional, behavior_fields_equal=True)
tree=subprocess.check_output(["git","-C",str(STAGE),"write-tree"],text=True).strip()
assert tree=="fc4a72c0cdab582c5212c35c40d0ff76b3e98ba4"
print(json.dumps(dict(tree=tree, target_parent_tree=manifest["tree"], checks=results, physical_H200=False),indent=2))
