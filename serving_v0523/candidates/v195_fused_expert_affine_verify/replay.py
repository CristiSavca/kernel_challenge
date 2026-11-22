"""Rerun both donors' numerical and lifecycle suites against the merged checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

here=Path(__file__).resolve().parent
expert=here.parent/"v191_decode_gate_up_epilogue"
att=here.parent/"v192_affine_verify_tiles"
scratch=Path(sys.argv[1]).resolve()
stage=scratch/"sglang"
tests=[(expert,"dispatch_check.py","decode.json",[stage]),(expert,"independent_epilogue.py","independent_epilogue.json",[stage]),
       (expert,"cpu_check.py","cpu.json",[stage]),(expert,"startup.py","startup.json",[scratch]),
       (att,"independent_vmm.py","independent_vmm.json",[stage,*sys.argv[2:]]),
       (att,"independent_graph.py","independent_graph.json",[stage])]
sha=lambda b:hashlib.sha256(b).hexdigest()
with tempfile.TemporaryDirectory() as tmp:
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",GIT_INDEX_FILE=str(Path(tmp)/"index"))
    # The graph review compares startup field order to V190, its original source baseline.
    subprocess.run(["git","-C",stage,"read-tree","fc4a72c0cdab582c5212c35c40d0ff76b3e98ba4"],check=True,env=env)
    result={}
    for donor,script,receipt,args in tests:
        raw=subprocess.check_output([sys.executable,donor/script,*args],env=env)
        data=json.loads(raw);expected=json.loads((donor/receipt).read_bytes())
        assert data==expected,script
        result[script]=dict(donor=donor.name,script_sha256=sha((donor/script).read_bytes()),receipt=receipt,
                            matched_all_fields=True,receipt_sha256=sha((donor/receipt).read_bytes()))
print(json.dumps(dict(actual_device=False,combined_tree="0481ee37be4bff6d675a4a8fb6144ab73091977e",tests=result),indent=2))
