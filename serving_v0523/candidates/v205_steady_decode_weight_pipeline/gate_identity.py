"""Compare the other reader of the shared K-pair header with its frozen actual cubins."""
import hashlib
import json
from pathlib import Path
import sys
scratch=Path(sys.argv[1]);here=Path(__file__).resolve().parent
old=json.loads((here.parent/"v197_grouped_prompt_gate_up/sm90a.json").read_bytes())["kernels"]
out={}
for name,reference in old.items():
    p=scratch/"final"/name
    hashes={ext+"_sha256":hashlib.sha256(p.with_suffix("."+ext).read_bytes()).hexdigest() for ext in ("sass","cubin")}
    assert all(hashes[k]==reference[k] for k in hashes),(name,hashes,reference)
    out[name]=hashes
print(json.dumps(dict(humming="0.1.12",toolchain="NVCC 12.9.86 SM90a",identical_to_V204_inherited_W13=out),indent=2))
