"""Bind combined code generation to the newly compiled W13 and frozen W2 cubins."""
import hashlib
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
images=Path(sys.argv[1])
w2=json.loads((here.parent/"v274_four_block_prompt_multicast/sm90a.json").read_bytes())["variants"]
w13=json.loads((here/"machine.json").read_bytes())["images"]
expected={k:v for k,v in w2.items() if k!="DiagnosticGroup8"}
expected.update({v["kernel"]:v for v in w13})
assert len(expected)==6
rows={}
for name,r in expected.items():
    source=hashlib.sha256((images/(name+".cu")).read_bytes()).hexdigest()
    assert source==r["source_sha256"]
    rows[name]=dict(source_sha256=source,cubin_sha256=r["cubin_sha256"],registers=r["registers"])
fences=json.loads((here/"fence.json").read_bytes())
assert all(rows[x["image"][:-3]]["source_sha256"]==x["sha256"] for x in fences["images"])
print(json.dumps(dict(production_images=rows,physical_GPU=False),indent=2))
