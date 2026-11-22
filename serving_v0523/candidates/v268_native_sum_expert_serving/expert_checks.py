"""Run the normal grouped decode route and the host-compiled W13 header on this export."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
V263=HERE.parent/"v263_token_major_expert_decode"
if __name__=="__main__":
    outputs={}
    for key, driver in (("decode","dispatch.py"),("writer","independent_token_writer.py")):
        fn=runpy.run_path(str(V263/driver))["main"]
        fn.__globals__["HERE"]=HERE
        if key=="decode":
            buf=io.StringIO()
            with redirect_stdout(buf):
                fn()
            outputs[key]=json.loads(buf.getvalue())
        else:
            outputs[key]=fn()
    assert outputs["decode"]["patch_sha256"]==outputs["writer"]["patch_sha256"]
    print(json.dumps(outputs,indent=2))
