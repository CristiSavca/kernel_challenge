"""Repeat the integrated expert replays with a checked, non-BLAS reference contraction."""
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = Path(sys.argv[1]).resolve()
GENERIC = HERE.parent / "v66_humming_grouped_decode/cpu_check.py"
original = Path.read_text
old = "return bf16(inputs.astype(np.float64) @ weight.astype(np.float64).T)"
new = """assert np.isfinite(inputs).all() and np.isfinite(weight).all()
    result = np.einsum('...k,nk->...n',inputs.astype(np.float64),weight.astype(np.float64),optimize=False)
    assert np.isfinite(result).all()
    return bf16(result)"""


def checked_read(path,*a,**kw):
    content=original(path,*a,**kw)
    if path.resolve()==GENERIC:
        assert content.count(old)==1
        return content.replace(old,new)
    return content


suites = {}
for name, script, args in [
    ("decode", HERE.parent/"v191_decode_gate_up_epilogue/dispatch_check.py", []),
    ("linked", HERE.parent/"v195_fused_expert_affine_verify/linked_check.py", [HERE/"independent_attention_output.json"]),
]:
    out = io.StringIO()
    with patch.object(Path,"read_text",checked_read), patch.object(sys,"argv",[str(script),str(STAGE),*map(str,args)]), redirect_stdout(out), warnings.catch_warnings():
        warnings.simplefilter("error",RuntimeWarning)
        runpy.run_path(str(script),run_name="__main__")
    value=json.loads(out.getvalue())
    reference=json.loads((HERE/(name+".json")).read_bytes())
    assert value==reference,name
    suites[name]=dict(ordinary_reference_sha256=sha256((HERE/(name+".json")).read_bytes()).hexdigest(),matched=True)
words=json.loads((HERE/"independent_attention_output.json").read_bytes())["values"]
bits=np.asarray(words,dtype="<u2")
payload=sha256(bits.tobytes()).hexdigest()
assert bits.shape==(8,32) and payload==reference["attention_bf16_sha256"]
print(json.dumps(dict(stage_checked=True,interpreter_slice_bf16_sha256=payload,slice_words=bits.size,distinct_words=int(np.unique(bits).size),
    ordinary_reference_suite=suites,physical_h200=False,scope="Independent NumPy contraction, explicit finite checks on live GEMM inputs, weights and outputs, all runtime warnings promoted to failures. Includes concrete outputs from segmented attention fed to the native decoder-layer fixture."),indent=2))
