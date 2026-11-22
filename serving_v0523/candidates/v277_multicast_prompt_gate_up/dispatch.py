"""Exercise production dispatch including ordinary, missing, stale and failed launches."""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

jit=ModuleType("sglang.kernels.jit.utils")
jit.load_jit=None
sys.modules[jit.__name__]=jit
stage=Path(sys.argv[1])
run=runpy.run_path(str(stage/"python/sglang/srt/layers/muse_spark/decode_gate_up.py"))["run_gate_up"]
calls=[]
class Tensor:
    def __init__(self,shape,ptr=17):
        self.shape,self.ptr=shape,ptr
    def new_empty(self,shape):
        return Tensor(shape)
    def data_ptr(self):
        return self.ptr
    def int(self):
        return self

def call(name):
    def wrapper(*args):
        calls.append((name,*args[-4:]))
        return accepted
    return wrapper

launcher=SimpleNamespace(run_prompt_multicast=call("cluster"),run_prompt=call("prompt"),run=call("decode"))
tuning="native tune"
weights,scales=Tensor((32,256,16384),1),Tensor((32,256,8192),2)
bindings=tuple((t.data_ptr(),t.shape) for t in (weights,scales))
layer=SimpleNamespace(w13_weight=weights,w13_weight_scale=scales,w13_global_scale=Tensor((32,)),intermediate_size_per_partition=4096)
cases=[]
for prompt,rows,cluster,stale,accepted,expected in (
    (True,704,True,False,True,"prompt"),(True,705,True,False,True,"cluster"),(True,4096,False,False,True,"prompt"),
    (True,4096,True,True,True,None),(False,4096,True,False,True,"decode"),(True,4096,True,False,False,"cluster")):
    kernels={tuning:(None,11,128000,176)}
    if cluster: kernels[tuning,"multicast"]=(None,21,128000,176,128)
    state=launcher,Tensor((256,)),bindings,kernels,132
    setattr(layer,"humming_prompt_gate_up" if prompt else "humming_decode_gate_up",state)
    weights.ptr=91 if stale else 1
    calls.clear()
    result=run(layer,Tensor((rows,4096)),Tensor((rows,)),Tensor((33,)),tuning,prompt=prompt)
    assert (result is not None)==(accepted and not stale)
    assert ([] if expected is None else [expected])==[c[0] for c in calls]
    if expected: assert calls[0][1:]==((21,128000,176,128) if expected=="cluster" else (11,128000,176,132))
    cases.append(dict(rows=rows,prompt=prompt,cluster_available=cluster,stale=stale,accepted=accepted,calls=calls.copy()))
print(json.dumps(cases,indent=2))
