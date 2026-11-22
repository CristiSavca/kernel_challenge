"""Test the lone shape relaxation against the native checker on this specialization's eligible metadata."""
import ast
from hashlib import sha256
from io import BytesIO
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen
from zipfile import ZipFile
import sys

here=Path(__file__).resolve().parent
stage=Path(sys.argv[1])
wheel="https://files.pythonhosted.org/packages/ec/a2/f96912ee7d68f56e61c15555657ffa057b8afbaa21241c63342b5969bb74/humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl"
buf=urlopen(wheel).read()
assert sha256(buf).hexdigest()=="cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331"
with ZipFile(BytesIO(buf)) as z:
    native=z.read("humming/kernel/humming.py").decode()
production=(stage/"python/sglang/srt/layers/muse_spark/prompt_weight_pipeline.py").read_text()
def method(src,classname):
    cls=next(x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.ClassDef) and x.name==classname)
    return ast.unparse(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="check_shape"))
power2=lambda n:n>0 and not n&(n-1)
scope=dict(jit_utils=SimpleNamespace(is_power_of_two=power2),MmaType=SimpleNamespace(WGMMA="wgmma"))
native_check=method(native,"HummingKernel")
guard="assert self.warp_shape[1] >= 32"
assert native_check.count(guard)==1
exec(native_check.replace(guard,"assert self.warp_shape[1] >= 16"),scope)
reference=scope["check_shape"]
exec(method(production,"WidePromptWeightPipeline"),scope)
new=scope["check_shape"]
cases=accepted=0
for n,k in itertools.product([0,8,128,255,256,384,512,4096,8192], [0,16,32,63,64,65,96,128,4096]):
    geometry=SimpleNamespace(sm_version=90, block_shape=(256,128,64),warp_shape=(256,16,64), mma_type="wgmma",
        problem_shape=(0,n,k), pad_shape=(0,0,0), shape_n=n,shape_k=k, a_dtype=SimpleNamespace(num_bits=16))
    def test(fn):
        try: fn(geometry)
        except (AssertionError,ValueError): return False
        return True
    got,ref=test(new),test(reference)
    assert not got or ref
    assert got==(ref and n%256==0)
    cases+=1
    accepted+=got
print(json.dumps(dict(cases=cases,accepted=accepted,geometry="256 x 128 x 64 block, 256 x 16 x 64 warp, group-16 BF16/FP4, WGMMA, no padding",
    native_check_sha256=sha256(native_check.encode()).hexdigest(),accepted_shapes_pass_all_other_vendor_assertions=True),indent=2))

