"""Verify V273 decode and wide-prompt codegen closure and Humming cache boundaries."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import os
from types import SimpleNamespace

here = Path(__file__).resolve().parent
root = Path(sys.argv[1])
vendor = root/'humming'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
v269 = json.loads((here.parent/'v269_warp_expert_schedule/assembly.json').read_bytes())['cases']
v270 = json.loads((here.parent/'v270_wide_prompt_weight_reuse/sm90a.json').read_bytes())['variants']
runtime = vendor/'jit/runtime.py'
compiler = vendor/'jit/compiler.py'
digest = vendor/'utils/jit.py'
launcher = vendor/'csrc/launcher/launcher.cpp'
native = next(c for c in ast.parse(runtime.read_text()).body if isinstance(c,ast.ClassDef) and c.name=='KernelRuntime')
native.decorator_list=[]
native.body = [n for n in native.body if getattr(n,'name','')=='__new__']
env={'dtypes':SimpleNamespace(DataType=type('DataType',(),{}))}
exec(compile(ast.Module(body=[native],type_ignores=[]),str(runtime),'exec'),env)
base = env['KernelRuntime']
base._instances = {}
ctx = 1
base.current_context = staticmethod(lambda:ctx)
names = ['LiveDecodeDownKernel','WidePromptWeightPipeline','LiveDecodeGateUpKernel','GroupedResidualPromptGateUpKernel']
classes = [type(n,(base,),{}) for n in names]
ref = {}
for ctx in (1,2,1,2):
    for cls in classes+classes[::-1]:
        # Identical kwargs deliberately isolate class name and CUDA context.
        k=cls.__new__(cls, shape=256, block=[8,128,256])
        k.inited=True
        key=ctx,cls.__name__
        if key in ref:
            assert ref[key] is k
        else:
            assert all(other is not k for other in ref.values())
            ref[key]=k
assert len(ref)==8
hashfn = next(n for n in ast.parse(digest.read_text()).body if getattr(n,'name','')=='hash_to_hex')
hashenv={'hashlib':hashlib}
exec(compile(ast.Module(body=[hashfn],type_ignores=[]),str(digest),'exec'),hashenv)
assert 'signature = f"{cls.__name__}$${cls.signature()}$${flags}$${kernel_expr}$${code}"' in compiler.read_text()
assert 'context_path_ids.find(cubin_path);' in launcher.read_text() and 'g_path_ids[context]' in launcher.read_text()
sources = {}
for f in sorted((root/'live').glob('*.cu')):
    entry=v269[f.stem]['V269']
    assert sha(f)==entry['sha256']['cu']
    sources[f.stem]=dict(source_sha256=sha(f),matched_cubin_sha256=entry['sha256']['cubin'])
for f in sorted((root/'wide').glob('*.cu')):
    entry=v270[f.stem]
    assert sha(f)==entry['source_sha256']
    sources[f.stem]=dict(source_sha256=sha(f),matched_cubin_sha256=entry['cubin_sha256'])
keys=[]
for name,row in sources.items():
    f=(root/('live' if name.startswith('w') else 'wide')/(name+'.cu')).read_text()
    code,stub=f.rsplit('\nauto ptr = reinterpret_cast<void*>(&',1)
    expr=stub.removesuffix(');\n')
    # The other fields are held constant to probe expression/source separation.
    h=hashenv['hash_to_hex']('NVCCCompiler$$toolchain$$flags$$'+expr+'$$'+code+'$$headers')
    keys.append(h)
assert len(keys)==len(set(keys))==9
patch_sha = sha(here/'patch.diff')
stage = root/'sglang'
tree = subprocess.check_output(['git','-C',str(stage),'write-tree'],text=True).strip()
assert patch_sha == "5273a9889f2820086737169740af412644b2862f6ba1e48a5f1c3b14767b8500"
with tempfile.TemporaryDirectory() as temp:
    env = {**os.environ,'GIT_INDEX_FILE':str(Path(temp)/'index')}
    def git(*args):
        return subprocess.check_output(['git','-C',str(stage),*args],env=env)
    git('read-tree','297e54fd105e974cdf401e5b4d20253a22c16f90')
    git('apply','--cached','--whitespace=error-all',str(here/'patch.diff'))
    assert git('write-tree').decode().strip()==tree
assert not subprocess.check_output(['git','-C',str(stage),'diff','--name-only'])
startup = {}
for receipt, reference in [('decode-startup.json','v269_warp_expert_schedule/startup.json'),
                           ('wide-startup.json','v270_wide_prompt_weight_reuse/startup.json')]:
    check = root/receipt
    assert json.loads(check.read_bytes()) == json.loads((here.parent/reference).read_bytes())
    startup[receipt] = sha(check)
print(json.dumps(dict(full_patch_sha256=patch_sha, staged_tree=tree, sources=sources, startup_receipts=startup, vendor_closure={str(p.relative_to(vendor)):sha(p) for p in (runtime,compiler,digest,launcher)},
    native_instance_cache=dict(contexts=2,classes=names,alternating_calls=32,distinct_instances=8),
    source_expression_probe_distinct_keys=9,
    dispatch_scope="Six V269 decode and all three V270 W2 prompt CUDA sources regenerated from the V273 checkout. Public Python instance key replay used identical dummy arguments across class names; source keys used a common toolchain, flag and header label. The C++ path registry was source-inspected. No CUDA-context hardware or timing test."),indent=2))
