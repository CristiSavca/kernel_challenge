"""Exercise V177's module check and Humming's actual ELF loader against both compiled modules."""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as S

stage=Path(sys.argv[1])
vendor=stage/"vendor/humming"
adapter=(stage/"sglang/python/sglang/srt/layers/muse_spark/prefill_gate_up.py").read_text()
source=(vendor/"jit/runtime.py").read_text()
elf=(vendor/"utils/cubin.py").read_text()
cuda=(vendor/"utils/cuda.py").read_text()
compiler=(vendor/"jit/compiler.py").read_text()
def method(src,name,cls=None):
    tree=ast.parse(src)
    if cls:tree=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name==cls)
    n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    n.decorator_list=[]
    return ast.unparse(n)
read_names=next(n.name for n in ast.parse(elf).body if getattr(n,"name",None)=="get_cubin_kernel_names")
ns=dict(Path=Path,struct=struct)
exec(method(elf,read_names),ns)
symbols=ns[read_names]
calls=[]
driver=S(cuModuleLoad=lambda fname:(0,b"module"),cuModuleGetFunction=lambda module,name:(calls.append(name) or (0,7)))
native=dict(cbd=driver,get_cubin_kernel_names=symbols);exec(method(source,"load_cubin","KernelRuntime"),native)
file=None
def read(cmd,text):
    assert cmd==["cuobjdump","-sass",str(file)] and text is True
    return file.with_suffix(".sass").read_text()
ctx={"NVCCCompiler":S(_get_env=lambda:{"binaries":{"cuobjdump":"cuobjdump"}}),
     "subprocess":S(check_output=read),"KernelRuntime":S(load_cubin=native["load_cubin"])}
exec(method(adapter,"load_cubin","GroupedGateUpKernel"),ctx)
cases=[]
for width in (136,152,176):
    for binary in ("nvcc","nvrtc"):
        suffix=".nvrtc" if binary=="nvrtc" else ""
        file=stage/"machine/generated"/f"gate_{width}{suffix}.cubin"
        if not file.exists() or not file.with_suffix(".sass").exists():continue
        names=[x for x in symbols(str(file)) if "humming" in x];assert len(names)==1
        obj=S(block_shape=[width,256,64],name="humming",kernel_filename=str(file),cubin_loaded=False)
        calls.clear()
        try:
            ctx["load_cubin"](obj);ctx["load_cubin"](obj);accepted=True
        except RuntimeError:
            accepted=False
        assert accepted==(binary=="nvcc") and len(calls)==int(accepted)
        if accepted:assert calls==[names[0].encode()] and obj.func==7 and obj.cubin_loaded
        cases.append(dict(band=width,compiler=binary,accepted=accepted,module_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                          native_named_symbols=len(names),driver_handle_resolutions=len(calls)))
assert any(c["accepted"] for c in cases) and any(not c["accepted"] for c in cases)
assert 'required_binaries=["nvcc"]' in compiler and '"cuobjdump"' in cuda
assert '(cls.__name__, cls.current_context(),' in source and 'not cls._instances[signature].inited' in source
assert 'kernel_expr}$${code}' in compiler
print(json.dumps(dict(compiled_modules=cases,adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest(),
                      vendor_version="0.1.12",actual_nvcc_or_nvrtc_execution=False,
                      physical_driver=False,scope="Actual ELF names, actual SASS, actual wrapper and native loader control flow; driver returns a test handle."),indent=2))
