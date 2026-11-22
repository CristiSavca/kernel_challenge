"""Compile the actual CUDA driver launcher and inspect its public TVM-FFI symbols offline."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument("stage", type=Path)
parser.add_argument("base", type=Path)
parser.add_argument("guest", nargs="+")
args = parser.parse_args()
directory = "python/sglang/kernels/jit/"
source = args.stage / "python/sglang/srt/layers/muse_spark/decode_gate_up.py"
header = args.stage / directory / "csrc/muse_spark/decode_gate_up_launch.cuh"
spec = args.base / directory / "utils/compile/spec.py"
native = ast.parse(spec.read_bytes())
defs = [n for n in native.body if getattr(n,"name",None) == "_wrapper_source" or
        isinstance(n,ast.Assign) and getattr(n.targets[0],"id",None) == "_FFI_INCLUDES"]
assert len(defs)==2
env={"Tuple":Tuple}
exec(compile(ast.Module(body=defs,type_ignores=[]),str(spec),"exec"),env)
setup = next(n for n in ast.parse(source.read_bytes()).body if getattr(n,"name",None)=="_launcher")
call = next(n for n in ast.walk(setup) if isinstance(n,ast.Call) and getattr(n.func,"id",None)=="load_jit")
wrappers=ast.literal_eval(next(n.value for n in call.keywords if n.arg=="cuda_wrappers"))
files=ast.literal_eval(next(n.value for n in call.keywords if n.arg=="cuda_files"))
assert files==["muse_spark/"+header.name]
unit=env["_wrapper_source"](tuple(files),tuple(wrappers)).encode()

def guest(*args_, data=None):
    return subprocess.check_output([*args.guest,*args_],input=data)
scratch=guest("mktemp","-d","/tmp/okc-v297-ffi.XXXXXXXX").decode().strip()
assert scratch.startswith("/tmp/okc-v297-ffi.")
paths={f:m for f,m in [(files[0],header.read_bytes()),("cuda.cu",unit)]}
base_commit="297e54fd105e974cdf401e5b4d20253a22c16f90"
for path in subprocess.check_output(["git","-C",args.base,"ls-tree","-r","--name-only",base_commit,"--",directory+"include"]).decode().splitlines():
    paths[path.removeprefix(directory)]=subprocess.check_output(["git","-C",args.base,"show",base_commit+":"+path])
bundle=io.BytesIO()
with tarfile.open(fileobj=bundle,mode="w") as tar:
    for name,content in paths.items():
        info=tarfile.TarInfo(name)
        info.size=len(content)
        tar.addfile(info,io.BytesIO(content))
guest("tar","-C",scratch,"-xf","-",data=bundle.getvalue())
build=r"""
set -eu
cd "$1"
tool="$HOME/nvcc126"
nvcc="$tool/toolchain/usr/local/cuda-12.9/bin/nvcc"
"$nvcc" -O3 -std=c++20 -DSGL_CUDA_ARCH=900 --expt-relaxed-constexpr -gencode=arch=compute_90a,code=sm_90a \
  -Xcompiler -fPIC -I"$tool/ffi/tvm_ffi/include" -I"$tool/compat" -Iinclude -c cuda.cu -o cuda.o
"$nvcc" --version
nm -g cuda.o | awk '$2 == "T" && $3 ~ /__tvm_ffi_/ || $1 == "U" && $2 ~ /cu(TensorMap|FuncSet|LaunchKernel)/'
sha256sum cuda.o
"""
output=guest("bash","-c",build,"ffi",scratch).decode()
symbols={line.split()[-1] for line in output.splitlines() if "__tvm_ffi_" in line}
assert symbols=={"__tvm_ffi_"+name for name,_ in wrappers},(output,symbols)
for ref in ("cuTensorMapEncodeTiled","cuFuncSetAttribute","cuLaunchKernel"):
    assert ref in output
hashpath=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(launcher_sha256=hashpath(header),python_sha256=hashpath(source),native_wrapper_builder_sha256=hashpath(spec),
    wrapper_sha256=hashlib.sha256(unit).hexdigest(),object_sha256=output.splitlines()[-1].split()[0],
    compiler=next(line for line in output.splitlines() if "release " in line),exports=sorted(symbols),
    driver_references=["cuTensorMapEncodeTiled","cuFuncSetAttribute","cuLaunchKernel"],
    scope="Compiled and inspected the complete CUDA driver launcher object with the published SGLang headers and TVM-FFI. No GPU execution.")
# Inspect exactly this isolated scratch before retiring the final object and copied sources.
observed=set(guest("find",scratch,"-type","f","-printf","%P\\n").decode().splitlines())
assert observed=={*paths,"cuda.o"}
cleanup = r"""
import json, os, shutil, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
manifest = json.loads(sys.stdin.read())
assert str(root).startswith("/tmp/okc-v297-ffi.")
children = sorted(root.rglob("*"), key=lambda p:len(p.parts), reverse=True)
assert set(manifest) == {str(p.relative_to(root)) for p in children if p.is_file()}
assert all(not p.is_symlink() for p in children)
for process in Path("/proc").glob("[0-9]*"):
    for entry in [process/"cwd",*list((process/"fd").glob("*"))]:
        try:
            target = Path(os.readlink(entry))
        except OSError:
            continue
        assert target != root and root not in target.parents
before = shutil.disk_usage(root.parent).free
for p in children:
    p.rmdir() if p.is_dir() else p.unlink()
root.rmdir()
after = shutil.disk_usage(root.parent).free
print(json.dumps(dict(path=str(root),files=len(manifest),directory_removed=not root.exists(),
    fs_free_before_bytes=before,fs_free_after_bytes=after,actual_free_delta_bytes=after-before)))
"""
retired=guest("python3","-c",cleanup,scratch,data=json.dumps(sorted(observed)).encode())
report["cleanup"]=json.loads(retired)
print(json.dumps(report,indent=2))
