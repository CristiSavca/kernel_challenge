"""Generate production TVM wrappers for saved and rank-major FFN boundary layouts."""
import ast
import json
import runpy
import subprocess
import sys
from pathlib import Path

root,dst=map(Path,sys.argv[1:])
jit=root/"python/sglang/kernels/jit"
cpp=runpy.run_path(str(jit/"utils/compile/cpp_args.py"))["make_cpp_args"]
spec=(jit/"utils/compile/spec.py").read_text()
ns={"Tuple":tuple}
exec(spec[spec.index("_FFI_INCLUDES ="):spec.index("def resolve_sources(")],ns)
src=root/"python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py"
header=jit/"csrc/muse_spark/norm_residual.cuh"
parent="fab843fba01f208bed745d72b99c718f137f5785"
def original(path):
    return subprocess.check_output(["git","-C",str(root),"show",parent+":"+str(path.relative_to(root))],text=True)


def definitions(text):
    tree=ast.parse("from __future__ import annotations\n"+text)
    nodes=[tree.body[0]]
    for node in tree.body:
        if getattr(node,"name","")=="_jit_module":
            node.decorator_list=[]
            nodes.append(node)
    scope={"is_arch_support_pdl":lambda:True,"make_cpp_args":cpp,"load_jit":lambda *args,**kw:kw}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),"exec"),scope)
    return scope["_jit_module"]

native=definitions(original(src))
candidate=definitions(src.read_text())
dst.mkdir(exist_ok=True)
for label, text, generator, worlds in (("old",original(header),native,(1,)),("new",header.read_text(),candidate,(1,4,8))):
    where=dst/label
    where.mkdir(exist_ok=True)
    (where/"norm_residual.cuh").write_text(text)
    for threads, prefetch in ((256,True),(512,False)):
        for world in worlds:
            args=[8192,threads,prefetch,1]
            functions=generator(*args,world) if label=="new" else generator(*args)
            body=ns["_wrapper_source"](("norm_residual.cuh",),functions["cuda_wrappers"])
            (where/f"r{world}_t{threads}.cu").write_text(body)
print(json.dumps(dict(layouts=["row-major","rank-major TP4","rank-major TP8"],ordered_capture_threads=[256,512],
                      width=8192,orig_wrappers=native(8192,256,True,1)["cuda_wrappers"],PDL=True),indent=2))
