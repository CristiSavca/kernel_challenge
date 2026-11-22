"""Assemble exact emitted grouped-decode specializations with Triton and ptxas, without a GPU."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from compile_decode_sm90 import load_jit_module


source = Path(sys.argv[1])
assembler = Path(triton.__file__).parent / "backends/nvidia/bin/ptxas"


def kernel_hashes(path):
    names = {"_decode_route_expert", "_plan_grouped_decode", "_gather_grouped_decode", "_stage_grouped_decode"}
    tree = ast.parse(path.read_bytes())
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            for n in tree.body if getattr(n,"name",None) in names}


def build(kernel, tag, dynamic, constants, warps, folder):
    sig = dict(zip(kernel.arg_names, dynamic + ["constexpr"]*len(constants), strict=True))
    binary = triton.compile(ASTSource(kernel, sig, constexprs=constants),
                            target=GPUTarget("cuda",90,32),options={"num_warps":warps})
    ptx=folder/"test.ptx"
    ptx.write_text(binary.asm["ptx"])
    info=subprocess.run([assembler,"-v","-arch=sm_90a",ptx,"-o",folder/"test.cubin"],check=True,capture_output=True,text=True).stderr
    spills=re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",info)
    return dict(case=tag,kernel=kernel.__name__,constants=constants,dynamic_signature=dynamic,
                warps=warps,shared_bytes=binary.metadata.shared,registers=int(re.search(r"Used (\d+) registers",info)[1]),
                spill_bytes=list(map(int,spills.groups())),cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest())



def stage_warps(source, slots, hidden_size, experts=None):
    """Read the launch expression from the submitted module."""
    body = ast.parse(source.read_bytes())
    launchers = [n for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "_stage_grouped_decode"]
    launcher, = launchers
    expression, = [kw.value for kw in launcher.keywords if kw.arg == "num_warps"]
    safe = (ast.IfExp, ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE,
            ast.Eq, ast.NotEq, ast.Constant, ast.Name, ast.Load)
    assert all(isinstance(node, safe) for node in ast.walk(expression))
    values = {"slots":slots, "span":triton.next_power_of_2(slots), "hidden_size":hidden_size, "experts":experts}
    return eval(compile(ast.Expression(expression), str(source), "eval"), {"__builtins__":{}}, values)


if __name__ == "__main__":
    cases = []
    sensitivity = []
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary)
        module=load_jit_module(source,folder)
        configurations=[
            (1,1,1,1024,False), (8,8,32,4096,True), (16,8,32,4096,True), (32,8,32,4096,True),
            (64,8,32,4096,True), (128,8,32,4096,True), (256,8,32,4096,True),
            (64,4,7,1025,True), (128,4,17,3073,False), (256,16,64,2048,True), (256,8,256,4096,False)]
        for span,topk,experts,hidden,mapping in configurations:
            for dtype in ("i32","i64"):
                pointers=["*"+dtype,"*i32" if mapping else "*"+dtype]+["*i32"]*4
                c=dict(EXPERTS=experts,GLOBAL_EXPERTS=experts*8 if mapping else experts,
                       HAS_EXPERT_MAP=mapping,SLOT_SPAN=span,EXPERT_SPAN=triton.next_power_of_2(experts+1))
                gather=dict(TOPK=topk,HIDDEN=hidden,SLOT_SPAN=span,BLOCK=triton.next_power_of_2(hidden))
                tag=f"span{span}_top{topk}_e{experts}_h{hidden}_map{int(mapping)}_{dtype}"
                cases.append(build(module._stage_grouped_decode,tag,pointers+["*bf16","*bf16","i32"],
                                   {**c,"TOPK":topk,"HIDDEN":hidden,"BLOCK":gather["BLOCK"]},
                                   stage_warps(source,span,hidden,experts),folder))
                if (span,experts,hidden,mapping)==(256,32,4096,True):
                    for warp_count in (2,4,8):
                        sensitivity.append(build(module._stage_grouped_decode,tag,pointers+["*bf16","*bf16","i32"],
                            {**c,"TOPK":topk,"HIDDEN":hidden,"BLOCK":gather["BLOCK"]},warp_count,folder))
                if dtype=="i32" and mapping and experts==32:
                    cases.append(build(module._plan_grouped_decode,tag,pointers+["i32"],c,4,folder))
                    cases.append(build(module._gather_grouped_decode,tag,["*bf16","*i32","*i32","*bf16","i32"],
                                       gather,8 if hidden>=4096 else 4,folder))
    print(json.dumps(dict(triton=triton.__version__,physical_gpu=False,architecture="sm90",cases=cases,controlled_warp_sensitivity=sensitivity,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),kernel_ast_sha256=kernel_hashes(source)),indent=2))
