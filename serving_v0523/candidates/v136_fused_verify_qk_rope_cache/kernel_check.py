"""Run the production producer and native norm bodies with identical inputs."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import triton
import triton.language as tl


def module(path, work):
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "_qk_rms_norm_fwd_kernel")
    lines = source.splitlines(keepends=True)
    content = "".join(lines[node.decorator_list[0].lineno - 1:node.end_lineno])
    out = work / (path.stem + "_jit.py")
    out.write_text("import triton\nimport triton.language as tl\nFP8_E4M3_MAX=448.0\n" + content)
    spec = importlib.util.spec_from_file_location(out.stem, out)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._qk_rms_norm_fwd_kernel


def compile_cases(kernel, work):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource

    results = []
    for heads, kv, d, bm in ((16, 2, 64, 4), (16, 2, 64, 8), (8, 4, 64, 4),
                              (5, 1, 80, 8), (8, 2, 128, 16), (8, 4, 256, 32)):
        for rope in (True, False):
            for pdl in (False, True):
                c = dict(NUM_Q_HEADS=heads, NUM_KV_HEADS=kv, D=d, BD=triton.next_power_of_2(d),
                         BLOCK_M=bm, HAS_ROPE=rope, HAS_KV_STORE=True, FP8_KV_STORE=True, USE_PDL=pdl, FP8_MAX=448.0)
                dyn = [a for a in kernel.arg_names if a not in c]
                types = ["*bf16","*u8","*bf16","*i64","*fp32","*u8","*u8","*i64","fp32","i32","i32","*i64","*u8"]
                sig = dict(zip(dyn, types, strict=True))
                sig.update({k:"constexpr" for k in c})
                binary = triton.compile(ASTSource(kernel, sig, constexprs=c),
                    target=GPUTarget("cuda",90,32), options={"num_warps":4,"launch_pdl":pdl})
                ptx = work/"kernel.ptx"
                ptx.write_text(binary.asm["ptx"])
                assert "@bad trap;" in binary.asm["ptx"]
                assert ("griddepcontrol.launch_dependents" in binary.asm["ptx"]) == pdl
                res = subprocess.run([get_ptxas(90).path,"-arch=sm_90a","-v",str(ptx),"-o",str(work/"kernel.cubin")],
                                      check=True,capture_output=True,text=True)
                results.append(dict(q_heads=heads,kv_heads=kv,dim=d,block_m=bm,rope=rope,pdl=pdl,
                                    unsigned_release_trap=True, registers=int(re.search(r"Used (\d+) registers",res.stderr)[1]),
                                    spills=re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",res.stderr).groups(),
                                    cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest()))
    return dict(mode="release SM90 offline AOT, without CUDA device execution", cases=results)


def interpret(kernel, stock, lut):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runs"))
    import numpy as np
    from interpret_decode_kernels import buf as Buf

    # The interpreter cannot execute NVIDIA inline PTX. Substitute just this instruction,
    # compare its unsigned result, and separately require the trap in every real cubin.
    saved = tl.inline_asm_elementwise
    def checked(*args, **kwargs):
        assert "setp.ge.u64" in kwargs["asm"] and "@bad trap;" in kwargs["asm"]
        idx, bound = kwargs["args"]
        if np.any(idx.handle.data.astype(np.uint64) >= bound.handle.data.astype(np.uint64)):
            raise ValueError("physical bound")
        return idx
    tl.inline_asm_elementwise = checked
    rng = np.random.default_rng(780)
    cases = []
    try:
        geom = ((1,1,1,16),(7,5,1,80),(8,16,2,64),(32,16,2,64),(57,16,2,64),(128,8,4,64),(256,16,2,64))
        for rows, h, g, d in geom:
            for rope in (False,True):
                for special in (False,True):
                    cap, qn, kn = rows + 5, h*d, g*d
                    pitch=qn+2*kn+(13 if rows%2 else qn)
                    finite = rng.uniform(-10,10,(rows,pitch)).astype(np.float32)
                    x = (finite.view(np.uint32) >> 16).astype(np.uint16)
                    if special:
                        vals = np.array([0,0x8000,0x7f80,0xff80,0x7fc0,0xffc1,0x7f81,0x0001,0x43e0,0xc3e1],np.uint16)
                        x[:,::19] = np.resize(vals,x[:,::19].shape)
                    pos = rng.integers(0,21,rows,dtype=np.int64)
                    cs=rng.normal(size=(21,d)).astype(np.float32)
                    idtype=np.int32 if rows%2 else np.int64
                    slots=rng.permutation(np.arange(1,cap,dtype=idtype))[:rows].copy()
                    slots[-1]=0
                    q,k=np.zeros((rows,qn),np.uint16), np.zeros((rows,kn),np.uint16)
                    out=np.full_like(q,234,dtype=np.uint8)
                    kc=rng.integers(0,256,(cap+3,kn),dtype=np.uint8)
                    vc=rng.integers(0,256,(cap+3,kn),dtype=np.uint8)
                    ek,ev=kc.copy(),vc.copy()
                    bh=4 if rows*(h+g)<1024 else 8
                    xw=Buf(x,tl.bfloat16); pw=Buf(pos,tl.int64); cw=Buf(cs,tl.float32)
                    loc=Buf(slots,tl.int32 if idtype==np.int32 else tl.int64)
                    const=dict(NUM_Q_HEADS=h,NUM_KV_HEADS=g,D=d,BD=triton.next_power_of_2(d),BLOCK_M=bh,HAS_ROPE=rope,
                               USE_PDL=False,FP8_MAX=448.0)
                    old_rows=rows*(h+g)
                    stock[(triton.cdiv(old_rows,bh),)](xw,Buf(q,tl.bfloat16),Buf(k,tl.bfloat16),pw,cw,xw,xw,loc,
                         1e-6,pitch,old_rows,cap,HAS_KV_STORE=False,FP8_KV_STORE=False,**const)
                    new_rows=rows*(h+2*g)
                    limit=np.array([cap],np.int64)
                    args=[xw,Buf(out,tl.uint8),xw,pw,cw,Buf(kc,tl.uint8),Buf(vc,tl.uint8),loc,1e-6,pitch,new_rows,Buf(limit,tl.int64)]
                    newconst=dict(**const,HAS_KV_STORE=True,FP8_KV_STORE=True,ENCODINGS=Buf(lut,tl.uint8))
                    kernel[(triton.cdiv(new_rows,bh),)](*args,**newconst)
                    live=slots>0
                    ek[slots[live]]=lut[k[live]]
                    ev[slots[live]]=lut[x[live,qn+kn:qn+2*kn]]
                    assert np.array_equal(out,lut[q]), (rows,rope,special,"q",np.sum(out!=lut[q]))
                    assert np.array_equal(kc,ek), (rows,rope,special,"k",np.sum(kc!=ek))
                    assert np.array_equal(vc,ev), (rows,rope,special,"v")
                    cases.append(dict(rows=rows,heads=h,kv_heads=g,d=d,rope=rope,special=special,
                                      checked_bytes=out.size+kc.size+vc.size))
        faults=0
        for invalid in (-1,cap,cap+2):
            slots[0]=invalid
            try:
                kernel[(triton.cdiv(new_rows,bh),)](*args,**newconst)
            except Exception as e:
                if "physical bound" not in str(e):
                    raise
                faults+=1
        # Replay the same launch and tensors after VMM publishes a different
        # backed prefix. Slot 20 fits the reserved K/V arrays throughout.
        slots[:] = 0
        slots[0] = 20
        replay_faults = 0
        for effective in (1, 24, 20):
            limit[0] = effective
            try:
                kernel[(triton.cdiv(new_rows,bh),)](*args,**newconst)
                assert effective == 24
                assert np.array_equal(vc[20],lut[x[0,qn+kn:qn+2*kn]])
            except Exception as e:
                if effective == 24 or "physical bound" not in str(e):
                    raise
                replay_faults += 1
        assert replay_faults == 2
        assert faults==3
    finally:
        tl.inline_asm_elementwise=saved
    return dict(mode="Triton interpreter, identical stock producer followed by lookup/scatter reference",
                cases=cases,bytes_compared=sum(c["checked_bytes"] for c in cases),faults=faults,
                replayed_stable_capacity_cases=3, table_sha256=hashlib.sha256(lut.tobytes()).hexdigest())


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("new", type=Path)
    a.add_argument("--original",type=Path)
    a.add_argument("--table",type=Path)
    args=a.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp)
        new=module(args.new,work)
        if args.original:
            import numpy as np
            result=interpret(new,module(args.original,work),np.fromfile(args.table,dtype=np.uint8))
        else:
            result=compile_cases(new,work)
    result.update(source_sha256=hashlib.sha256(args.new.read_bytes()).hexdigest(),triton=triton.__version__)
    print(json.dumps(result,indent=2))
