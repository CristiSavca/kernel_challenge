"""Compile the shipped query-only kernel or run it against V229's producer in the interpreter."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import triton
import triton.language as tl
from triton.runtime.jit import TensorWrapper


def source_kernel(path, work):
    source = path.read_text()
    f = next(n for n in ast.parse(source).body if getattr(n, "name", "") == "_qk_rms_norm_fwd_kernel")
    out = work / (path.stem + "_jit.py")
    lines = source.splitlines(True)[f.decorator_list[0].lineno - 1:f.end_lineno]
    out.write_text("import triton\nimport triton.language as tl\n" + "".join(lines))
    spec = importlib.util.spec_from_file_location(out.stem, out)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._qk_rms_norm_fwd_kernel


def aot(kernel, work, baseline=None):
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource

    data = []
    for q_heads, kv_heads, dim, block in ((16,2,64,32), (16,2,64,16), (5,2,80,16), (8,2,128,32), (8,2,256,32)):
        for rope in (False, True):
            for pdl in (False, True):
                c = dict(NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, D=dim, BD=triton.next_power_of_2(dim),
                         BLOCK_M=block, HAS_ROPE=rope, HAS_KV_STORE=False, FP8_KV_STORE=False,
                         USE_PDL=pdl, FP8_MAX=448.0, ENCODINGS=None)
                dyn = [k for k in kernel.arg_names if k not in c]
                sig = dict(zip(dyn, ("*bf16","*u8","*bf16","*i64","*fp32","*bf16","*bf16","*i64",
                                     "fp32","i32","i32","i32","*u8"), strict=True))
                sig.update({k:"constexpr" for k in c})
                binary = triton.compile(ASTSource(kernel,sig,constexprs=c),target=GPUTarget("cuda",90,32),
                                        options={"num_warps":4,"launch_pdl":pdl})
                ptx = binary.asm["ptx"]
                assert ("griddepcontrol.launch_dependents" in ptx) == pdl
                assert "@bad trap;" not in ptx
                path = work / "q.ptx"
                path.write_text(ptx)
                asm = subprocess.run([get_ptxas(90).path,"-arch=sm_90a","-v",str(path),"-o",str(work/"q.cubin")],
                                     capture_output=True,text=True,check=True).stderr
                spill = tuple(map(int,re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads",asm).groups()))
                data.append(dict(heads=q_heads,kv=kv_heads,dim=dim,block=block,rope=rope,pdl=pdl,
                                 registers=int(re.search(r"Used (\d+) registers",asm)[1]),spill_bytes=spill,
                                 cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest()))
    result = dict(mode="Offline release SM90 PTX and cubin, physical_H200=False", cases=data)
    if baseline is not None:
        comparison = []
        for row in [r for r in data if r["heads"] == 16 and r["dim"] == 64 and r["block"] == 32]:
            c = dict(NUM_Q_HEADS=16, NUM_KV_HEADS=2, D=64, BD=64, BLOCK_M=32, HAS_ROPE=row["rope"],
                     HAS_KV_STORE=False, FP8_KV_STORE=False, USE_PDL=row["pdl"], FP8_MAX=448.0, ENCODINGS=None)
            dyn = [k for k in baseline.arg_names if k not in c]
            sig = dict(zip(dyn, ("*bf16","*bf16","*bf16","*i64","*fp32","*bf16","*bf16","*i64",
                                 "fp32","i32","i32","i32"), strict=True))
            sig.update({k:"constexpr" for k in c})
            src = ASTSource(baseline,sig,constexprs=c)
            binary = triton.compile(src,target=GPUTarget("cuda",90,32),options={"num_warps":4,"launch_pdl":row["pdl"]})
            path=work/"parent.ptx"; path.write_text(binary.asm["ptx"])
            asm = subprocess.run([get_ptxas(90).path,"-arch=sm_90a","-v",str(path),"-o",str(work/"parent.cubin")],
                                 capture_output=True,text=True,check=True).stderr
            comparison.append(dict(rope=row["rope"],pdl=row["pdl"],old_registers=int(re.search(r"Used (\d+) registers",asm)[1]),
                new_registers=row["registers"],old_cubin_sha256=hashlib.sha256(binary.asm["cubin"]).hexdigest(),
                new_cubin_sha256=row["cubin_sha256"]))
        result["public_large_kernel"]=comparison
    return result


class HostBuffer:
    device = "cpu"
    def __init__(self, a):
        self.a = a
        self.data = a
        self.shape = a.shape
    def data_ptr(self): return self.a.ctypes.data
    def untyped_storage(self): return self
    def cpu(self): return self
    def size(self): return self.shape
    def stride(self): return tuple(s // self.a.itemsize for s in self.a.strides)
    def storage_offset(self): return 0
    def new_empty(self, size, device): return HostBuffer(np.empty(size, self.a.dtype))
    def set_(self, storage, offset, size, stride):
        assert offset == 0 and storage.shape == size and storage.stride() == stride
        self.a, self.shape = storage.a, size
    def copy_(self, other): np.copyto(self.a, other.a)


def buf(a, dtype):
    return TensorWrapper(HostBuffer(a), dtype)


def compare(new, base, table):
    rng = np.random.default_rng(1109)
    dtype_bf, guard_bf, guard_fp = np.uint16, 0x4512, 0xe4
    configs = ((1,1,1,16),(7,5,2,80),(16,16,2,64),(33,16,2,64),(57,8,2,128),
               (149,8,2,256),(257,16,2,64),(4096,16,2,64))
    records = []
    for rows, heads, kv, d in configs:
        for rope in (False,True):
            for poison in (False,True) if rows < 4096 else (False,):
                qwidth, kwidth = heads*d, kv*d
                pitch = qwidth + 2*kwidth + (qwidth if rope else 13)
                weights = rng.uniform(-7,7,(rows,pitch)).astype(np.float32)
                source = (weights.view(np.uint32) >> 16).astype(np.uint16)
                if poison:
                    types = np.array([0,0x8000,0x0001,0x007f,0xff80,0x7f80,0x7fa1,0xffa1,0x43e0,0xffc0],np.uint16)
                    source[:,::21] = np.resize(types,source[:,::21].shape)
                orig = source.copy()
                positions = rng.integers(0,37,rows,dtype=np.int64)
                cs = rng.uniform(-1.5,1.5,(37,d)).astype(np.float32)
                q1 = np.full((rows+2,qwidth),guard_bf,dtype_bf)
                q2 = np.full((rows+2,qwidth),guard_fp,np.uint8)
                k1 = np.full((rows+2,kwidth),guard_bf,dtype_bf)
                k2 = k1.copy()
                dummy = np.array([41],np.int32)
                bm = 32 if rows*(heads+kv)>=65536 else 16 if rows*(heads+kv)>=8192 else 8 if rows*(heads+kv)>=1024 else 4
                nrows = rows*(heads+kv)
                geometry = dict(NUM_Q_HEADS=heads,NUM_KV_HEADS=kv,D=d,BD=triton.next_power_of_2(d),BLOCK_M=bm,
                                HAS_ROPE=rope,HAS_KV_STORE=False,FP8_KV_STORE=False,USE_PDL=False)
                for func,q,k,qt,keys in ((base,q1,k1,tl.bfloat16,{}),
                                         (new,q2,k2,tl.uint8,dict(QUERY_ENCODINGS=buf(table,tl.uint8)))):
                    func[(triton.cdiv(nrows,bm),)](buf(source,tl.bfloat16),buf(q,qt),buf(k,tl.bfloat16),
                        buf(positions,tl.int64),buf(cs,tl.float32),buf(dummy,tl.int32),buf(dummy,tl.int32),
                        buf(dummy,tl.int32),1e-6,pitch,nrows,0,**geometry,**keys)
                assert np.array_equal(table[q1[:rows]],q2[:rows])
                assert np.array_equal(k1,k2) and np.array_equal(source,orig)
                assert np.all(q2[rows:]==guard_fp) and np.all(k2[rows:]==guard_bf) and dummy.tolist()==[41]
                records.append(dict(rows=rows,heads=heads,kv=kv,dim=d,rope=rope,special=poison,
                                    query_bytes=rows*qwidth,key_bytes=rows*kwidth*2))
    return dict(mode="Triton interpreter, original V229 norm + table vs V231 fused encode",cases=records,
                table_bytes=table.nbytes,table_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
                query_bytes=sum(r["query_bytes"] for r in records), key_bytes=sum(r["key_bytes"] for r in records),
                physical_H200=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("--parent",type=Path)
    parser.add_argument("--aot-parent",type=Path)
    parser.add_argument("--table",type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp)
        kernel=source_kernel(args.source,work)
        if args.parent:
            result=compare(kernel,source_kernel(args.parent,work),np.fromfile(args.table,np.uint8))
            result["parent_source_sha256"]=hashlib.sha256(args.parent.read_bytes()).hexdigest()
        else:
            result=aot(kernel,work,source_kernel(args.aot_parent,work) if args.aot_parent else None)
    result.update(triton=triton.__version__,source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest())
    print(json.dumps(result,indent=2))
