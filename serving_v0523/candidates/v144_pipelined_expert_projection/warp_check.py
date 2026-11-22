"""Run the exact production prefetch, drain, MMA operand and write code against dense BF16 oracle."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

data=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(data/"v129_routed_projection_pair"))
import warp_check as parent
import order_check as cancellation
base=parent.base


def compile_device(stage,temp):
    p=stage/"python/sglang/kernels/jit/csrc/muse_spark/expert_route_pair.cuh"
    src=p.read_text();pre=src[src.index("namespace sglang {"):src.index("SGL_DEVICE void contract(")]
    pre,count=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',r"\1 = pair_mul(\2, \3);",pre)
    assert count==2
    c=base.body(src,"void contract(")
    c,count=re.subn(r"asm volatile\(.*?\)\);","co_yield Op{&d,{a0,a1,a2,a3},{cur.lo,cur.hi}};",c,flags=re.S);assert count==1
    body=base.body(src,"void project(")
    def inject(m):
        out,frag=m.groups()
        return "{auto& out="+out+"; const Fragment& frag=cur."+frag+";"+c+"}"
    body,count=re.subn(r"contract\((\w+), cur\.(\w+), cur.lo, cur.hi\);",inject,body);assert count==4
    types=src[src.index("SGL_DEVICE float scaled_bf16"):src.index("template <bool Gate>\nSGL_DEVICE void project(")]
    fn="template<bool Gate> Routine sim(const Projection& p,const int32_t* job,unsigned column){"+body+"}"
    code=parent.v86.PRELUDE+base.v89.HOST_MUL+pre+types+fn+"\n}}\n"+parent.v86.POSTLUDE.split('extern "C" void simulate')[0]+parent.DRIVER
    cpp=temp/"warp.cc";lib=temp/"warp.so";cpp.write_text(code)
    subprocess.run(["c++","-std=c++20","-O2","-fno-strict-aliasing","-fPIC","-shared",str(cpp),"-o",str(lib)],check=True)
    dll=ctypes.CDLL(str(lib));dll.simulate.argtypes=[ctypes.c_void_p]*10+[ctypes.c_uint32]*4;dll.simulate.restype=None
    return dll,hashlib.sha256(src.encode()).hexdigest()


if __name__=="__main__":
    rng=np.random.default_rng(144)
    fixtures=[(128,128,2,np.full((7,3),-1),1),(256,128,4,rng.integers(-1,5,(11,3)),1),
              (128,512,3,np.zeros((19,2),dtype=np.int32),2),(512,256,3,rng.integers(-2,4,(8,8)),2),
              (384,640,3,np.array([0]*8+[1]*16+[2]*17+[-1]*4).reshape(9,5),3),
              (4096,4096,1,np.array([[-1,0],[-1,-1]]),1)]
    with tempfile.TemporaryDirectory() as d:
        dll,digest=compile_device(Path(sys.argv[1]),Path(d))
        cases=[parent.run_case(dll,rng,*t) for t in fixtures]
        sensitive=[cancellation.check(dll,n) for n in (2,4)]
    print(json.dumps(dict(header_sha256=digest,production_warp_execution=True,cases=cases,
                          ascending_k_cancellation=sensitive,physical_hardware=False),indent=2))
