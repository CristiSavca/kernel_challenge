"""Run the actual worklist producer against the native C++ consumer's scheduling body."""

import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


SRT="python/sglang/srt/layers/muse_spark/muse_spark_humming_moe.py"
HDR="python/sglang/kernels/jit/csrc/muse_spark/expert_skinny_gate_up.cuh"


def function_body(text,name):
    pos=text.index("{",text.index("void "+name+"(")); depth=1
    for end in range(pos+1,len(text)):
        depth+=(text[end]=="{")-(text[end]=="}")
        if depth==0:
            return text[pos+1:end]
    raise ValueError(name)


def compile_scheduler(header,tmp):
    body=function_body(header,"run_planned_kernel")
    # Replace only launch coordinates and the tile computation with a
    # bound-checked counter. The loop and all its pointer arithmetic are exact.
    code="""
#include <cstdint>
struct Params { const int32_t* offsets; uint32_t experts; };
struct Dim { uint32_t x, y; };
constexpr uint32_t kWidth = 64;
extern "C" int visit(const int32_t* layout, uint32_t experts, uint32_t groups,
                     uint32_t bands, int32_t* visits) {
  const Params p{layout, experts};
  const Dim gridDim{bands, groups};
  if (layout[experts+1]<0 || uint32_t(layout[experts+1])>experts) return 1;
  bool invalid=false;
  auto project_tile = [&](const Params&, uint32_t expert, uint32_t column) {
    if (expert>=experts || column/kWidth>=bands) { invalid=true; return; }
    ++visits[expert*bands+column/kWidth];
  };
  for (uint32_t x=0;x<bands;++x) for (uint32_t y=0;y<groups;++y) {
    const Dim blockIdx{x, y};
""" + body + """
  }
  return int(invalid);
}
"""
    src=tmp/"schedule.cc"; out=tmp/"schedule.so"; src.write_text(code)
    subprocess.run(["c++","-std=c++17","-O2","-shared","-fPIC",str(src),"-o",str(out)],check=True)
    visit=ctypes.CDLL(str(out)).visit
    i32=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
    visit.argtypes=[i32,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,i32]
    visit.restype=ctypes.c_int
    return visit


def run(stage):
    root=Path(__file__).resolve().parent.parent
    v81=root/"v81_token_owned_expert_inputs/cpu_check.py"
    env={"__file__":str(v81)}
    exec(compile(v81.read_text().split("adapter = stage")[0],str(v81),"exec"),env)
    tl,ptr,tensor=env["Language"],env["Pointer"],env["tensor"]
    source=stage/SRT; funcs={"tl":tl,"math":__import__("math")}
    env["scope"]["extract"](source,("_stage_token_routes","_active_expert_groups"),funcs)
    header=(stage/HDR).read_text()
    rng=np.random.default_rng(174142)
    pow2=lambda n:1<<(n-1).bit_length()
    tests=dict(features=0,kernel_calls=0,rows_and_slots=0,consumer_grids=0,
               replay_layouts=0,nonzero_programs_without_layout_writes=0)
    with tempfile.TemporaryDirectory(prefix="v116-audit-") as tmp:
        visit=compile_scheduler(header,Path(tmp))
        for e,k,tokens,mapped in itertools.product((1,3,16,32,47),(1,3,8),(1,9,32),(False,True)):
            global_e=e*8 if mapped else e
            mapping=np.full(global_e,-1,np.int32)
            owned=rng.choice(global_e,size=e,replace=False) if mapped else np.arange(e)
            mapping[owned]=np.arange(e,dtype=np.int32)
            slots=tokens*k; h=5; width=pow2(h)
            layout=np.full(2*(e+1),-777,np.int32)
            x=env["bf16"](rng.normal(size=(tokens,h)).astype(np.float32))
            w=env["bf16"](rng.normal(size=h).astype(np.float32))
            for tag in ("all","none","duplicate","random","last"):
                if tag=="all": ids=np.resize(owned,slots)
                elif tag=="none": ids=np.full(slots,1<<40,np.int64)
                elif tag=="duplicate": ids=np.repeat(rng.choice(owned,size=tokens),k)
                elif tag=="last": ids=np.full(slots,int(owned[-1]),np.int64)
                else: ids=rng.integers(-1,global_e+2,slots,dtype=np.int64)
                valid=(ids>=0)&(ids<global_e); local=np.full(slots,-1,np.int32)
                local[valid]=mapping[ids[valid]]
                counts=np.bincount(local[local>=0],minlength=e)
                live=np.flatnonzero(counts); remote=slots-counts.sum()
                expected=np.r_[remote,remote+np.cumsum(counts)]
                def stage_one(buff,on):
                    fields=[np.full(slots,-1991,np.int32) for _ in range(3)]
                    output=np.full((slots,h),-735.,np.float32)
                    args=[ptr(tensor(a)) for a in (ids,mapping,x,w,fields[0],buff,fields[1],fields[2])]
                    args.append(ptr(tensor(output),dtype="bf16"))
                    for token in rng.permutation(tokens):
                        tl.pid=int(token);before=buff.copy()
                        funcs["_stage_token_routes"](*args,np.float32(1e-5),slots,global_e,e,mapped,k,pow2(k),h,
                                                     pow2(slots),pow2(e+1),width,on)
                        if token:
                            np.testing.assert_array_equal(buff,before)
                            tests["nonzero_programs_without_layout_writes"]+=1
                        tests["kernel_calls"]+=1
                    return fields+[output]
                oldbuf=np.full(e+1,-287,np.int32); prev_tail=layout[e+2:].copy()
                oldout=stage_one(oldbuf,False)
                out=stage_one(layout,True)
                for a,b in zip(oldout,out):
                    np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(oldbuf,expected)
                np.testing.assert_array_equal(layout[:e+1],expected)
                assert layout[e+1]==len(live)
                np.testing.assert_array_equal(layout[e+2:e+2+len(live)],live)
                np.testing.assert_array_equal(layout[e+2+len(live):],prev_tail[len(live):])
                host=funcs["_active_expert_groups"](tokens,k,global_e,e)
                for groups in set((1,e,min(24,e),max(1,host))):
                    bands=3
                    visits=np.zeros((e,bands),np.int32)
                    assert visit(layout,e,groups,bands,visits)==0
                    np.testing.assert_array_equal(visits,np.tile((counts>0)[:,None],(1,bands)))
                    tests["consumer_grids"]+=1
                tests["features"]+=1;tests["rows_and_slots"]+=slots
                tests["replay_layouts"]+=tag!="all"
    return {"host":"Real production Triton staging function executed on checked CPU array pointers with reordered program IDs. Exact C++ consumer scheduling body compiled by host compiler, with CUDA block indices supplied and project_tile replaced by coverage counters.",
            "counts":tests,"header_sha256":hashlib.sha256(header.encode()).hexdigest(),
            "stager_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"h200_execution":False}


if __name__=="__main__":
    print(json.dumps(run(Path(sys.argv[1])),indent=2))
