"""Replay the native V174 multirow pipelines against a dense reference with stale, bounded routing plans."""
import ctypes
import hashlib
import importlib.util
import json
import math
import mmap
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
header_dir=STAGE/"python/sglang/kernels/jit/csrc/muse_spark"
s=importlib.util.spec_from_file_location("v173_independent",HERE.parent/"v173_medium_expert_tma/independent_warp.py")
prev=importlib.util.module_from_spec(s);s.loader.exec_module(prev)
ref=prev.reference

def make_guarded(contents):
    """Put the last physical plan slot immediately before a no-access page."""
    page=mmap.PAGESIZE
    used=contents.nbytes
    boundary=math.ceil(used/page)*page
    region=mmap.mmap(-1,boundary+page)
    addr=ctypes.addressof(ctypes.c_char.from_buffer(region))
    protect=ctypes.CDLL(None,use_errno=True).mprotect
    protect.argtypes=(ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int)
    assert protect(addr+boundary,page,0)==0,ctypes.get_errno()
    out=np.ndarray(contents.shape,dtype=contents.dtype,buffer=region,offset=boundary-used)
    out[:]=contents
    return out

class Case(prev.Case):
    def __init__(self,streamed,dense,planner):
        super().__init__(streamed,dense,planner,False)
        self.transfers=[0,0,0]
        self.unchanged=0
        self.tails=0

    def plan(self,ids,experts,rng):
        self.ids=ids.astype(np.int32,copy=False).ravel()
        stride=self.ids.size+1
        # Initialize each unused entry with an actual positive destination. A
        # subsequent native unpadded planner must publish only count and prefix.
        old=(np.arange(experts*stride,dtype=np.int32)+1).reshape(experts,stride)%max(self.ids.size,1)
        old=np.maximum(old,1) if self.ids.size>1 else old
        self.data=old.copy()
        self.planner(self.ids.ctypes.data,self.data.ctypes.data,experts,self.ids.size,174)
        for e,row in enumerate(self.data):
            slots=np.flatnonzero(self.ids==e)
            count=slots.size
            assert row[0]==count
            np.testing.assert_array_equal(row[1:1+count],slots)
            np.testing.assert_array_equal(row[1+count:],old[e,1+count:])
            self.tails+=int((row[1+count:]>0).sum())
            self.cohort_kinds.add(count)
        return make_guarded(np.r_[np.array([experts,experts,self.ids.size],np.int32),self.data.ravel()])

    def simulate(self,*args):
        hidden,inner,top,_=args[-4:]
        act=np.ctypeslib.as_array((ctypes.c_uint16*(self.ids.size*inner)).from_address(args[7])).reshape(-1,inner)
        out=np.ctypeslib.as_array((ctypes.c_uint16*(self.ids.size*hidden)).from_address(args[8])).reshape(-1,hidden)
        live=(self.ids>=0)&(self.ids<len(self.data))
        for phase,eligible in enumerate(self.eligible):
            a0,o0=act.copy(),out.copy()
            if eligible:
                self.sparse.phase(phase,0)
                counters=lambda:[self.sparse.metric(i) for i in (2,3,4)]
                before=counters()
                self.sparse.simulate(*args)
                after=counters()
                observed=[a-b for a,b in zip(after,before)]
                # A CTA reads each bank slab once per output band and row tile.
                rows=32 if phase else 16
                chunks=(inner if phase else hidden)//128
                bands=(hidden if phase else inner)//128
                planes=1 if phase else 2
                tiles=sum(math.ceil(int(count)/rows) for count in self.data[:,0])
                row_moves=int(live.sum())*chunks*bands
                tensor_moves=tiles*planes*chunks*bands*2
                expected=[tensor_moves,row_moves,tiles*planes*chunks*bands*(8192+1024)+row_moves*256]
                assert observed==expected,(phase,observed,expected)
                self.transfers=[a+b for a,b in zip(self.transfers,observed)]
            else:
                self.dense.phase(phase,0)
                self.dense.simulate(*args)
            np.testing.assert_array_equal(act[~live],a0[~live])
            np.testing.assert_array_equal(out[~live],o0[~live])
            np.testing.assert_array_equal(act if phase else out,a0 if phase else o0)
            self.unchanged+=(~live).sum().item()*(inner+hidden)+(act.size if phase else out.size)

def main():
    rng=np.random.default_rng(174)
    def route(counts,pad=0,top=1):
        data=np.repeat(np.arange(len(counts),dtype=np.int32),counts)
        extras=np.resize(np.array([-11,-1,len(counts),90],dtype=np.int32),pad)
        data=np.r_[data,extras]
        return rng.permutation(data).reshape(-1,top)
    mixed=[0,1,7,8,9,15,16,17,23,24,25,31,32,33,47,48,49,63,64,65]
    fixtures=[(128,128,len(mixed),route(mixed,(-sum(mixed))%8+8,8)),
              (384,512,3,route([2,17,33],4,8)),(512,384,4,route([15,16,31,32],2,8)),
              (128,256,3,route([0,0,0],16,8)),
              (4096,4096,1,route([9],7,8))]
    # The absolute slot bound deliberately ends inside the maximum group width.
    fixtures.extend((128,128,1,route([n])) for n in (1,7,8,9,15,16,17,31,32,33))
    with tempfile.TemporaryDirectory(prefix="v174-numeric-") as name:
        tmp=Path(name)
        stream,hashes=build_stream(tmp)
        dense,mdig=build_dense(tmp)
        # Use the exact release specialization named by the native run_medium wrapper.
        tma=(header_dir/"expert_tma_route_pair.cuh").read_text()
        assert "MuseMediumRoutePair::run_with<false>" in tma
        planner=prev.adapted(prev.build_planner,[("tasks[threadIdx.x]=plan_rows<true>(",
                                 "tasks[threadIdx.x]=plan_rows<false>(")],{})(tmp)
        groups=[]
        for order in (0,1,2):
            stream.configure(order)
            case=Case(stream,dense,planner)
            with patch.object(ref,"plan_jobs",case.plan):
                tests=[ref.run_case(case,rng,*f,1) for f in fixtures]
                for flags in ((True,False),(False,True),(False,False)):
                    case.eligible=flags
                    tests.append(dict(streamed=flags,**ref.run_case(case,rng,*fixtures[0],1)))
                case.eligible=(True,True)
                special=[cancel(case,n) for n in (2,4)]
            groups.append(dict(schedule=order,cases=tests,cancellation=special,
                positive_stale_tail_slots=case.tails,per_phase_unchanged_values=case.unchanged,
                list_lengths=sorted(case.cohort_kinds),tensor_copies=case.transfers[0],
                input_row_copies=case.transfers[1],all_copied_bytes=case.transfers[2]))
    print(json.dumps(dict(header_sha256=hashes,medium_sha256=mdig,hardware=False,
                          protected_plan_ends=True,schedules=groups),indent=2))

def cancel(case,split):
    # Repeat the baseline non-associativity discriminator in all four 8-row
    # fragments and on the first fragment of the second 32-row W2 tile.
    fn=prev.adapted(prev.cancellation,[
      ("[0]+[1]*9+[-1]*6","[0]*16+[1]*33+[-1]*7"),
      (".reshape(2,8)",".reshape(7,8)"),("(2,width),np.float32","(7,width),np.float32"),
      ("sparse_slots=1,dense_slots=9","tested_routes=49")],{})
    return fn(case,split)

def build_dense(tmp):
    return prev.build_dense(tmp)

def build_stream(tmp):
    """Retain address math, branching and fragment order; model device intrinsics on a CPU."""
    import re
    low=(header_dir/"expert_route_pair.cuh").read_text()
    src=(header_dir/"expert_tma_route_pair.cuh").read_text()
    helper=prev.medium.base
    common=low[low.index("namespace sglang {"):low.index("using Words =")]
    common,n=re.subn(r'asm\("mul\.rn\.bf16x2 %0, %1, %2;" : "=r"\((\w+)\) : "r"\((\w+)\), "r"\((\w+)\)\);',
                     r"\1 = pair_mul(\2, \3);",common)
    assert n==2
    body=low[low.index("SGL_DEVICE float scaled_bf16"):low.index("struct Slice {")]
    body+=low[low.index("constexpr uint32_t kChannelWarps"):low.index("template <bool Gate>\nstruct alignas(16)")]
    body+=low[low.index("SGL_DEVICE void contract_fragment"):low.index("struct VectorStage")]
    body+=low[low.index("template <bool Gate>\nSGL_DEVICE void finish_channels"):low.index("template <bool Gate, class Stage")]
    body,n=re.subn(r'asm volatile\(\s*"mma.sync.*?\)\);','return Op{&sum,{a0,a1,a2,a3},{lo,hi}};',body,flags=re.S)
    assert n==1
    body=body.replace("void contract_fragment(","Op contract_fragment(")
    shim=prev.replace_once(prev.stream.SHIM,'threadIdx.x%32>=8','threadIdx.x%32>=32')
    body+=shim+src[src.index("template <bool Gate, uint32_t Rows = kRows>"):src.index("SGL_DEVICE uint32_t shared_addr")]
    producer=src[src.index("template <bool Gate, uint32_t Rows>\nSGL_DEVICE void produce("):src.index("template <bool Gate, bool ExpertList")]
    producer=producer.replace("uint64_t* full","MBar* full").replace(
        "const uint32_t lane = threadIdx.x % 32, bar = shared_addr(full);","const uint32_t lane=threadIdx.x%32; MBar* bar=full;")
    producer,n=re.subn(r'asm volatile\("mbarrier.arrive.expect_tx.*? : "memory"\);','arrive(bar,total);',producer,flags=re.S)
    assert n==1
    producer,n=re.subn(r'asm volatile\("cp.async.bulk.shared::cluster.*? : "memory"\);',
                       'issue_input(&tile.inputs[lane][0],src,bytes,bar);',producer,flags=re.S)
    assert n==1
    consumer=helper.body(src,"void consume_multirow(")
    consumer=prev.replace_once(consumer,"wait_phase(full + index, (sequence / 2) & 1);",
        "while(!ready(full+index,(sequence/2)&1)) co_yield Op{};")
    consumer,n=re.subn(r'asm volatile\("mbarrier.arrive.shared::cta.*? : "memory"\);','arrive(empty+index);',consumer)
    assert n==1
    consumer=consumer.replace("contract_fragment(a[","co_yield contract_fragment(a[").replace(
                               "contract_fragment(b[","co_yield contract_fragment(b[")
    cta=helper.body(src,"void multirow_projection(")
    cta=prev.replace_once(cta,"if (rows == 0) return;","if (rows == 0) co_return;")
    cta=prev.replace_once(cta,"__shared__ uint64_t full[2], empty[2];","__shared__ MBar full[2], empty[2];")
    cta,n=re.subn(r'asm volatile\("mbarrier.init.*? : "memory"\);','init(full+index,1);init(empty+index,kChannelWarps);',cta,flags=re.S)
    assert n==1
    cta=prev.replace_once(cta,"if (sequence >= 2) wait_phase(empty + index, ((sequence / 2) - 1) & 1);",
        "while(sequence>=2 && !ready(empty+index,((sequence/2)-1)&1)) co_yield Op{};")
    cta=prev.replace_once(cta,"chunk, slot, live);","chunk, slot, live); co_yield Op{};")
    cta=cta.replace("min(rows_per_tile, rows - first)","std::min(rows_per_tile, rows - first)")
    cta,n=re.subn(r'consume_multirow<(Gate|true|false), (\d+)>\(p, job, live, expert, column, ring, full, empty, sequence\);',
        lambda m:"{constexpr uint32_t Groups="+m[2]+";"+consumer+"}",cta)
    assert n==5
    body+=producer+"\ntemplate<bool Gate> Routine sim(const Projection& p,const BankMaps& maps,uint32_t stride) {"+cta+"}"
    body=body.replace("__syncthreads();","co_yield Op{};").replace("__syncwarp();","")
    assert "asm volatile" not in body and "wait_phase(" not in body
    driver=prev.stream.DRIVER
    a,b=driver.index("void invoke("),driver.index(" const unsigned width")
    driver=driver[:a]+"void invoke(const Projection& p,unsigned E,unsigned stride) {\n"+driver[b:]
    driver="int phase_filter=0;\nextern \"C\" void phase(unsigned p,unsigned){phase_filter=p;}\n"+driver
    driver=prev.replace_once(driver,'gridDim={p.output_width/128,groups}','gridDim={p.output_width/128,E}')
    driver=prev.replace_once(driver,'sim<Gate>(p,maps)','sim<Gate>(p,maps,stride)')
    driver=prev.replace_once(driver,'gscale,plan,h,inner,top},dn{act,out,{down,ds},dscale,plan,inner,h,1}',
                             'gscale,plan+3,h,inner,top},dn{act,out,{down,ds},dscale,plan+3,inner,h,1}')
    driver=prev.replace_once(driver,'invoke<true>(up,groups);invoke<false>(dn,groups);',
                             'if(!phase_filter)invoke<true>(up,plan[1],plan[2]+1);else invoke<false>(dn,plan[1],plan[2]+1);')
    driver=driver.replace('if(policy==2 && (clock_tick%7)','if(rounds>1 && policy==2 && (clock_tick%7)')
    code=ref.v86.PRELUDE+'\n#include <algorithm>\n#include <vector>\n#include <array>\n#include <cstring>\n'
    code+=ref.base.v89.HOST_MUL+common+body+'\n}}\n'+ref.v86.POSTLUDE.split('extern "C" void simulate')[0]+driver
    cpp,lib=tmp/"multirow.cpp",tmp/"multirow.so"
    cpp.write_text(code)
    subprocess.run(["c++","-std=c++20","-O2","-shared","-fPIC","-fno-strict-aliasing",cpp,"-o",lib],check=True)
    dll=ctypes.CDLL(str(lib));dll.simulate.argtypes=[ctypes.c_void_p]*10+[ctypes.c_uint32]*4
    dll.simulate.restype=None;dll.metric.argtypes=[ctypes.c_uint32];dll.metric.restype=ctypes.c_uint64
    return dll,[hashlib.sha256(x.encode()).hexdigest() for x in (low,src)]

if __name__=="__main__":
    main()
