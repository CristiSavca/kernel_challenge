"""Execute V173's two expert-list consumers with the published planner layout and poisoned buffers."""
import ctypes
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
STAGE=Path(sys.argv[1]).resolve()
HEADERS=STAGE/"python/sglang/kernels/jit/csrc/muse_spark"

def load(name, file):
    spec=importlib.util.spec_from_file_location(name,HERE.parent/file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

stream=load("stream165","v165_streamed_expert_stages/warp_check.py")
medium=load("medium141","v141_adaptive_expert_projection/warp_check.py")
reference=stream.ref.parent

def replace_once(text, old, new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def adapted(fn, substitutions, env):
    """Adapt the older simulator's launch ABI; it still loads the actual candidate functions."""
    code=inspect.getsource(fn)
    for old,new in substitutions:
        code=replace_once(code,old,new)
    ns=fn.__globals__|env
    exec(compile(code, str(HERE/"independent_warp.py"),"exec"),ns)
    return ns[fn.__name__]

def build_sparse(tmp):
    driver='\n'.join(['int phase_filter=0;', 'extern "C" void phase(unsigned p,unsigned){phase_filter=p;}', stream.DRIVER])
    begin=driver.index("void invoke(")
    end=driver.index(" const unsigned width",begin)
    driver=driver[:begin]+"""void invoke(const Projection& p,unsigned E,unsigned stride) {
"""+driver[end:]
    driver=replace_once(driver,'gridDim={p.output_width/128,groups}','gridDim={p.output_width/128,E}')
    driver=replace_once(driver,'sim<Gate>(p,maps)','sim<Gate,true>(p,maps,stride)')
    driver=replace_once(driver,'gscale,plan,h,inner,top},dn{act,out,{down,ds},dscale,plan,inner,h,1}',
                        'gscale,plan+3,h,inner,top},dn{act,out,{down,ds},dscale,plan+3,inner,h,1}')
    driver=replace_once(driver,'invoke<true>(up,groups);invoke<false>(dn,groups);',
                        'if(!phase_filter)invoke<true>(up,plan[1],plan[2]+1);else invoke<false>(dn,plan[1],plan[2]+1);')
    # Denied sparse lists return coherently in every warp before initializing barriers.
    early="    body=body.replace('if (blockIdx.y >= jobs) return;',"
    substitutions=[(early,"    body=body.replace('if (rows == 0 || rows > kRows) return;', 'if (rows == 0 || rows > kRows) co_return;')\n"+early)]
    return adapted(stream.build,substitutions,{"DRIVER":driver})(STAGE,tmp)

def build_dense(tmp):
    # The new header defines one extra boolean template. Expose that same
    # parameter to the inherited launch driver and select the actual dense grid.
    driver=medium.DRIVER.replace('run_grid<false>(p,stride)','run_grid<false,true>(p,stride)').replace(
                                 'run_grid<true>(p,stride)','run_grid<true,true>(p,stride)')
    driver='\n'.join(['int phase_filter=0; bool sparse_companion=true;',
                      'extern "C" void phase(unsigned p,unsigned companion){phase_filter=p;sparse_companion=companion;}',
                      driver])
    driver=replace_once(driver,"const Projection& p=phase?dn:up;", "if(phase!=phase_filter) continue; const Projection& p=phase?dn:up;")
    driver=replace_once(driver,"if(phase) run_grid<false,true>(p,stride);else run_grid<true,true>(p,stride);",
                        "if(sparse_companion){if(phase) run_grid<false,true>(p,stride); else run_grid<true,true>(p,stride);}"
                        "else{if(phase) run_grid<false,false>(p,stride); else run_grid<true,false>(p,stride);}")
    substitutions=[('template<bool Gate> void run_grid','template<bool Gate, bool SkipSparse> void run_grid'),
                   ('orig.index("template <bool Gate>")','orig.index("struct Slice {")')]
    return adapted(medium.compile_kernel,substitutions,{"DRIVER":driver})(STAGE,tmp)

def build_planner(tmp):
    medium_source=(HEADERS/"expert_medium_route_pair.cuh").read_text()
    body=medium_source[medium_source.index("template <bool PadSparse"):medium_source.index("struct Inputs")]
    body=replace_once(body,"__global__ void plan_rows","Task plan_rows")
    cpp=(HERE.parent/"v141_adaptive_expert_projection/independent_plan.cpp").read_text()
    cpp=cpp[:cpp.index("constexpr unsigned kWarps")].replace("PLANNER_BODY","constexpr uint32_t kRows=8;\n"+body)
    cpp=replace_once(cpp,"tasks[threadIdx.x]=plan_rows(","tasks[threadIdx.x]=plan_rows<true>(")
    cpp+="""
extern "C" void plan(const int32_t* values, int32_t* output, int experts,int slots,unsigned seed) {
    std::vector<int32_t> ids(values,values+slots);std::mt19937 rng(seed);
    for(int e=0;e<experts;++e)run_plan(ids,e,output,rng);
}
"""
    own=tmp/"planner.cc";lib=tmp/"planner.so";own.write_text(cpp)
    subprocess.run(["c++","-std=c++20","-O2","-shared","-fPIC",str(own),"-o",str(lib)],check=True)
    fn=ctypes.CDLL(str(lib)).plan;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_uint];fn.restype=None
    return fn

class Case:
    def __init__(self, sparse, dense, planner, order):
        self.sparse,self.dense,self.planner,self.order=sparse,dense,planner,order
        self.counts={k:0 for k in ("sparse_slots","dense_slots","absent_slots","unchanged_middle_values","dirty_tail_entries")}
        self.cohort_kinds=set()
        self.eligible=(True,True)

    def plan(self,ids,experts,rng):
        self.ids=ids.astype(np.int32,copy=False).ravel()
        stride=self.ids.size+1
        data=np.full((experts,stride),0x3fffffff,dtype=np.int32)
        # A dense prior run may have left valid positive original slot IDs in a
        # future sparse tail. Reuse the buffer once before publishing live IDs.
        previous=(np.arange(len(self.ids))%experts).astype(np.int32)
        self.planner(previous.ctypes.data,data.ctypes.data,experts,len(self.ids),31)
        self.planner(self.ids.ctypes.data,data.ctypes.data,experts,len(self.ids),173)
        for row,expert in zip(data,range(experts)):
            loc=np.flatnonzero(self.ids==expert)
            count=len(loc)
            assert row[0]==count and np.array_equal(row[1:1+count],loc)
            if count<8:
                assert (row[1+count:1+min(8,len(self.ids))]==-1).all()
            self.counts["dirty_tail_entries"]+=int((row[1+max(count,8):]>0).sum())
            self.cohort_kinds.add(count)
        self.data=data
        return np.r_[np.array([experts,experts,len(self.ids)],np.int32),data.ravel()]

    def simulate(self,*args):
        hidden,inner,top,_=args[-4:]
        act=np.ctypeslib.as_array((ctypes.c_uint16*(self.ids.size*inner)).from_address(args[7])).reshape(-1,inner)
        out=np.ctypeslib.as_array((ctypes.c_uint16*(self.ids.size*hidden)).from_address(args[8])).reshape(-1,hidden)
        live=(self.ids>=0)&(self.ids<len(self.data))
        in_range=np.clip(self.ids,0,len(self.data)-1)
        sparse=live & (self.data[in_range,0]<=8)
        dense=live & ~sparse
        self.counts["sparse_slots"]+=int(sparse.sum());self.counts["dense_slots"]+=int(dense.sum())
        self.counts["absent_slots"]+=int((~live).sum())
        for phase, eligible in enumerate(self.eligible):
            use_sparse=eligible and self.ids.size>=8
            launches=[(self.sparse,sparse),(self.dense,dense)] if use_sparse else [(self.dense,live)]
            if self.order: launches.reverse()
            for dll,mask in launches:
                dll.phase(phase,int(use_sparse))
                before_a,before_o=act.copy(),out.copy()
                dll.simulate(*args)
                np.testing.assert_array_equal(act[~mask],before_a[~mask])
                np.testing.assert_array_equal(out[~mask],before_o[~mask])
                if phase==0:
                    np.testing.assert_array_equal(out,before_o)
                    assert np.all((act[mask]!=0x7fc1).any(axis=1))
                else:
                    np.testing.assert_array_equal(act,before_a)
                    assert np.all((out[mask]!=0x7fc2).any(axis=1))
                self.counts["unchanged_middle_values"]+=int((~mask).sum())*(inner+hidden)


def cancellation(case,split):
    # Strict ascending K preserves the sub-ULP contribution in W2 and discards
    # it in W13. Reassociating K stripes changes both observed BF16 results.
    v=reference.v86
    width=128
    ids=np.array([0]+[1]*9+[-1]*6,dtype=np.int32).reshape(2,8)
    gates=np.zeros((2,2*width,width),np.uint8); downs=np.zeros((2,width,width),np.uint8)
    eps=2.**-27
    x=np.zeros((2,width),np.float32)
    x[:,[0,16,64]]=[1,eps,1]
    x=v.bf16(x)
    positions=[1,width//split+1,width//split+17]
    gates[:,0,[0,16,64]]=[2,2,10]
    gates[:,width,0]=2
    for n,src in zip(positions,[0,0,16]):
        gates[:,n,0]=2;gates[:,width+n,src]=2
    downs[:,0,positions]=[2,10,2]
    gates,downs=map(v.pack_codes,[gates,downs])
    gs=np.full((2,width//16,2*width),56,np.uint8)
    ds=np.full((2,width//16,width),56,np.uint8)
    glob=v.bf16([1,1])
    act=np.full((ids.size,width),0x7fc1,np.uint16)
    out=np.full_like(act,0x7fc2)
    plan=case.plan(ids,2,None)
    case.simulate(*(a.ctypes.data for a in [x,gates,gs,glob,downs,ds,glob,act,out,plan]),width,width,8,1)
    for slot in np.flatnonzero(ids.ravel()>=0):
        sample=v.floats(act[slot,positions])
        assert act[slot,0]==0 and sample[0]==sample[1]>0 and sample[2]==sample[0]*eps
        assert v.floats(out[slot,0])==sample[2]
    return dict(split=split,sparse_slots=1,dense_slots=9,gate_bf16_bits=int(act[0,0]),
                down_bf16_bits=int(out[0,0]),preserved_nonzero=float(v.floats(out[0,0])))

if __name__=="__main__":
    rng=np.random.default_rng(173)
    edge_counts=[0,1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,45]
    def ids(counts):
        src=np.repeat(np.arange(len(counts),dtype=np.int32),counts)
        src=np.r_[src,np.full(8-src.size%8,-1,dtype=np.int32),np.array([-17,-1,88,99,99,-1,79,-1],dtype=np.int32)]
        return rng.permutation(src).reshape(-1,8)
    fixtures=[(128,128,len(edge_counts),ids(edge_counts)),(384,256,6,ids([8,0,1,33,2,9])),
              (256,384,4,ids([1,8,9,0])),(128,128,3,ids([0,0,0])),
              (256,256,7,ids([9,10,16,17,31,32,33])),(4096,4096,1,ids([1])),
              (128,128,2,np.array([[0,1,0,1,-1,-1,0]],dtype=np.int32)),
              (128,128,2,np.array([[0,1,0,1,-1,-1,0,0]],dtype=np.int32))]
    results=[];sensitivity=[]
    with tempfile.TemporaryDirectory(prefix="v173-numeric-") as td:
        path=Path(td)
        sparse,sha=build_sparse(path);dense,med_sha=build_dense(path);planner=build_planner(path)
        for schedule in (0,1,2):
            sparse.configure(schedule)
            case=Case(sparse,dense,planner,order=schedule%2)
            with patch.object(reference,"plan_jobs",case.plan):
                runs=[]
                for f in fixtures:
                    runs.append(reference.run_case(case,rng,*f,1))
                for eligible in ((True,False),(False,True),(False,False)):
                    case.eligible=eligible
                    runs.append(dict(eligibility=eligible,**reference.run_case(case,rng,*fixtures[0],1)))
                case.eligible=(True,True)
                sensitivity.extend(cancellation(case,n) for n in (2,4))
            results.append(dict(schedule=schedule,counts=case.counts,expert_list_lengths=sorted(case.cohort_kinds),cases=runs))
        copy_stats=[sparse.metric(i) for i in range(6)]
    print(json.dumps(dict(headers_sha256=[*sha,med_sha],schedules=results,ascending_k=sensitivity,copy_statistics=copy_stats,
                         hardware=False),indent=2))
