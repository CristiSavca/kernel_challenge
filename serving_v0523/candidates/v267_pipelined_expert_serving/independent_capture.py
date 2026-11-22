"""Execute both attention communication endpoints under the pinned breakable stream tracker."""
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
BASE = runpy.run_path(str(HERE.parent / "v260_pipelined_prompt_qkv/independent_transport.py"))
FILES = {"qkv":"layers/communication/sp_half_qkv.py", "output":"layers/communication/sp_attention_return.py",
         "sp":"layers/layernorm_sp.py", "model":"models/muse_spark.py",
         "graph":"model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py",
         "runner":"model_executor/runner/prefill_cuda_graph_runner.py"}
native=BASE["native"]


class Hazard(Exception):
    pass


class Schedule:
    def __init__(self, src, mode, remove):
        self.edges, self.ops = Counter(), Counter()
        self.access = defaultdict(list)
        self.current = None
        self.stage = "init"
        self.omit = remove
        self.capture = ContextVar("capture", default=None)
        self.forks = ContextVar("forks", default=None)
        # The native hook uses a set of stream objects.
        class Stream:
            def __init__(self,i):
                self.lane = self.stream_id = i
                self.clock=[0,0]
        self.main, self.side = Stream(0), Stream(1)
        self.current = self.main
        if mode == "breakable":
            self.capture.set(self.main)
            self.forks.set(set())
        def merge(dst, other):
            self.edges[(self.stage,dst.lane,other.lane)] += 1
            dst.clock[:] = map(max,dst.clock,other.clock)
        hooked = native(src["graph"],"_hooked_wait_stream",dict(
            _original_wait_stream=merge, _is_stream_capturing=lambda s:s in self.forks.get(),
            _forked_streams_var=self.forks, _current_stream_var=self.capture))
        for stream in (self.main,self.side):
            def wait(other, dst=stream):
                if self.omit != (self.stage, dst.lane):
                    hooked(dst,other)
            stream.wait_stream=wait

    @contextmanager
    def using(self,stream):
        old,self.current=self.current,stream
        try: yield
        finally: self.current=old

    def visit(self,t,writes,name):
        if not t.numel(): return
        current=self.current
        current.clock[current.lane]+=1
        now=tuple(current.clock)
        lower=t.storage_offset()
        upper=lower+1+sum((n-1)*s for n,s in zip(t.shape,t.stride()))
        storage=t.untyped_storage()
        history=self.access[storage]
        for a,b,write,when in history:
            if lower<b and a<upper and (write or writes) and any(u>v for u,v in zip(when,now)):
                raise Hazard((name,self.stage,lower,upper,when,now))
        history.append((lower,upper,writes,now))
        self.ops[(self.stage,current.lane,name)]+=1

    def boundary(self):
        if self.current is not self.main or self.forks.get():
            raise Hazard(("side stream still forked",self.stage))
        if any(a>b for a,b in zip(self.side.clock,self.main.clock)):
            raise Hazard(("owner can retire while side stream is outstanding",self.stage))


def replay(cfg, src, rank, mode, count, opts, fail=None, remove=None):
    size=8
    sched=Schedule(src,mode,remove)
    cuda=NS(current_stream=lambda _:sched.current, stream=sched.using)
    def make(shape):
        return torch.empty(shape,dtype=torch.bfloat16,device="meta")
    state=NS(projects=0,segments=0,entered=0)
    def matmul(x,w,*,out):
        sched.visit(x,False,"gemm input")
        sched.visit(w,False,"weight")
        sched.visit(out,True,"gemm output")
        state.projects+=1
        if fail==(sched.stage,state.projects):
            raise ValueError("injected GEMM")
        return torch.mm(x,w,out=out)
    api=NS(cuda=cuda,empty_like=torch.empty_like,mm=matmul)
    default=NS(is_cutedsl=lambda:False,is_gemv=lambda:False)
    class Comm:
        def change_state(self,enable): return nullcontext()
        def group_start(self): pass
        def group_end(self): pass
        def send(self,t,peer): sched.visit(t,False,"send")
        def recv(self,t,peer): sched.visit(t,True,"recv")
    group=NS(world_size=size, rank_in_group=rank,pynccl_comm=Comm(),sp_qkv_cohort_stream=sched.side,
             is_symmetric_memory_enabled=lambda:False, _resolve_outplace_all_reduce_method=lambda t:None,
             _can_use_flashinfer_allreduce=lambda t:False)
    def reduce_along(x,dim):
        sched.visit(x,False,"reduce input")
        result=make((x.shape[0]//size,x.shape[1]))
        sched.visit(result,True,"reduce output")
        return result
    group.reduce_scatter_along_dim=reduce_along
    rowsum=native(src["sp"],"reduce_scatter_token_rows",dict(torch=torch,os=os,ceil_align=lambda n,d:(n+d-1)//d*d))
    modules={"sglang.srt.layers.layernorm_sp":NS(reduce_scatter_token_rows=rowsum),
             "sglang.srt.layers.quantization.unquant":NS(get_bf16_gemm_backend=lambda:default)}
    qenv={"torch":api, "get_parallel":lambda:NS(tp_group=group)}
    for name in ("_exchange_half","_exchange_peer","_project","half_qkv"):
        qenv[name]=native(src["qkv"],name,qenv)
    project_output=native(src["output"],"striped_attention_return",dict(torch=api))
    forward=native(src["model"],"forward",dict(torch=torch),"MuseSpark1xAttention")
    qw,ow=make((2304,8192)),make((8192,1024))
    def qkv(x):
        sched.stage="qkv"
        state.projects=0
        if opts[0]:
            result=qenv["half_qkv"](x,qw,size,False)
        else:
            result=make((x.shape[0]*size,2304))
            sched.visit(x,False,"ordinary input")
            sched.visit(result,True,"ordinary output")
        sched.boundary()
        return result,None
    def attn(q,k,v,batch,save_kv_cache):
        sched.stage="attention"
        for t in (q,k,v):sched.visit(t,False,"attention input")
        out=make(tuple(q.shape))
        sched.visit(out,True,"attention output")
        return out
    def gate(x,og):
        for t in (x,og):sched.visit(t,False,"gate input")
        out=make(tuple(x.shape))
        sched.visit(out,True,"gate output")
        return out
    def output(x):
        sched.stage="output"
        state.projects=0
        if opts[1]:
            result=project_output(x,ow,group,sched.side)
        else:
            result=make((x.shape[0]//8,8192))
            sched.visit(x,False,"ordinary input")
            sched.visit(result,True,"ordinary output")
        sched.boundary()
        return result,None
    model=NS(qkv_proj=qkv, q_size=1024,kv_size=128,use_o_gate=True,rotary_emb=None,_fused_qk_norm=None,
             attn=attn,_fused_o_norm_gate=gate,o_proj=output)
    copy=torch.Tensor.copy_
    def checked_copy(dst,other,*a,**kw):
        sched.visit(other,False,"copy source")
        sched.visit(dst,True,"copy destination")
        return copy(dst,other,*a,**kw)
    with patch.dict(sys.modules,modules), patch.dict(os.environ,NCCL_NVLS_ENABLE="0"), patch.object(torch.Tensor,"copy_",checked_copy):
        data=make((2048,8192))
        try:
            for _ in range(count):
                state.entered+=1
                data=forward(model,None,data,NS())
                sched.boundary()
                assert tuple(data.shape)==(2048,8192)
                # Graph segments can end between layers. A correctly joined
                # endpoint leaves no work for the graph manager to auto-join.
                state.segments+=1
        except ValueError as exc:
            assert fail and str(exc)=="injected GEMM"
            sched.boundary()
        else:
            assert fail is None
    return dict(rank=rank, mode=mode, qkv=opts[0], output=opts[1], entered=state.entered, completed=state.segments,
                failed=fail, checked_accesses=sum(sched.ops.values()), waits=sum(sched.edges.values()))


def main():
    loader=BASE["read_export"]
    loader.__globals__.update(HERE=HERE,FILES=FILES)
    cfg,src=loader()
    # The model enables the QKV and O linears in one setup call, and the same
    # stream object is assigned by that startup operation to both endpoints.
    setup=src["qkv"]
    assert "def enable_sp_attention_pipeline" in setup and "group.sp_qkv_cohort_stream = stream" in setup
    assert "fc._sp_return_stream = stream" in setup and setup.count("torch.cuda.Stream(")==1
    assert "enable_sp_attention_pipeline(" in src["model"]
    assert "with self.backend.capture_session(self.stream)" in src["runner"] and "for num_tokens in capture_range" in src["runner"]
    cases=[replay(cfg,src,rank,mode,62,opts) for rank in (0,3,4,7) for mode in ("eager","breakable")
           for opts in ((True,True),(True,False),(False,True))]
    failures=[replay(cfg,src,rank,"breakable",2,(True,True),fail=f) for rank in (0,7)
              for f in (("qkv",1),("output",2))]
    mutants=[]
    for stage in ("qkv","output"):
        for dst in (0,1):
            try: replay(cfg,src,0,"breakable",1,(True,True),remove=(stage,dst))
            except Hazard as error: mutants.append({"removed_wait":[stage,dst],"detected":str(error)})
            else: raise AssertionError(("missing dependency accepted",stage,dst))
    return dict(tree=cfg["tree"], patch_sha256=cfg["patch_sha256"], source_sha256={f:hashlib.sha256(s.encode()).hexdigest() for f,s in src.items()},
                cases=cases, failure_joins=failures, missing_wait_mutations=mutants, completed_layer_forwards=sum(r["completed"] for r in cases),
                checked_accesses=sum(r["checked_accesses"] for r in cases), physical_gpu=False,
                scope="Actual frozen Muse attention method and both communication endpoints. Modeled compute/one shared communication stream with native breakable wait hook, vector clocks and real native-dimension tensor storage metadata. Attention, CUDA compute and transport are modeled; GPU stream capture and memory residency require hardware.")


if __name__=="__main__":
    print(json.dumps(main(),indent=2))
