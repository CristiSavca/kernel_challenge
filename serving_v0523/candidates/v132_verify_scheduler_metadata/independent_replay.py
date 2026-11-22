"""Model CUDA schedule capture using the actual V132 host methods and a replayable CPU instruction tape."""
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile

from independent_geometry import build_oracle, evaluate

BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
HERE=Path(__file__).resolve().parent


class Device:
    def __init__(self,index): self.index=index


class Tensor:
    def __init__(self, shape, device, dtype="e4m3"):
        self.shape=shape
        self.ndim=len(shape)
        self.device=device
        self.dtype=dtype
        self.is_cuda=True
        self.data=[]

    def is_contiguous(self): return True
    def numel(self): return math.prod(self.shape)


class Cuda:
    stream=17
    def current_device(self): return 0
    def current_stream(self,device): return SimpleNamespace(cuda_stream=self.stream)


cuda=Cuda()
torch=SimpleNamespace(float8_e4m3fn="e4m3",int32="int32",cuda=cuda,
                      compiler=SimpleNamespace(is_compiling=lambda:False))


def method(stage, filename, cls, name):
    path=stage/"python/sglang/srt"/filename
    tree=ast.parse(path.read_text())
    klass=next(c for c in tree.body if getattr(c,"name","")==cls)
    fun=next(f for f in klass.body if getattr(f,"name","")==name)
    unit=ast.parse("from __future__ import annotations")
    unit.body.append(fun)
    namespace={"torch":torch}
    exec(compile(ast.fix_missing_locations(unit),str(path),"exec"),namespace)
    return namespace[name], hashlib.sha256(ast.unparse(fun).encode()).hexdigest()


def pinned(filename):
    return subprocess.check_output(["git","-C",str(HERE.parents[1]/"sglang"),"show",
                                    BASE+":python/sglang/srt/"+filename],text=True)


def run(stage, vendor):
    get_schedule, method_hash=method(stage,"layers/attention/flashattention_backend.py","FlashAttentionBackend","_verify_schedule")
    forward, model_hash=method(stage,"models/muse_spark.py","MuseSparkTextForCausalLM","forward")
    assert "return super().forward(" in pinned("models/muse_spark_omni.py")
    runner=pinned("model_executor/runner/decode_cuda_graph_runner.py")
    assert "out = forward(" in runner and "output = self.backend.replay(" in runner
    for backend in ("full_cuda_graph_backend","breakable_cuda_graph_backend"):
        assert ".replay()" in pinned(f"model_executor/runner_backend/{backend}.py")
    device=Device(0)
    layer=SimpleNamespace(tp_q_head_num=16,tp_k_head_num=2,head_dim=64,v_head_dim=64,logit_cap=0)
    fb=SimpleNamespace(spec_info=SimpleNamespace(num_tokens_per_req=8,ragged_verify_layout=None),
                       forward_mode=SimpleNamespace(is_target_verify=lambda: True),_fa3_verify_schedules=None)
    host_calls=0
    tape=[]
    handles=[]
    events=[]
    requests=4
    q=Tensor((requests*8,1024),device)
    k=Tensor((128,160,2,64),device)
    v=Tensor(k.shape,device)
    table=Tensor((requests,820),device,"int32")
    seq=Tensor((requests,),device,"int32")
    cu=Tensor((requests+1,),device,"int32")
    class Backend:
        page_size=160
        _verify_schedule_sms=132
        def _get_scheduler_metadata(self, **kw):
            handle={"work":None,"semaphore":-1}
            def prepare():
                lengths=tuple(kw["cache_seqlens"].data)
                offsets=tuple(kw["cu_seqlens_q"].data)
                handle.update(work=(lengths,offsets,kw["window_size"]),semaphore=0)
            tape.append(prepare)
            handles.append(handle)
            return handle
    backend=Backend()
    windows=[(-1,-1)]*16+[(2047,0)]*46
    model=SimpleNamespace(_share_verify_schedule=True,capture_aux_hidden_states=False,lm_head=None,
                          logits_processor=lambda ids,states,head,batch,aux:states)
    def layers(ids,positions,batch,input_embeds=None):
        nonlocal host_calls
        host_calls+=1
        for window in windows:
            schedule=get_schedule(backend,batch._fa3_verify_schedules,q,k,v,table,seq,cu,8,layer,window,batch)
            assert schedule is not None
            def attention(schedule=schedule,window=window):
                assert schedule["semaphore"]==0
                schedule["semaphore"]=1
                assert schedule["work"]==(tuple(seq.data),tuple(cu.data),window)
                events.append(schedule["work"])
            def combine(schedule=schedule):
                assert schedule["semaphore"]==1
                schedule["semaphore"]=0
            tape.extend((attention,combine))
        return ids
    model.model=layers
    outer_cache={"outer context":"keep"}
    fb._fa3_verify_schedules=outer_cache
    forward(model,"ids",None,fb)
    assert fb._fa3_verify_schedules is outer_cache and len(handles)==2 and host_calls==1
    first_tape=tape[:]
    first_handles=handles[:]
    tape.clear();handles.clear()
    forward(model,"ids",None,fb)
    assert host_calls==2 and len(handles)==2 and all(x is not y for x in handles for y in first_handles)
    for replay in range(12):
        seq.data=[100+replay,1 if replay%2 else 131000,33*replay,8000-replay]
        cu.data=[0,8,16,24,32]
        if replay%3==0: cu.data=[0,8,16,16,16]
        before=len(events)
        for op in (first_tape if replay%2 else tape): op()
        assert len(events)==before+62
    assert host_calls==2
    model.model=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError("injected"))
    try: forward(model,"ids",None,fb)
    except RuntimeError as e: assert str(e)=="injected"
    assert fb._fa3_verify_schedules is outer_cache

    # Explicitly hold a separate mutable allocation per stream, length tensor,
    # and backend. Device-index mismatch must decline before the native builder.
    cache={}
    a=get_schedule(backend,cache,q,k,v,table,seq,cu,8,layer,windows[0],fb)
    cuda.stream=31
    b=get_schedule(backend,cache,q,k,v,table,seq,cu,8,layer,windows[0],fb)
    c=get_schedule(backend,cache,q,k,v,table,Tensor(seq.shape,device,"int32"),cu,8,layer,windows[0],fb)
    d=get_schedule(Backend(),cache,q,k,v,table,seq,cu,8,layer,windows[0],fb)
    assert len({id(a),id(b),id(c),id(d)})==4
    q.device=Device(1)
    assert get_schedule(backend,cache,q,k,v,table,seq,cu,8,layer,windows[0],fb) is None
    q.device=device
    wrong_q=Tensor((requests*8,2048),device)
    assert get_schedule(backend,cache,wrong_q,k,v,table,seq,cu,8,layer,windows[0],fb) is None
    # Evaluate the actual Python guard against C++ static splits, including
    # partial pages, tail windows, and thresholds on either side of half occupancy.
    cases=list(itertools.product((1,2,4,8,16,32,33,48,64), (2,4,8,9), (1,128,160),
                                 (1,4,5,13,412,820),(-1,31,2047),(108,132),(0,)))
    with tempfile.TemporaryDirectory(prefix="okc-v132-eligibility-") as t:
        actual=evaluate(build_oracle(vendor,Path(t)),cases)
    selected=0
    for (bs,width,page,cols,left,sms,_),native in zip(cases,actual):
        backend.page_size=page
        backend._verify_schedule_sms=sms
        fb.spec_info.num_tokens_per_req=width
        k=Tensor((1000,page,2,64),device)
        meta=get_schedule(backend,{},Tensor((bs*width,1024),device),k,k,Tensor((bs,cols),device,"int32"),
                          Tensor((bs,),device,"int32"),Tensor((bs+1,),device,"int32"),
                          width,layer,(left,-1 if left<0 else 0),fb)
        if meta is not None:
            selected+=1
            assert native[5]>1 and not native[4],(bs,width,page,cols,left,sms,native)
    assert selected>0
    return dict(model_forward_ast_sha256=model_hash,get_schedule_ast_sha256=method_hash,
                graph_model_host_calls=host_calls,independent_graph_captures=2,simulated_replays=12,
                captured_prepares_per_graph=2,simulated_attention_calls=len(events),mutable_workspaces_checked=4,
                compared_eligibility_against_vendor_cpp=len(cases),accepted_only_native_multisplit=selected,
                rejected_current_device_mismatch=True,rejected_reshaped_row_mismatch=True,restored_on_nested_and_exception=True,
                runtime_emulation="CPU instruction tape; actual Python V132 methods and pinned native C++ host heuristics",
                gpu_replay=False)


if __name__=="__main__":
    print(json.dumps(run(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
