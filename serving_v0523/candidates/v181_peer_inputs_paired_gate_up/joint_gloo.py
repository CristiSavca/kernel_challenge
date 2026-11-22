"""Follow one live peer-read tensor through paired W13 and V167 return on actual Gloo groups."""
import inspect
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

here, stage = Path(__file__).resolve().parent, Path(sys.argv[1])


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


peer=load("peer_check",here.parent/"v179_peer_prompt_inputs/check_peer.py")
jit=ModuleType("sglang.kernels.jit.utils");jit.load_jit=None
with patch.dict(sys.modules,{"sglang.kernels.jit.utils":jit}):
    fused=load("gate_up",stage/"python/sglang/srt/layers/muse_spark/prefill_gate_up.py")
original_bank=peer.integration.bank
original_fixture=peer.prior.fixture
original_load=peer.load_module
frame={}


def worker(rank,size,root,slabs):
    torch.set_num_threads(1)
    joint=Counter()

    def fixture(stats):
        f=original_fixture(stats)
        native_return=f.env["_return_partials"]
        def returning(partials,*args):
            if frame.get("fused") and frame.get("addr")==partials.untyped_storage().data_ptr():
                joint["peer_w13_return_calls"]+=1
                joint["peer_w13_return_rows"]+=len(partials)
                joint["peer_w13_return_values"]+=partials.numel()
            frame.clear()
            return native_return(partials,*args)
        def gate(e,hidden,rows,ids,runs):
            result=fused.run_gate_up(e,hidden,rows,ids,runs)
            if result is not None:
                joint["paired_windows"]+=1
                if frame.get("addr")==hidden.untyped_storage().data_ptr():
                    frame["fused"]=True
                    joint["paired_peer_routes"]+=len(rows)
            return result
        f.env.update(run_gate_up=gate,_return_partials=returning)
        return f

    def load_peer(name,path):
        mod=original_load(name,path)
        if name=="prod_peer":
            old=mod.PeerPromptInputs.read
            def read(self,*args):
                actual=old(self,*args)
                frame.clear()
                frame["addr"]=actual.untyped_storage().data_ptr()
                return actual
            mod.PeerPromptInputs.read=read
        return mod

    def bank(f,r,g,h):
        e=original_bank(f,r,g,h)
        e.hidden_size=h
        e.w13_weight=torch.empty((e.num_local_experts,h//16,4*h),dtype=torch.int32)
        e.w13_weight_scale=torch.empty((e.num_local_experts,h//16,2*h),dtype=torch.uint8).view(torch.float8_e4m3fn)
        class Launch:
            def run(self,x,a,ids,offsets,scales,maps,handle,shared,width,sms):
                assert x.shape[1]==a.shape[1]==128 and x.data_ptr()!=a.data_ptr()
                assert offsets[0]==0 and offsets[-1]==len(ids)
                gate=x.new_empty((len(x),2*h))
                f.env["HummingMethod"].forward_layer(e,inputs=x,outputs=gate,expert_layout=offsets,
                    compute_config="grouped",tuning_config=width,sublayer_name="w13")
                e.humming_prefill_silu.run(gate,a,ids,scales)
                return True
        config={w:(None,1,128000,w or 176) for w in (None,136,152,176)}
        bindings=tuple((t.data_ptr(),t.shape) for t in (e.w13_weight,e.w13_weight_scale))
        e.humming_prefill_gate_up=Launch(),None,bindings,config,132
        return e

    # Keep every production branch intact; use the same fixture at a supported 128-wide layout.
    test_source=inspect.getsource(peer.integration.exercise).replace("hidden, topk = 12, 8","hidden, topk = 128, 8")
    original=peer.integration.exercise
    exec(test_source,peer.integration.__dict__)
    full=peer.integration.exercise

    def exercise(f,rank,size,n,stats):
        if n==1025:
            full(f,rank,size,n,stats)
    with patch.object(peer.prior,"fixture",fixture),patch.object(peer,"load_module",load_peer), \
         patch.object(peer.integration,"exercise",exercise),patch.object(peer.integration,"bank",bank):
        peer.worker(rank,size,root,slabs)
    peer.integration.exercise=original
    assert joint["peer_w13_return_calls"] and joint["peer_w13_return_values"] and joint["paired_peer_routes"]
    (Path(root)/f"joint_{rank}.json").write_text(json.dumps(joint))


if __name__=="__main__":
    torch.set_num_threads(1)
    summary={}
    for size in (4,8):
        with tempfile.TemporaryDirectory() as scratch:
            slabs=torch.empty((size,512,128),dtype=torch.bfloat16).share_memory_()
            mp.start_processes(worker,(size,scratch,slabs),nprocs=size,start_method="fork")
            def collect(pattern):
                return sum((Counter(json.loads(p.read_bytes())) for p in Path(scratch).glob(pattern)),Counter())
            summary[size]=dict(joint=collect("joint_*.json"),native=collect("rank*.json"))
    print(json.dumps(dict(gloo=summary,hidden=128,intermediate=128,prompt_tokens=1025,
         hardware=False,simulated="NVLink symmetric maps on shared CPU slabs; GPU GEMMs and finalizer via CPU arithmetic",
         torch=torch.__version__),indent=2))
