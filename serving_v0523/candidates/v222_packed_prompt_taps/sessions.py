"""Execute the native capture session, capture_one and replay against a CPU graph primitive."""
from collections import namedtuple
from contextlib import contextmanager, nullcontext
import ast
import json
import sys
from types import SimpleNamespace as NS

import torch

from replay import BACKEND, RUNNER, classes, get_source, load


def test(stage):
    rt,body,env=classes(stage,False,0,8)
    backend=rt.backend
    pool=object()
    active=[]
    class Graph:
        def __init__(self,*args):
            self.run=None
        def replay(self):
            self.run()
    @contextmanager
    def recording(cuda_graph,**kwargs):
        active.append(cuda_graph)
        try: yield
        finally: active.pop()
    env.update(BreakableCUDAGraph=Graph,BreakableCUDAGraphCapture=recording,
        get_or_create_global_graph_memory_pool=lambda device:pool,set_graph_pool_id=lambda p:None,
        graph_pool_capture_scope=nullcontext,graph_pool_replay_scope=nullcontext,
        enable_breakable_cuda_graph=nullcontext)
    bcg=ast.parse(get_source(stage,BACKEND))
    for name in ("capture_one","replay"):
        setattr(type(backend),name,load(bcg,name,"BreakableCudaGraphBackend",env))
    for name in ("capture_session","replay_session"):
        setattr(type(backend),name,contextmanager(load(bcg,name,"BreakableCudaGraphBackend",env)))
    backend._pool=None
    backend._outputs={};backend._capture_inputs={}
    backend.deduped_cuda_graph=False
    backend._device_module=NS(synchronize=lambda:None)
    backend._tp_group=NS(barrier=lambda:None)
    backend._precarve=NS(measure=nullcontext,mint=lambda:None)
    backend.begin_cuda_graph_capture=lambda:None
    backend.end_cuda_graph_capture=lambda:None
    rt.cuda_graph_output_capacity_rows=lambda out:None
    backend._cuda_graph_runner.cuda_graph_output_capacity_rows=rt.cuda_graph_output_capacity_rows
    flags=NS(dp=NS(capturing_prefill_graph=False,prefill_graph_has_dp_gather=False))
    env.update(get_flags=lambda:flags,get_exec=lambda:NS(graph=NS(enable_cudagraph_gc=False)),
               freeze_gc=lambda _:nullcontext(),get_or_create_global_graph_capture_stream=lambda:None,
               graph_capture=lambda **kw:nullcontext(NS(stream=object())))
    rt.warmup=lambda:None
    rt._bind_muse_prompt_taps=rt.bind
    capture=load(ast.parse(get_source(stage,RUNNER)),"_capture","PrefillCudaGraphRunner",env)
    Key=namedtuple("Key","size")
    live={}
    def capture_keys():
        for n in (264,129,17,7):
            local=(n+7)//8
            batch=NS(forward_mode=rt.capture_forward_mode,dflash_prefill_raw_buffer=(
                None if rt._muse_prompt_taps is None else rt._muse_prompt_taps[:local]))
            item=torch.full((n,16),2.)
            live[n]=(batch,item)
            key=Key(n)
            def forward(batch=batch,item=item,local=local):
                def model_forward():
                    return body(None,None,batch,input_embeds=item)
                if active:
                    tree=backend._shared_output_buffer
                    active[-1].run=lambda:backend._copy_output_to_buffer(model_forward(),tree,local)
                return model_forward()
            backend.capture_one(key,forward,capture_inputs=batch)
    rt._capture_one_stream=capture_keys
    runs=0;buffers=[];observed=[]
    for epoch in range(3):
        if epoch==2: h=rt.layer_model.register_forward_hook(lambda m,a,o:o)
        with torch.inference_mode():
            capture(rt)
            selected=rt._muse_prompt_taps is not None
            assert selected==(epoch<2)
            if selected: buffers.append(rt._muse_prompt_taps.untyped_storage().data_ptr())
            observed.append(backend._shared_output_buffer[0].shape[0])
            for n in (7,129,17,264,7):
                batch,item=live[n]
                runs+=1;item.fill_(runs+0.5)
                out=backend.replay(Key(n),batch)
                tmp=NS(forward_mode=rt.capture_forward_mode,dflash_prefill_raw_buffer=None)
                native=body(None,None,tmp,input_embeds=item)
                assert native[0].equal(out[0]) and torch.cat(native[1],-1).equal(torch.cat(out[1],-1))
                if selected:
                    got=env["_pack_muse_prompt_taps"](out[1],rt._muse_prompt_taps)
                    assert got.is_set_to(rt._muse_prompt_taps[:(n+7)//8])
        if epoch==2:h.remove()
    assert len(set(buffers))==1 and observed==[33,33,264]
    assert not flags.dp.capturing_prefill_graph
    return dict(real_runner_capture_epochs=3,actual_backend_replays=runs,shared_leading_rows=observed,
                raw_storages_across_clean_epochs=len(set(buffers)),previous_keys_present_at_recapture=True,
                post_registration_hook_disables_next_epoch=True,physical_cuda=False)


if __name__=="__main__":
    print(json.dumps(test(sys.argv[1]),indent=2))
