"""Exercise graph allocation, binding, replay and layer dispatch from the frozen backend."""
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "v132_verify_scheduler_metadata"))
from verify_dispatch import Hopper, Mode, method, tensors


def check(stage):
    backend = (stage / "python/sglang/srt/layers/attention/flashattention_backend.py").read_text()
    attn = (stage / "python/sglang/kernels/ops/attention/paged_verify_attention.py").read_text()
    meta = (stage / "python/sglang/kernels/ops/attention/metadata.py").read_text()
    calls = []
    native = []
    mode = Mode()
    mode.is_decode_or_idle = lambda: False
    alg = NS(is_dflash=lambda: True)
    def work(q, k, v, pages, lens, cu, **kw):
        calls.append((pages, kw.get("origins"), kw["window_left"]))
        return q
    def fallback(**kw):
        native.append(kw)
        return kw["q"]
    env = dict(torch=torch, dataclass=dataclass, get_spec=lambda:NS(speculative_algorithm="DFLASH"),
               SpeculativeAlgorithm=NS(from_string=lambda name:alg), maybe_create_verify_mask=lambda **kw:None,
               is_cp_active=lambda fb:False, ForwardMode=Mode, paged_verify_attention=work,
               AttentionType=NS(ENCODER_ONLY="enc",DECODER_BIDIRECTIONAL="bi"),flash_attn_with_kvcache=fallback)
    cls = next(n for n in ast.parse(backend).body if getattr(n, "name", None) == "FlashAttentionMetadata")
    exec("from __future__ import annotations\n"+ast.unparse(cls),env)
    names = ("init_cuda_graph_state", "_bind_metadata_buffers", "_set_decode_page_metadata",
             "_apply_cuda_graph_metadata", "forward_extend", "_mxfp8_sf_kwargs")
    for name in names:
        exec(method(backend, name, "FlashAttentionBackend"), env)
    Backend = type("Backend", (), {n:env[n] for n in names})
    def initialize(window=2047, use_pool=True, translated=False):
        op=Backend()
        op.__dict__.update(device="cpu", max_context_len=4096, page_size=1, _get_scheduler_metadata=None,
            local_attn_builder=None, use_sliding_window_kv_pool=use_pool, topk=1, has_swa=True,
            is_encoder_decoder=False, is_prefill_aware_swa=False, _kv_shard_pool=None, skip_prefill=False,
            speculative_step_id=0, speculative_num_draft_tokens=8, is_draft_runner=False,
            kv_index_translator=NS(is_translating=translated,reads_are_translated=translated),
            _hopper_verify_sms=132, use_mla=False, req_to_token=torch.zeros((32,4120),dtype=torch.int32),
            sliding_window_size=window, _draft_window_left=-1, fa_impl_ver=3, num_splits=1,
            fa_skip_kv_cache=False, kv_cache_is_mxfp8=False, _disable_scheduler_metadata_precompute=False)
        op.prepare_paged_mha_query=lambda q,qr,kr,*a,**kw:(q,qr,kr,None,None)
        op.init_cuda_graph_state(32,256)
        return op
    assert "swa_verify_tile_origins" not in initialize(use_pool=False).target_verify_metadata
    assert "swa_verify_tile_origins" not in initialize(translated=True).target_verify_metadata
    assert "swa_verify_tile_origins" not in initialize(window=None).target_verify_metadata
    op=initialize()
    buf=op.target_verify_metadata["swa_verify_tile_origins"]
    assert buf.shape==(32,34) and buf.numel()*buf.element_size()==4352
    class Launch:
        def __getitem__(self,grid):
            def run(*args, **kw):
                assert kw["swa_origins"].data_ptr()==buf.data_ptr() and kw["SWA_ORIGIN_STRIDE"]==34
                assert kw["SWA_WINDOW"]==2047 and kw["VERIFY_WIDTH"]==8
                assert kw["origins"].data_ptr()==op.target_verify_metadata["verify_tile_origins"].data_ptr()
                assert kw["WINDOW_LEFT"]==-1 and grid[0]==args[23]
                observed["metadata_launches"]+=1
            return run
    env.update(triton=NS(cdiv=lambda n,k:(n+k-1)//k),_fused_decode_metadata_kernel=Launch())
    exec(method(meta,"normal_decode_set_metadata"),env)
    op.token_to_kv_pool=NS(full_to_swa_index_mapping=torch.arange(5120,dtype=torch.int32))
    observed=Counter()
    for b in (1,4,16,32):
        _,fb,layer,q,_,extra=tensors(b,8,1,4120,16,2)
        fb.forward_mode,fb.spec_algorithm=mode,alg
        op.get_paged_mha_kv_cache=lambda layer:(q,q)
        op._bind_metadata_buffers(b,8*b,None,mode,fb.spec_info,torch.device("cpu"))
        for step in (0,1):
            op._apply_cuda_graph_metadata(None,b,torch.arange(b),torch.arange(b)+700+32*step,
                                          None,mode,fb.spec_info,None)
            md=op.forward_metadata
            assert md.swa_verify_tile_origins.shape==(b,34) and md.swa_verify_tile_origins.data_ptr()==buf.data_ptr()
            for i in range(62):
                layer.sliding_window_size=-1 if i%4==0 else 2047
                op.forward_extend(q,None,None,layer,fb)
                pages,cert,window=calls[-1]
                assert pages is (md.page_table if window<0 else md.swa_page_table)
                assert cert is (md.verify_tile_origins if window<0 else md.swa_verify_tile_origins)
                observed["full_layer" if window<0 else "sliding_layer"]+=1
            layer.sliding_window_size=255
            op.forward_extend(q,None,None,layer,fb)
            assert calls[-1][1] is None
            layer.sliding_window_size=2047
            md.max_seq_len_q=4
            op.forward_extend(q,None,None,layer,fb)
            assert calls[-1][1] is None
            md.max_seq_len_q=8
            md.swa_spec_metadata=extra()
            op.forward_extend(q,None,None,layer,fb)
            assert calls[-1][1] is None
            md.swa_spec_metadata=None
            fb.spec_info.ragged_verify_layout=object()
            op.forward_extend(q,None,None,layer,fb)
            assert native and native[-1]["page_table"] is md.swa_page_table
            fb.spec_info.ragged_verify_layout=None
            observed["fallback_cases"]+=4
    env["TensorDescriptor"]=lambda tensor,shape,strides,block:(shape,strides,block)
    exec(method(attn,"_direct_tile_args"),env)
    def gpu(t):return t.as_subclass(Hopper)
    for rows in (65,128,4120):
        b=2
        pages=gpu(torch.empty(b,rows,dtype=torch.int32))
        k=gpu(torch.empty((370,1,2,64),dtype=torch.float8_e4m3fn))
        for window in (-1,0,63,2047):
            tn=64 if window>=0 else 128
            span=rows if window<0 else min(rows,window+8+63)
            caps=gpu(torch.full((b,(span+tn-1)//tn),-1,dtype=torch.int32))
            plan=env["_direct_tile_args"](k,k,pages,caps,window,tn,8)
            assert plan["Origins"] is caps and plan["CAPACITY"]==370 and plan["KDesc"][2]==[tn,64]
            assert not env["_direct_tile_args"](k,k,pages,caps.to(torch.int64),window,tn,8)
            observed["descriptor_cases"]+=2
    return dict(source_sha256={k:hashlib.sha256(v.encode()).hexdigest()
                               for k,v in dict(backend=backend,metadata=meta,attention=attn).items()},
                counts=dict(observed),physical_h200=False,
                scope="Production graph and layer methods on CPU tensors with an emulated attention and kernel launch boundary.")


if __name__ == "__main__":
    print(json.dumps(check(Path(sys.argv[1])),indent=2))
