"""Replay target verification's native metadata writers, per-layer dispatch and split-plan launch."""
import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import inspect
import itertools
import json
import math
from pathlib import Path
import sys
import subprocess
import os
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import PropertyMock,patch

import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'v132_verify_scheduler_metadata'))
from verify_dispatch import Mode,method,tensors
BACK='python/sglang/srt/layers/attention/flashattention_backend.py'
PAGED='python/sglang/kernels/ops/attention/paged_verify_attention.py'
META='python/sglang/kernels/ops/attention/metadata.py'
DIV=lambda n,b:-(-n//b)
TR=NS(cdiv=DIV,next_power_of_2=lambda n:1<<(n-1).bit_length())

def body(source,name,ns,cls=None):
    exec(method(source,name,cls),ns)
    return ns[name]

def oracle(lengths,cu,left,p):
    lens=lengths.tolist();q=cu.diff().tolist();b=len(lens);tile=64 if left>=0 else 128
    starts=[max(0,n-w-left)//tile if left>=0 else 0 for n,w in zip(lens,q)]
    counts=[(DIV(n,tile)-first) if w else 0 for n,w,first in zip(lens,q,starts)]
    cumulative=[0,*itertools.accumulate(counts)]
    width=max(1,DIV(cumulative[-1],p))
    owners=[sum(end<=part*width for end in cumulative[1:]) for part in range(p)]
    return [*cumulative,*starts,width,*owners]

class Call:
    def __init__(self,f):self.f=f
    def __getitem__(self,grid):
        return lambda *a,**kw:self.f(grid,*a,**kw)

def setup(stage):
    back,paged,meta=[(stage/p).read_text() for p in (BACK,PAGED,META)]
    alg=NS(is_dflash=lambda:True);mode=Mode()
    mode.is_decode_or_idle=lambda:False;mode.is_extend=lambda:True
    mode.is_draft_extend_v2=lambda:False;mode.is_dllm_extend=lambda:False
    events=[];receipts=[];fail=NS(window=None)
    def raster(grid,lens,cu,out,b,p,left,tile,bb,bp,**kw):
        if left==fail.window:raise RuntimeError('injected raster launch failure')
        assert grid==(DIV(p,16),) and tile==(64 if left>=0 else 128)
        out.copy_(torch.tensor(oracle(lens,cu,left,p),dtype=out.dtype))
        events.append(('build',left,p,out.data_ptr()))
    def maps(grid,*args,**kw):
        lens=args[0]+kw['SEQ_LEN_DELTA']
        args[7].copy_(lens);args[9][0]=0;args[9][1:].copy_(lens.cumsum(0))
        events.append(('metadata',))
    def partition(grid,*args,**kw):
        plan=kw.get('Plan')
        left=args[24];lens,cu=args[4:6];p=grid[0]
        if plan is not None:assert plan.tolist()==oracle(lens,cu,left,p)
        events.append(('partition',left,p,plan.data_ptr() if plan is not None else None))
    def merge(grid,*args,**kw):
        plan=kw.get('Plan')
        if plan is not None:assert plan.tolist()==oracle(args[2],args[3],args[12],args[15])
        events.append(('merge',plan.data_ptr() if plan is not None else None))
    def native(**kw):
        events.append(('native',))
        return kw['q']
    env=dict(torch=torch,math=math,triton=TR,TensorDescriptor=None,_partition_attention=Call(partition),
        _merge_partitions=Call(merge),_build_verify_raster=Call(raster),_fused_decode_metadata_kernel=Call(maps),
        dataclass=dataclass,get_spec=lambda:NS(speculative_algorithm='DFLASH'),SpeculativeAlgorithm=NS(from_string=lambda _:alg),IdSpaceKind=NS(SLIDING_WINDOW=1),
        maybe_create_verify_mask=lambda **kw:None,is_cp_active=lambda fb:False,ForwardMode=Mode,
        AttentionType=NS(ENCODER_ONLY='enc',DECODER_BIDIRECTIONAL='bi'),flash_attn_with_kvcache=native)
    for fn in ('verify_partition_workspace','refresh_verify_rasters','_direct_tile_args','paged_verify_attention'):body(paged,fn,env)
    body(meta,'normal_decode_set_metadata',env)
    ragged_path='python/sglang/srt/speculative/ragged_verify.py'
    source=subprocess.check_output(['git','-C',str(stage),'show','HEAD:'+ragged_path],text=True)
    env['RaggedTargetVerifyGeometry']=NS
    body(source,'build_ragged_target_verify_geometry',env)
    cls=next(n for n in ast.parse(back).body if getattr(n,'name',None)=='FlashAttentionMetadata')
    exec('from __future__ import annotations\n'+ast.unparse(cls),env)
    names=('init_cuda_graph_state','init_forward_metadata_out_graph','_init_full_cg_decode_metadata',
           '_bind_metadata_buffers','_set_decode_page_metadata','_apply_cuda_graph_metadata','_capture_unified_verify_metadata',
           'forward_extend','_mxfp8_sf_kwargs','_refresh_verify_rasters','init_forward_metadata','_get_swa_write_locations')
    attrs={n:body(back,n,env,'FlashAttentionBackend') for n in names}
    Backend=type('Backend',(),attrs)
    def make():
        op=Backend()
        op.__dict__.update(device='cpu',max_context_len=8192,page_size=1,_get_scheduler_metadata=None,
            local_attn_builder=None,use_sliding_window_kv_pool=True,topk=1,has_swa=True,needs_cpu_seq_lens=False,head_dim=64,num_kv_heads=2,
            is_encoder_decoder=False,is_prefill_aware_swa=False,_kv_shard_pool=None,skip_prefill=False,
            speculative_step_id=0,speculative_num_draft_tokens=8,is_draft_runner=False,
            kv_index_translator=NS(is_translating=False,reads_are_translated=False,write_ids=lambda *args:None),_hopper_verify_sms=132,use_mla=False,
            req_to_token=torch.zeros((32,8240),dtype=torch.int32),sliding_window_size=2047,_draft_window_left=-1,
            fa_impl_ver=3,num_splits=1,fa_skip_kv_cache=False,kv_cache_is_mxfp8=False,_disable_scheduler_metadata_precompute=False)
        op.prepare_paged_mha_query=lambda q,qr,kr,*a,**kw:(q,qr,kr,None,None)
        op.token_to_kv_pool=NS(full_to_swa_index_mapping=torch.arange(8240,dtype=torch.int32),translate_loc_from_full_to_swa=lambda t:t)
        op.req_to_token_pool=NS(req_to_token=op.req_to_token)
        op._maybe_init_local_attn_metadata=lambda *a:None
        op.init_cuda_graph_state(32,256)
        return op
    def batch(b):
        _,fb,layer,_,_,_=tensors(b,8,1,8240,16,2)
        fb.forward_mode=mode;fb.spec_algorithm=alg;fb.positions=torch.arange(8*b)
        fb.req_pool_indices=torch.arange(b);fb.encoder_lens=None;fb.kv_loc_plan=None
        fb.seq_lens=fb.seq_lens_cpu=torch.arange(b,dtype=torch.int32)*370+1
        fb.out_cache_loc=None;fb.swa_out_cache_loc=None
        q=torch.empty((b*8,16,64),dtype=torch.float8_e4m3fn)
        layer.k_scale=layer.v_scale=None
        kv=torch.empty((32,1,2,64),dtype=q.dtype)
        return fb,layer,q,(kv,kv)
    return make,batch,env,events,receipts,fail

def check(stage):
    make,batch,env,events,_,fail=setup(stage);op=make()
    counts=Counter();per_size={}
    def bind(op,fb,capture):
        events.clear();op.init_forward_metadata_out_graph(fb,in_capture=capture)
        return op.forward_metadata,events.copy()
    def caller(op,fb,layer,q,kv,left,expect):
        layer.sliding_window_size=left;op.get_paged_mha_kv_cache=lambda _:kv
        events.clear();op.forward_extend(q,None,None,layer,fb)
        if expect=='native':assert events==[('native',)],events
        else:
            assert [x[0] for x in events]==['partition','merge'],events
            assert events[0][3]==events[1][1]
            assert (events[0][3] is not None)==expect,events
        counts[str(expect)]+=1
        return events.copy()
    with patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True):
        for b in (1,3,15,32):
            fb,layer,q,kv=batch(b)
            md,ev=bind(op,fb,True)
            assert ev[0][0]=='metadata'
            assert [e[1] for e in ev[1:]]==([-1,2047] if b>1 else [])
            ptrs={k:v.data_ptr() for k,v in md.verify_rasters[2].items()} if b>1 else {}
            per_size[b]=dict(buffer_count=len(ptrs),bytes=sum(v.nbytes for v in md.verify_rasters[2].values()) if b>1 else 0)
            for rep in range(2):
                fb.seq_lens=torch.tensor([((i*997+rep*4123)%7400) if i%4 else 0 for i in range(b)],dtype=torch.int32)
                fb.seq_lens_cpu=fb.seq_lens
                repeated,ev=bind(op,fb,False)
                assert repeated is md
                assert [e[0] for e in ev]==['metadata']+(['build']*2 if b>1 else [])
                assert not ptrs or ptrs=={k:v.data_ptr() for k,v in md.verify_rasters[2].items()}
                for i in range(62):caller(op,fb,layer,q,kv,-1 if i%4==0 else 2047,b>1)
                caller(op,fb,layer,q,kv,255,False)
                fb.spec_info.ragged_verify_layout=object()
                caller(op,fb,layer,q,kv,-1,'native')
                fb.spec_info.ragged_verify_layout=None
        # Distinct shape-compatible metadata owners have independent plan storage.
        fb,layer,q,kv=batch(3);other=make()
        eager=make()
        events.clear();eager.init_forward_metadata(fb)
        eager_md=eager.forward_metadata
        assert [e[0] for e in events]==['build','build']
        caller(eager,fb,layer,q,kv,-1,True)
        fb.seq_lens=fb.seq_lens_cpu=torch.tensor([512,3001,0],dtype=torch.int32)
        eager.init_forward_metadata(fb)
        assert eager_md is not eager.forward_metadata and eager_md.verify_rasters[2][-1,132].data_ptr()!=eager.forward_metadata.verify_rasters[2][-1,132].data_ptr()
        caller(eager,fb,layer,q,kv,2047,True)
        m1,_=bind(op,fb,False);m2,_=bind(other,fb,True)
        assert all(m1.verify_rasters[2][k].data_ptr()!=v.data_ptr() for k,v in m2.verify_rasters[2].items())
        captured_plan_refs=dict(m1.verify_rasters[2])
        m1,ev=bind(op,fb,True)
        assert all(m1.verify_rasters[2][k].data_ptr()==buf.data_ptr() for k,buf in captured_plan_refs.items())
        before={k:v.clone() for k,v in m1.verify_rasters[2].items()}
        fb.seq_lens+=3000;bind(other,fb,False)
        assert all(torch.equal(m1.verify_rasters[2][k],v) for k,v in before.items())
        # A new captured role variant at the same B binds new views to the old
        # length storage. Its runtime callback, and old graph device pointers,
        # both observe the producer's new (possibly padded) lengths.
        refs=dict(m1.verify_raster_storage[2])
        lens=torch.tensor([2,7,0],dtype=torch.int32)
        ragged=NS(verify_lens=lens,qo_indptr_device=torch.tensor([0,2,9,9]),verify_lens_cpu=None)
        ragged.padded_to_bucket=lambda padded_bs:ragged if padded_bs==3 else None
        fb.spec_info.ragged_verify_layout=ragged
        m1,ev=bind(op,fb,True)
        assert m1.cu_seqlens_q.tolist()==[0,2,9,9] and [e[0] for e in ev]==['metadata','build','build']
        assert all(m1.verify_raster_storage[2][k].data_ptr()==buf.data_ptr() and buf.tolist()==oracle(m1.cache_seqlens_int32,m1.cu_seqlens_q,*k) for k,buf in refs.items())
        caller(op,fb,layer,q,kv,-1,'native')
        fb.spec_info.ragged_verify_layout=None
        m1,ev=bind(op,fb,True)
        # Queries with different length or cu tensors must never use an unrelated plan.
        saved=m1.cache_seqlens_int32;m1.cache_seqlens_int32=saved.clone()
        caller(op,fb,layer,q,kv,-1,False);m1.cache_seqlens_int32=saved
        saved=m1.cu_seqlens_q;m1.cu_seqlens_q=saved.clone()
        caller(op,fb,layer,q,kv,-1,False);m1.cu_seqlens_q=saved
        # A physical head geometry producing different partition counts stays indexed.
        layer.tp_k_head_num=layer.tp_v_head_num=1;layer.tp_q_head_num=8
        small=torch.empty((32,1,1,64),dtype=q.dtype)
        caller(op,fb,layer,q[:,:8].contiguous(),(small,small),-1,False)
        layer.tp_k_head_num=layer.tp_v_head_num=2;layer.tp_q_head_num=16
        for info in (m1,m2):
            x=info.cu_seqlens_q; x.copy_(torch.tensor([0,0,4,4],dtype=torch.int32))
            env['refresh_verify_rasters'](info,2047,2,132)
            assert info.verify_rasters[2][-1,132].tolist()==oracle(info.cache_seqlens_int32,x,-1,132)
        # Reconfigure the prewarmed head geometry and sliding width. A stale
        # prior key must be absent or must have been refreshed to the new lengths.
        fb.seq_lens=fb.seq_lens_cpu=torch.tensor([811,5199,411],dtype=torch.int32)
        op.sliding_window_size=2046;op.num_kv_heads=1
        bind(op,fb,False)
        assert not m1.verify_rasters[2]
        assert all(m1.verify_raster_storage[2][k].data_ptr()==buf.data_ptr() and buf.tolist()==oracle(m1.cache_seqlens_int32,m1.cu_seqlens_q,*k) for k,buf in refs.items())
        for (w,p),buf in m1.verify_rasters[2].items():
            assert buf.tolist()==oracle(m1.cache_seqlens_int32,m1.cu_seqlens_q,w,p),(w,p)
        op.sliding_window_size=2047;op.num_kv_heads=2
        bind(op,fb,False)
        # A failed refresh must not publish old buffers with newly changed lengths.
        fb.seq_lens=fb.seq_lens_cpu=torch.tensor([91,918,122],dtype=torch.int32)
        frozen_refs=dict(m1.verify_rasters[2])
        fail.window=2047
        try:bind(op,fb,False)
        except RuntimeError as err:assert str(err)=='injected raster launch failure'
        else:raise AssertionError('injected runtime did not propagate')
        candidate=m1.verify_rasters
        if candidate is not None:
            for (w,p),buf in candidate[2].items():
                assert buf.tolist()==oracle(m1.cache_seqlens_int32,m1.cu_seqlens_q,w,p),(w,p,'stale failed refresh')
        fail.window=None
        bind(op,fb,False);caller(op,fb,layer,q,kv,-1,True)
        assert all(m1.verify_rasters[2][k].data_ptr()==buf.data_ptr() and
                   buf.tolist()==oracle(m1.cache_seqlens_int32,m1.cu_seqlens_q,*k) for k,buf in frozen_refs.items()),'captured graph points to superseded allocations after retry'
    manifest=json.loads((HERE/'validation.json').read_bytes())
    full=(HERE/'patch.diff').read_bytes()
    assert sha256(full).hexdigest()==manifest['patch_sha256']
    with tempfile.TemporaryDirectory() as scratch:
        opts={**os.environ,'GIT_INDEX_FILE':str(Path(scratch)/'index')}
        def git(*args,data=None):
            return subprocess.check_output(['git','-C',str(stage),*args],input=data,env=opts)
        git('read-tree',manifest['base']);git('apply','--cached','-',data=full)
        assert git('write-tree').decode().strip()==manifest['tree']
        assert all(git('show',':'+p)==(stage/p).read_bytes() for p in (PAGED,META,BACK))
    return dict(tree=manifest['tree'],patch_sha256=manifest['patch_sha256'],per_size=per_size,dispatches=counts,
        regressions=['same-pointer lengths and two replay steps per bucket','independent concurrent metadata owners',
            'eager metadata creation and replacement','same-bucket recapture and ragged padded q offsets',
            'different length/cu tensors and changed head geometry','historical captured addresses remain current when current policy changes',
            'injected second-raster launch error followed by same-address retry'],source_sha256={p:sha256((stage/p).read_bytes()).hexdigest() for p in (PAGED,META,BACK)},
        scope='Production backend graph/capture update, real pager and per-layer dispatch. Native metadata, raster and attention CUDA launches use checked host probes.',physical_H200=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',type=Path);args=ap.parse_args()
    torch.set_num_threads(1);print(json.dumps(check(args.stage),indent=2))
