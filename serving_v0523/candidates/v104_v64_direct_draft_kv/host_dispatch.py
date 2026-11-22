"""Production DFlash worker callback and materialize routing with actual Torch CPU matmul."""
import argparse
import ast
import collections
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch


def method(path,classname,name,env):
    cls=next(n for n in ast.parse(path.read_text()).body if getattr(n,'name',None)==classname)
    node=next(n for n in cls.body if getattr(n,'name',None)==name)
    module=ast.parse('from __future__ import annotations')
    module.body.append(node)
    exec(compile(module,str(path),'exec'),env)
    return env[name]


def run(root):
    module=root/'python/sglang/kernels/ops/speculative/fused_kv_materialize.py'
    worker=root/'python/sglang/srt/speculative/dflash_worker_v2.py'
    calls=collections.Counter()
    captured={}
    class Launch:
        def __getitem__(self,grid):
            def go(proj,*args):
                captured.update(proj=proj.clone(),loc=args[-5],lens=args[-4],width=args[-3],fp8=args[-2],grid=grid)
                calls['direct']+=1
            return go

    def together(*args,**kwargs):
        calls['workspace']+=1
        captured['proj']=args[0].clone()
        return kwargs['k_out'],kwargs['v_out']

    scope=dict(torch=torch,triton=NS(next_power_of_2=lambda x:1<<(x-1).bit_length()),
               _fused_norm_rope_kernel_stacked=Launch(),_fused_norm_rope_stacked=together)
    configure=method(module,'FusedKVMaterializeHelper','configure_prefix_cache',scope)
    materialize=method(module,'FusedKVMaterializeHelper','materialize',scope)
    append=method(worker,'DFlashWorkerV2','_append_target_hidden_fused',dict(torch=torch,KVWriteLoc=lambda x,**kw:x))
    gpu=torch.device('cuda:0')
    class Meta:
        def __init__(self,shape,dtype,device=gpu):
            self.shape,self.dtype,self.device,self.ndim=shape,dtype,device,len(shape)
            self.contiguous=True
        def is_contiguous(self):
            return self.contiguous
    class Pool:
        def set_kv_buffer_prefix_valid(self,*args):
            calls['prefix_cb']+=1
        def set_kv_buffer(self,*args):
            calls['full_cb']+=1
    mock=ModuleType('sglang.srt.mem_cache.memory_pool')
    mock.MHATokenToKVPool=Pool
    runtime_layers=[NS(self_attn=NS(attn=NS(layer_id=i,k_scale=1.,v_scale=1.))) for i in range(5)]
    checks={}
    with patch.dict(sys.modules,{'sglang.srt.mem_cache.memory_pool':mock}), patch.object(torch.cuda,'get_device_capability',return_value=(9,0)):
        for fp8 in (False,True):
            for fault in ('ok','subclass','layout','type','invalid_dim','v_dim','head_num','layers','start',
                          'scale_buf','ptr_dtype','ptr_shape','ptr_stride','buf_shape','buf_device','buf_dtype',
                          'buf_stride','patched_writer','weight_dtype','host_weights'):
                class Child(Pool): pass
                pool=(Child if fault=='subclass' else Pool)()
                pool.dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16
                pool.kv_cache_layout='nhd' if fault!='layout' else 'hnd'
                pool.start_layer=int(fault=='start')
                pool.layer_num=5+int(fault=='layers')
                pool.head_num=1+int(fault=='head_num')
                pool.head_dim=128+int(fault=='invalid_dim')
                pool.v_head_dim=128+int(fault=='v_dim')
                pool.k_scale_buffer=1 if fault=='scale_buf' else None
                pool.v_scale_buffer=None
                typ=torch.uint8 if fp8 else torch.bfloat16
                pool.k_buffer=[Meta((64,1,128),typ) for _ in runtime_layers]
                pool.v_buffer=[Meta((64,1,128),typ) for _ in runtime_layers]
                pool.k_data_ptrs=Meta((5,),torch.uint64)
                pool.v_data_ptrs=Meta((5,),torch.uint64)
                helper=NS(flat_kv_weight_t=Meta((7,1280),torch.bfloat16),head_dim=128,num_kv_heads=1,n_layers=5,_prefix_cache=None)
                if fault=='type':pool.dtype=torch.float16
                if fault=='ptr_dtype':pool.k_data_ptrs.dtype=torch.int64
                if fault=='ptr_shape':pool.k_data_ptrs.shape=(6,)
                if fault=='ptr_stride':pool.v_data_ptrs.contiguous=False
                if fault=='buf_shape':pool.v_buffer[1].shape=(64,2,64)
                if fault=='buf_device':pool.k_buffer[1].device=torch.device('cpu')
                if fault=='buf_dtype':pool.v_buffer[2].dtype=torch.float
                if fault=='buf_stride':pool.k_buffer[2].contiguous=False
                if fault=='weight_dtype':helper.flat_kv_weight_t.dtype=torch.float16
                if fault=='host_weights':helper.flat_kv_weight_t.device=torch.device('cpu')
                if fault=='patched_writer':pool.set_kv_buffer_prefix_valid=lambda *a:None
                configure(helper,pool,runtime_layers)
                assert (helper._prefix_cache is not None)==(fault=='ok'),fault
                checks['configured_or_rejected']=checks.get('configured_or_rejected',0)+1
    torch.manual_seed(77)
    for fp8 in (False,True):
        for condition in ('direct','no_flag','prefill','strided_loc','strided_lens','strided_positions',
                          'replaced_table','stale_pool','changed_scale','tensor_scale','mismatched_lens'):
            B,W,H,L,D=2,4,9,5,8
            rows=B*W
            x=torch.randn(rows,H).bfloat16()
            weight=torch.randn(H,2*L*D).bfloat16()
            positions=torch.arange(rows)
            loc=torch.arange(rows,dtype=torch.int64).reshape(B,W)
            lens=torch.tensor([4,1],dtype=torch.int32)
            k=torch.zeros((L,rows,1,D),dtype=torch.bfloat16)
            v=torch.empty_like(k)
            pool=Pool()
            pool.k_data_ptrs=torch.arange(L)
            pool.v_data_ptrs=torch.arange(L)
            scaled=tuple(l.self_attn.attn for l in runtime_layers) if fp8 else ()
            for l in runtime_layers:
                l.self_attn.attn.k_scale=1.
            helper=NS(_prefix_cache=(pool,pool.k_data_ptrs,pool.v_data_ptrs,scaled),device=torch.device('cpu'),
                      _proj_workspace=torch.empty((rows,2*L*D),dtype=torch.bfloat16),_k_workspace=k,_v_workspace=v,
                      flat_kv_weight_t=weight,k_norm_weights=torch.ones((L,D)),eps_values=torch.ones(L),
                      n_layers=L,num_kv_heads=1,head_dim=D,kv_size=D,layer_out_dim=2*D,rotary_dim=D,
                      max_position_hint=rows,_mm_out_supported=True,_ensure_workspace=lambda *z:None,
                      _ensure_rope_cache=lambda _:torch.ones((rows,D)))
            helper.materialize=lambda **kw:materialize(helper,**kw)
            wk=NS(_fused_kv_helper=helper,draft_model_runner=NS(token_to_kv_pool=pool),draft_model=NS(layers=runtime_layers))
            if condition=='no_flag': helper._prefix_cache=None
            if condition=='prefill':loc2=None
            else:loc2=loc
            if condition=='strided_loc':loc2=torch.arange(rows*2).view(B,2*W)[:,::2]
            if condition=='strided_lens':lens=torch.arange(B*2,dtype=torch.int32)[::2]
            if condition=='strided_positions':positions=torch.arange(rows*2)[::2]
            if condition=='replaced_table':pool.k_data_ptrs=pool.k_data_ptrs.clone()
            if condition=='stale_pool':
                other=Pool();other.k_data_ptrs=pool.k_data_ptrs;other.v_data_ptrs=pool.v_data_ptrs
                wk.draft_model_runner.token_to_kv_pool=other
            if condition=='changed_scale':runtime_layers[1].self_attn.attn.k_scale=2.
            if condition=='tensor_scale':runtime_layers[3].self_attn.attn.k_scale=torch.tensor(1.)
            if condition=='mismatched_lens':lens=lens[:1]
            calls.clear()
            append(wk,x,positions,loc.reshape(-1),loc2,lens)
            assert torch.equal(captured['proj'],(x@weight).reshape(rows,L,D*2))
            direct=condition=='direct' or (not fp8 and condition in ('changed_scale','tensor_scale'))
            if direct:
                assert calls=={'direct':1} and captured['grid']==(rows,1,L) and captured['fp8']==fp8
                assert captured['loc'] is loc and captured['lens'] is lens
            else:
                assert calls=={'workspace':1,'full_cb' if condition=='prefill' else 'prefix_cb':L},(condition,calls)
            checks['worker_and_projection']=checks.get('worker_and_projection',0)+1
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    print(json.dumps({"torch":torch.__version__,**run(p.parse_args().root)},indent=2))
