"""Exercise the certified mixed full-tail planner, eligibility, split and grouped launcher."""
import argparse
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import PropertyMock, patch

import torch

HERE = Path(__file__).resolve().parent
PAGED = 'python/sglang/kernels/ops/attention/paged_verify_attention.py'
BACKEND = 'python/sglang/srt/layers/attention/flashattention_backend.py'
PLANNER = 'python/sglang/srt/layers/attention/direct_prompt_kv.py'
EXPECTED = 'c1b938cd93f2cfdbaba9b0e6dce7e5d16f95410d811db9a8e8570693739f0acf'


def source(stage):
    m = json.loads((HERE/'validation.json').read_text())
    patch_bytes=(HERE/'patch.diff').read_bytes()
    assert hashlib.sha256(patch_bytes).hexdigest()==m['patch_sha256']==EXPECTED
    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ,'GIT_INDEX_FILE':str(Path(scratch)/'index')}
        def git(*args,data=None):
            return subprocess.check_output(['git','-C',str(stage),*args],input=data,env=env)
        git('read-tree',m['base'])
        git('apply','--cached','-',data=patch_bytes)
        assert git('write-tree').decode().strip()==m['tree']
        result={path:git('show',':'+path) for path in (PAGED,BACKEND,PLANNER)}
    assert all(content==(stage/path).read_bytes() for path,content in result.items())
    return result


def code(nodes,name):
    return compile(ast.Module(body=nodes,type_ignores=[]),name,'exec')


def get(nodes,name):
    return next(n for n in nodes if getattr(n,'name',None)==name)


class SwappablePool:
    pass


class Probe:
    def __init__(self,call):
        self.call=call

    def __getitem__(self,grid):
        return lambda *a,**kw:self.call(grid,*a,**kw)


def run_case(contents,tails,reason):
    qh,h,d=16,2,64
    if reason=='unsupported_groups':
        qh=18
    lengths=[4096]+[8]*tails
    if reason=='ragged':
        lengths[-1]=7
    b=len(lengths)
    cut=lengths[0]
    live=sum(lengths)
    prefixes=[83]+[64+(13*i)%127 for i in range(tails)]
    totals=[n+p for n,p in zip(lengths,prefixes)]
    def ints(v):
        return torch.tensor(v,dtype=torch.int32)
    cu=ints([0,*itertools.accumulate(lengths)])
    cuk=ints([0,*itertools.accumulate(totals)])
    lengths_gpu=ints(totals)
    cols=max(totals)
    table=torch.arange(b*(cols+4),dtype=torch.int32).view(b,-1)[:,:cols]
    cache=[torch.empty((b*(cols+4),1,h,d),dtype=torch.float8_e4m3fn) for _ in range(2)]
    write=[torch.empty((live,h,d),dtype=torch.float8_e4m3fn) for _ in range(2)]
    k=write[0].bfloat16()
    v=write[1].bfloat16()
    q=torch.empty((live,qh,d),dtype=torch.float8_e4m3fn)
    output=torch.full((live+11,qh,d),-2,dtype=torch.bfloat16)
    dst=output[3:3+live]
    mode=NS(is_target_verify=lambda:False)
    pool=SwappablePool()
    pool.layers_mapping=[(0,False)]
    pool.full_kv_pool=NS()
    meta=NS(cache_seqlens_int32=lengths_gpu,cu_seqlens_q=cu,cu_seqlens_k=cuk,page_table=table,
            image_token_ranges=None,swa_spec_metadata=None,max_seq_len_q=max(lengths),max_seq_len_k=max(totals))
    backend=NS(forward_metadata=meta,full_cg_prefill_metadata=None,token_to_kv_pool=pool,
        kv_cache_dtype=q.dtype,topk=1,_disable_scheduler_metadata_precompute=False,_hopper_verify_sms=132,
        speculative_num_draft_tokens=8,fa_impl_ver=3,num_splits=1,kv_index_translator=NS(is_translating=False))
    layer=NS(tp_q_head_num=qh,tp_k_head_num=h,tp_v_head_num=h,head_dim=d,v_head_dim=d,is_cross_attention=False,
             sliding_window_size=-1,layer_id=0,attn_type='decoder',logit_cap=2.0 if reason=='softcap' else 0.0,
             scaling=.125)
    batch=NS(batch_size=b,forward_mode=mode,dcp_kv_mask=None,mixed_verify_logits_indices=ints([0]),
             mixed_verify_is_causal=True,spec_info=None,mixed_verify_prompt_count=1,_attn_output=dst,
             extend_seq_lens_cpu=lengths,extend_prefix_lens_cpu=prefixes,seq_lens_cpu=lengths_gpu,
             spec_algorithm=NS(is_dflash=lambda:True))
    scope=dict(torch=torch,ForwardMode=NS(EXTEND=mode),AttentionType=NS(ENCODER_ONLY='enc',DECODER_BIDIRECTIONAL='bi'),
               SWAKVPool=SwappablePool,is_in_prefill_graph_scope=lambda:True,is_in_full_prefill_graph=lambda:False,
               get_prefill_raw_num_tokens=lambda:live)
    exec(code([get(ast.parse(contents[PLANNER]).body,'prepare_batched_prompt_kv')],PLANNER),scope)
    with torch.inference_mode(),patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True),patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
        layout=scope['prepare_batched_prompt_kv'](backend,q,k,v,layer,batch,NS(loc=ints(range(live))))
    assert layout is not None and layout[1][0]==cut and layout[1][3].tolist()==[x-cut for x in cu[1:].tolist()]
    calls=[]
    info={}
    scales=dict(k_descale=torch.arange(b*h,dtype=torch.float32).view(b,h)+1,
                v_descale=torch.tensor([1.25]).expand(b,h))

    def partition(grid,*args,**kwargs):
        assert grid==(132,2) and kwargs=={'num_warps':4}
        assert args[0].data_ptr()==q[cut:].data_ptr() and args[3].data_ptr()==table[1:].data_ptr()
        assert args[1] is cache[0] and args[2] is cache[1] and args[4].data_ptr()==lengths_gpu[1:].data_ptr()
        assert args[5].tolist()==cu[1:].sub(cut).tolist() and args[6].data_ptr()==scales['k_descale'][1:].data_ptr()
        calls.append('partition')
        info.update(scratch_bytes=args[7].nbytes+args[8].nbytes)

    def merge(grid,*args,**kwargs):
        assert kwargs=={'num_warps':4}
        rows=2 if tails==3 else 4
        assert grid==(tails,2,64//rows) and args[-1]==rows and args[-2]==16
        assert args[2].data_ptr()==lengths_gpu[1:].data_ptr() and args[3] is layout[1][3]
        assert args[4].stride()==(0,0) and args[4].shape==(tails,h)
        assert args[5].data_ptr()==dst[cut:].data_ptr()
        args[5].fill_(11)
        info.update(merge_grid=grid,rows=rows)
        calls.append('merge')

    paged=ast.parse(contents[PAGED]).body
    ctx=dict(torch=torch,math=math,triton=NS(cdiv=lambda n,b:-(-n//b),next_power_of_2=lambda n:1<<(n-1).bit_length()),
             _partition_attention=Probe(partition),_merge_partitions=Probe(merge),TensorDescriptor=None)
    exec(code([get(paged,'_direct_tile_args'),get(paged,'paged_verify_attention')],PAGED),ctx)
    def native(**kw):
        assert kw['out'].data_ptr()==dst[cut:].data_ptr() and kw['q'].data_ptr()==q[cut:].data_ptr()
        assert kw['cu_seqlens_q'].tolist()==cu[1:].sub(cut).tolist() and kw['page_table'].data_ptr()==table[1:].data_ptr()
        assert kw['v_descale'].stride()==(0,0) and kw['softcap']==layer.logit_cap
        kw['out'].fill_(11)
        calls.append('native')
        return kw['out']
    def pack(*a):
        assert a[0] is cache[0] and a[1] is cache[1] and a[4].shape[0]==1
        calls.append('pack')
        return write
    def dense(**kw):
        assert kw['q'].data_ptr()==q.data_ptr() and kw['q'].shape[0]==cut and kw['out'].data_ptr()==dst.data_ptr()
        assert kw['k_descale'].shape==(1,h)
        kw['out'].fill_(7)
        calls.append('dense')
        return kw['out']
    backend.flash_attn_varlen_func=dense
    cls=get(ast.parse(contents[BACKEND]).body,'FlashAttentionBackend')
    fwd=get(cls.body,'forward_extend')
    eligibility=next(n for n in ast.walk(fwd) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='verify_eligible')
    branch=next(n for n in ast.walk(fwd) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('batched_prompt_kv is not None and query.dtype'))
    branch.body=[n for n in branch.body if not isinstance(n,ast.ImportFrom)]
    env=dict(self=backend,forward_batch=batch,layer=layer,use_local_attn=False,return_lse=False,causal=True,
             kwargs=scales.copy(),query=q,key_cache=cache[0],value_cache=cache[1],batched_prompt_kv=layout,
             _fa_out=dst,page_table=table,cache_seqlens=lengths_gpu,cu_seqlens_q=cu,
             write_k=write[0],write_v=write[1],pack_prompt_rows=pack,flash_attn_with_kvcache=native,
             window_size=(-1,-1),paged_verify_attention=ctx['paged_verify_attention'])
    exec(code([eligibility,branch],BACKEND),env)
    assert env['result'].data_ptr()==dst.data_ptr()
    assert dst[:cut].eq(7).all() and dst[cut:].eq(11).all() and output[:3].eq(-2).all() and output[3+live:].eq(-2).all()
    assert calls==(['partition','merge','pack','dense'] if reason=='eligible' else ['native','pack','dense'])
    charged=write[0].nbytes+write[1].nbytes
    if reason=='eligible':
        assert info['scratch_bytes']==(132+tails-1)*h*64*(d+1)*4 and info['scratch_bytes']+charged<=32*1024**2
    return dict(tail_requests=tails,mode=reason,eligible_predicate=bool(env['verify_eligible']),
                calls=calls,charged_write_bytes=charged,**info)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',type=Path)
    args=parser.parse_args()
    contents=source(args.stage)
    torch.set_num_threads(1)
    cases=[run_case(contents,n,reason) for n,reason in itertools.product((3,15,31),('eligible','softcap','ragged','unsupported_groups'))]
    print(json.dumps(dict(full_sha256=EXPECTED,source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in contents.items()},
          cases=cases,scope='Production planner, actual forward_extend eligibility and split, real paged launcher, CPU torch storage. CUDA partition, merge and FA3 boundaries replaced by checked probes.'),indent=2))


if __name__=='__main__':
    main()
