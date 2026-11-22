"""Pair the actual prompt preparation/query conversion with a physical-slot CPU attention reference."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch, PropertyMock

import torch


def extract(path, names, scope):
    tree = ast.parse(path.read_text())
    for name in names:
        node = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        for arg in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]:
            arg.annotation=None
        node.returns=None
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)


def check(stage):
    attention=stage/'python/sglang/srt/layers/attention'
    direct=attention/'direct_prompt_kv.py'; fa=attention/'flashattention_backend.py'
    class Plain: pass
    class Hybrid: pass
    state=NS(scope=False,full=False,raw=None)
    scope=dict(torch=torch,ForwardMode=NS(EXTEND="extend"),AttentionType=NS(ENCODER_ONLY="enc",DECODER_BIDIRECTIONAL="dec"),
               MHATokenToKVPool=Plain,SWAKVPool=Hybrid,is_in_prefill_graph_scope=lambda:state.scope,
               is_in_full_prefill_graph=lambda:state.full,get_prefill_raw_num_tokens=lambda:state.raw)
    extract(direct,['prepare_direct_prompt_kv','complete_direct_prompt_kv'],scope)
    extract(fa,['prepare_paged_mha_query'],scope)
    prepare,complete,query=(scope[n] for n in ('prepare_direct_prompt_kv','complete_direct_prompt_kv','prepare_paged_mha_query'))
    torch.manual_seed(7178); torch.set_num_threads(1)
    fp8=torch.float8_e4m3fn
    cache_bytes=out_values=perms=0
    accepted=declines=0
    comparisons=[]
    for n,p,h,d,window,scale,live,same,mode in [
        (4096,0,2,64,-1,False,True,False,'eager'),(4097,151,2,64,-1,True,True,False,'eager'),
        (4096,9,4,64,2047,True,True,True,'breakable'),(4097,2222,4,64,2047,True,True,False,'breakable'),
        (4096,4099,4,64,0,False,True,False,'eager'),(4096,2081,4,64,31,True,True,False,'eager'),
        (4096,512,2,64,-1,True,True,False,'padded'),(4096,512,2,64,-1,True,True,False,'full'),
        (4096,512,2,64,-1,True,True,False,'warmup'),(4096,512,2,64,-1,True,False,False,'eager'),
    ]:
        state.scope=mode in ('padded','full','warmup','breakable');state.full=mode=='full'
        state.raw=None if mode=='warmup' else n
        heads=2*h;total=n+p; cap=total+37
        q=torch.randn(n + (9 if mode=='padded' else 0),heads,d).bfloat16()
        storage=torch.randn(n,h,2*d).bfloat16()
        k,v=storage[...,::2],storage[...,1::2]
        if same:v=k
        original_q=q.clone();original_k=k.clone();original_v=v.clone()
        ks=torch.tensor([0.38]) if scale else None
        vs=torch.tensor([1.73]) if scale else None
        full=(1+torch.randperm(cap-1)[:total]).int().unsqueeze(0)
        swa=(1+torch.randperm(cap-1)[:total]).int().unsqueeze(0)
        lengths=torch.tensor([total],dtype=torch.int32)
        md=NS(page_table=full,swa_page_table=swa,max_seq_len_q=n,max_seq_len_k=total,
              cache_seqlens_int32=lengths,cu_seqlens_q=torch.tensor([0,n],dtype=torch.int32),
              image_token_ranges=None,swa_spec_metadata=None)
        pool=Hybrid();pool.full_kv_pool=Plain();pool.swa_kv_pool=Plain();pool.layers_mapping={3:(0,window>=0)}
        be=NS(kv_cache_dtype_str="fp8_e4m3",kv_cache_is_mxfp8=False,fa_impl_ver=3,kv_cache_dtype=fp8,
              forward_metadata=md,full_cg_prefill_metadata=None,token_to_kv_pool=pool)
        layer=NS(layer_id=3,tp_q_head_num=heads,tp_k_head_num=h,tp_v_head_num=h,head_dim=d,v_head_dim=d,
                 sliding_window_size=window,attn_type='causal',is_cross_attention=False,k_scale=ks,v_scale=vs)
        batch=NS(forward_mode='extend',batch_size=1,dcp_kv_mask=None,extend_seq_lens_cpu=[n],extend_prefix_lens_cpu=[p],
                 req_pool_indices_cpu=torch.tensor([3]) if live else None,seq_lens_cpu=torch.tensor([total]))
        write=NS(loc=full[0,p:],swa_loc=swa[0,p:])
        with torch.inference_mode(),patch.object(torch.Tensor,'is_cuda',new_callable=PropertyMock,return_value=True),              patch.object(torch.cuda,'is_current_stream_capturing',return_value=False):
            packed=prepare(be,q,k,v,layer,batch,write,ks,vs)
            if mode in ('padded','full','warmup') or not live:
                assert packed is None and torch.equal(k,original_k) and torch.equal(v,original_v)
                declines+=1
                continue
            assert packed is not None
            accepted+=1
            qs,_,_,kdes,vdes=query(be,q,None,None,layer,logical_batch_size=1,kv_head_num=h,is_prefill=True)
        expected_k,expected_v=original_k.clone(),original_v.clone()
        if same: expected_v=expected_k
        if scale: expected_k.div_(ks);expected_v.div_(vs)
        expected_k=expected_k.to(fp8).view(torch.uint8);expected_v=expected_v.to(fp8).view(torch.uint8)
        logical=(swa if window>=0 else full)[0].long()
        cs=[torch.randn(cap,h,d).to(fp8).view(torch.uint8) for _ in range(2)]
        canonical=[c.clone() for c in cs]
        for c,rowvals in zip(canonical,(expected_k,expected_v)):
            c[logical[p:]]=rowvals
        for c,new in zip(cs,packed[:2]):
            c[logical[p:]]=new[-n:].view(torch.uint8)
        assert all(torch.equal(a,b) for a,b in zip(cs,canonical))
        dense_k,dense_v,klen=complete(md,packed,*[c.view(fp8).unsqueeze(1) for c in cs],logical.unsqueeze(0),lengths)
        omit=max(0,p-window) if window>=0 else 0
        assert klen.item()==total-omit and dense_k.shape==(1,total-omit,h,d)
        records=[canonical[j].index_select(0,logical).view(fp8) for j in range(2)]
        for original,dense in zip(records,(dense_k,dense_v)):
            assert torch.equal(original[omit:].view(torch.uint8),dense[0].view(torch.uint8))
            cache_bytes+=original[omit:].numel()
        assert torch.equal(qs.view(torch.uint8),original_q.to(fp8).view(torch.uint8))
        if scale:
            assert torch.equal(kdes,ks.expand(1,h)) and torch.equal(vdes,vs.expand(1,h))
        else: assert kdes is None and vdes is None
        # One independent physical request-table traversal for each logical causal query.
        # Accumulate attention in fp64 to keep kernel/tile approximations out of the oracle.
        for qi in (0,1,n//2,n-1):
            end=p+qi+1; begin=0 if window<0 else max(0,end-1-window)
            e=klen.item()-n+qi+1; b=0 if window<0 else max(0,e-1-window)
            assert begin==b+omit and end==e+omit
            for head in (0,heads-1):
                kh=head//(heads//h)
                qvec=qs[qi,head].double()
                def attn(key,value):
                    z=(key[:,kh].double()@qvec)*(1/d**.5)*(ks.item() if scale else 1.)
                    score=31*torch.tanh(z/31)
                    return (torch.softmax(score,0)@value[:,kh].double())*(vs.item() if scale else 1.)
                x=attn(dense_k[0,b:e],dense_v[0,b:e])
                y=attn(records[0][begin:end],records[1][begin:end])
                assert torch.equal(x,y) and torch.isfinite(x).all()
                out_values+=d
        perms+=total
        comparisons.append([p,n,window,same,scale])
    # Check conversion extremes that would make a numerical attention output NaN.
    allbits=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    tests=0
    for scale in (.125,.75,1.,1.75):
        x=allbits.clone().div_(scale)
        a=torch.empty_like(x,dtype=fp8);a.copy_(x)
        assert torch.equal(a.view(torch.uint8),x.to(fp8).view(torch.uint8))
        tests+=x.numel()
    return dict(helper_sha256=hashlib.sha256(direct.read_bytes()).hexdigest(),backend_sha256=hashlib.sha256(fa.read_bytes()).hexdigest(),
                accepted=accepted,declines=declines,cases=comparisons,logical_rows=perms,matched_dense_cache_bytes=cache_bytes,
                attention_fp64_outputs=out_values,bf16_extreme_conversions=tests,torch=torch.__version__,physical_gpu=False)


if __name__=='__main__':
    print(json.dumps(check(Path(sys.argv[1])),indent=2))
