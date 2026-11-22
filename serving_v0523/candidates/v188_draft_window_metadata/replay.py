"""Replay the pinned graph-metadata and FA4 calls, preserving model positions and pool aliases."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional, Union, Tuple, Callable
import torch

STAGE=Path(sys.argv[1])
SRC=STAGE/'python/sglang/srt/layers/attention/flashattention_backend.py'
PINNED=Path(sys.argv[2]) if len(sys.argv)>2 else STAGE
FA=PINNED/'python/sglang/kernels/ops/attention/flash_attention_v4.py'


def executable(path,names,ns):
    mod=ast.parse(path.read_text())
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias('annotations')],level=0)]
    for name in names:
        n=next(n for n in ast.walk(mod) if isinstance(n,ast.FunctionDef) and n.name==name)
        n.decorator_list=[];nodes.append(n)
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,[])),str(path),'exec'),ns)
    return {n:ns[n] for n in names}


def device_metadata(lengths,cu,table,reqmap,indices,width,prefix,delta,page,swa,**kw):
    assert page==1 and swa is None and not kw['skip_page_table']
    win=kw['window_left']
    start=(prefix-win).clamp(min=0)//128*128 if win>=0 else torch.zeros_like(prefix)
    lengths.copy_(prefix+delta-start);cu[0]=0;cu[1:]=lengths.cumsum(0)
    for dest,req,offset,length in zip(table,indices,start,lengths):
        dest[:length]=reqmap[req,offset:offset+length]


def native_masked_attention(**a):
    q=a['q'];k=a['k'];v=a['v'];table=a['page_table'];lengths=a['seqused_k'];rows=[]
    assert a['window_size']==(2047,2047) and a['max_seqlen_q']==8 and not a['causal']
    # Native CuTe traverses identical 128-key tiles; independent physical row comparison below.
    for b,n in enumerate(lengths.tolist()):
        for i in range(8):
            query=n-8+i
            phys=table[b,max(0,query-2047):min(n,query+2048)].long()
            kk=k.index_select(0,phys).float(); vv=v.index_select(0,phys).float()
            scores=kk @ q[b*8+i].float()
            rows.append((scores.softmax(0) @ vv).bfloat16())
    return torch.stack(rows),torch.zeros(q.shape[0])


def main():
    torch.set_num_threads(1); rng=torch.Generator().manual_seed(18885)
    ns=dict(normal_decode_set_metadata=device_metadata,torch=torch)
    ns.update(executable(SRC,['_set_decode_page_metadata','_apply_cuda_graph_metadata'],ns))
    Backend=type('Backend',(),{k:ns[k] for k in ('_set_decode_page_metadata','_apply_cuda_graph_metadata')})
    real=executable(FA,['flash_attn_with_kvcache'],dict(torch=torch,flash_attn_varlen_func=native_masked_attention))
    call=real['flash_attn_with_kvcache']; cases=values=physical=full_calls=changed_replays=0
    mode=NS(is_decode_or_idle=lambda:False,is_target_verify=lambda:True)
    for B in (1,4,16,32):
        size=4200; H=8; reqmap=torch.randperm(size*(B+2),generator=rng).int().reshape(B+2,size)
        be=[Backend(),Backend()]
        for s,win in zip(be,[-1,2047]):
            s.page_size=1;s.speculative_num_draft_tokens=8;s._draft_window_left=win;s.topk=0;s.req_to_token=reqmap
            s.use_sliding_window_kv_pool=False;s.kv_index_translator=NS(reads_are_translated=False)
            s.target_verify_metadata={B:NS(cache_seqlens_int32=torch.empty(B,dtype=torch.int32),
                cu_seqlens_k=torch.empty(B+1,dtype=torch.int32),page_table=torch.full((B,size),-42,dtype=torch.int32),
                swa_page_table=None)}
        for iteration in range(5):
            P=torch.randint(0,size-8,(B,),generator=rng,dtype=torch.int32)
            P[:min(B,4)]=torch.tensor(([0,2047,2174,2175],[1,2048,2176,4192],[4091,4092,4093,4094],
                                       [1919,2046,2048,2304],[4192,2175,0,1])[iteration][:min(B,4)])
            reqs=torch.randperm(B+2,generator=rng)[:B]
            q=torch.randn(B*8,H,generator=rng).bfloat16()
            physical_pool=torch.randn((B+2)*size,H*2,generator=rng).bfloat16()
            K,V=physical_pool[:,:H],physical_pool[:,H:]
            for s in be:
                s._apply_cuda_graph_metadata(None,B,reqs,P,None,mode,NS(),None)
            origin=torch.clamp(P-2047,min=0)//128*128
            l,r=[s.forward_metadata for s in be]
            for idx,req in enumerate(reqs):
                n=int(r.cache_seqlens_int32[idx]);o=int(origin[idx])
                assert torch.equal(r.page_table[idx,:n],l.page_table[idx,o:o+n])
                physical+=n
            for layer in range(5):
                outputs=[]
                for m in (l,r):
                    outputs.append(call(q=q,k_cache=K,v_cache=V,cache_seqlens=m.cache_seqlens_int32,
                        page_table=m.page_table,cu_seqlens_q=torch.arange(0,(B+1)*8,8,dtype=torch.int32),
                        cu_seqlens_k_new=m.cu_seqlens_k,max_seqlen_q=8,window_size=(2047,2047),causal=False))
                assert torch.equal(*outputs);q=(q+outputs[0]).bfloat16()
                values+=q.numel(); full_calls+=2
            cases+=1;changed_replays+=iteration>0
    return dict(replay_pairs=cases,fresh_followup_replays=changed_replays,native_FA4_wrapper_calls=full_calls,
                physical_suffix_page_ids=physical,matched_BF16_components=values,torch=torch.__version__,
                scope='Production full-graph metadata body+FA4 wrapper; host kernel substitutes for metadata and softmax.',
                hardware=False)


if __name__=='__main__':print(json.dumps(main(),indent=2))
