"""Replay the pinned FA4 mask and split boundaries over a V255 physical-page suffix."""
import argparse
import ast
from collections import Counter, namedtuple
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]/"sglang"
PIN = "297e54fd105e974cdf401e5b4d20253a22c16f90"
CUTE = "python/sglang/kernels/ops/attention/flash_attn/cute/"
META = "python/sglang/kernels/ops/attention/metadata.py"
WRAP = "python/sglang/kernels/ops/attention/flash_attention_v4.py"
BACK = "python/sglang/srt/layers/attention/flashattention_backend.py"
sys.path.insert(0,str(HERE.parent/"v248_shared_verify_raster"))
from independent_geometry import TritonSubset, Pointer

class Ops(TritonSubset):
    def load(self, p, mask=True, other=0, **kwargs):
        return super().load(p, mask, other)
    def store(self, p, val, mask=True, **kwargs):
        return super().store(p, val, mask)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--manifest", type=Path, default=HERE/"validation.json")
    parser.add_argument("--output", type=Path, default=HERE/"independent_geometry.json")
    return parser.parse_args()

def run(root, manifest, patch_sha):
    sources = {}
    def source(path):
        if (root/path).is_file():
            raw=(root/path).read_bytes()
        else:
            raw=subprocess.check_output(["git","-C",BASE,"show",f"{PIN}:{path}"])
        sources[path]=sha256(raw).hexdigest()
        return raw.decode()
    tl=Ops()
    ns=dict(torch=torch, tl=tl, triton=NS(cdiv=lambda x,y:(x+y-1)//y),
        FwdConfig=namedtuple("FwdConfig","m n rs ol"), Int32=int,
        cute=NS(ceil_div=lambda x,y:(x+y-1)//y),const_expr=bool,
        cutlass=NS(min=min,max=max,const_expr=bool,range=lambda n,**kw:range(n)))
    def bind(path,name,owner=None):
        nodes=ast.parse(source(path)).body
        if owner: nodes=next(n for n in nodes if getattr(n,"name","")==owner).body
        node=next(n for n in nodes if getattr(n,"name","")==name)
        node.decorator_list=[]
        exec("from __future__ import annotations\n"+ast.unparse(node),ns)
        return ns[name]
    tile=bind(CUTE+"interface.py","_tile_size_fwd_sm90")
    split_count=bind(CUTE+"interface.py","num_splits_heuristic")
    bounds=bind(CUTE+"block_info.py","get_n_idx_left_right","BlockInfo")
    blocks=bind(CUTE+"block_info.py","get_n_block_min_max","BlockInfo")
    page_load=bind(CUTE+"paged_kv.py","load_page_table","PagedKVManager")
    wrapper=bind(WRAP,"flash_attn_with_kvcache")
    assert tile(128,128,False,True)[:2]==(128,128)
    canonical=bind(CUTE+"interface.py","_resolve_causal_local_window")
    assert all(canonical(False,w,w)==(False,True,w,w) for w in (0,1,127,511,2047))
    wrapper_src=source(WRAP)
    assert "seqused_k=cache_seqlens" in wrapper_src
    assert "seqlen_k = num_pages * page_size" in source(CUTE+"interface.py")
    torch.set_num_threads(1)
    rng=torch.Generator().manual_seed(255)
    width=8
    counts=Counter()
    cases=(0,1,7,127,128,129,2046,2047,2048,2174,2175,2176,2177,16439,32763,65529,131064)
    # Confirm that the production draft-block input satisfies the per-forward
    # gate. Capture uses the same width and sets custom_mask=None for a draft.
    ns.update(DFlashVerifyInput=lambda **kw:NS(**kw),CaptureHiddenMode=NS(NULL="null"))
    make_info=bind("python/sglang/srt/speculative/draft_worker_common.py","make_draft_block_spec_info")
    spec=make_info(draft_token_num=8,device=torch.device("cpu"))
    fixed=bind(BACK,"_fixed_draft_window","FlashAttentionBackend")
    host=NS(_draft_window_left=2047,speculative_num_draft_tokens=8)
    assert fixed(host,spec,None)==2047
    for info,enc in ((None,None),(spec,torch.tensor([0])),(NS(**{**vars(spec),"draft_token_num":4}),None),
                     (NS(**{**vars(spec),"custom_mask":torch.empty(0)}),None),
                     (NS(**vars(spec),ragged_verify_layout=object()),None)):
        assert fixed(host,info,enc)==-1
        counts["runtime_fallbacks"]+=1
    cap=source("python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py")
    assert "if (self.model_runner.is_draft_worker or not build_custom_mask)" in cap
    model=source("python/sglang/srt/models/dflash.py")
    assert "self.attn(q, k, v, forward_batch)" in model and "sinks=self.attention_sink_bias" in model
    for p in (*cases,*range(2032,2440),*range(4096,4352)):
        # Include two different model-wide maxima, heterogeneous inner radii and
        # all residues around each cutoff. All public layers use 2,047.
        for radius in (127,2047):
            origin=max(0,p-radius)//128*128
            stock=NS(seqlen_q=8,seqlen_k=p+8)
            small=NS(seqlen_q=8,seqlen_k=p+8-origin)
            for win in sorted({0,1,127,radius//2,radius}):
                for pack in (1,8):
                    info=NS(tile_m=128,tile_n=128,is_local=True,is_causal=False,
                        window_size_left=win,window_size_right=win,qhead_per_kvhead_packgqa=pack,is_split_kv=True)
                    for q in range(8):
                        a,b=bounds(info,stock,q*pack)
                        c,d=bounds(info,small,q*pack)
                        assert (a,min(b,stock.seqlen_k))==(origin+c,origin+min(d,small.seqlen_k))
                        counts["query_masks"]+=1
                    nblocks=(2*win+129+127)//128
                    for splits in {split_count(bs,132,nblocks,128) for bs in (1,4,16,32)}:
                        for j in range(splits):
                            a,b=blocks(info,stock,0,j,splits)
                            c,d=blocks(info,small,0,j,splits)
                            assert (a,b)==(origin//128+c,origin//128+d)
                            counts["ordered_split_intervals"]+=1
    # Other eligible model block widths can occupy several packed query tiles.
    # Split numbering and left/right bounds must also survive on those M tiles.
    for width_alt in (1,7,12,32,129):
        for p in (0,2047,2175,8191,65529):
            origin=max(0,p-2047)//128*128
            stock=NS(seqlen_q=width_alt,seqlen_k=p+width_alt)
            short=NS(seqlen_q=width_alt,seqlen_k=p+width_alt-origin)
            for pack in (1,8,32):
                for win in (0,511,2047):
                    info=NS(tile_m=128,tile_n=128,is_local=True,is_causal=False,window_size_left=win,
                        window_size_right=win,qhead_per_kvhead_packgqa=pack,is_split_kv=True)
                    for m in range((width_alt*pack+127)//128):
                        for splits in (1,4,8):
                            for j in range(splits):
                                a,b=blocks(info,stock,m,j,splits)
                                c,d=blocks(info,short,m,j,splits)
                                assert (a,b)==(c+origin//128,d+origin//128)
                                counts["other_width_multi_m_split_intervals"]+=1
    bind(META,"_resolve_metadata_page_ids")
    fused=bind(META,"_fused_decode_metadata_kernel")
    programs=[0]
    class Kernel:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                kwargs.pop("num_warps",None);kwargs.pop("num_stages",None)
                ptr=[Pointer(a.numpy().reshape(-1)) if isinstance(a,torch.Tensor) else a for a in args]
                for i in range(grid[0]):
                    for j in range(grid[1]):
                        tl.pid=(i,j);fused(*ptr,**kwargs);programs[0]+=1
            return launch
    ns["_fused_decode_metadata_kernel"]=Kernel()
    produce=bind(META,"normal_decode_set_metadata")
    # Changing logical request order and an independent physical permutation on each pass
    # expose stale pages or a mistaken second translation.
    extent=131072
    qrows=torch.tensor([2,0,1],dtype=torch.int64)
    req=torch.arange(3*extent,dtype=torch.int32).view(3,extent)+1
    mapped=torch.empty((3*extent+2,),dtype=torch.int64)
    mapped[-1]=-1
    physical=16384
    k=torch.randn((physical,1,1,12),dtype=torch.bfloat16,generator=rng)
    v=torch.randn((physical,1,1,12),dtype=torch.bfloat16,generator=rng)
    k[0]=0;v[0]=0
    q=torch.randn((3*width,1,12),dtype=torch.bfloat16,generator=rng)
    full_ids=torch.empty(3,2182,dtype=torch.int32);swa_ids=full_ids.clone()
    lens=torch.empty(3,dtype=torch.int32);cu=torch.empty(4,dtype=torch.int32)
    pointers=tuple(tensor.data_ptr() for tensor in (lens,cu,full_ids,swa_ids))
    states=((0,2047,131064),(4096,127,16388),(8191,2176,16),(2048,0,32765))
    for step,row_lengths in enumerate(states):
        p=torch.tensor(row_lengths,dtype=torch.int32)
        order=qrows.roll(step)
        mapped[:-1].random_(1,physical,generator=rng);mapped[0]=0
        # Poison compact slack and clear a mix of mapped physical slots,
        # including some in the retained prefix. The reference sees the same
        # sink for a missing slot. Valid serving live rows have physical owners.
        for b,prefix in enumerate(row_lengths):
            idx=int(order[b])
            for c in {0,max(0,prefix-2077),max(0,prefix-127),prefix+5}:
                mapped[req[idx,c]]=0
        full_ids.fill_(-6231);swa_ids.fill_(-7193)
        produce(seq_lens=p,req_to_token=req,req_pool_indices=order,cache_seqlens_int32=lens,
            cu_seqlens_k=cu,page_table=full_ids,seq_len_delta=width,page_size=1,
            max_seq_pages=full_ids.size(1),swa_page_table=swa_ids,full_to_swa_index_mapping=mapped,window_left=2047)
        assert tuple(t.data_ptr() for t in (lens,cu,full_ids,swa_ids))==pointers
        expected_lens=[]
        dense=torch.zeros((3,max(row_lengths)+width),dtype=torch.int32)
        for b,prefix in enumerate(row_lengths):
            raw=req[order[b],:prefix+width]
            physical_ids=mapped[raw.long()].int()
            dense[b,:raw.numel()]=physical_ids
            origin=max(0,prefix-2047)//128*128
            suffix=slice(origin,None);l=raw.numel()-origin;expected_lens.append(l)
            assert torch.equal(full_ids[b,:l],raw[suffix])
            assert torch.equal(swa_ids[b,:l],physical_ids[suffix])
            counts["physical_owner_rows"]+=l
            # The compact eager rail has no allocation extending it through
            # the tile end. Execute SM90's per-thread paged loader on the last
            # tile against a guard-page-like strict table; no tail load may
            # use the poisoned suffix or run off the allocation.
            class Checked:
                def __getitem__(self,i):
                    assert 0<=i<l
                    return int(swa_ids[b,i])
            block=(l-1)//128
            for t in range(128):
                loader=NS(page_entry_per_thread=1,num_threads=128,thread_idx=t,gmem_threads_per_row=8,
                    n_block_size=128,leftpad_k=0,page_size_divmod=1,seqlen_k=l,mPageTable=Checked(),
                    tPrPage=[None],tPrPageOffset=[None])
                page_load(loader,block)
                c=block*128+(t%8)*16+t//8
                assert loader.tPrPage[0]==(int(physical_ids[origin+c]) if c<l else 0)
                counts["sm90_tail_page_lanes"]+=1
        assert lens.tolist()==expected_lens and cu.tolist()==[0,*np.cumsum(expected_lens)]
        for radius in (0,7,127,511,2047):
            # Observe the real FA4 wrapper passing identities and lengths to
            # the device boundary. The substitute implements local attention
            # from native CuTe query intervals over the physical K/V address rail.
            info=NS(is_local=True,is_causal=False,window_size_left=radius,window_size_right=radius,qhead_per_kvhead_packgqa=1)
            def attention(**kw):
                assert kw["window_size"]==(radius,radius) and kw["causal"] is False
                assert kw["k"] is k and kw["v"] is v and kw["q"] is q
                assert kw["max_seqlen_q"]==width and "max_seqlen_k" not in kw
                values=[]
                for row,length in enumerate(kw["seqused_k"].tolist()):
                    geom=NS(seqlen_q=width,seqlen_k=length)
                    for t in range(width):
                        lo,hi=bounds(info,geom,t)
                        ids=kw["page_table"][row,lo:min(hi,length)].long()
                        assert (ids>=0).all() and (ids<physical).all()
                        a=k[ids,0,0].float();wts=torch.softmax(a @ q[row*width+t,0].float()/12**0.5,dim=0)
                        values.append((wts@v[ids,0,0].float()).bfloat16())
                return torch.stack(values)
            ns["flash_attn_varlen_func"]=attention
            common=dict(q=q,k_cache=k,v_cache=v,max_seqlen_q=width,causal=False,window_size=(radius,radius))
            base=wrapper(**common,page_table=dense,cache_seqlens=p+width)
            got=wrapper(**common,page_table=swa_ids,cache_seqlens=lens)
            assert torch.equal(base,got)
            counts["fa4_wrapper_calls"]+=2;counts["attention_bf16_outputs"]+=got.numel()
    return dict(patch_sha256=patch_sha,source_sha256=sources,counts=dict(counts),programs=programs[0],
        fresh_mapping_per_pass=len(states),prefix_max=max(max(r) for r in states),
        scope="Native mask, tile, split arithmetic, fused Triton body and FA4 wrapper executed on CPU. A CPU BF16 output reference replaces FA4 device arithmetic. Leading zero-mapped slots and empty/short prefixes are compared to native address semantics. No physical H200 run.")

if __name__=="__main__":
    args=parse_args()
    if args.manifest.is_file() and args.manifest.stat().st_size:
        manifest=json.loads(args.manifest.read_bytes())
        patch=args.manifest.with_name("patch.diff")
        patch_sha=sha256(patch.read_bytes()).hexdigest()
        assert patch_sha==manifest["patch_sha256"]
        with tempfile.TemporaryDirectory(prefix="okc-v255-independent-") as td:
            root=Path(td)
            for path in (META,BACK):
                dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(subprocess.check_output(["git","-C",BASE,"show",f"{PIN}:{path}"]))
            subprocess.run(["git","apply","--no-index",f"--include={META}",f"--include={BACK}",str(patch.resolve())],
                cwd=root,check=True)
            if args.stage:
                assert all((args.stage/name).read_bytes()==(root/name).read_bytes() for name in (META,BACK))
            receipt=run(root,manifest,patch_sha)
        args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    else:
        assert args.stage and not args.output.exists()
        receipt=run(args.stage,None,None)
    print(receipt["counts"])
