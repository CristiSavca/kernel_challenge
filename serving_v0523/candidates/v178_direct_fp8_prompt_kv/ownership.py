"""Execute the pinned page-one allocation, suffix publication, and full/SWA read translation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as S

import torch

repo = Path(sys.argv[1])
pin = "297e54fd105e974cdf401e5b4d20253a22c16f90"
ns=dict(torch=torch, support_triton=lambda *_:False, attention_backends=lambda:("cpu",None))
sources={}


def read(path):
    filename="python/sglang/"+path
    data=subprocess.check_output(["git","-C",str(repo),"show",pin+":"+filename])
    sources[filename]=hashlib.sha256(data).hexdigest()
    return data.decode()


def methods(src,cls,*wanted):
    nodes=ast.parse(src).body
    if cls:
        nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==cls).body
    result={}
    for name in wanted:
        n=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
        exec("from __future__ import annotations\n"+ast.unparse(n),ns)
        result[name]=ns[name]
    return result


allocation=read("srt/mem_cache/allocation.py")
kernel=read("kernels/ops/memory/common.py")
token=read("srt/mem_cache/allocator/token.py")
allocator=read("srt/mem_cache/allocator/swa.py")
pool=read("srt/mem_cache/swa_memory_pool.py")
backend=read("srt/layers/attention/flashattention_backend.py")
# Both the GPU writer and the executable non-Triton implementation take their new suffix IDs
# from the same out_cache_loc. The attention table and the SWA writer translate through one map.
extract=methods(allocation,None,"write_cache_indices")["write_cache_indices"]
alloc=ast.unparse(next(n for n in ast.parse(allocation).body if getattr(n,"name",None)=="alloc_for_extend"))
assert "write_cache_indices(out_cache_loc," in alloc and "return (out_cache_loc," in alloc
fn=ast.unparse(next(n for n in ast.parse(kernel).body if getattr(n,"name",None)=="write_req_to_token_pool_triton"))
assert "tl.load(out_cache_loc +" in fn and "tl.store(" in fn and "offset + pre_len" in fn
assert "swa_out_cache_loc = swa_out_cache_loc" in backend and "translate_loc_from_full_to_swa(" in backend
assert "get" in read("srt/layers/radix_attention.py") and "q[:n]" in read("srt/layers/radix_attention.py")
pgraph=read("srt/model_executor/runner/prefill_cuda_graph_runner.py")
assert "extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu" in pgraph
assert "extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu" in pgraph and "seq_lens_cpu=forward_batch.seq_lens_cpu" in pgraph
Token=type("Token",(),methods(token,"TokenToKVPoolAllocator","clear","alloc","available_size"))
Pair=type("Pair",(),methods(allocator,"SWATokenToKVPoolAllocator","alloc","set_full_to_swa_mapping"))
translate=methods(pool,"SWAKVPool","translate_loc_from_full_to_swa")["translate_loc_from_full_to_swa"]
torch.manual_seed(391)
cases=0;count=0
for n in (1,7,513,4096,16384):
    for p in (0,31,2048):
        idx=3
        total=n+p
        a,b=Token(),Token()
        for ar in (a,b):
            ar.size=total+17; ar.device="cpu"; ar.need_sort=False; ar.clear()
            ar.free_pages=ar.free_pages[torch.randperm(ar.free_pages.numel())]
        pair=Pair()
        pair.page_size=1;pair.full_attn_allocator=a;pair.swa_attn_allocator=b
        pair.full_to_swa_index_mapping=torch.zeros(total+19,dtype=torch.int64)
        past=pair.alloc(p);new=pair.alloc(n)
        storage=torch.full((5,total+7),-3,dtype=torch.int32)
        arr=lambda x:torch.tensor([x],dtype=torch.int64)
        req=S(write=lambda pos,value:storage[pos].copy_(value))
        extract(new,arr(idx),arr(idx),arr(p),arr(p),arr(total),arr(total),arr(n),arr(n),[past],req)
        f=storage[idx,:total]
        s=translate(pair,f.long()).int()
        sw=translate(pair,new)
        assert torch.equal(f[p:],new.int()) and torch.equal(s[p:],sw.int())
        assert torch.equal(f[:p],past.int())
        for t in (f,s):
            assert (t>0).all() and t.unique().numel()==len(t)
        assert torch.all(storage[idx,total:]==-3)
        cases+=1;count+=2*total
print(json.dumps(dict(pin=pin,real_paired_allocations=cases,distinct_positive_slots_validated=count,
                      triton_publication_source_checked=True,breakable_real_row_scope_and_host_mirrors_checked=True,
                      source_sha256=sources),indent=2))
