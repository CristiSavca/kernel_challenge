"""Replay host allocator → MHA device backup → staged multi-chunk transfer with CPU tensors."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from functools import wraps
import torch

BASE="297e54fd105e974cdf401e5b4d20253a22c16f90"
HERE=Path(__file__).resolve().parent
SG=HERE.parents[1]/"sglang"
REL="python/sglang/"
BASE_PATH=REL+"srt/mem_cache/pool_host/base.py"
HOST_PATH=REL+"srt/mem_cache/pool_host/mha.py"
WRAP_PATH=REL+"kernels/ops/kvcache/hicache.py"


def source(file):
    return subprocess.check_output(["git","-C",str(SG),"show",BASE+":"+file],text=True)


def functions(txt, env, names, cls=None):
    owner=ast.parse(txt)
    if cls is not None: owner=next(n for n in owner.body if isinstance(n,ast.ClassDef) and n.name==cls)
    for name in names:
        node=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec("from __future__ import annotations\n"+ast.unparse(node),env)


def test():
    torch.set_num_threads(1)
    sources={f:source(f) for f in (BASE_PATH,HOST_PATH,WRAP_PATH)}
    env=dict(torch=torch,wraps=wraps,debug_kernel_api=lambda fn:fn,DEFAULT_BLOCK_QUOTA=12,
             _default_unroll=lambda width:1)
    functions(sources[BASE_PATH],env,["synchronized"])
    base_namespace=dict(env)
    fields=("clear","available_size","_merge_release_slots","logical_size","logical_page_size","alloc","free")
    functions(sources[BASE_PATH],base_namespace,fields,"HostKVCache")
    HostAlloc=type("HostAlloc",(),{k:base_namespace[k] for k in fields})
    functions(sources[WRAP_PATH],env,["transfer_hicache_all_layer_staged_lf_pf"])
    env["jit_transfer_hicache_all_layer_staged_lf_pf"]=env["transfer_hicache_all_layer_staged_lf_pf"]
    functions(sources[HOST_PATH],env,["backup_from_device_all_layer","load_to_device_per_layer"],"MHATokenToKVPoolHost")
    fallback=[]
    def record(name):
        def observe(*args,**kw):fallback.append(name)
        return observe
    for name in ("jit_transfer_hicache_all_layer","transfer_kv_all_layer","transfer_kv_all_layer_lf_pf",
                 "transfer_kv_all_layer_lf_ph","transfer_kv_direct","transfer_kv_all_layer_direct_lf_pf",
                 "jit_transfer_hicache_one_layer","transfer_kv_per_layer_lf_lf","transfer_kv_per_layer_pf_lf"):
        env[name]=record(name)
    Host=type("Host",(HostAlloc,),{k:env[k] for k in ("backup_from_device_all_layer","load_to_device_per_layer")})
    cases=[]
    def run(page, count, layers, width, layout, dtype):
        h=Host();h.dcp_size=1;h.page_size=page;h.device="cpu";h.lock=threading.RLock()
        h.size=(2*count+13)*page;h.clear()
        if layout in ("fragmented","mixed-runs"):
            all_slots=h.alloc(h.size).reshape(-1,page)
            indices_to_free=(all_slots[4::3],all_slots[2::3],all_slots[:1]) if layout=="fragmented" else (all_slots[3:17],all_slots[25:32],all_slots[50:97],all_slots[115:])
            for indices in indices_to_free:
                if indices.numel()>0:h.free(indices.reshape(-1))
            dst=h.alloc(count*page); assert dst is not None
            if count>4: assert torch.any(torch.diff(dst[::page])!=page)
        else:
            h.alloc(3*page);dst=h.alloc(count*page)
        assert h.slot_used[dst].all()
        total_phys=2*count+13
        order=torch.randperm(total_phys,generator=torch.Generator().manual_seed(83+count))[:count]*page
        ids=(order[:,None]+torch.arange(page)).flatten()
        # Noncontiguous source index tensor with real strides; the wrapper contiguates page headers.
        padded=torch.empty((2*len(ids),),dtype=torch.int64);padded[::2]=ids;ids=padded[::2]
        h.layout="page_first";h.can_use_write_back_jit=True;h.can_use_jit=True;h.device_rows_packed=True
        h.dtype=dtype;h.layer_num=h.target_layer_num=layers;h.element_dim=width
        h.layout_dim=h.token_stride_size=h.device_row_stride_bytes=width*torch.empty((),dtype=dtype).element_size()
        h.head_num=2;h.host_kv_data_refs=[];h.k_data_ptrs=object();h.v_data_ptrs=object()
        h._require_packed_device_rows=lambda _name:None
        num_data=h.size*width
        def bank(lane):
            return [(((torch.arange(num_data,dtype=torch.int32).view(h.size,width)*17+l*31+lane*67)%251).to(dtype)) for l in range(layers)]
        h.device_k_buffers=bank(0);h.device_v_buffers=bank(1)
        h.device_k_data_ptrs=h.device_k_buffers;h.device_v_data_ptrs=h.device_v_buffers
        h.k_buffer=torch.full((h.size,layers,width),241,dtype=dtype)
        h.v_buffer=torch.full_like(h.k_buffer,242)
        h.staging_k_buffer=torch.empty((page*64,layers,width),dtype=dtype)
        h.staging_v_buffer=torch.empty_like(h.staging_k_buffer)
        expected=[]
        for src,target in ((h.device_k_buffers,h.k_buffer),(h.device_v_buffers,h.v_buffer)):
            oracle=target.clone()
            oracle.index_copy_(0,dst,torch.stack([layer.index_select(0,ids) for layer in src],dim=1))
            expected.append(oracle)
        chunks=[];runs=[]
        def launch(dk,dv,target_ids,sk,sv,src_pages,kbank,vbank,pg):
            assert pg==page and len(src_pages)<=64 and src_pages.is_contiguous()
            chunks.append(len(src_pages))
            slot=(src_pages[:,None]+torch.arange(page)).flatten()
            t=target_ids.view(-1,page)
            assert torch.equal(t,t[:,:1]+torch.arange(page))
            boundaries=[0]+[i for i in range(1,len(t)) if int(t[i,0])!=int(t[i-1,0])+page]+[len(t)]
            runs.append(len(boundaries)-1)
            for host,stage,bank in ((dk,sk,kbank),(dv,sv,vbank)):
                for layer, values in enumerate(bank):
                    stage[:,layer].copy_(values.index_select(0,slot))
                for first,end in zip(boundaries,boundaries[1:]):
                    host.narrow(0,int(t[first,0]),(end-first)*page).copy_(stage[first*page:end*page])
        env["_jit_hicache_staged_module"]=lambda **kw:SimpleNamespace(launch_all_lf_pf_staged=launch)
        h.backup_from_device_all_layer(None,dst,ids,"kernel")
        assert chunks==([64]*(count//64)+([count%64] if count%64 else []))
        for original,got in zip(expected,(h.k_buffer,h.v_buffer)):
            assert torch.equal(original.view(torch.uint8),got.view(torch.uint8))
        assert not fallback
        cases.append(dict(page=page,pages=count,layers=layers,width=width,dtype=str(dtype),layout=layout,
                          chunk_sizes=chunks,copies_before=2*count,copies_with_coalescing=2*sum(runs),
                          compared_bytes=2*h.k_buffer.numel()*h.k_buffer.element_size()))
        return h,dst,ids
    for q,dtype in ((16,torch.uint8),(46,torch.uint8),(5,torch.bfloat16)):
        for count in (0,1,63,64,65,129):
            run(1,count,q,128,"ascending",dtype)
    for page,n in ((1,63),(1,65),(1,191),(4,65),(4,131),(16,3)):
        run(page,n,3,16,"fragmented",torch.uint8)
    for pg in (1,4):
        run(pg,133,3,16,"mixed-runs",torch.uint8)
    h,dst,ids=run(1,7,3,16,"ascending",torch.uint8)
    for layout,backend,staged,packed,expected in (
        ("page_first","kernel",False,True,"transfer_kv_all_layer_lf_pf"),
        ("page_first","kernel",False,False,"jit_transfer_hicache_all_layer"),
        ("layer_first","kernel",True,True,"jit_transfer_hicache_all_layer"),
        ("layer_first","direct",True,True,"transfer_kv_direct"),
        ("page_first_direct","direct",True,True,"transfer_kv_all_layer_direct_lf_pf"),
        ("page_head","kernel",True,True,"transfer_kv_all_layer_lf_ph")):
        h.layout=layout;h.can_use_write_back_jit=staged;h.device_rows_packed=packed
        h.backup_from_device_all_layer(None,dst,ids,backend)
        assert fallback[-1]==expected,(layout,backend,fallback)
    h.layout="page_first";h.device_rows_packed=True;h.can_use_write_back_jit=True
    h.device_row_stride_bytes=h.token_stride_size
    h.k_data_refs=h.k_buffer.transpose(0,1);h.v_data_refs=h.v_buffer.transpose(0,1)
    h.load_to_device_per_layer(None,dst,ids,0,"kernel")
    assert fallback[-1]=="jit_transfer_hicache_one_layer"
    return dict(allocator="pinned HostKVCache.alloc/free and ordered release queue",normal_entry="pinned MHATokenToKVPoolHost with actual multi-chunk wrapper",
                backend="CPU tensor mock at CUDA JIT launch interface; independent full-byte slot-index oracle",
                stage_capacity_pages=64,layout_and_restore_controls=fallback,cases=cases,
                all_compared_bytes=sum(c["compared_bytes"] for c in cases),
                pinned_sources={path:hashlib.sha256(text.encode()).hexdigest() for path,text in sources.items()},hardware=False)


if __name__=="__main__":print(json.dumps(test(),indent=2))
