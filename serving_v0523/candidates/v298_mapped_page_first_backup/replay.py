"""Execute source-bound MHA backup, pool selection, scatter and per-layer restore on CPU."""
from contextlib import contextmanager
from types import SimpleNamespace as S
import ast
import json
import torch
from verify import BASE, COMMON, MHA, candidate, digest

def expose(src, names, env, cls=None):
    root=ast.parse(src)
    if cls:root=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name==cls)
    result={}
    for name in names.split():
        fn=next(n for n in root.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec("from __future__ import annotations\n"+ast.unparse(fn),env)
        result[name]=env[name]
    return result

def verify_mappings(src):
    env=dict(torch=torch,_CUDA_HOST_REGISTERED_RANGES_ATTR="native_ranges")
    valid=expose(src,"has_linear_host_mapping",env)["has_linear_host_mapping"]
    buf=torch.empty(36,dtype=torch.uint8)
    base=buf.data_ptr()
    results={}
    for case in ("identity","translated","discontinuous","gap","overlap","short","empty","missing","unmapped"):
        spans=[(0,4),(4,17),(21,15)]
        if case=="gap":spans[1]=(5,16)
        if case=="overlap":spans[1]=(3,18)
        if case=="short":spans[-1]=(21,14)
        if case=="empty":spans=[]
        buf.native_ranges=[(base+i,n) for i,n in spans]
        def lookup(t,device):
            assert device==2
            if case=="unmapped":raise RuntimeError("unmapped")
            return t.data_ptr()+8192*(case=="translated" or (case=="discontinuous" and t.data_ptr()>base))
        env["_resolve_device_accessible_ptr_fn"]=lambda:None if case=="missing" else lookup
        results[case]=valid(buf,torch.device("cuda:2"))
        assert results[case]==(case in ("identity","translated"))
    del buf.native_ranges
    results["no registration"]=valid(buf,torch.device("cuda:2"))
    assert not results["no registration"]
    return results

def main():
    torch.set_num_threads(1)
    host=BASE+"pool_host/base.py";engine=BASE+"l2_transfer.py"
    wrap="python/sglang/kernels/ops/kvcache/hicache.py"
    kernel="python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh"
    with candidate() as (_,tree,git):
        src={p:git("show",tree+":"+p).decode() for p in (host,engine,wrap,kernel,MHA,COMMON)}
    env=dict(torch=torch,debug_kernel_api=lambda f:f,_default_unroll=lambda size:4,DEFAULT_BLOCK_QUOTA=2)
    pool_methods=expose(src[MHA],"backup_from_device_all_layer load_to_device_per_layer",env,"MHATokenToKVPoolHost")
    pool_methods.update(expose(src[host],"backup_from_device_all_layer_physical load_to_device_per_layer_physical",env,"HostKVCache"))
    Host=type("Host",(),pool_methods)
    env["jit_transfer_hicache_all_layer_staged_lf_pf"]=expose(src[wrap],"transfer_hicache_all_layer_staged_lf_pf",env)["transfer_hicache_all_layer_staged_lf_pf"]
    events=[]
    @contextmanager
    def submit(self,transfers,stream,name,start_event=None):
        events.append(stream+" wait"); yield transfers,name; events.append(stream+" complete")
    engine_methods=expose(src[engine],"submit_device_to_host submit_host_to_device",env,"L2TransferEngine")
    engine_methods.update(_submission=submit,io_backend="kernel",device_to_host_stream="write",host_to_device_stream="read")
    run=type("Engine",(),engine_methods)()
    env["envs"]=S(SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES=S(get=lambda:0))
    results=[]
    for count,layers,width,dtype,case in (
      (16384,16,128,torch.uint8,"ascending"),(513,46,128,torch.uint8,"gaps"),(769,5,128,torch.bfloat16,"ascending"),
      (130,4,128,torch.uint8,"descending"),(181,3,128,torch.uint8,"duplicate"),(127,3,128,torch.uint8,"ascending"),
      (128,3,128,torch.uint8,"ascending"),(257,2,128,torch.uint8,"mapping refused"),(129,4,128,torch.uint8,"strided"),
      (128,3,128,torch.uint8,"mixed dtype"),(513,3,128,torch.uint8,"layer_first"),(0,3,128,torch.uint8,"ascending")):
        h=Host();h.page_size=h.dcp_size=1;h.layout="layer_first" if case=="layer_first" else "page_first"
        h.can_use_jit=h.can_use_write_back_jit=h.device_rows_packed=True;h.can_use_mapped_backup=case!="mapping refused"
        h._transfer_device=torch.device("cpu");h.staging_page_capacity=64
        h.dtype=dtype;h.layer_num=h.target_layer_num=layers;h.element_dim=width
        h.token_stride_size=h.device_row_stride_bytes=width*dtype.itemsize;h.layout_dim=layers*h.token_stride_size
        slots=count*(2 if case=="gaps" else 1)+7
        shape=(2,slots,layers,width) if h.layout=="page_first" else (2,layers,slots,width)
        storage=torch.full(shape,169,dtype=dtype)
        h.k_buffer,h.v_buffer=storage
        tables=storage.transpose(1,2) if h.layout=="page_first" else storage
        h.k_data_refs,h.v_data_refs=tables
        h.k_data_ptrs,h.v_data_ptrs=tables
        gen=torch.Generator().manual_seed(722)
        payload=torch.randint(0,256,(2,layers,slots,width*dtype.itemsize),dtype=torch.uint8,generator=gen).view(dtype)
        h.device_k_buffers,h.device_v_buffers=[tuple(x) for x in payload]
        h.device_k_data_ptrs,h.device_v_data_ptrs=h.device_k_buffers,h.device_v_buffers
        h.staging_k_buffer,h.staging_v_buffer=torch.empty(2,64,layers,width,dtype=dtype)
        source=torch.randperm(slots,generator=gen)[:count]
        dest=torch.arange(count)+1
        if case=="gaps":dest=2*dest
        if case=="descending":dest=dest.flip(0)
        if case=="duplicate":dest[17]=dest[4]
        if case=="strided":
            t=torch.empty(2*count,dtype=source.dtype);t[::2]=source;source=t[::2]
        if case=="mixed dtype":source=source.int()
        calls=[]
        def direct(**kw):
            calls.append("mapped" if h.layout=="page_first" else "native layer-first")
            assert kw["kv_cache_dst_stride_bytes"]==tables.stride(2)*dtype.itemsize
            assert kw["kv_cache_src_stride_bytes"]==payload.stride(2)*dtype.itemsize
            for dst,origin in ((kw["k_ptr_dst"],kw["k_ptr_src"]),(kw["v_ptr_dst"],kw["v_ptr_src"])):
                for l in range(layers):dst[l][kw["indices_dst"]]=origin[l][kw["indices_src"].long()]
        def staged(k,v,ids,sk,sv,indices,ks,vs,ps):
            calls.append("staged")
            for j,row in enumerate(ids):
                for l in range(layers):
                    k[row,l]=ks[l][indices[j]];v[row,l]=vs[l][indices[j]]
        def restore(**kw):
            for s in ("k","v"):kw[s+"_cache_dst"][kw["indices_dst"].long()]=kw[s+"_cache_src"][kw["indices_src"]]
        env.update(jit_transfer_hicache_all_layer=direct,_jit_hicache_staged_module=lambda **_:S(launch_all_lf_pf_staged=staged),
                   jit_transfer_hicache_one_layer=restore)
        tx=S(host_pool=h,device_pool=None,host_indices=dest,device_indices=source,layer_mapper=None,is_draft=False)
        events.clear();run.submit_device_to_host([tx])
        assert events==["write wait","write complete"]
        want="native layer-first" if case=="layer_first" else ("mapped" if case in ("ascending","gaps") and count>=128 else "staged")
        assert calls==([want] if want!="staged" else [want]*((count+63)//64)),(case,calls)
        for side in range(2):
            for l in range(layers):
                expected=torch.full_like(payload[side,l],169)
                if case=="duplicate":
                    for i,j in zip(source,dest):expected[j]=payload[side,l,i]
                else:expected[dest]=payload[side,l,source.long()]
                assert torch.equal(tables[side,l].view(torch.uint8),expected.view(torch.uint8))
        n=min(count,2);payload[:,:,:n]=0
        back=S(**dict(vars(tx),host_indices=dest[:n],device_indices=torch.arange(n)))
        done=[];run.submit_host_to_device([back],transfer_layer_id_max=layers,on_layer_done=done.append)
        assert done==list(range(layers)) and events[-2:]==["read wait","read complete"]
        assert torch.equal(payload[:,:,:n].view(torch.uint8),tables[:,:,dest[:n]].view(torch.uint8))
        results.append(dict(case=case,rows=count,layers=layers,execution=want,operator_calls=len(calls),
                            checked_host_bytes=storage.nbytes,checked_device_bytes=2*layers*n*width*dtype.itemsize))
    print(json.dumps(dict(tree=tree,source_sha256={k:digest(v.encode()) for k,v in src.items()},mapping=verify_mappings(src[COMMON]),
                    cases=results,checked_bytes=sum(r["checked_host_bytes"]+r["checked_device_bytes"] for r in results),
                    scope="Native transfer and MHA pool dispatch, staged slicing, per-layer restoration, CPU stand-ins for streams and CUDA launches."),indent=2))

if __name__=="__main__":main()
