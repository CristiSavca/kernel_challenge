"""Compare V133 prompt cache bytes to the pinned native MHA store and exercise its host dispatch."""
import argparse
import ast
import collections
import contextlib
import ctypes
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import torch

CASES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASES / "v104_v64_direct_draft_kv"))
import bytes_cpu as evaluator
from host_dispatch import method

MODULE = "python/sglang/kernels/ops/speculative/fused_kv_materialize.py"
WORKER = "python/sglang/srt/speculative/dflash_worker_v2.py"
POOL = "python/sglang/srt/mem_cache/memory_pool.py"
STORE = "python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh"
PREFIX = "python/sglang/kernels/ops/kvcache/cache_move.py"
PTR = evaluator.Pointer


def pinned(root, path):
    return subprocess.check_output(["git", "-C", root, "show", "HEAD:"+path])


def native_store(root, tmp):
    # Compile the source's actual index checks, reserved slot branch and pointer
    # arithmetic. Mock only CUDA/PDL and the coalesced warp byte-copy primitive.
    src = pinned(root, STORE).decode()
    src = src[src.index("struct StoreKVCacheParams"):src.index("template <int64_t kKBytes, int64_t kVBytes, uint32_t kNumThreads, bool kUsePDL>\nstruct StoreKVCacheKernel")]
    shim = r"""
#include <array>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#define assert(x) do { if (!(x)) throw std::out_of_range("bad slot"); } while(false)
#define __global__
#define __grid_constant__
struct Idx {int x, y;}; Idx blockIdx, threadIdx, blockDim;
namespace device {
constexpr int kWarpThreads=32;
template<bool B> void PDLWaitPrimary(){}
template<bool B> void PDLTriggerSecondary(){}
}
namespace warp {
struct LoadStorePattern {enum class type {WARP_UNIFORM_16B};};
template<int bytes,LoadStorePattern::type pattern>
std::array<uint8_t,bytes> load_bytes(const void* p) {
 std::array<uint8_t,bytes> r; std::memcpy(r.data(),p,bytes); return r;
}
template<int bytes,LoadStorePattern::type pattern>
void store_bytes(void* p, const std::array<uint8_t,bytes>& v) { std::memcpy(p,v.data(),bytes); }
}
namespace pointer {
template<class T> auto offset(T* p,int64_t a,int64_t b) {
  using V=std::conditional_t<std::is_const_v<T>,const uint8_t,uint8_t>;
  return reinterpret_cast<V*>(p)+a+b;
}
}
"""
    driver = r"""
template<int B> void run(const StoreKVCacheParams& p) {
  blockDim.y=2;
  for(unsigned row=0;row<p.batch_size;++row) {
    blockIdx.x=row;
    for(threadIdx.y=0;threadIdx.y<2;++threadIdx.y) store_kvcache_kernel<B,B,64,false,int64_t>(p);
  }
}
extern "C" int hoststore(const void* k,const void* v,void* kc,void* vc,const void* ids,
  int64_t sk,int64_t sv,int64_t skc,int64_t svc,int64_t si,int n,int limit,int bytes) {
  StoreKVCacheParams p{k,v,kc,vc,ids,sk,sv,skc,svc,si,(uint32_t)n,limit,0};
  try {
    switch(bytes) {
      case 64:run<64>(p);break;case 96:run<96>(p);break;case 128:run<128>(p);break;
      case 192:run<192>(p);break;case 256:run<256>(p);break;case 384:run<384>(p);break;
      default:std::abort();
    }
  } catch(const std::out_of_range&) {return 1;}
  return 0;
}
"""
    cpp = tmp / "store.cpp"; cpp.write_text(shim+src+driver)
    dll = tmp / "store.so"
    subprocess.run(["c++","-std=c++20","-O2","-shared","-fPIC",str(cpp),"-o",str(dll)],check=True)
    op = ctypes.CDLL(str(dll)).hoststore
    op.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int64]*5 + [ctypes.c_int]*3
    op.restype = ctypes.c_int

    def copy(k,v,kc,vc,ids,*,row_bytes,v_row_bytes,size_limit):
        assert row_bytes==v_row_bytes and ids.dtype==torch.int64
        tensors=(k,v,kc,vc,ids)
        err=op(*(t.data_ptr() for t in tensors),*(t.stride(0)*t.element_size() for t in tensors[:4]),
               ids.stride(0),ids.numel(),size_limit,row_bytes)
        if err:
            raise IndexError("native store bounds")
    return copy


def actual_pool_writer(root, tmp, native):
    path=tmp/"memory_pool.py"; path.write_bytes(pinned(root,POOL))
    # Keep native set_kv_buffer, NHD physical dispatch and the native store.
    # The CPU cases provide the same flags and physical-location object.
    scope=dict(torch=torch,unwrap_write_loc=lambda loc:(loc,None,None),
               maybe_detect_oob=lambda *args:None,_is_cuda=True,_is_hip=False,
               can_use_store_cache=lambda *args:True,store_cache=native)
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if getattr(n,"name",None)=="_set_kv_buffer_impl")
    ns=ast.parse("from __future__ import annotations"); ns.body.append(node)
    exec(compile(ns,str(path),"exec"),scope)
    store=method(path,"MHATokenToKVPool","_store_kv_layer",scope)
    setbuf=method(path,"MHATokenToKVPool","set_kv_buffer",scope)
    def run(pool,layer,loc,k,v):
        pool._store_kv_layer=lambda *args:store(pool,*args)
        setbuf(pool,NS(layer_id=layer),loc,k,v,2.0,0.5)
    return run


def prompt_bytes(root, stock, tmp):
    pid = [0,0,0]
    stock_op=evaluator.source(stock,{"_fused_norm_rope_kernel_stacked"},pid)._fused_norm_rope_kernel_stacked
    new_op=evaluator.source(root/MODULE,{"_cache_round","_fused_norm_rope_kernel_stacked"},pid)._fused_norm_rope_kernel_stacked
    def inline(asm,constraints,args,dtype,is_pure,pack):
        assert asm == "{ .reg .pred bad; setp.ge.u64 bad, $1, $2; @bad trap; mov.b64 $0, $1; }"
        assert constraints == "=l,l,l" and dtype==torch.int64 and not is_pure and pack==1 and len(args)==2
        slot=args[0]
        if len(args)==2 and not (0<=int(slot)<int(args[1])):raise IndexError("direct store bounds")
        if len(args)==1:raise IndexError("direct store bounds")
        return slot
    new_op.__globals__["tl"].inline_asm_elementwise=inline
    native=native_store(root,tmp)
    write=actual_pool_writer(root,tmp,native)
    rng=torch.Generator().manual_seed(133)
    torch.set_num_threads(1)
    metric=collections.Counter()
    for m,n,heads,dim,rot in ((1,5,1,128,128),(7,2,2,64,32),(19,2,1,96,48),(33,3,3,32,24),(65,1,1,32,16)):
        width=heads*dim
        ctxstride, layerstride, nstride, rstride = n*(2*width+8)+16, 2*width+8, dim+6, rot+4
        proj=(torch.randn(m*ctxstride,generator=rng)*14).bfloat16()
        trig=torch.randn(196*rstride,generator=rng).bfloat16()
        norm=torch.randn(n*nstride,generator=rng).bfloat16()
        pos=torch.randint(0,195,(m,),generator=rng)
        eps=torch.linspace(1e-6,0.0002,n)
        # Include raw special BF16 V patterns. K goes through real RMSNorm and RoPE.
        special=torch.tensor([0,0x8000,0x7f80,0xff80,0x7fc1,0xffd4,0x3f80,0x47ff],dtype=torch.int32).to(torch.int16)
        proj[width:width+8].view(torch.int16).copy_(special)
        tk=torch.empty((n,m+11,heads,dim),dtype=torch.bfloat16);tv=torch.empty((n,m+17,heads,dim),dtype=torch.bfloat16)
        shared=list(map(PTR,(proj,norm,eps,trig,pos)))
        strides=[ctxstride,layerstride,nstride,rstride]
        spec=[m,n,heads,dim,width,rot,rot//2,1<<(dim-1).bit_length()]
        grid=[(i,j,k) for k in range(n) for j in range(heads) for i in range(m)]
        for it in grid:
            pid[:]=it
            stock_op(*shared,PTR(tk),PTR(tv),*strides,*tk.stride()[:3],*tv.stride()[:3],*spec)
        capacity=3*m+17
        for layout in ("sequential","holes","zero_repeat","all_padding"):
            k=torch.randint(0,65536,(n,capacity+4,heads,dim),generator=rng,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
            v=torch.randint(0,65536,(n,capacity+7,heads,dim),generator=rng,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
            dk,dv=k.clone(),v.clone()
            key=list(k[:,2:2+capacity]);value=list(v[:,3:3+capacity])
            direct_k=list(dk[:,2:2+capacity]);direct_v=list(dv[:,3:3+capacity])
            pool=NS(_check_physical_write_loc=lambda *args:None,size=capacity-3,page_size=3,start_layer=0,
                    is_quantized_kv_cache=False,store_dtype=torch.bfloat16,dtype=torch.bfloat16,use_hnd=False,
                    kv_cache_layout="nhd",row_dim=width,v_row_dim=width,k_buffer=key,v_buffer=value,alt_stream=None,
                    device_module=None)
            for replay in range(2):
                loc=torch.arange(1,1+m) if layout=="sequential" else 1+torch.randperm(capacity-1,generator=rng)[:m]
                if layout=="zero_repeat":loc[::3]=0
                if layout=="all_padding":loc.zero_()
                if layout=="holes":loc[-1]=capacity-1;loc[:-1]=loc[:-1].where(loc[:-1]!=capacity-1,torch.tensor(0))
                # Offset nonzero remains contiguous and is relative to the actual Loc data pointer.
                physical=torch.full((m+4,),-10004,dtype=torch.int64);physical[2:2+m]=loc;loc=physical[2:2+m]
                for layer in range(n):write(pool,layer,loc,tk[layer,:m],tv[layer,:m])
                for it in reversed(grid):
                    pid[:]=it
                    new_op(*shared,PTR(direct_k),PTR(direct_v),*strides,0,0,0,0,0,0,*spec,
                           Loc=PTR(loc),Lengths=None,WIDTH=0,CACHE_FP8=False,size_limit=torch.tensor(capacity,dtype=torch.int64))
                assert torch.equal(k.view(torch.uint8),dk.view(torch.uint8)) and torch.equal(v.view(torch.uint8),dv.view(torch.uint8)),(m,layout,replay)
                metric["byte_cases"]+=1;metric["cache_bytes"]+=(dk.numel()+dv.numel())*2
                metric["live_kv_heads"]+=int((loc!=0).sum())*n*heads
                metric["zero_kv_heads"]+=int((loc==0).sum())*n*heads
            for invalid in (-1,capacity,2**40):
                loc=loc.clone(); loc[0]=invalid
                try:write(pool,0,loc,tk[0,:m],tv[0,:m])
                except IndexError:pass
                else:raise AssertionError("stock should trap")
                pid[:]=[0,0,0]
                try:new_op(*shared,PTR(direct_k),PTR(direct_v),*strides,0,0,0,0,0,0,*spec,
                           Loc=PTR(loc),Lengths=None,WIDTH=0,CACHE_FP8=False,size_limit=torch.tensor(capacity,dtype=torch.int64))
                except IndexError:pass
                else:raise AssertionError("direct should trap")
                metric["matched_oob_assertions"]+=1
    return dict(metric)


def host_dispatch(root):
    state={}; calls=collections.Counter()
    class Launch:
        def __getitem__(self,grid):
            def launch(*a,**kw):
                calls["direct"]+=1
                state["native"]=(a[-5],a[-4],a[-3],a[-2],kw.get("size_limit"),grid)
                state["proj"]=a[0].clone()
            return launch
    def workspace(*a,**kw):
        calls["workspace"]+=1; state["proj"]=a[0].clone()
        return kw["k_out"],kw["v_out"]
    ns=dict(torch=torch,triton=NS(next_power_of_2=lambda x:1<<(x-1).bit_length()),
            _fused_norm_rope_kernel_stacked=Launch(),_fused_norm_rope_stacked=workspace)
    materialize=method(root/MODULE,"FusedKVMaterializeHelper","materialize",ns)
    append=method(root/WORKER,"DFlashWorkerV2","_append_target_hidden_fused",
                  dict(torch=torch,KVWriteLoc=lambda x,**kw:x))
    append_outer=method(root/WORKER,"DFlashWorkerV2","_append_target_hidden_to_draft_kv_by_loc",
                        dict(torch=torch,draft_tp_context=lambda _:contextlib.nullcontext(),logger=logging.getLogger()))
    class Pool:
        def set_kv_buffer(self,*a):calls["full_cb"]+=1
        def set_kv_buffer_prefix_valid(self,*a):calls["prefix_cb"]+=1
    class Child(Pool):pass
    mock=ModuleType("sglang.srt.mem_cache.memory_pool")
    mock.MHATokenToKVPool=Pool
    def can_native(k,v):
        assert (k,v)==(64,64)
        return state["mode"]!="native-fail"
    mock.can_use_store_cache=can_native
    environ=ModuleType("sglang.srt.environ")
    environ.envs=NS(SGLANG_ENABLE_ASYNC_ASSERT=NS(get=lambda:state["mode"]=="async-assert"))
    modes=("direct","odd-slot-view","strided-hidden","empty","strided-loc","strided-pos","fp8",
           "wrong-store-dtype","quantized","stale-pool","replaced-k-table","replaced-v-table","overridden-store",
           "overridden-writer","async-assert","capture","native-fail","nonprompt","unconfigured","subclass",
           "empty-limit","changed-limit","trim")
    checked=collections.Counter()
    direct_modes=("direct","odd-slot-view","strided-hidden","changed-limit","trim")
    with patch.dict(sys.modules,{"sglang.srt.mem_cache.memory_pool":mock,"sglang.srt.environ":environ}), \
         patch.object(torch.cuda,"is_current_stream_capturing",side_effect=lambda:state["mode"]=="capture"):
        for m in (1,5,17):
            for mode in modes:
                if mode=="empty" and m!=1: continue
                rows=0 if mode=="empty" else m
                state["mode"]=mode;calls.clear()
                pool=(Child if mode=="subclass" else Pool)()
                pool.size,pool.page_size=63,4
                if mode=="empty-limit":pool.size=-pool.page_size
                pool.store_dtype=torch.uint8 if mode=="wrong-store-dtype" else torch.bfloat16
                pool.dtype=torch.float8_e4m3fn if mode=="fp8" else torch.bfloat16
                pool.is_quantized_kv_cache=mode=="quantized"
                pool.k_data_ptrs=torch.arange(5);pool.v_data_ptrs=torch.arange(5)
                if mode=="overridden-store":pool._store_kv_layer=lambda *a:None
                if mode=="overridden-writer":pool.set_kv_buffer=lambda *a:calls.update(full_cb=1)
                layers=[NS(self_attn=NS(attn=NS(layer_id=i,k_scale=1.,v_scale=1.))) for i in range(5)]
                ctx=torch.randn(rows+(3 if mode=="trim" else 0),10,dtype=torch.bfloat16)
                if mode=="strided-hidden":ctx=torch.randn(rows,20,dtype=torch.bfloat16)[:,::2]
                loc=torch.arange(1,1+rows); pos=torch.arange(rows)
                if mode=="odd-slot-view":loc=torch.arange(rows+4)[1:rows+1]
                if mode=="strided-loc":loc=torch.arange(2*rows+2)[1:2*rows+1:2]
                if mode=="strided-pos":pos=torch.arange(2*rows)[::2]
                weight=torch.randn((10,5*64),dtype=torch.bfloat16)
                helper=NS(device=torch.device("cpu"),max_position_hint=rows,
                          flat_kv_weight_t=weight,n_layers=5,num_kv_heads=1,head_dim=32,kv_size=32,
                          layer_out_dim=64,rotary_dim=32,_mm_out_supported=True,
                          _ensure_rope_cache=lambda _:torch.ones((max(1,rows),32)),
                          eps_values=torch.ones(5),k_norm_weights=torch.ones((5,32)))
                scaled=tuple(l.self_attn.attn for l in layers) if mode=="fp8" else ()
                helper._prefix_cache=(pool,pool.k_data_ptrs,pool.v_data_ptrs,scaled)
                if mode=="unconfigured":helper._prefix_cache=None
                if mode=="replaced-k-table":pool.k_data_ptrs=pool.k_data_ptrs.clone()
                if mode=="replaced-v-table":pool.v_data_ptrs=pool.v_data_ptrs.clone()
                if mode=="changed-limit":pool.size+=32
                def allocate(n,dtype):
                    calls["ensure"]+=1
                    helper._proj_workspace=torch.empty((n,320),dtype=dtype)
                    helper._k_workspace=torch.empty((5,n,1,32),dtype=dtype)
                    helper._v_workspace=torch.empty_like(helper._k_workspace)
                helper._ensure_workspace=allocate
                helper.materialize=lambda **kw:materialize(helper,**kw)
                if mode=="stale-pool":
                    stale=Pool();stale.__dict__.update(pool.__dict__);pool=stale
                model=NS(layers=layers,hidden_norm=lambda x:x,project_target_hidden=lambda x:x)
                wk=NS(model_runner=NS(device=torch.device("cpu")),draft_model_runner=NS(token_to_kv_pool=pool),
                      draft_parallel_layout=None,draft_model=model,_verify_projection=None,lilicorr=None,
                      _fused_kv_helper=helper,_use_fused_kv_materialize=True,_logged_padding_trim=True)
                wk._append_target_hidden_fused=lambda **kw:append(wk,**kw)
                lens=torch.tensor([rows],dtype=torch.int32) if mode!="nonprompt" else None
                append_outer(wk,target_hidden=ctx,cache_loc=loc,positions=pos,extend_lens=lens,prefill_fc_done=True)
                if mode=="empty":
                    assert not calls and not hasattr(helper,"_proj_workspace")
                elif mode in direct_modes or (rows==1 and mode in ("strided-pos","strided-loc")):
                    assert calls=={"direct":1} and not hasattr(helper,"_proj_workspace"),(m,mode,calls)
                    ptrs,valid,width,fp8,limit,grid=state["native"]
                    assert ptrs is loc and valid is None and width==0 and not fp8 and grid==(rows,1,5)
                    assert limit==pool.size+pool.page_size
                else:
                    assert calls=={"ensure":1,"workspace":1,"full_cb":5},(m,mode,calls)
                if rows:
                    assert torch.equal(state["proj"],(ctx[:rows]@weight).view(rows,5,64))
                checked["ordinary_prompt_worker_cases"]+=1
                checked["direct_without_workspace"]+=calls["direct"]
                checked["normal_fallbacks"]+=calls["workspace"]
    return dict(checked)


def legacy_host(root):
    source = CASES/"v104_v64_direct_draft_kv/host_dispatch.py"
    text = source.read_text()
    old = "            def go(proj,*args):"
    assert text.count(old) == 1
    text = text.replace(old, "            def go(proj,*args,**kw):\n                assert kw == {'size_limit': None}")
    # The stock V104 recorder predates the appended runtime bound keyword.
    # Keep all its worker/preparation inputs and original expectations.
    scope = {"__name__": "legacy_dispatch", "__file__": str(source)}
    exec(compile(text,str(source),"exec"),scope)
    return scope["run"](root)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",type=Path)
    args=parser.parse_args(); root=args.stage.resolve()
    torch.manual_seed(133);torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="v133-prompt-check-") as folder:
        tmp=Path(folder);stock=tmp/"stock.py";stock.write_bytes(pinned(root,MODULE))
        prefix=tmp/"prefix.py";prefix.write_bytes(pinned(root,PREFIX))
        byte=prompt_bytes(root,stock,tmp)
        host=host_dispatch(root)
        old=evaluator.run(root/MODULE,stock,prefix)
        legacy_dispatch=legacy_host(root)
    print(json.dumps(dict(kernel_sha256=hashlib.sha256((root/MODULE).read_bytes()).hexdigest(),
                          worker_sha256=hashlib.sha256((root/WORKER).read_bytes()).hexdigest(),
                          native_store_sha256=hashlib.sha256(pinned(root,STORE)).hexdigest(),
                          prompt=byte,host=host,legacy_prefix=old,legacy_dispatch=legacy_dispatch,cuda_execution=False),indent=2))


if __name__=="__main__":
    main()
