"""Exercise native VMM finalization, MHA views and V192's descriptor guard using a CPU storage/driver stand-in."""
import ast
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as S
from unittest.mock import patch

import torch

stage=Path(sys.argv[1])
def read(rel):
    p=stage/rel
    return p.read_text() if p.is_file() else subprocess.check_output(["git","-C",stage,"show","HEAD:"+rel],text=True)
def methods(src,names,cls=None):
    body=ast.parse(src).body
    if cls is not None:body=next(n.body for n in body if getattr(n,"name",None)==cls)
    return "\n".join(ast.unparse(n) for n in body if getattr(n,"name",None) in names)
def install(src,params):
    exec(compile("from __future__ import annotations\n"+src,"pinned-production","exec"),params)
def sha(data):return hashlib.sha256(data.encode()).hexdigest()

if len(sys.argv)==4:
    util=ModuleType("triton._utils");exec(Path(sys.argv[3]).read_text(),util.__dict__)
    sys.modules[util.__name__]=util
    td={};exec(Path(sys.argv[2]).read_text(),td)
    descriptor=td["TensorDescriptor"]
else:
    from triton.tools.tensor_descriptor import TensorDescriptor as descriptor
pool=read("python/sglang/srt/mem_cache/memory_pool.py")
vmm=read("python/sglang/srt/mem_cache/kv_vmm_backing.py")
paged=read("python/sglang/srt/mem_cache/layout/paged_view.py")
fa=read("python/sglang/srt/layers/attention/flashattention_backend.py")
kernel=read("python/sglang/kernels/ops/attention/paged_verify_attention.py")
ns=dict(torch=torch,TensorDescriptor=descriptor,triton=S(cdiv=lambda a,b:(a+b-1)//b))
install(methods(kernel,{"_direct_tile_args"}),ns)
install(methods(paged,{"paged_view","paged_kv_view"}),ns)
install(methods(fa,{"get_paged_mha_kv_cache"},"FlashAttentionBackend"),ns)
install(methods(pool,{"_kv_buffer_shapes","_assign_post_capture_tensors","_get_key_buffer","get_key_buffer",
                      "_get_value_buffer","get_value_buffer","get_kv_buffer","_finalize_backing_tokens"},"MHATokenToKVPool"),ns)
install(methods(vmm,{"finalize","ensure_prefix"},"KvVmmBufferOwner"),ns)
def bind(obj,methods_):
    for name in methods_:setattr(obj,name,ns[name].__get__(obj))
class DeviceView:
    # Only the CUDA-device flag is simulated. Geometry, strides, pointer and dtype are real PyTorch.
    is_cuda=True
    def __init__(self,t):self.t=t
    def __getattr__(self,k):return getattr(self.t,k)

args=ns["_direct_tile_args"]
accept=checks=refusals=0
for heads in (1,2,4):
    for kpad,vpad in ((0,0),(16,32),(64,16)):
        rows=641;page=1;d=64
        owner=S(_reserved_num_tokens=rows-1,page_size=1,_specs=[])
        owner._back_spans=lambda spans: spans
        bind(owner,["ensure_prefix","finalize"])
        pool_=S(head_num=heads,head_dim=d,v_head_dim=d,use_hnd=False,size=rows-1,page_size=page,
            dtype=torch.float8_e4m3fn,store_dtype=torch.uint8,start_layer=0,layer_num=1,
            layer_transfer_counter=None,is_quantized_kv_cache=False,_muse_verify_capacity=None,_post_capture_owner=owner)
        bind(pool_,["_kv_buffer_shapes","_assign_post_capture_tensors","_get_key_buffer","get_key_buffer","_get_value_buffer",
                    "get_value_buffer","get_kv_buffer","_finalize_backing_tokens"])
        layer=S(layer_id=0,tp_k_head_num=heads,tp_v_head_num=heads,head_dim=d,v_head_dim=d)
        backend=S(page_size=1,token_to_kv_pool=pool_);bind(backend,["get_paged_mha_kv_cache"])
        tensors=[torch.empty_strided(shape,(heads*d+padding,d,1),dtype=torch.uint8)
                 for shape,padding in zip(pool_._kv_buffer_shapes(),(kpad,vpad))]
        for t in tensors:owner._specs.append(S(desc=S(prefix_span_bytes=lambda n,p:n,final_span_bytes=lambda n,p:n+p)))
        owner.ensure_prefix(1);pool_._assign_post_capture_tensors(tensors)
        table=torch.zeros((2,513),dtype=torch.int32);origins=torch.empty((2,5),dtype=torch.int32)
        def capture():
            return args(*(DeviceView(t) for t in backend.get_paged_mha_kv_cache(layer)),table,origins,-1,128)
        before=capture(); assert before["CAPACITY"]==rows
        # Change the advertised physical extent after real view/descriptor creation.
        pool_._finalize_backing_tokens(398)
        assert pool_.size==398 and owner._final_num_tokens==398 and pool_._muse_verify_slot_limit==399
        after=capture();assert after["CAPACITY"]==rows
        for label,pad,bank in zip(("KDesc","VDesc"),(kpad,vpad),tensors):
            desc=before[label];assert desc.base.data_ptr()==after[label].base.data_ptr()==bank.data_ptr()
            assert desc.shape==[rows,heads*d] and desc.strides==[heads*d+pad,1] and desc.block_shape==[128,d]
            assert desc.base.dtype==torch.float8_e4m3fn
            for start in (1,2,128,271):
                seq=torch.arange(start,start+128)
                assert (seq<399).all()  # Mirrors native backed allocator IDs.
                for h in range(heads):
                    expected=seq[:,None]*bank.stride(0)+torch.arange(h*d,(h+1)*d)[None,:]
                    actual=(start+torch.arange(128)[:,None])*desc.strides[0]+h*d+torch.arange(d)[None,:]
                    assert torch.equal(actual,expected)
                    checks+=actual.numel()
        accept+=1
        k,v=map(DeviceView,backend.get_paged_mha_kv_cache(layer))
        for ka,va,pt,orig,win,tile in ((k,v,table,None,-1,128),(k,v,table,origins,64,128),
            (k,v,table,origins,-1,64),(k,v,table,origins[:,:4],-1,128),
            (k,v,table,origins.to(torch.int64),-1,128),(k,v,table,torch.zeros(2,10,dtype=torch.int32)[:,::2],-1,128),
            (k.t,v,table,origins,-1,128),(k,DeviceView(v.t.view(torch.uint8)),table,origins,-1,128)):
            assert args(ka,va,pt,orig,win,tile)=={};refusals+=1
        for newstride in (heads*d-1,heads*d+1):
            bad=DeviceView(torch.empty_strided((rows,1,heads,d),(newstride,newstride,d,1),dtype=torch.float8_e4m3fn))
            assert args(bad,v,table,origins,-1,128)=={};refusals+=1
        backing=torch.empty(k.t.numel()+1,dtype=torch.float8_e4m3fn)
        unaligned=DeviceView(backing[1:].view_as(k.t))
        assert args(k,unaligned,table,origins,-1,128)=={};refusals+=1
        with patch.object(torch.compiler,"is_compiling",return_value=True):
            assert args(k,v,table,origins,-1,128)=={};refusals+=1
assert "valid &= (tiles + 1) * 128 <= num_live_pages" in read("python/sglang/kernels/ops/attention/metadata.py")
print(json.dumps(dict(actual_cuda=False,descriptor="Triton 3.8.0 TensorDescriptor",positive_pools=accept,
    finalizations=accept,unchanged_descriptor_base_addresses=2*accept,adverse_guard_cases=refusals,
    one_byte_kv_address_comparisons=checks,rows_reserved=641,rows_physically_advertised_after_backing=399,
    guard_sha256=sha(methods(kernel,{"_direct_tile_args"}))),indent=2))
