"""Production host-pool initialization, preallocated staging and per-registration admission."""
import json
from types import SimpleNamespace as S
import torch
from replay import expose
from verify import candidate,digest,BASE,MHA,COMMON

with candidate() as (_,tree,git):
    sources={p:git("show",tree+":"+p).decode() for p in (MHA,COMMON,BASE+"pool_host/base.py")}
env=dict(torch=torch,_is_cuda=True,_is_hip=False,_WRITE_BACK_STAGING_PAGE_CHUNK=64,
  _CUDA_HOST_REGISTERED_RANGES_ATTR="ranges",current_platform=S(capabilities=S(hicache_device_kernels=True)))
env["has_linear_host_mapping"]=expose(sources[COMMON],"has_linear_host_mapping",env)["has_linear_host_mapping"]
expose(sources[MHA],"prepare_mha_write_back_staging",env)
methods=expose(sources[MHA],"_initialize_transfer_views _init_write_back_staging_buffers",env,"MHATokenToKVPoolHost")
methods.update(expose(sources[BASE+"pool_host/base.py"],"_init_device_row_stride",env,"HostKVCache"))
Host=type("Host",(),methods)
result=[]
for case in ("full","swa","draft","layer_first","page_four","unregistered","unverified alias","fragmented alias","no JIT",
             "noncuda","not pinned","no staging support","strided device","disjoint translated base","reused preparation"):
    heads,dim=2,64
    layers=46 if case=="swa" else 5 if case=="draft" else 16
    dtype=torch.bfloat16 if case=="draft" else torch.uint8
    h=Host();h._transfer_device=torch.device("cuda:3");h.dtype=dtype;h.head_num=heads;h.head_dim=dim;h.layer_num=layers
    h.pin_memory=case!="not pinned";h.page_size=4 if case=="page_four" else 1;h.page_num=193
    h.layout="layer_first" if case=="layer_first" else "page_first"
    dims=(2,layers,193,heads,dim) if h.layout=="layer_first" else (2,193,layers,heads,dim)
    h.kv_buffer=torch.empty(dims,dtype=dtype);h.k_buffer,h.v_buffer=h.kv_buffer
    val=torch.empty(3,heads,dim,dtype=dtype)
    h.device_rows_packed=case!="strided device"
    if not h.device_rows_packed:val=torch.empty(6,heads,dim,dtype=dtype)[::2]
    h.device_k_buffers=h.device_v_buffers=(val,)*layers
    size=h.kv_buffer.nbytes;base=h.kv_buffer.data_ptr();cut=(size//2)//dtype.itemsize*dtype.itemsize
    if case!="unregistered":h.kv_buffer.ranges=[(base,cut),(base+cut,size-cut)]
    def resolve(t,rank):
        assert rank==3
        return t.data_ptr()+(case=="disjoint translated base" or (case=="fragmented alias" and t.data_ptr()>=base+cut))*4096
    env["_resolve_device_accessible_ptr_fn"]=lambda:None if case=="unverified alias" else resolve
    # Native CPU tensors stand in for GPU K/V storage and pointer table. The certifier still receives the explicit CUDA index.
    env["make_kernel_ptr_table"]=lambda seq,dev,host_memory_registered:[resolve(t,dev.index) for t in seq]
    env["can_use_hicache_jit_kernel"]=lambda **_:case!="no JIT"
    env["can_use_write_back_jit_kernel"]=lambda **_:case!="no staging support"
    env["_is_cuda"]=case!="noncuda"
    info=S(head_num=heads,head_dim=dim,v_head_dim=dim,dtype=dtype,device=torch.device("cpu"))
    staging=env["prepare_mha_write_back_staging"](info,layer_num=layers,page_size=h.page_size) if case=="reused preparation" else None
    h._initialize_transfer_views(info,staging)
    accept=case in ("full","swa","draft","disjoint translated base","reused preparation")
    assert h.can_use_mapped_backup==accept,(case,h.can_use_mapped_backup)
    if h.can_use_mapped_backup:
        for l in range(layers):
            assert h.k_data_ptrs[l]==resolve(h.k_buffer[:,l],3)
            assert h.v_data_ptrs[l]==resolve(h.v_buffer[:,l],3)
        if staging:assert h.staging_k_buffer.data_ptr()==staging[0].data_ptr()
    result.append(dict(path=case,mapped=h.can_use_mapped_backup,staged=h.can_use_write_back_jit))
print(json.dumps(dict(tree=tree,cases=result,scope="Native pool setup, stride computation, byte-range certifier and staging preparation. CUDA mapping queries, JIT capability and pointer table modeled with Python; includes a nonidentity affine alias."),indent=2))
