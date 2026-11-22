"""Time just the two Python submit loops on CPU; the native CUDA boundary returns immediately."""
from statistics import median
from time import perf_counter_ns
import json
import platform
import torch
from replay import expose
from verify import MHA,candidate

torch.set_num_threads(1)
wrap="python/sglang/kernels/ops/kvcache/hicache.py"
with candidate() as (parent,tree,git):
    source=git("show",tree+":"+wrap).decode()
    native=git("show",tree+":"+MHA).decode()
env=dict(torch=torch,debug_kernel_api=lambda f:f,DEFAULT_BLOCK_QUOTA=2,_default_unroll=lambda _:4)
staged=expose(source,"transfer_hicache_all_layer_staged_lf_pf",env)["transfer_hicache_all_layer_staged_lf_pf"]
backup=expose(native,"backup_from_device_all_layer",env,"MHATokenToKVPoolHost")["backup_from_device_all_layer"]
class Module:
    def launch_all_lf_pf_staged(self,*args):pass
module=Module()
env.update(_jit_hicache_staged_module=lambda **_:module,jit_transfer_hicache_all_layer=lambda **_:None,
    jit_transfer_hicache_all_layer_staged_lf_pf=staged)
class H:pass
h=H();h.device_k_buffers=h.device_v_buffers=();h.device_k_data_ptrs=h.device_v_data_ptrs=()
h.k_data_ptrs=h.v_data_ptrs=();h.device_rows_packed=h.can_use_write_back_jit=True
h._transfer_device=torch.device("cpu");h.staging_page_capacity=64;h.layout="page_first";h.page_size=1
n=16384;dest=torch.arange(n);origin=dest.flip(0)
def run(mapped):
    h.can_use_mapped_backup=mapped
    for width,layers in ((128,16),(128,46),(256,5)):
        h.dtype=torch.uint8;h.device_row_stride_bytes=h.element_dim=width;h.layout_dim=width*layers
        h.staging_k_buffer=h.staging_v_buffer=torch.empty(64,layers,width,device="meta")
        h.k_buffer=h.v_buffer=torch.empty(n,layers,width,device="meta")
        backup(h,None,dest,origin,"kernel")
for path in (False,True):run(path)
dur={False:[],True:[]}
for i in range(41):
    for path in ((True,False) if i%2 else (False,True)):
        start=perf_counter_ns();run(path);dur[path].append((perf_counter_ns()-start)/1e6)
baseline=median(dur[False]);new=median(dur[True])
print(json.dumps(dict(tree=tree,machine=platform.machine(),system=platform.system(),torch=torch.__version__,alternating_pairs=41,
    geometry="All three full 16K backups; date-independent synthetic addresses and exact wrapper slices.",
    staged_control_python_ms=baseline,mapped_dispatch_python_ms=new,median_paired_difference_ms=median(a-b for a,b in zip(dur[False],dur[True])),
    scope="Single-thread CPU, meta storage, existing warmed Torch, real Python wrappers, no-op JIT calls. Excludes CUDA launch, index uploads, transfers and GPU time."),indent=2))
