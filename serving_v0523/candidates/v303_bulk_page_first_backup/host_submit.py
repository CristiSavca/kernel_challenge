"""Measure the production CPU planner and wrapper slice overhead with immediate FFI returns."""
from bisect import bisect_right
from statistics import median
from time import perf_counter_ns
import json
import platform
from types import SimpleNamespace as S

import torch

from replay import expose
from verify import MHA, candidate

WRAPPER = "python/sglang/kernels/ops/kvcache/hicache.py"

def build(label, sources, count):
    module = S(launch_all_lf_pf_staged=lambda *a: None, launch_all_lf_pf_bulk=lambda *a: None)
    env = dict(torch=torch, debug_kernel_api=lambda f:f, DEFAULT_BLOCK_QUOTA=2, _default_unroll=lambda _:4,
               _jit_hicache_staged_module=lambda **_:module, bisect_right=bisect_right, jit_transfer_hicache_all_layer=lambda **_:None)
    env["jit_transfer_hicache_all_layer_staged_lf_pf"] = expose(sources[WRAPPER],"transfer_hicache_all_layer_staged_lf_pf",env)["transfer_hicache_all_layer_staged_lf_pf"]
    names = "backup_from_device_all_layer" + (" _try_bulk_page_first_backup" if label=="bulk" else "")
    Pool = type("Pool", (), expose(sources[MHA], names, env, "MHATokenToKVPoolHost"))
    pools = []
    for layers,width in ((16,128),(46,128),(5,256)):
        h = Pool()
        h.dtype = torch.uint8
        h.device_row_stride_bytes = h.element_dim = width
        h.layout_dim = width*layers
        h.device_k_buffers=h.device_v_buffers=h.device_k_data_ptrs=h.device_v_data_ptrs=h.k_data_ptrs=h.v_data_ptrs=()
        h.device_rows_packed=h.can_use_write_back_jit=True
        h.can_use_mapped_backup=label!="stock staged"
        h._transfer_device=torch.device("cpu")
        h.staging_page_capacity=64
        h.size=count
        h.bulk_backup_row_ends=(count,)
        h.layout="page_first"
        h.page_size=1
        h.staging_k_buffer=h.staging_v_buffer=torch.empty(64,layers,width,device="meta")
        h.k_buffer=h.v_buffer=torch.empty(count,layers,width,device="meta")
        pools.append(h)
    return pools

def main():
    torch.set_num_threads(1)
    with candidate() as (parent,tree,git):
        source = {ref:{f:git("show",ref+":"+f).decode() for f in (WRAPPER,MHA)} for ref in (parent,tree)}
    n=16384
    labels = ["stock staged","V298","bulk"]
    variants = {label:build(label, source[tree if label=="bulk" else parent], n) for label in labels}
    slots=torch.arange(n)
    src=slots.flip(0)
    cases = []
    for order,dest in (("contiguous",slots),("two out-of-order runs",slots.roll(7000))):
        durations={label:[] for label in labels}
        for rep in range(44):
            for label in labels[rep%3:]+labels[:rep%3]:
                start=perf_counter_ns()
                for pool in variants[label]:
                    pool.backup_from_device_all_layer(None,dest,src,"kernel")
                if rep>=3:durations[label].append((perf_counter_ns()-start)/1e6)
        cases.append(dict(host_slots=order,median_ms={k:median(v) for k,v in durations.items()},
            bulk_minus_V298_paired_median_ms=median(a-b for a,b in zip(durations["bulk"],durations["V298"]))))
    print(json.dumps(dict(tree=tree,parent=parent,machine=platform.machine(),system=platform.system(),torch=torch.__version__,
        repetitions=41,cases=cases,scope="Full/SWA/draft at 16K each. Single-thread Apple CPU, real Python and Torch CPU index work, synthetic shuffled or linear host slots, meta staging. CUDA calls are no-ops. Allocation, GPU execution and copy time excluded."),indent=2))

if __name__=="__main__":main()
