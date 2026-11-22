"""Replay Humming 0.1.12's grouped scheduler token counts against real V169 plans."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import torch

stage, vendor = map(Path, sys.argv[1:])
tiles = stage / "python/sglang/srt/layers/muse_spark/prefill_expert_tiles.py"
sp = importlib.util.spec_from_file_location("production", tiles)
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
src = (vendor/"include/humming/scheduler.cuh").read_text()
normal = next(x.strip() for x in src.splitlines() if "expert_tokens[index] = ctx.smem.expert_offset[index + 1]" in x)
last = next(x.strip() for x in src.splitlines() if "expert_tokens[kNumExperts - 1] = ctx.params.shape_m" in x)
# Keep both upstream expressions byte for byte inside the CPU wrapper.
native = """
struct Ctx { struct { int *expert_tokens; const int *expert_offset; } smem; struct { int shape_m; } params; };
extern "C" void run(const int *offsets, int *out, int kNumExperts, int rows) {
    Ctx ctx={{out, offsets}, {rows}};
    %s
    for (int index=0; index<kNumExperts-1;index++) { %s }
}
""" % (last, normal)
rng = random.Random(1697)
torch.manual_seed(1697)
torch.set_num_threads(1)
sizes = {136:{}, 152:{}, 176:{}}
checked, dual, zero_last, empty_bins, unaligned_cut, full_m_rows, checked_slots = (0,)*7
widths = set()
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    (tmp/"check.cc").write_text(native)
    subprocess.run(["c++","-std=c++17","-O2","-shared","-fPIC",str(tmp/"check.cc"),"-o",str(tmp/"check.so")],check=True)
    fn=ctypes.CDLL(str(tmp/"check.so")).run
    ptr=ctypes.POINTER(ctypes.c_int)
    fn.argtypes=(ptr,ptr,ctypes.c_int,ctypes.c_int)
    for case in range(170):
        experts=32
        counts=[rng.choice([0, 1, 132, 136, 150, 152, 163, 176, 259, 272, 299, 302, 350, 712]) for _ in range(experts)]
        counts[-1]=0 if case%2 else rng.randrange(1,350)
        if case%7==0:
            counts=[rng.randrange(65,136) for _ in counts]
        raw=torch.repeat_interleave(torch.arange(experts,dtype=torch.int32),torch.tensor(counts))
        remote=8-len(raw)%8 + 8*(case%6)
        ids=torch.cat((raw,torch.full((remote,),-1,dtype=torch.int32)))
        ids=ids[torch.randperm(ids.numel())].view(-1,8)
        state=(sizes,(64,64),132)
        plan=m.choose_tiles(state,counts)
        if plan is None:
            continue
        sorted_keys,slots,runs=m.sort_tile_cohorts(ids,plan,state)
        live=slots[-sum(counts):]
        assert torch.equal(live.sort().values,(ids.flatten()>=0).nonzero().flatten())
        dual+=len(runs)>1
        dense=ids.flatten()[live]
        assert torch.equal(sorted_keys[-sum(counts):]%experts,dense)
        for (width,_,_),(start,end,off,_) in zip(plan[0],runs):
            assert off.is_contiguous() and off.dtype==torch.int32 and off.shape==(experts+1,)
            assert int(off[0])==0 and int(off[-1])==end-start
            out=torch.empty(experts,dtype=torch.int32)
            fn(ctypes.cast(off.data_ptr(),ptr),ctypes.cast(out.data_ptr(),ptr),experts,end-start)
            torch.testing.assert_close(out,torch.bincount(dense[start:end].long(),minlength=experts).int(),rtol=0,atol=0)
            for e,(lo,hi) in enumerate(zip(off[:-1],off[1:])):
                lo,hi=int(lo),int(hi)
                assert (dense[start+lo:start+hi]==e).all()
                index=live[start+lo:start+hi]
                assert index.numel()<2 or (index[1:]>index[:-1]).all()
            # TMA uses each narrowed tensor's data pointer. One real 4096-wide
            # row shifts a BF16 pointer by 8192B, preserving required alignment.
            arena=torch.empty((end+1,4096),dtype=torch.bfloat16)
            rows=arena[start:end]
            assert rows.is_contiguous() and rows.data_ptr()==arena.data_ptr()+start*8192 and rows.data_ptr()%16==0
            empty_bins+=int((out==0).sum());zero_last+=int(out[-1]==0)
            widths.add(width);unaligned_cut+=bool(start%16);full_m_rows+=end-start
        checked+=1;checked_slots+=ids.numel()
assert dual and zero_last and unaligned_cut and widths==set(sizes)
print(json.dumps(dict(groupings=checked,two_cohorts=dual,last_expert_empty=zero_last,zero_expert_bins=empty_bins,
    nonaligned_row_cuts=unaligned_cut,live_rows=full_m_rows,routed_slots=checked_slots,widths=sorted(widths),
    scheduler_source_sha256=hashlib.sha256(src.encode()).hexdigest(),planner_sha256=hashlib.sha256(tiles.read_bytes()).hexdigest(),
    physical_GPU=False),indent=2))
