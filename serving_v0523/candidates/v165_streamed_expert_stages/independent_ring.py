"""Check native launch dispatch, shuffled ring epochs, and all padded activation masks."""
from __future__ import annotations
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

stage=Path(sys.argv[1]).resolve()
p=stage/'python/sglang/kernels/jit/csrc/muse_spark'
src=(p/'expert_tma_route_pair.cuh').read_text()
base=(p/'expert_route_pair.cuh').read_text()
# These statements tie the independent state model below to the audited CUDA.
for phrase in (
    'warp == kChannelWarps','lane < kRows ? job[1 + lane] : -1',
    'const uint32_t live = __popc(__ballot_sync(0xffffffffu, slot >= 0))',
    'const uint32_t total = sizeof(tile.weights) + sizeof(tile.scales) + live * bytes',
    'if (slot >= 0)', 'uint32_t(slot) / p.top_k', 'size_t(row) * p.input_width + chunk * (16 * kKSteps)',
    's * 8 + lane % 4', 'const uint32_t lo = live ? x[0] : 0u, hi = live ? x[4] : 0u',
    'sequence >= stages) wait_phase(empty + index, ((sequence / stages) - 1) & 1)',
    'wait_phase(full + index, (sequence / stages) & 1)',
    'if (lane == 0) {\n          asm volatile("mbarrier.arrive.shared::cta',
    'uint32_t sequence = 0;\n  for (uint32_t j = blockIdx.y; j < jobs; j += gridDim.y)',
    '32 * (kChannelWarps + 1), 0)', 'return static_cast<uint32_t>(sms * blocks)',
):
    assert phrase in src,phrase
start=src.index('template <bool Gate>\ninline void launch_stream(');stop=src.index('inline auto encoder()',start)
code='''
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <iostream>
struct Projection {unsigned output_width;};
struct BankMaps{};
struct DLDevice{};
struct dim3 { unsigned x,y; dim3(unsigned x,unsigned y):x(x),y(y){} };
constexpr unsigned kChannelWarps=8;
unsigned selected,nthreads,gx,gy;
template<bool Gate>void tensor_projection(Projection,BankMaps) {selected=Gate?1:2;}
template<bool Gate>void streamed_projection(Projection,BankMaps) {selected=Gate?3:4;}
namespace host{struct LaunchKernel{LaunchKernel(dim3 g,unsigned nt,DLDevice){gx=g.x;gy=g.y;nthreads=nt;}
void operator()(void(*fn)(Projection,BankMaps),Projection p,BankMaps m){fn(p,m);}};}
''' +src[start:stop]+'''
int main() {
  unsigned cases=0,stream=0;
  for(unsigned sms:{0u,1u,8u,16u,64u,132u,144u})for(unsigned occupancy:{0u,1u,2u,3u,4u,8u})
  for(unsigned width:{128u,512u,4096u,8192u})for(unsigned rows:{1u,2u,4u,8u,16u,32u})
  for(unsigned top:{1u,2u,4u,8u})for(unsigned erank:{1u,4u,16u,32u})for(bool gate:{true,false}) {
    unsigned slots=rows*top, used=std::min(erank,slots), bound=used+(slots-used)/8;
    unsigned groups=std::max(1u,std::min(used,(slots*erank+255)/256));
    if(gate)launch_stream<true>({width},{},groups,bound,occupancy*sms,{});
    else launch_stream<false>({width},{},groups,bound,occupancy*sms,{});
    unsigned capacity=occupancy*sms/(width/128);
    bool dispatch=groups<capacity;
    assert(selected==(dispatch?(gate?3u:4u):(gate?1u:2u)));
    assert(gx==width/128 && gy==(dispatch?std::min(bound,capacity):groups));
    assert(nthreads==(dispatch?288u:256u) && gy>0 && gy<=bound);
    for(unsigned jobs=0;jobs<=bound;jobs++){
      for(unsigned job=0;job<jobs;job++){unsigned sy=job%gy, iter=job/gy;assert(sy<gy && sy+iter*gy==job);}
    }
    ++cases;stream+=dispatch;
  }
  std::cout<<cases<<" "<<stream;
}
'''
with tempfile.TemporaryDirectory(prefix='v165-host-dispatch-') as tmp:
    f=Path(tmp)/'test.cpp';o=Path(tmp)/'test';f.write_text(code);subprocess.run(['c++','-std=c++17','-O2',f,'-o',o],check=True)
    cases,special=map(int,subprocess.check_output([o]).split())


def random_schedule(depth,chunks,jobs,seed):
    rng=random.Random(seed);total=jobs*chunks;issued=0;next_read=[0]*8;delays=[None]*depth
    consumed=[[-1]*8 for _ in range(depth)]
    last_copy=[-1]*depth;last_full=[-1]*depth;cross=0;wait=0
    # Copies of K weights, scales and each separate live activation may complete
    # in arbitrary order. A full epoch becomes visible only after all bytes land.
    while min(next_read)<total:
        ready=[];i=issued%depth
        if issued<total and (issued<depth or all(v==issued-depth for v in consumed[i])):
            ready.append(('produce',0))
        for index,pending in enumerate(delays):
            if pending:ready.append(('copy',index))
        for w,n in enumerate(next_read):
            if n<total and last_full[n%depth]==n:ready.append(('read',w))
        assert ready
        kind,n=rng.choice(ready)
        if kind=='produce':
            assert delays[i] is None
            if issued>=depth:
                assert all((v//depth)&1==((issued//depth)-1)&1 for v in consumed[i]);wait+=1
            m=rng.randrange(256);planes=2 if depth==2 else 1
            payload=[8192,1024]*planes+[256]*m.bit_count()
            rng.shuffle(payload);delays[i]=payload;last_copy[i]=issued;issued+=1
        elif kind=='copy':
            delays[n].pop()
            if not delays[n]:last_full[n]=last_copy[n];delays[n]=None
        else:
            pos=next_read[n];ix=pos%depth
            assert last_full[ix]==last_copy[ix]==pos and (last_full[ix]//depth)&1==(pos//depth)&1
            assert consumed[ix][n]==(pos-depth if pos>=depth else -1)
            if pos and pos%chunks==0:cross+=1
            consumed[ix][n]=pos;next_read[n]+=1
    assert issued==total and all(t==total for t in next_read)
    return cross,wait


cross=wait=schedules=0
for stages in (2,3):
    for chunks in (1,2,3,5,16,32):
        for jobs in (1,2,5,9):
            for seed in range(5):
                c,w=random_schedule(stages,chunks,jobs,seed);cross+=c;wait+=w;schedules+=1

# The same input bit slices are consumed by all eight MMA warps; the four-word
# inter-row pad never participates, including a missing row following a live row.
masks=addresses=active=0
for gate in (False,True):
    for mask in range(256):
        n_live=mask.bit_count();nplanes=2 if gate else 1
        full=sum((8192+1024) for _ in range(nplanes))+256*n_live
        assert full==nplanes*(8*256*4+8*64*2)+n_live*256
        for row in range(8):
            offset=row*68*4
            assert offset%16==0 and offset+256<=(row+1)*68*4
            for k in range(8):
                for part in range(4):
                    for high in (0,4):
                        index=k*8+part+high
                        assert index<64 and (offset+4*index-offset)//4==index
                        addresses+=1;active+=bool(mask&(1<<row))
        masks+=1
print(json.dumps(dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),
        direct_native_host_launches=cases,selected_stream_launches=special,
        shuffled_async_schedules=schedules,consumer_job_boundaries=cross,validated_empty_epochs=wait,
        exhaustive_route_masks=masks,local_activation_word_checks=addresses,active_word_checks=active,
        hardware=False),indent=2))
