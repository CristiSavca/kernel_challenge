"""Replay the production BF16 epilogue against the pinned native writer and descriptor ABI."""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CPP = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using std::min;
struct alignas(16) int4 { uint16_t v[8]; };
using __nv_bfloat16=uint16_t;
float __bfloat162float(uint16_t value){uint32_t bits=uint32_t(value)<<16;float r;std::memcpy(&r,&bits,4);return r;}
uint16_t __float2bfloat16_rn(float x){
  uint32_t b;std::memcpy(&b,&x,4);
  if((b&0x7fffffff)>0x7f800000)return 0x7fc0;
  return (b+0x7fff+((b>>16)&1))>>16;
}
struct {unsigned x=0;} threadIdx;
#define CUDA_INLINE
#define PRAGMA_UNROLL
template<class C> struct EpilogueArithmetic {};
template<class C,class A> class EpilogueGmemWriter;
struct SharedStorageType {char pad[128*PAD];int4 reduce[MUSE_GATE_M*32];};
struct MuseGateCtx {
  SharedStorageType smem;static constexpr uint32_t kNumMathThreads=512;
  struct { const void *c,*bs2; const int32_t *expert_ids_ptr; uint32_t top_k; } params;
};
struct Tensor{
  uint64_t e,k;
  uint64_t size(int i){return i==0?e:k;}
  void* data_ptr(){return nullptr;}
};
using cuuint64_t=uint64_t;using cuuint32_t=uint32_t;
struct CUtensorMap {uint64_t d[4],s[3];uint32_t b[4],el=0;};
enum {CU_TENSOR_MAP_DATA_TYPE_UINT32=4,CU_TENSOR_MAP_DATA_TYPE_UINT8=1,
      CUDA_SUCCESS=0,CU_TENSOR_MAP_INTERLEAVE_NONE=0,CU_TENSOR_MAP_SWIZZLE_NONE=0,
      CU_TENSOR_MAP_L2_PROMOTION_NONE=0, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE=0};
int cuTensorMapEncodeTiled(CUtensorMap* m,int el,int ndim,void*,const uint64_t* d,
                          const uint64_t* s,const uint32_t* b,const uint32_t* steps,int a,int sw,int c,int fill){
    assert(ndim==4&&a==0&&sw==0&&c==0&&fill==0);
    for(int i=0;i<4;++i){assert(steps[i]==1);m->d[i]=d[i];m->b[i]=b[i];}
    std::copy(s,s+3,m->s);m->el=el;return 0;}
"""
MAIN = r"""
uint64_t layout_count=0;
void descriptors(int inner,int k,int e){
  for(bool weight:{false,true}){
    unsigned el=weight?4:1, pack=weight?2:1, count=weight?16:8;
    CUtensorMap m; assert(bank_map(Tensor{uint64_t(e),uint64_t(k/16)},inner,weight,&m));
    assert(m.el==el&&m.d[0]==16&&m.d[1]==unsigned(inner*pack/16)&&m.d[2]==2&&m.d[3]==unsigned(e*k/16));
    assert(m.b[0]==16&&m.b[1]==count&&m.b[2]==2&&m.b[3]==8);
    for(int expert:{0,e-1})for(int kt:{0,k/128-1})for(int nt=0;nt<inner/128;++nt)
      for(unsigned slab=0;slab<8;++slab)for(unsigned half=0;half<2;++half)
        for(unsigned j=0;j<count*16;++j){
          uint64_t x=j%16,y=nt*count+j/16,t=expert*(k/16)+kt*8+slab;
          uint64_t source=x*m.el+y*m.s[0]+half*m.s[1]+t*m.s[2];
          uint64_t native=el*(t*2*inner*pack+half*inner*pack+nt*128*pack+j);
          assert(source==native);
          auto dest=(slab*2*count+half*count+j/16)*16+j%16;
          assert(dest==slab*(256*pack)+half*128*pack+j);
          ++layout_count;
        }
  }
}
uint16_t value(unsigned r,unsigned c,unsigned h,unsigned band){
  return uint16_t((band*128*MUSE_GATE_M+r*128+c)*(h?2969u:3389u)+(h?11239:48731));
}
float rscale(uint16_t b,float scale){
  return __bfloat162float(__float2bfloat16_rn(__bfloat162float(b)*scale));
}
int main(){
  uint64_t scalars=0,untouched=0; bool gate_seen[65536]={},up_seen[65536]={};
  for(auto i:{128,384,4096})for(auto k:{128,4096})descriptors(i,k,8);
  constexpr unsigned M=MUSE_GATE_M,I=HUMMING_SHAPE_N/2, start=START_ROW, rows=SLOT_COUNT;
  auto sent=int4{{0x5213,0xf112,0x3f7e,0x0,0x8001,0x1753,0xf2e1,0x1112}};
  std::vector<int32_t> ids(rows);for(unsigned i=0;i<rows;++i)ids[i]=-1;
  unsigned key_shift=0;for(unsigned tmp=rows-1;tmp;tmp>>=1)++key_shift;
  for(unsigned i=start;i<start+M;++i)ids[i]=int((4u<<key_shift)|((i*37u)%rows));
  for(unsigned live:{1u,M-1,M}){
    for(uint16_t factor:{uint16_t(0),uint16_t(0x8000),uint16_t(0x0001),uint16_t(0x3e81),uint16_t(0xc110),uint16_t(0x7f7f)}){
      uint16_t scales[4]={0x7fc1,0x7fc1,0x7fc1,0x7fc1};scales[3]=factor;
      std::vector<int4> result(rows*I/8+4,sent);
      MuseGateCtx ctx{};ctx.params={result.data()+2,scales,ids.data(),key_shift};
      EpilogueArithmetic<MuseGateCtx> ar;EpilogueGmemWriter<MuseGateCtx,EpilogueArithmetic<MuseGateCtx>> writer(ctx,ar);
      for(unsigned nt=0;nt<I/128;++nt){
        for(unsigned r=0;r<M;++r)for(unsigned c=0;c<256;++c){
          // Invert the pinned native write_legacy mapping from logical C[row,col].
          unsigned linear=(c/64)*M+r;
          unsigned word=linear*8+(((c%64)/8)^((linear+offsetof(SharedStorageType,reduce)/128)%8));
          ctx.smem.reduce[word].v[c%8]=value(r,c%128,c/128,nt);
        }
        writer.seek(0,nt,start+live,start);
        for(threadIdx.x=0;threadIdx.x<512;++threadIdx.x)writer.write(0,1,0);
      }
      for(unsigned pos=0;pos<result.size();++pos){
        if(pos<2+start*I/8 || pos>=2+(start+live)*I/8) {
          assert(!std::memcmp(&result[pos],&sent,sizeof(int4)));untouched+=8;continue;
        }
        unsigned r=(pos-2)/(I/8)-start, cbase=(pos-2)%(I/8)*8;
        for(unsigned j=0;j<8;++j){
          auto c=cbase+j;auto g=value(r,c%128,0,c/128),u=value(r,c%128,1,c/128);
          gate_seen[g]=1;up_seen[u]=1;float x=rscale(g,__bfloat162float(factor)), y=rscale(u,__bfloat162float(factor));
          auto expected=__float2bfloat16_rn((x/(1.f+expf(-x)))*y);
          assert(result[pos].v[j]==expected);++scalars;
        }
      }
    }
  }
  printf("%llu %llu %llu %u %u\n",(unsigned long long)scalars,(unsigned long long)untouched,
         (unsigned long long)layout_count,unsigned(std::count(gate_seen,gate_seen+65536,true)),
         unsigned(std::count(up_seen,up_seen+65536,true)));
}
"""


def function(src, key):
    start = src.index(key)
    pos = src.index("{",start)
    depth = 1
    while depth:
        pos += 1
        depth += (src[pos]=="{") - (src[pos]=="}")
    return src[start:pos+1]


def main(root):
    device = root/"python/sglang/kernels/jit/csrc/muse_spark/decode_gate_up.cuh"
    launch = device.with_name("decode_gate_up_launch.cuh")
    source = device.read_text()
    epi = source[source.index("template <> class EpilogueGmemWriter"):]
    # The native RN multiply instruction is emulated by a binary32 product.
    epi,n = re.subn(r'asm\("mul\.rn\.f32[^;]+;[^\n]+\);', "product = __bfloat162float(value) * scale;", epi)
    assert n == 1
    header = CPP + function(launch.read_text(), "static bool bank_map(") + "\n" + epi
    counts = []
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/"check.cpp"; path.write_text(header + MAIN)
        for m,inner,pad,slots,start in ((8,384,1,64,13),(8,128,7,65,57),(16,4096,6,128,112),(16,384,2,255,0),(24,4096,3,257,233),(24,128,7,2048,1023)):
            cmd = ["c++","-std=c++17","-O2","-fno-strict-aliasing",f"-DMUSE_GATE_M={m}",f"-DSTART_ROW={start}",f"-DSLOT_COUNT={slots}",
                   f"-DHUMMING_SHAPE_N={2*inner}",f"-DPAD={pad}",str(path),"-o",temp+"/check"]
            subprocess.run(cmd,check=True)
            rows = list(map(int,subprocess.check_output([temp+"/check"],text=True).split()))
            counts.append(dict(tile_m=m,intermediate=inner,allocated_route_slots=slots,first_sorted_row=start,base_swizzle=pad,bf16_outputs=rows[0],guard_words=rows[1],
                               packed_weight_scale_addresses=rows[2],gate_patterns=rows[3],up_patterns=rows[4]))
    assert any(c["gate_patterns"]==c["up_patterns"]==65536 for c in counts)
    return dict(device_sha256=hashlib.sha256(device.read_bytes()).hexdigest(),
                launcher_sha256=hashlib.sha256(launch.read_bytes()).hexdigest(),
                cases=counts,cuda=False,native_math="CPU expf and software RN BF16; both oracle and source use same primitive",
                scope="Actual packed map builder, shared BF16 epilogue read/store, precision-sensitive packed route keys, row- and tile-relative output addresses, independently assembled shared bands.")


if __name__=="__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
