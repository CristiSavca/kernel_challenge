// Portable SM90 fragment callback. The included helper is the submitted function.
// Build with -I pointing at python/sglang/kernels/jit/csrc/muse_spark.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define CUDA_INLINE inline
#define PRAGMA_UNROLL
struct uint4 { unsigned x[4]; };
struct int4 { int x[4]; };
struct Storage { struct StageStorage { char data[256]; }; StageStorage stages[4]; };
struct Ctx {
  using SharedStorage = Storage;
  struct BlockShape { enum {M=176,N=256,K=64}; };
  struct WarpShape { enum {M=176,N=32,K=64}; };
  enum {kIsGroupedContiguousGemm=1,kNumMathThreads=256,kWarpIters=4};
  Storage smem;
};
struct MMA {
  enum {kSwizzleBytes=128,kUsePackedKLayout=0,kPackedKFactor=1};
  static constexpr int final_regs_c_index() { return 0; }
  unsigned regs_b[2][2][1][4]{};
  float regs_c[1][2][1][88]{};
  unsigned smem_offset=8192;
  Ctx ctx{};
};
static unsigned k, stage_id, iter_id, tid, stride;
static unsigned long long commits, waits, fences, calls;
static std::vector<float> activation, weight;
unsigned cast_smem_ptr_to_uint(const void*) { return 32768; }
template<int Bytes> unsigned long long make_wgmma_smem_desc(unsigned x) {static_assert(Bytes==128);return x;}
void wgmma_fence() {}
void wgmma_commit() {++commits;}
template<int N> void wgmma_wait() {static_assert(N==0);++waits;}
void warpgroup_fence_operand(unsigned&) {++fences;}
#include "residual_prompt_mma.cuh"

unsigned row(unsigned lane,unsigned d) {return (lane%4)*2 + d%2 + 8*(d/4);}
unsigned chan(unsigned lane,unsigned d) {return (lane/32)*16 + lane%32/4 + 8*(d/2%2);}

template<unsigned Width> struct Op {
  static void fma(uint64_t desc, const unsigned* b, float* c, bool accumulate) {
    assert(desc==32768+8192+stage_id*sizeof(Storage::StageStorage)+iter_id*32);
    assert(accumulate && b[0]<2 && b[1]==k && b[2]==tid && b[3]==0xc0ffee);
    for (unsigned reg=0; reg<Width/2; reg++) {
      const auto nr=chan(tid,reg)+64*b[0], mr=row(tid,reg);
      float acc=c[reg];
      for(unsigned i=0;i<16;i++) acc=std::fma(weight[nr*stride+k+i],activation[mr*stride+k+i],acc);
      c[reg]=acc;
    }
    ++calls;
  }
};

static float bf16(float v) {
  uint32_t bits;std::memcpy(&bits,&v,4);
  bits=(bits+0x7fffu+((bits>>16)&1u))&0xffff0000u;
  std::memcpy(&v,&bits,4);return v;
}

template<bool Paired, unsigned Width> unsigned long long trial(unsigned live,unsigned K,unsigned seed) {
  stride=K;
  const std::array<float,8> code={0,.5,1,1.5,2,3,4,6};
  activation.resize(176*K);weight.resize(128*K);
  auto generator=[&]() {seed=1664525u*seed+1013904223u;return seed;};
  for(auto &v:activation) v=bf16(std::ldexp(float(int(generator()>>17)-16384),int(generator()%16)-14));
  for(auto &v:weight) {
    unsigned raw=8+generator()%119, nibble=generator()>>28;
    float scale=std::ldexp(1.f+(raw%8)*.125f,int(raw/8)-7);
    v=std::copysign(code[nibble%8]*scale,nibble>=8 ? -1.f:1.f);
    assert(v==bf16(v));
  }
  std::vector<bool> seen(128*live);
  std::array<MMA,128> lanes{};
  unsigned long long checked=0;
  for(tid=0;tid<128;tid++) {
    auto &m=lanes[tid];
    for(k=0;k<K;k+=16) {
      stage_id=k/64%4;iter_id=k/16%4;
      for(unsigned j=0;j<2;j++) {
        auto &b=m.regs_b[iter_id%2][j][0];b[0]=j;b[1]=k;b[2]=tid;b[3]=0xc0ffee;
      }
      auto c0=commits,w0=waits,f0=fences,a0=calls;
      muse_residual_mma<Op<Width>,Paired,Ctx>(m,stage_id,iter_id);
      assert(calls-a0==2 && commits-c0==(!Paired || iter_id%2) && waits-w0==commits-c0);
      assert(fences-f0==(Paired && iter_id%2 ? 16u:0u));
    }
    for(unsigned j=0;j<2;j++) for(unsigned d=0;d<88;d++) {
      unsigned mr=row(tid,d),nr=chan(tid,d)+64*j;
      if(mr>=Width){assert(m.regs_c[0][j][0][d]==0);continue;}
      if(mr>=live)continue;
      auto index=mr*128+nr;assert(!seen[index]);seen[index]=true;
      float expected=0;
      for(unsigned i=0;i<K;i++) expected=std::fma(weight[nr*K+i],activation[mr*K+i],expected);
      assert(expected==m.regs_c[0][j][0][d]);++checked;
    }
  }
  assert(checked==128*live);
  return checked;
}
int main() {
  unsigned long long values=0;
  for(unsigned live=1;live<=96;live++) {
    unsigned K=(live==32 || live==33) ? 4096:64;
    for(bool paired:{false,true}) {
      if(live<=32) values+=paired?trial<true,32>(live,K,live):trial<false,32>(live,K,live);
      else values+=paired?trial<true,96>(live,K,live):trial<false,96>(live,K,live);
    }
  }
  std::printf("{\"live_FP32\":%llu,\"native_fma_call_abi_checks\":%llu,\"drains\":%llu,\"register_fences\":%llu,\"cases\":192,\"K\":[64,4096]}\n",values,calls,waits,fences);
}
