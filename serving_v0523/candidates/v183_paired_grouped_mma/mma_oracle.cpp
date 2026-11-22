// CPU delayed-completion replay. Includes the submitted two-slice method itself.
// Build with -DHEADER='"/.../grouped_mma_pair.cuh"' and -DMUSE_OVERLAP_M={136,152,176}.
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#define CUDA_INLINE inline
#define PRAGMA_UNROLL
#define HUMMING_SHAPE_N 4096
#define HUMMING_SHAPE_K 4096
template <int...> struct Shape {};
struct BFloat16 {}; struct Float4E2M1 {}; struct Float8E4M3 {};
struct LayerConfig {}; struct ComputeConfig {}; struct TuningConfig {};
struct int4 { unsigned x[4]; };
struct Storage { struct StageStorage { char x[8192]; }; StageStorage stages[4]; };
template <typename...> struct KernelContext { Storage smem{}; };
template <class> struct MainloopArithmetic {};
enum class MmaType { WGMMA };
template<MmaType, class, class> struct MmaSelector {};
uint32_t cast_smem_ptr_to_uint(const void*) { return 0; }
template <unsigned> uint64_t make_wgmma_smem_desc(unsigned x) { return x; }
void warpgroup_fence_operand(uint32_t&) {}
void wgmma_fence() {}
uint64_t drains, dot_products, registers_checked, calls;
std::array<std::array<float,8>,16> act;
struct Pending {
  uint64_t desc;
  const uint32_t* src;
  std::array<uint32_t,4> saved;
  float* dest;
};
std::vector<Pending> pending;
unsigned max_unretired;
struct MmaOpClass {
  static void fma(uint64_t desc, uint32_t* src, float* dest, bool add) {
    assert(add);
    Pending p{desc,src,{},dest};
    std::memcpy(p.saved.data(),src,16);
    pending.push_back(p);++calls;
    max_unretired=std::max<unsigned>(max_unretired,pending.size());
  }
};
void wgmma_commit() {}
template<unsigned Groups> void wgmma_wait() {
  static_assert(Groups==0);
  assert(!pending.empty());
  ++drains;
  for (auto p:pending) {
    assert(std::memcmp(p.saved.data(),p.src,16)==0);
    registers_checked+=4;
    unsigned stage=p.desc/sizeof(Storage::StageStorage);
    unsigned kslice=(p.desc%sizeof(Storage::StageStorage))/(2*sizeof(int4));
    auto &a=act[4*stage+kslice];
    for(unsigned m=0;m<MUSE_OVERLAP_M/2;m++) {
      for(unsigned i=0;i<8;++i) {
        uint32_t bf=(p.src[i/2] >> ((i%2)*16))&65535;
        p.dest[m]=std::fma(a[(m+i)%8],std::bit_cast<float>(bf<<16),p.dest[m]);
        ++dot_products;
      }
    }
  }
  pending.clear();
}
template<class Context,class Arith> struct WGMMA {
  using SharedStorage=Storage;
  struct Holder { Storage smem{}; } ctx;
  uint32_t regs_b[2][2][1][4]{};
  float regs_c[1][2][1][MUSE_OVERLAP_M/2]{};
  unsigned smem_offset=0;
  static constexpr unsigned kWarpIters=4,kPackedKFactor=1,kRegsBKDim=1,kSwizzleBytes=128;
  static constexpr bool kUsePackedKLayout=false;
  static constexpr unsigned final_regs_c_index(){ return 0; }
  void run(unsigned stage,unsigned iter) {
    uint64_t desc=stage*sizeof(SharedStorage::StageStorage)+iter*2*sizeof(int4);
    for(unsigned j=0;j<2;++j) MmaOpClass::fma(desc,regs_b[iter%2][j][0],regs_c[0][j][0],true);
    wgmma_commit();wgmma_wait<0>();
  }
};
#include HEADER
using Base=WGMMA<MuseOverlapCtx,MainloopArithmetic<MuseOverlapCtx>>;
template<class Op> auto compute(int stages,int seed) {
  Op op{}; std::mt19937 rng(seed);
  for(int s=0;s<stages;++s) {
    for(unsigned i=0;i<4;++i) {
      for(auto &x : act[(s%4)*4+i]) x=std::ldexp(float(int(rng()%255)-127),int(rng()%15)-7);
      for(auto &halves : op.regs_b[i%2]) for(auto &pair : halves[0]) {
        // Includes signed zero, cancellation and distinct positive/negative scale magnitudes.
        uint32_t vals=0;
        for(unsigned k=0;k<2;++k) {
          float w=std::ldexp(float(int(rng()%13)-6),int(rng()%17)-8);
          if(seed%3==0 && (k+i)%2==0) w=-0.0f;
          vals|=(std::bit_cast<uint32_t>(w)>>16)<<(k*16);
        }
        pair=vals;
      }
      op.run(s%4,i);
      if(i>=2) assert(pending.empty()); // native shared-stage release + final stage exit.
    }
  }
  std::array<float,MUSE_OVERLAP_M> output;
  std::memcpy(output.data(),op.regs_c,sizeof(output));
  return output;
}
int main() {
  uint64_t outputs=0, cases=0;
  for(unsigned seed=0;seed<11;++seed) for(int s : {1,2,3,4,5,15,16,63,64}) {
    auto base=compute<Base>(s,seed);
    auto test=compute<MuseOverlapMma>(s,seed);
    assert(std::memcmp(base.data(),test.data(),sizeof(base))==0);
    assert(pending.empty());outputs+=base.size();++cases;
  }
  assert(max_unretired==4);
  std::printf("{\"cases\":%llu,\"fp32_outputs\":%llu,\"scalar_cpu_fmas\":%llu,\"matrix_instructions\":%llu,\"delayed_source_register_checks\":%llu,\"max_pending\":%u}\n",
    (unsigned long long)cases,(unsigned long long)outputs,(unsigned long long)dot_products,(unsigned long long)calls,(unsigned long long)registers_checked,max_unretired);
}
