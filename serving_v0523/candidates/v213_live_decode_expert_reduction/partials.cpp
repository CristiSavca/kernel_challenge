// Both production reducers are included unmodified except for cooperative barrier yields.
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>
#define CUDA_INLINE inline
#define PRAGMA_UNROLL
#define MAX(a,b) ((a)>(b)?(a):(b))
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
struct alignas(16) int4 {uint32_t v[4];};
struct alignas(16) float4 {float x,y,z,w;};
struct Float16{}; struct BF16{}; struct half2{}; struct nv_bfloat162{};
float __fadd_rn(float a,float b) {return a+b;}
struct {unsigned x;} threadIdx;
struct Task {
 struct promise_type {
  Task get_return_object(){return {std::coroutine_handle<promise_type>::from_promise(*this)};}
  std::suspend_always initial_suspend(){return {};}
  std::suspend_always final_suspend() noexcept{return {};}
  void return_void(){} void unhandled_exception(){std::terminate();}
 };
 std::coroutine_handle<promise_type> h;
};
template<unsigned R,unsigned G> struct Ctx {
 struct BlockShape {enum {K=64*G,N=512/G,M=R};};
 struct WarpShape {enum {K=64,N=32,M=R};};
 struct MmaOpClass {using ValTypeC=float;struct MmaShape{};};
 using ElementC=BF16;
 struct TuningConfig {enum {kReduceOverlapLastStageOnly=0};};
 enum {kNumWriteSplits=1,kNumMathThreads=512,kIsGroupedContiguousGemm=1,kUseWgmma=1,kUseTmaC=0,kUseStreamK=0};
 enum {kWeightScaleGroupSize=16,kInputScaleGroupSize=0,kUseFusedE8m0Scale=0,kUseIntWeightScale=0};
 struct SharedStorage {
   alignas(128) union {
    int4 stages[8192];
    int4 reduce[2048];
   };
 };
 SharedStorage& smem;
 unsigned warp_id()const{return threadIdx.x/32;}
 unsigned lane_id()const{return threadIdx.x%32;}
};
template<unsigned R>struct Matrix {using CRegistersArrayType=float[2][1][R/2];};
struct Arith{};
template<class C,class M,class A> struct EpiloguePipeline {
 C &ctx;
 struct SharedWriter { void write(unsigned*,unsigned,unsigned){} } smem_writer;
 struct GlobalWriter { void write(unsigned,unsigned,unsigned){} } gmem_writer;
 explicit EpiloguePipeline(C& c):ctx(c){}
 void seek(unsigned,unsigned,unsigned,unsigned,unsigned){}
};
#include "native.h"
#include "submitted.h"
uint64_t cells=0,order_sensitive=0,zeros=0,spectra=0,cases=0;
template<unsigned R,unsigned G>void exercise(unsigned rows,unsigned seed){
 using C=Ctx<R,G>;using M=Matrix<R>;constexpr unsigned parts=R,owners=512/G;
 typename C::SharedStorage sa{},sb{};C ca{sa},cb{sb};
 auto old=EpilogueSmemReducer<C,M>(ca);auto fresh=MuseLiveDecodeEpilogue<C,M,Arith>(cb);
 fresh.seek(0,0,0,rows+33,33);
 alignas(16) std::array<std::array<float,parts>,512>a{},b{};
 std::mt19937 rng(seed);
 constexpr uint32_t probes[]={0,0x80000000u,0x00800000u,0x80800000u,0x7f7fffffu,0xff7fffffu,
  0x3f800000u,0xbf800000u,0x40000000u,0xc0000000u,0x71800000u,0xf1800000u,0x7f800000u,0xff800000u,0x7fc00000u};
 for(unsigned g=0;g<G;g++)for(unsigned t=0;t<owners;t++)for(unsigned i=0;i<parts;i++){
   uint32_t n=rng();float f=std::bit_cast<float>((n & 0x807fffffu) | ((rng()%240+5)<<23));
   if(i%5==0) f=std::bit_cast<float>(probes[rng()%(sizeof(probes)/4)]);
   if((t%7)==0 && G==4)f=(g==0?0x1p100f:g==1?-0x1p100f:g==2?1.f:0.f);
   a[g*owners+t][i]=b[g*owners+t][i]=f;
 }
 auto original=a;
 auto run=[&](bool candidate){
   std::array<Task,512> ts;
   for(threadIdx.x=0;threadIdx.x<512;threadIdx.x++) {
     auto* p=reinterpret_cast<unsigned*>((candidate?b:a)[threadIdx.x].data());
     ts[threadIdx.x]=candidate?fresh.call(p):old.reduce(p);
   }
   std::array<unsigned,512> schedule{};std::iota(schedule.begin(),schedule.end(),0);
   unsigned stages=0;
   for(;;) {
     std::shuffle(schedule.begin(),schedule.end(),rng);
     unsigned done=0;
     for(unsigned t:schedule){threadIdx.x=t;ts[t].h.resume();done+=ts[t].h.done();}
     assert(done==0 || done==512);
     if(done)break;
     stages++;
   }
   for(auto t:ts)t.h.destroy();
   assert(candidate ? stages==4 : stages==(R>16?2u:1u)*(G==4?4u:2u));
 };
 run(false);run(true);
 for(unsigned lane=0;lane<owners;lane++)for(unsigned j=0;j<parts;j++){
   unsigned reg=j%(R/2);
   unsigned token=(lane%4)*2+reg%2+8*(reg/4);
   if(token>=rows) continue;
   uint32_t a_bits=std::bit_cast<uint32_t>(a[lane][j]), b_bits=std::bit_cast<uint32_t>(b[lane][j]);
   assert(a_bits==b_bits);
   if((b_bits&0x7fffffff)==0)zeros++;
   if((b_bits&0x7f800000)==0x7f800000)spectra++;
   if(G==4) {
      float alt=original[lane][j];
      for(unsigned g=1;g<G;g++)alt+=original[g*owners+lane][j];
      order_sensitive+=std::bit_cast<uint32_t>(alt)!=b_bits;
   }
   cells++;
 }
 cases++;
}
int main(){
 for(unsigned i=0;i<=32;i++)for(unsigned seed:{1u,20019u,570u,937u}){
   exercise<8,2>(i,seed);exercise<8,4>(i,seed);exercise<16,2>(i,seed);exercise<16,4>(i,seed);
   exercise<24,2>(i,seed);exercise<24,4>(i,seed);
 }
 assert(order_sensitive>1000 && zeros>0);
 printf("{\"cases\":%llu,\"FP32_bit_comparisons\":%llu,\"linear_K_order_counterexamples\":%llu,\"signed_or_positive_zeros\":%llu,\"nonfinite_outputs\":%llu}\n",
  (unsigned long long)cases,(unsigned long long)cells,(unsigned long long)order_sensitive,(unsigned long long)zeros,(unsigned long long)spectra);
}
