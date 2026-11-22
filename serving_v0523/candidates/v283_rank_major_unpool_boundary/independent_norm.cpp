// Run the literal single-CTA CUDA kernel body as a host coroutine per lane.
#include <array>
#include <bit>
#include <cmath>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using bf16_t = uint16_t;
using fp32_t = float;
struct fp32x2_t { float x,y; };
struct bf16x2_t { uint16_t x,y; };
constexpr int kMaxVecBytes=16, kWarpThreads=32;
inline uint16_t bf16(float v) {
  auto b=std::bit_cast<uint32_t>(v);
  if((b&0x7fffffffU)>0x7f800000U)return 0x7fff;
  return uint16_t((b+0x7fffU+((b>>16)&1))>>16);
}
template<class To,class From> To cast(From v) {
  if constexpr(std::is_same_v<To,fp32x2_t>)
    return {std::bit_cast<float>(uint32_t(v.x)<<16),std::bit_cast<float>(uint32_t(v.y)<<16)};
  else return {bf16(v.x),bf16(v.y)};
}
template<class T,int N> struct AlignedVector {
  T elem[N];
  template<class U> void load(const U *ptr,int index) {std::memcpy(elem,reinterpret_cast<const char*>(ptr)+index*sizeof(elem),sizeof(elem));}
  template<class U> void store(U *ptr,int index) const {std::memcpy(reinterpret_cast<char*>(ptr)+index*sizeof(elem),elem,sizeof(elem));}
  T& operator[](int i){return elem[i];}
  const T& operator[](int i)const{return elem[i];}
};
template<bool> void PDLWaitPrimary(){}
template<bool> void PDLTriggerSecondary(){}
namespace math { inline float rsqrt(float x){ return 1.0f/std::sqrt(x); } }
struct Index { int x; };
thread_local Index blockIdx{},threadIdx{};
thread_local std::array<float,1024> partial{},broadcast{};
struct Step {
  struct promise_type {
    Step get_return_object(){return {std::coroutine_handle<promise_type>::from_promise(*this)};}
    std::suspend_always initial_suspend() noexcept {return {};}
    std::suspend_always final_suspend() noexcept {return {};}
    void unhandled_exception(){std::abort();}
    void return_void(){}
  };
  std::coroutine_handle<promise_type> h;
};
struct Reduce {
  float input;
  bool await_ready() const{return false;}
  void await_suspend(std::coroutine_handle<>) const{partial[threadIdx.x]=input;}
  float await_resume() const{return broadcast[threadIdx.x];}
};
static std::array<float,32> warp_sum(std::array<float,32> x){
  for(int mask=16;mask;mask>>=1){
    auto old=x;
    for(int i=0;i<32;++i)x[i]=old[i]+old[i^mask];
  }
  return x;
}
static void cta_sum(int threads){
  std::array<float,32> scratch{};
  for(int w=0;w<threads/32;++w){
    std::array<float,32> warp;
    std::memcpy(warp.data(),partial.data()+32*w,sizeof(warp));
    scratch[w]=warp_sum(warp)[0];
  }
  auto reduced=warp_sum(scratch);
  for(int t=0;t<threads;++t)broadcast[t]=reduced[t%32];
}
#define SGL_DEVICE inline
#define __shared__
#define __grid_constant__
#include "common.inc"
namespace original {
#include "before.inc"
}
namespace candidate {
#include "after.inc"
}
namespace incorrect {
#include "wrong.inc"
}

template<int T,int R,bool Native,bool Wrong,bool Scale,bool Router,bool Weight,bool Gate>
static void testrow(NormResidualParams p,int row){
  blockIdx.x=row;
  std::vector<Step> lanes;
  lanes.reserve(T);
  for(int t=0;t<T;++t) {
    if constexpr(Native) lanes.push_back(original::muse_spark_norm_residual_kernel<8192,T,false,T==256,Scale,Router,Weight,Gate>(p));
    else if constexpr(Wrong) lanes.push_back(incorrect::muse_spark_norm_residual_kernel<8192,T,false,T==256,Scale,Router,Weight,Gate,R>(p));
    else lanes.push_back(candidate::muse_spark_norm_residual_kernel<8192,T,false,T==256,Scale,Router,Weight,Gate,R>(p));
  }
  for(int phase=0;phase<3;++phase){
    for(int t=0;t<T;++t) {
      threadIdx.x=t; lanes[t].h.resume();
      if(lanes[t].h.done()!=(phase==2))std::abort();
    }
    if(phase!=2)cta_sum(T);
  }
  for(auto &v:lanes)v.h.destroy();
}

template<int T,int R,bool Native,bool Wrong> static int launch(int mode,NormResidualParams p,int row){
  if(mode==0)testrow<T,R,Native,Wrong,true,true,true,true>(p,row);
  else if(mode==1)testrow<T,R,Native,Wrong,false,false,false,false>(p,row);
  else if(mode==2)testrow<T,R,Native,Wrong,false,true,true,true>(p,row);
  else if(mode==3)testrow<T,R,Native,Wrong,true,false,true,true>(p,row);
  else return 2;
  return 0;
}
template<int T> static int dispatch(int fmt,int mode,NormResidualParams p,int row){
  switch(fmt){
    case -1:return launch<T,1,true,false>(mode,p,row);
    case 1:return launch<T,1,false,false>(mode,p,row);
    case 4:return launch<T,4,false,false>(mode,p,row);
    case 8:return launch<T,8,false,false>(mode,p,row);
    case -4:return launch<T,4,false,true>(mode,p,row);
  }
  return 3;
}
extern "C" int norm_row(int threads,int fmt,int mode,int row,NormResidualParams p){
  return threads==256?dispatch<256>(fmt,mode,p,row):threads==512?dispatch<512>(fmt,mode,p,row):4;
}
