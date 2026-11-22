// Coroutine scheduler for the staged CUDA kernel. Each CUDA thread pauses at
// its shuffle and barriers; all 128 values become visible before the next step.
#include <array>
#include <bit>
#include <cmath>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
using bf16_t = uint16_t;
using Vec = std::array<bf16_t, 4>;
template <typename T> T load_as(const void* p, int index) {
  T out; std::memcpy(&out, static_cast<const char*>(p)+sizeof(T)*index,sizeof(T)); return out;
}
template <typename T> void store_as(void* p, T x, int index) {
  std::memcpy(static_cast<char*>(p)+sizeof(T)*index,&x,sizeof(T));
}
template <typename To, typename From> To cast(From x) {
  if constexpr (std::is_same_v<To,float>) return std::bit_cast<float>(uint32_t(x)<<16);
  const auto bits=std::bit_cast<uint32_t>(float(x));
  if((bits&0x7fffffffU)>0x7f800000U) return uint16_t(0x7fff);
  return uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
float __fmaf_rn(float x,float y,float z){return std::fma(x,y,z);}
float __fadd_rn(float x,float y){return x+y;}
float __fmul_rn(float x,float y){return x*y;}
float __fdiv_rn(float x,float y){return x/y;}
float rsqrtf(float x){return 1.0f/std::sqrt(x);}
struct Axis{int x;};
static Axis blockIdx,threadIdx;
static std::array<float,128> exchange;
struct Step {
  struct promise_type {
    Step get_return_object(){return {std::coroutine_handle<promise_type>::from_promise(*this)};}
    std::suspend_always initial_suspend() noexcept{return {};}
    std::suspend_always final_suspend() noexcept{return {};}
    void unhandled_exception(){std::abort();}
    void return_void(){}
  };
  std::coroutine_handle<promise_type> h;
};
struct Shuffle {
  float x; int distance;
  bool await_ready() const{return false;}
  void await_suspend(std::coroutine_handle<>) const{exchange[threadIdx.x]=x;}
  float await_resume() const;
};
static std::array<float,128> earlier;
float Shuffle::await_resume() const{
  int lane=threadIdx.x%32;
  return earlier[lane+distance<32 ? threadIdx.x+distance : threadIdx.x];
}
#define SGL_DEVICE inline
#define __grid_constant__
#define __shared__ static
constexpr int kThreads=128;
#include "kernel.inc"

extern "C" void run(const bf16_t* packed,const bf16_t* gain,const int32_t* map,
                    bf16_t* out,int ranks,int rows,int width,int topk,float epsilon) {
  int vectors=width/4, stripe=vectors/ranks;
  Params p{packed,gain,map,out,rows,vectors,stripe,std::countr_zero(unsigned(stripe)),topk,epsilon};
  for(int token=0;token<rows;++token) {
    // Reverse arrival of alternating CTAs to vary lane order within each barrier.
    blockIdx.x=token;
    std::array<Step,128> threads;
    for(int t=0;t<128;++t)threads[t]=normalize_to_routes(p);
    for(int phase=0;phase<8;++phase) {
      earlier=exchange;
      for(int i=0;i<128;++i) {
        int t=token%2 ? 127-i : i;
        threadIdx.x=t;
        threads[t].h.resume();
        if(threads[t].h.done() != (phase==7))std::abort();
      }
    }
    for(auto t:threads)t.h.destroy();
  }
}
