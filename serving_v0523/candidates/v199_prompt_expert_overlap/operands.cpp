#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <type_traits>
using std::uint32_t;
#define CUDA_INLINE inline
#define PRAGMA_UNROLL
struct uint2{uint32_t x,y;};
struct uint4{uint32_t x,y,z,w;};
uint2 make_uint2(uint32_t x,uint32_t y){return {x,y};}
template<int B,int E,int M> struct FP {
  static constexpr int kBits=B,kExponentBits=E,kMantissaBits=M;
  static constexpr bool kIsFloatingPointType=true,kIsSigned=true;
};
using BF16=FP<16,8,7>; using E2M1=FP<4,2,1>;using E4M3=FP<8,4,3>;using Float8E8M0=FP<8,8,0>;using BFloat16=BF16;
uint32_t lop3_and_or(uint32_t a,uint32_t b,uint32_t c){return (a&b)|c;}
template<class SourceType,class TargetType> @@CONVERT@@
template<class S,class T,bool A,bool B> uint32_t dequant_single(uint32_t v,uint32_t){return fp_to_fp<S,T>(v);}
template<class SourceType,class TargetType,bool kHasZeroPoint,bool kIsFpZeroPoint> @@UNPACK@@
template<class S,class T> void dequant(uint32_t* w,uint32_t* d,uint32_t j,uint32_t* zero){dequant_b1248<S,T,false,false>(w,d,j,zero);}
uint32_t bf(float f){auto bits=std::bit_cast<uint32_t>(f);return (bits+0x7fff+((bits>>16)&1))>>16;}
float as_float(uint32_t b){return std::bit_cast<float>(b<<16);}
uint32_t scalar_mul(uint32_t a,uint32_t b){return bf(as_float(a)*as_float(b));}
uint32_t mul2(uint32_t a,uint32_t b){return scalar_mul(a&65535,b&65535)|(scalar_mul(a>>16,b>>16)<<16);}
uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t mask){uint64_t src=uint64_t(b)<<32|a;uint32_t out=0;for(int i=0;i<4;++i)out|=((src>>(((mask>>(4*i))&7)*8))&255)<<(8*i);return out;}
@@LOADER@@
struct Context {
  static constexpr int kNumMathThreads=256,kNumLoadThreads=128,kWarpIters=4,kNumStages=4,N_WARPS=8,M_WARPS=1,K_WARPS=1,kMultiCastSize=1;
  static constexpr bool kUseTmaBS=false,kUseCpAsync=false,kUseTmaA=true,kUseTmaB=true;
  struct ProblemShape{static constexpr int N=4096;};
  using ElementA=BF16;using ElementB=E2M1;
  struct alignas(128) SharedStorage{uint32_t dummy[11];} smem;
  MusePromptScaleStages<Context> expanded;
  uint32_t tid;
  uint32_t load_thread_id()const{return tid;}
};
struct Loader {const uint32_t* gmem_ptr;int advance_count=0;void advance(){++advance_count;}};
template<class Ctx>struct Transform {uint4 regs_b[2][2]{};uint32_t regs_qb[2][2]{};uint2 scales[2];
  @@TRANSFORM@@
};
int main(){
  Context c{};
  static_assert(offsetof(Context,expanded)==sizeof(Context::SharedStorage));
  uint32_t packed[4096];uint64_t checked=0,zeros=0,negzero=0,words=0;
  bool seen[127][16]={};
  for(unsigned seed=0;seed<127;++seed){
    for(unsigned n=0;n<4096;++n){
      unsigned x=0;for(int h=0;h<4;++h){unsigned raw=(n*5+h*19+seed)%127;x|=(raw<8?0:raw)<<(8*h);}
      packed[n]=x;
    }
    unsigned stage=seed%4;
    for(c.tid=0;c.tid<128;++c.tid){
      Loader l{packed};
      if(stage&1)muse_load_scales<true>(c,l,stage);else muse_load_scales<false>(c,l,stage);
      assert(l.advance_count==int(stage&1));
    }
    for(unsigned k=0;k<4;++k)for(unsigned warp=0;warp<8;++warp)for(unsigned lane=0;lane<32;++lane){
      const unsigned column=(lane/4)*2+(warp%2)+(warp/2)*16;
      const uint32_t word=packed[1024*k+column];
      const auto pair=c.expanded.scales[stage][k][column];
      uint32_t scale_word[2]={word,0},zero[4]={},dq[4];
      dequant<E4M3,BF16>(scale_word,dq,0,zero);
      uint16_t original[8];for(unsigned j=0;j<8;++j)original[j]=(dq[j/2]>>(16*(j%2)))&65535;
      std::swap(original[1],original[2]);std::swap(original[5],original[6]);
      uint32_t ref[4];
      for(unsigned j=0;j<4;++j){ref[j]=scalar_mul(original[j],(120+127)<<7);assert(ref[j]==((j<2?pair.x:pair.y)>>(16*(j%2))&65535));++words;}
      for(unsigned offset=0;offset<16;++offset){
        Transform<Context> t{};const unsigned buf=k%2;t.scales[buf]=pair;
        for(unsigned j=0;j<2;++j)for(unsigned h=0;h<8;++h)t.regs_qb[buf][j]|=((h*5+j+offset)&15)<<(4*h);
        t.transform_b(buf,0);
        for(unsigned j=0;j<2;++j)for(unsigned i=0;i<4;++i)for(unsigned h=0;h<2;++h){
          const unsigned code=t.regs_qb[buf][j]>>(4*i+16*h)&15;
          const uint32_t tiny=((code&7)<<6)|((code&8)<<12);
          const uint32_t stock=scalar_mul(scalar_mul(tiny,(124+127)<<7),ref[2*j+(i&1)]);
          const uint32_t got=reinterpret_cast<uint32_t*>(&t.regs_b[buf])[4*j+i]>>(16*h)&65535;
          const unsigned raw=(5*(1024*k+column)+19*(2*j+(i&1))+seed)%127;
          const float scale=raw<8?0:std::ldexp(1+(raw%8)*0.125f,int(raw/8)-7);
          constexpr float fp4[]={0,0.5f,1,1.5f,2,3,4,6};
          const float logical=(code&8?-1.f:1.f)*fp4[code&7]*0.25f*scale;
          assert(got==stock && got==bf(logical));seen[raw][code]=true;++checked;
          zeros+=(got&32767)==0;negzero+=got==32768;
        }
      }
    }
  }
  for(auto &row:seen)for(bool value:row)assert(value);
  volatile float activation=std::ldexp(1.0f,126);
  volatile float internal=activation*1.0f,dense=activation*4.0f;
  assert(std::isfinite(internal)&&std::isinf(dense));
  std::printf("{\"compared_bf16_operands\":%llu,\"zero_results\":%llu,\"negative_zero\":%llu,\"native_scale_words_checked\":%llu,\"covered_input_code_pairs\":2032,\"bf16_dense_overflow_witness\":true}\n",
    (unsigned long long)checked,(unsigned long long)zeros,(unsigned long long)negzero,(unsigned long long)words);
}
