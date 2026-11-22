#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#define SGL_DEVICE inline
#define __global__
#define __grid_constant__
struct float4 { float x,y,z,w; };
inline float4 make_float4(float x,float y,float z,float w) {return {x,y,z,w};}
template<class T> struct DTypeTrait;
struct bf16_t {};
template<> struct DTypeTrait<bf16_t> {
  static uint16_t from(float x) {
    uint32_t b=std::bit_cast<uint32_t>(x);
    return (b+0x7fff+((b>>16)&1))>>16;
  }
};
struct Dim { unsigned x; };
inline thread_local Dim threadIdx,blockIdx;
struct Warp {
  std::barrier<> sync{32}; std::array<uint32_t,32> bits{};
};
inline thread_local Warp *warp_state;
inline unsigned lane() {return threadIdx.x % 32;}
template<typename T,typename F> T collective(T val, F combine) {
 auto &w=*warp_state; w.bits[lane()]=std::bit_cast<uint32_t>(val);w.sync.arrive_and_wait();
 T out=combine(w.bits);w.sync.arrive_and_wait();return out;
}
inline unsigned __ballot_sync(unsigned,int pred){
 return collective<unsigned>(pred,[](const auto &v){unsigned mask=0;for(unsigned i=0;i<32;i++) mask|=v[i]<<i;return mask;});
}
inline int __ffs(unsigned a){return __builtin_ffs(a);}
template<class T> T __shfl_sync(unsigned,T val,int src){
 return collective<T>(val,[src](const auto &v){return std::bit_cast<T>(v[src]);});
}
namespace device {
 constexpr int kWarpThreads=32;
 template<bool B> void PDLWaitPrimary(){}
 template<bool B> void PDLTriggerSecondary(){}
 namespace warp {
 constexpr unsigned kFullMask=~0u;
 inline float reduce_max(float a){return collective<float>(a,[](const auto &v){float y=-INFINITY; for(auto z:v)y=std::fmax(y,std::bit_cast<float>(z));return y;});}
 inline float reduce_sum(float a){return collective<float>(a,[](const auto &v){float y=0; for(auto z:v)y+=std::bit_cast<float>(z);return y;});}
 }
}
#include "original_warp.hpp"
#include "candidate_warp.hpp"

template<class T> bool same(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(T));}
static int cases=0,rows=0;
template<int K,int EPL,int Norm,int Warps>
void compare(int n,int m,bool use_map,bool padded,int pattern){
 std::mt19937 gen(254+n*7+K+pattern);
 std::vector<float> h(m*n),l(m*n),summed(m*n),bias(n),wa(m*K,-113.f),wb(wa);
 std::vector<int> idsA(m*K,313),idsB(idsA),pkA(m*K,-103),pkB(pkA),mp(n);
 float special[]={-0.f,0.f,0x1p-149f,-0x1p-149f,0x1p-126f,-0x1p-126f,1.f,-1.f,0.5f,0x1p-24f};
 for(int i=0;i<m*n;i++) {
    h[i]=int(gen()%31-15)/8.f;l[i]=int(gen()%57-28)*0x1p-18f;
    if(pattern==1){h[i]=special[gen()%10];l[i]=special[gen()%10];}
    if(pattern==2){h[i]=1.f;l[i]=(i%n==n-1)?0x1p-20f:0.f;}
    if(pattern==3){h[i]=(i%3)?16777216.f:-16777216.f;l[i]=-h[i]+float(int(gen()%5)-2);}
    summed[i]=h[i]+l[i];
 }
 for(int i=0;i<n;i++){bias[i]=(pattern==2)?0.f:(int(gen()%7)-3)*0x1p-15f;mp[i]=(i%3==0)?-1:int(gen()%32);}
 int count=padded?m-1:m;
 original::RoutingParams a{};candidate::RoutingParams b{};
 #define SET(p,input,weights,ids,packed) p.gating_output=(input).data();p.bias=bias.data();p.topk_weights=(weights).data();p.topk_ids=(ids).data();p.packed_topk_ids=(packed).data();p.local_expert_mapping=use_map?mp.data():nullptr;p.num_valid_tokens=padded?&count:nullptr;p.num_tokens=m;p.num_experts=n;p.stride_gm=n;p.stride_im=K;p.stride_pm=K;p.stride_wm=K;p.router_output_multiplier=pattern%2?1.f:.75f;p.route_scale=1.25f;p.division_eps=1e-7f
 SET(a,summed,wa,idsA,pkA);SET(b,h,wb,idsB,pkB);b.residual=l.data();
 #undef SET
 Warp warp;
 std::vector<std::jthread> threads;
 for(int i=0;i<32;i++)threads.emplace_back([&,i]{
   warp_state=&warp;
   for(int t=0;t<m;t++){
    threadIdx.x=(t%Warps)*32+i;blockIdx.x=t/Warps;
    original::routing_kernel<original::RoutingTrait<K,EPL,Norm,Warps>,false>(a);
    candidate::routing_kernel<candidate::RoutingTrait<K,EPL,Norm,Warps>,false,true>(b);
   }
 });
 threads.clear();
 if(!same(wa,wb)||!same(idsA,idsB)||!same(pkA,pkB))std::abort();
 cases++;rows+=m;
}
int main(){
 for(int p=0;p<4;p++)for(bool remap:{false,true})for(bool pad:{false,true}){
  compare<8,8,0,1>(256,3,remap,pad,p);
  compare<8,8,1,4>(252,5,remap,pad,p);
  compare<4,4,0,1>(124,2,remap,pad,p);
 }
 std::cout<<"{\"cases\":"<<cases<<",\"routed_rows\":"<<rows<<"}\n";
}
