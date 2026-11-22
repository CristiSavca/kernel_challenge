#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using U=uint32_t;
struct Dim {U x;} threadIdx;
struct uint4 {U x,y,z,w;};
uint4 make_uint4(U x,U y,U z,U w){return {x,y,z,w};}
#define CUDA_INLINE inline
#define __shared__ static
using Warp=std::array<U,32>;
using Journal=std::array<Warp,18>;
Journal prev,now;
U lane,ip;
bool invalid;
U save(U mask,U v){assert(mask==~0u&&ip<18); now[ip][lane]=v;return ip++;}
U __shfl_sync(U mask,U v,U l){return prev[save(mask,v)][l%32];}
U __shfl_up_sync(U mask,U v,U delta){return prev[save(mask,v)][lane<delta?lane:lane-delta];}
U __ballot_sync(U mask,bool v){auto &record=prev[save(mask,v)];U b=0;for(U i=0;i<32;++i)b|=U(!!record[i])<<i;
 if(std::popcount(b)!=1){invalid=true;if(!b)b=1;} return b;}
int __ffs(U bits){return bits?std::countr_zero(bits)+1:0;}
#include "candidate.inc"
template<U E>struct Context{
 struct BlockShape {static constexpr U M=256;};
 static constexpr U kNumExperts=E;
 static constexpr bool kIsGroupedContiguousGemm=true,kUseTmaC=false,kUseStreamK=false;
 struct {U expert_tokens[E],expert_offset[E+1];} smem{};
};
template<class F>void replay(U warp,F func){
 prev={};
 for(U p=0;p<19;++p){now={};invalid=false;
  for(lane=0;lane<32;++lane){ip=0;threadIdx.x=warp*32+lane;func();}
  if(prev==now && !invalid)return;
  prev=now;
 }
 std::abort();
}
struct Tile{U canonical,e,start,end,width;};
uint64_t cases=0,lookups=0,lanes=0,backward=0,rises=0,falls=0;
template<U E>void one(std::array<U,E> counts,U prefix){
 Context<E> ctx; std::vector<Tile> tiles;
 ctx.smem.expert_offset[0]=prefix;
 for(U e=0;e<E;e++){
  ctx.smem.expert_tokens[e]=counts[e];ctx.smem.expert_offset[e+1]=ctx.smem.expert_offset[e]+counts[e];
  for(U start=0;start<counts[e];start+=256){
    U rem=counts[e]-start;U wid=rem<=32?32:rem<=128?128:rem<=224?224:256;
    tiles.push_back({U(tiles.size()),e,ctx.smem.expert_offset[e]+start,ctx.smem.expert_offset[e+1],wid});
  }
 }
 auto oracle=tiles;
 std::stable_sort(oracle.begin(),oracle.end(),[](auto a,auto b){
   if(a.e/4!=b.e/4)return a.e/4<b.e/4;
   return ((a.e/4)%2)?a.width<b.width:a.width>b.width;
 });
 replay(0,[&]{muse_init_width_order(ctx);});
 U last=0;
 for(U i=0;i<oracle.size();i++){auto ref=oracle[i];
  if(i&&last>ref.canonical)backward++;
  if(i&&oracle[i-1].width<ref.width)rises++;
  if(i&&oracle[i-1].width>ref.width)falls++;
  last=ref.canonical;
  std::array<std::array<U,4>,32> res;
  replay((i*19+E)%12,[&]{auto &x=res[lane];x={i,~0u,~0u,~0u};muse_seek_width(ctx,x[0],x[1],x[2],x[3]);});
  for(auto &x:res){std::array<U,4> expect={ref.canonical,ref.e,ref.start,ref.end};assert(x==expect);}
  ++lookups;lanes+=32;
 }
 ++cases;
}
template<U E>void generate(std::mt19937 &r){
 std::array<U,E> a{};one<E>(a,0);
 std::vector<U> edges={0,1,31,32,33,127,128,129,223,224,225,255,256,257,511,512,513};
 for(U e=0;e<E;++e)for(U n:edges){a={};a[e]=n;one<E>(a,37);}
 for(int k=0;k<45;++k){for(auto &x:a)x=(k%2)?(r()%1000):edges[r()%edges.size()];one<E>(a,0);}
}
int main(){std::mt19937 r(280);generate<1>(r);generate<3>(r);generate<4>(r);generate<7>(r);generate<17>(r);generate<31>(r);generate<32>(r);
 assert(backward>0&&rises>0&&falls>0);
 std::cout<<"cases "<<cases<<" rows "<<lookups<<" lane-outputs "<<lanes<<" backwards "<<backward<<" rises "<<rises<<" falls "<<falls<<"\n";
}
