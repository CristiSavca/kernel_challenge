"""Execute the published group's real expert/row locator with the emitted V208 residual dispatch."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

root=Path(sys.argv[1])
native=(root/"humming/include/humming/scheduler.cuh").read_text()
cu=(root/"images/ResidualPromptWeightPipeline_176.cu").read_text()
fun=native[native.index("  void fetch_moe_group_block()"):native.index("  void fetch_moe_index_block()")]
fun=fun[:fun.index("  CUDA_INLINE")]
a=cu.index("      const auto live = scheduler.current_shape_m")
b=cu.index("      else compute(MuseFullWidth{});",a)+len("      else compute(MuseFullWidth{});")
test=r"""
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
struct BlockShape {enum {M=176};};
struct Ctx {
  enum {kIsGroupedMaskedGemm=0,kIsGroupedContiguousGemm=1};
  struct Table {std::array<unsigned,32> expert_tokens;std::array<unsigned,33> expert_offset;} smem{};
};
struct Scheduler {
 Ctx &ctx;
 uint32_t old_m_block_id=0,old_expert_id=-1,offset_in_expert=0,expert_id=0,m_block_id=0,current_shape_m=0,m_offset=0,expert_max_num_tokens=0;
 void update_tensor_map_c(){}
 FUNCTION
};
template<unsigned N> struct Tag {enum {width=N};};
using MuseResidual32=Tag<32>;using MuseResidual96=Tag<96>;using MuseFullWidth=Tag<176>;
int main(){
 std::mt19937 rng(281);
 uint64_t jobs=0,empty=0;
 for(unsigned repeat=0;repeat<500;repeat++){
   Ctx ctx;
   std::vector<std::array<unsigned,3>> tiles;
   for(unsigned e=0;e<32;e++){
     unsigned n= repeat==0?0:(rng()%3? rng()%1720:0);
     if(repeat<=176 && e==31)n=repeat;
     empty+=n==0;
     auto &s=ctx.smem;
     s.expert_tokens[e]=n;s.expert_offset[e+1]=s.expert_offset[e]+n;
     for(unsigned off=0;off<n;off+=176)tiles.push_back({e,s.expert_offset[e]+off,std::min(176u,n-off)});
   }
   for(unsigned bands:{16,32})for(unsigned grid:{1u,132u,264u}){
     unsigned work=tiles.size()*bands;
     for(unsigned block=0;block<std::min(grid,work);block++){
       Scheduler scheduler{ctx};
       for(unsigned job=block;job<work;job+=grid){
         scheduler.m_block_id=job/bands;
         scheduler.fetch_moe_group_block();
         auto expected=tiles[job/bands];
         assert(scheduler.expert_id==expected[0] && scheduler.m_offset==expected[1]);
         unsigned width=0;
         auto compute=[&](auto op){width=decltype(op)::width;};
         DISPATCH
         unsigned needed=expected[2];
         assert(needed>0 && width==(needed<=32?32:needed<=96?96:176));
         ++jobs;
       }
     }
   }
 }
 std::printf("{\"dispatched_CTA_jobs\":%llu,\"empty_groups\":%llu,\"histograms\":500,\"persistent_grid_sizes\":[1,132,264]}\n",
   (unsigned long long)jobs,(unsigned long long)empty);
}
"""
with tempfile.TemporaryDirectory() as tmp:
    cpp, exe=Path(tmp)/"scheduler.cpp",Path(tmp)/"scheduler"
    cpp.write_text(test.replace("FUNCTION",fun).replace("DISPATCH",cu[a:b]))
    subprocess.run(["c++","-O2","-std=c++17",str(cpp),"-o",str(exe)],check=True)
    obj=json.loads(subprocess.check_output([str(exe)]))
obj["vendor_scheduler_sha256"]=hashlib.sha256(native.encode()).hexdigest()
print(json.dumps(obj, indent=2))
