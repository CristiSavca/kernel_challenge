"""Exercise the published W2 persistent scheduler's expert addressing with a padded leading prefix."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

url = "https://files.pythonhosted.org/packages/ec/a2/f96912ee7d68f56e61c15555657ffa057b8afbaa21241c63342b5969bb74/humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl"
data = urlopen(url, timeout=10).read()
assert hashlib.sha256(data).hexdigest() == "cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331"
header=ZipFile(BytesIO(data)).read("humming/include/humming/scheduler.cuh")
s=header.decode()
body=s[s.index("  void fetch_moe_group_block() {"):]
body=body[:body.index("\n  };")+6]
count_last=next(line for line in s.splitlines() if "expert_tokens[kNumExperts - 1] =" in line)
count_earlier=next(line for line in s.splitlines() if "expert_tokens[index] = ctx.smem.expert_offset" in line)
tile_sum=next(line for line in s.splitlines() if "tmp_m_blocks += CEIL_DIV(ctx.smem.expert_tokens[index]" in line)
assert "if (has_next_block) fetch_moe_group_block();" in s
source=r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))
constexpr unsigned kNumExperts=32;
struct BlockShape {static constexpr unsigned M=176;};
struct Ctx {
  static constexpr bool kIsGroupedMaskedGemm=false, kIsGroupedContiguousGemm=true;
  struct {uint32_t shape_m=0;} params;
  struct {std::array<uint32_t,33> expert_offset{};std::array<uint32_t,32> expert_tokens{};} smem;
};
struct Scheduler {
  Ctx ctx;
  uint32_t old_m_block_id=0,m_block_id=0,offset_in_expert=0,expert_id=0,old_expert_id=0,expert_max_num_tokens=0,m_offset=0,current_shape_m=0;
  void update_tensor_map_c() {}  // Disabled for this non-TMA output specialization.
  uint32_t setup() {
    COUNT_LAST
    for (unsigned index=0; index<kNumExperts-1; ++index) {COUNT_EARLIER}
    unsigned tmp_m_blocks=0;
    for (unsigned index=0; index<kNumExperts; ++index) {TILE_SUM}
    return tmp_m_blocks;
  }
  BODY
};
int main(){
  std::mt19937 rng(308);
  uint64_t checked=0,layouts=0,empty=0,repeated=0,suffixes=0;
  for (unsigned shape: {12298u,16384u,24576u})
  for (unsigned i=0;i<100;++i){
    Scheduler init;
    auto &edges=init.ctx.smem.expert_offset;
    std::array<uint32_t,32> sizes{};
    unsigned live=i<2 ? 0 : (i%9==0 ? shape : rng()%shape);
    for (unsigned r=0;r<live;r++) ++sizes[i%7==0 ? 31 : i%5==0 ? 0 : rng()%32];
    edges[0]=shape-live;
    for(unsigned e=0;e<32;e++) edges[e+1]=edges[e]+sizes[e];
    init.ctx.params.shape_m=shape;
    std::vector<std::array<uint32_t,3>> expected;
    for(unsigned e=0;e<32;e++)
      for(uint32_t pos=edges[e];pos<edges[e+1];pos+=176)expected.push_back({e,pos,edges[e+1]});
    const auto n=init.setup();
    assert(n==expected.size());
    layouts++;empty+=!n;suffixes+=live && edges[0]>0;
    for (unsigned step: {1u,2u,7u,132u}) {
      for(unsigned start=0;start<std::min(step,n);start++){
        auto scheduler=init;
        for(unsigned j=start;j<n;j+=step){
          for(unsigned duplicate=0;duplicate<2;duplicate++){
            scheduler.m_block_id=j;scheduler.fetch_moe_group_block();
            assert(scheduler.expert_id==expected[j][0] && scheduler.m_offset==expected[j][1] &&
                   scheduler.current_shape_m==expected[j][2] && scheduler.m_offset>=edges[0]);
            ++checked;repeated+=duplicate;
          }
        }
      }
    }
  }
  std::cout<<layouts<<" "<<suffixes<<" "<<empty<<" "<<checked<<" "<<repeated<<"\n";
}
"""
source=source.replace("COUNT_LAST",count_last).replace("COUNT_EARLIER",count_earlier).replace("TILE_SUM",tile_sum).replace("BODY",body)
with TemporaryDirectory() as tmp:
    p=Path(tmp);(p/"scheduler.cpp").write_text(source)
    subprocess.run(["c++","-std=c++17","-O2",str(p/"scheduler.cpp"),"-o",str(p/"scheduler")],check=True)
    values=list(map(int,subprocess.check_output([p/"scheduler"]).split()))
print(json.dumps(dict(wheel_sha256=hashlib.sha256(data).hexdigest(),scheduler_sha256=hashlib.sha256(header).hexdigest(),
    wheel_source=url,layouts=values[0],partially_filled_suffixes=values[1],empty_layouts=values[2],
    checked_expert_tile_addresses=values[3],repeated_m_blocks=values[4],tile_height=176,
    scope="C++ host execution of the original expert addressing method and row-count expressions. CUDA launch, tensor maps and MMA not executed."),indent=2))
