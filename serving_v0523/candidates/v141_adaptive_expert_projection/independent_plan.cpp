#include <algorithm>
#include <array>
#include <cassert>
#include <coroutine>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

struct Dim { uint32_t x, y; };
Dim threadIdx{},blockIdx{};
bool votes[32],waiting[32];
uint32_t ballot;
struct Task {
  struct promise_type {
    Task get_return_object() { return {std::coroutine_handle<promise_type>::from_promise(*this)}; }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept { std::terminate(); }
  };
  std::coroutine_handle<promise_type> h;
};
struct Vote {
  bool vote;
  bool await_ready() const { return false; }
  void await_suspend(std::coroutine_handle<>) const {waiting[threadIdx.x]=true;votes[threadIdx.x]=vote;}
  uint32_t await_resume() const { return ballot; }
};
#define __ballot_sync(mask,p) co_await (Vote{p})
#define __popc(x) __builtin_popcount(x)
PLANNER_BODY

void run_plan(const std::vector<int32_t>& ids,uint32_t e,int32_t* p,std::mt19937& rng) {
  std::vector<unsigned> lanes(32);
  std::iota(lanes.begin(),lanes.end(),0);
  std::array<Task,32> tasks;
  blockIdx={e,0};
  for(threadIdx.x=0;threadIdx.x<32;++threadIdx.x) tasks[threadIdx.x]=plan_rows(ids.data(),p,ids.size());
  do {
    std::shuffle(lanes.begin(),lanes.end(),rng);
    for(auto lane:lanes) {
      threadIdx.x=lane;waiting[lane]=false;
      tasks[lane].h.resume();
    }
    if(tasks[0].h.done()) break;
    ballot=0;
    for(auto lane:lanes) {assert(waiting[lane]);ballot|=uint32_t(votes[lane])<<lane;}
  } while(true);
  for(auto t:tasks){assert(t.h.done());t.h.destroy();}
}

constexpr unsigned kWarps=2;
using std::min;
struct Projection {
  const int32_t* plan;
  unsigned output_width,top_k,count;
  const std::vector<int>& ids;
  std::vector<uint8_t>& written;
};
uint64_t checks=0, calls[3]{};
template<bool Gate,unsigned Groups>
void project_rows(const Projection& p, unsigned expert, const int32_t* slots,unsigned live,unsigned column) {
  unsigned lane=threadIdx.x%32;
  assert(live>0 && live<=Groups*8 && (Groups==1 || live>Groups*4));
  ++calls[Groups==1?0:Groups==2?1:2];
  for(unsigned g=0;g<Groups;++g) {
    unsigned readrow=8*g+lane/4;
    if(readrow<live) {
      auto s=slots[readrow];assert(s>=0 && unsigned(s)<p.count && p.ids[s]==int(expert));
      auto source=unsigned(s)/p.top_k;
      assert(source < (Gate ? p.count/p.top_k : p.count) && (Gate || source==unsigned(s)));
    }
    for(unsigned part=0;part<4;++part)for(unsigned i=0;i<4;++i) {
      ROW_WRITE_EXPRESSION
      if(row<live) {
        auto slot=slots[row];assert(slot>=0 && unsigned(slot)<p.count && p.ids[slot]==int(expert));
        COLUMN_WRITE_EXPRESSION
        assert(n<p.output_width && p.written[size_t(slot)*p.output_width+n]++==0);++checks;
      }
    }
  }
}
DISPATCH_BODY

int main(){
  int cases=0,replays=0,empty=0;std::mt19937 rng(141);
  constexpr int poison=-12345115;
  for(unsigned s:{1,7,8,9,15,16,17,31,32,33,256,257,511,512,1024,2047,2048})
  for(unsigned e:{1,3,8,32,64}) {
    const unsigned stride=s+1;std::vector<int32_t> backing(e*stride+2,poison);
    auto* plan=backing.data()+1;std::vector<int> ids(s);std::vector<unsigned> experts(e);
    std::iota(experts.begin(),experts.end(),0);
    for(unsigned mode=0;mode<10;++mode){
      for(unsigned j=0;j<s;++j){
        int id= mode==0?0:mode==1?-1:mode==2?int(j%e):mode==3?int(e-1):
                mode==4?(j%3==0?int(j%e):-1):mode==5?(j<31?0:j<48?int(e-1):-1):
                mode==6?int(j/8%e):mode==7?int(rng()%e):mode==8?(j%3?int(rng()%e):-2147483647):int(e+2);
        ids[j]=id;
      }
      std::shuffle(experts.begin(),experts.end(),rng);
      for(auto ex:experts)run_plan(ids,ex,plan,rng);
      for(auto ex:experts) {
        std::vector<int> want;
        for(unsigned j=0;j<s;++j)if(ids[j]==int(ex))want.push_back(j);
        assert(plan[ex*stride]==int(want.size()));
        empty+=want.empty();
        assert(std::equal(want.begin(),want.end(),plan+ex*stride+1));
      }
      assert(backing.back()==poison && backing.front()==poison);
      for(bool gate:{true,false}){
        const unsigned w=(mode+gate)%3?128:256;
        const unsigned top=gate && s%8==0?8:1;
        std::vector<uint8_t> visited(w*s);
        Projection p{plan,w,top,s,ids,visited};
        for(auto ex:experts)for(unsigned tile=0;tile<w/128;++tile){
          blockIdx={tile,ex};
          for(threadIdx.x=0;threadIdx.x<64;++threadIdx.x)
            if(gate)projection<true>(p,stride);else projection<false>(p,stride);
        }
        for(unsigned j=0;j<s;++j)for(unsigned n=0;n<w;++n)
          assert(visited[j*w+n]==(ids[j]>=0 && unsigned(ids[j])<e));
      }
      ++cases;replays+=mode>0;
    }
  }
  std::cout<<"{\"plan_cases\":"<<cases<<",\"dirty_buffer_replays\":"<<replays
           <<",\"empty_lists\":"<<empty<<",\"unique_gate_down_output_coordinates\":"<<checks
           <<",\"one_two_four_fragment_lane_calls\":["<<calls[0]<<','<<calls[1]<<','<<calls[2]<<"]}";
}
