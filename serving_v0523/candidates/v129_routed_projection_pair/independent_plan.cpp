// CPU execution harness. The Python driver inserts the production planner body.
#include <algorithm>
#include <array>
#include <cassert>
#include <coroutine>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <vector>

enum Point { cta, warp_point, vote, shuffle, done };
int current, input[256], result[256];
Point point[256];
struct Dim { int x; };
#define threadIdx (Dim{current})
constexpr uint32_t kRows = 8, kPlannerWarps = 8;
uint32_t next_job;
constexpr int canary = -19870113;
std::array<int32_t, 8192+2> storage;
int32_t* matches = storage.data()+1;

struct Task {
  struct promise_type {
    Task get_return_object() { return {std::coroutine_handle<promise_type>::from_promise(*this)}; }
    std::suspend_always initial_suspend() noexcept {return {};}
    std::suspend_always final_suspend() noexcept {return {};}
    void return_void() noexcept { point[current]=done; }
    void unhandled_exception() noexcept { std::terminate(); }
  };
  std::coroutine_handle<promise_type> h;
};
struct Collective {
  Point kind; int value;
  bool await_ready() const {return false;}
  void await_suspend(std::coroutine_handle<>) const {point[current]=kind;input[current]=value;}
  uint32_t await_resume() const {return static_cast<uint32_t>(result[current]);}
};
uint32_t atomicAdd(uint32_t* n,uint32_t x) {auto old=*n; *n+=x; return old;}
#define __popc(x) __builtin_popcount(x)
#define __syncthreads() co_await (Collective{cta,0})
#define __syncwarp() co_await (Collective{warp_point,0})
#define __ballot_sync(mask,val) co_await (Collective{vote,int(val)})
#define __shfl_sync(mask,v,src) co_await (Collective{shuffle,int(v)})

PRODUCTION_PLAN

std::vector<int> execute(const std::vector<int>& ids, int experts, std::mt19937& rng,
                         std::vector<int>& plan, int groups) {
  const int slots=static_cast<int>(ids.size());
  const int used=std::min(slots,experts), bound=used+(slots-used)/8;
  assert(int(plan.size())==1+9*bound+2);
  std::array<Task,256> tasks;
  storage.fill(canary);
  for(current=0;current<256;++current) {
    tasks[current]=plan_routes(ids.data(),plan.data()+1,slots,experts);
    tasks[current].h.resume();
  }
  std::array<int,32> lanes;
  std::iota(lanes.begin(), lanes.end(), 0);
  while(!std::all_of(point,point+256,[](Point p){return p==done;})) {
    std::vector<int> ready;
    for(int w=0;w<8;++w) {
      auto p=point[32*w];
      for(int l=1;l<32;++l) assert(point[32*w+l]==p);
      if(p!=cta && p!=done) ready.push_back(w);
    }
    if(ready.empty()) {
      assert(std::all_of(point,point+256,[](Point p){return p==cta;}));
      for(current=0;current<256;++current) tasks[current].h.resume();
      continue;
    }
    auto w=ready[rng()%ready.size()], i=32*w;
    uint32_t out=0;
    if(point[i]==vote) for(int l=0;l<32;++l) out |= uint32_t(input[i+l]!=0)<<l;
    else if(point[i]==shuffle) out=static_cast<uint32_t>(input[i]);
    for(int l=0;l<32;++l) result[i+l]=static_cast<int>(out);
    std::shuffle(lanes.begin(),lanes.end(),rng);
    for(auto l:lanes) {current=i+l;tasks[current].h.resume();}
  }
  for(auto x:tasks) x.h.destroy();
  assert(plan.front()==canary && plan.back()==canary && storage.front()==canary && storage.back()==canary);
  const int jobs=plan[1];
  assert(jobs>=0 && jobs<=bound);
  std::vector<int> visits(slots),job_experts,marks(jobs);
  std::vector<std::vector<int>> got(experts),want(experts);
  for(int s=0;s<slots;++s) if(ids[s]>=0) want[ids[s]].push_back(s);
  int expected_jobs=0;
  for(const auto& rows:want) expected_jobs+=(rows.size()+7)/8;
  assert(jobs==expected_jobs);
  // Consume the same grid-stride parallel job set as projection_kernel.
  for(int g=0;g<groups;++g) for(int j=g;j<jobs;j+=groups) {
    assert(++marks[j]==1);
    auto rec=plan.data()+2+9*j;
    assert(rec[0]>=0 && rec[0]<experts);
    int prior=-1;
    for(int k=0;k<8;++k) {
      int slot=rec[k+1];
      if(slot<0) {assert(slot==-1);prior=slots;continue;}
      assert(slot<slots && slot>prior && ids[slot]==rec[0] && ++visits[slot]==1);
      got[rec[0]].push_back(slot);prior=slot;
    }
  }
  // Job order across experts can change arbitrarily and the grid-stride
  // consumers can execute in any order. Within each expert's published records
  // the original slot order must be stable.
  for(int e=0;e<experts;++e) {
    auto normalized=got[e];std::sort(normalized.begin(),normalized.end());
    assert(normalized==want[e]);
    std::vector<int> slots_in_plan;
    for(int j=0;j<jobs;++j) {
      auto rec=plan.data()+2+9*j;
      if(rec[0]==e) for(int k=1;k<=8;++k) if(rec[k]>=0) slots_in_plan.push_back(rec[k]);
    }
    assert(slots_in_plan==want[e]);
  }
  for(int i=0;i<slots;++i) assert(visits[i]==(ids[i]>=0));
  for(int j=0;j<jobs;++j) job_experts.push_back(plan[2+9*j]);
  return job_experts;
}

int main() {
  std::mt19937 rng(129);
  int cases=0, full_bounds=0, graph_replays=0, distinct_orders=0;
  for(int slots: {1,7,8,9,31,32,33,63,64,65,127,128,255,256,257,511,1024})
    for(int experts: {1,2,7,8,9,16,32,63,64}) {
      int used=std::min(slots,experts), bound=used+(slots-used)/8;
      std::vector<int> plan(1+9*bound+2,canary), ids(slots);
      for(int mode=0;mode<8;++mode) {
        for(int s=0;s<slots;++s) {
          if(mode==0) ids[s]=s%experts;
          if(mode==1) ids[s]=-1;  // Reuses the previous nonempty plan allocation.
          if(mode==2) ids[s]=experts-1;
          if(mode==3) ids[s]=s<used ? s : 0;
          if(mode==4) ids[s]=s%2 ? -1 : s%experts;
          if(mode==5) ids[s]=s<32 ? 0 : (s%7==0 ? experts-1 : -1);
          if(mode==6) ids[s]=rng()%3 ? -1 : int(rng()%experts);
          if(mode==7) ids[s]=s%5 ? int(rng()%experts) : -1;
        }
        const int groups=1+int(rng()%used);
        auto order=execute(ids, experts,rng,plan,groups); ++cases;
        full_bounds+=int(order.size())==bound;
        graph_replays+=mode>0;
        if(mode==0 && experts>=2 && slots>=2) {
          std::set<std::vector<int>> orders{order};
          for(int i=0;i<3;++i) {
            orders.insert(execute(ids,experts,rng,plan,groups)); ++cases; ++graph_replays;
          }
          distinct_orders+=orders.size()>1;
        }
      }
    }
  assert(full_bounds>0 && distinct_orders>0);
  std::cout<<"{\"production_plan_executions\":"<<cases<<",\"stale_plan_replays\":"<<graph_replays
           <<",\"tight_bounds_observed\":"<<full_bounds<<",\"geometries_with_changed_job_order\":"<<distinct_orders<<"}\n";
}
