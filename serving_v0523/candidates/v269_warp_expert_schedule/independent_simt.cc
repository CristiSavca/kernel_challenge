// Host execution of the production device class with lockstep warp-intrinsic replay.
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

using U = uint32_t;
using Counts = std::array<U, 32>;
struct Index { U x; };
Index blockIdx{}, gridDim{}, threadIdx{};
U lane, slot;
bool seeded_ballot=false;
using Journal = std::array<Counts, 8>;
Journal prior, written;
U check(unsigned mask, U value) {
    assert(mask == ~0u && slot < prior.size());
    written[slot][lane] = value;
    return slot++;
}
U __shfl_up_sync(U mask, U value, U delta) {
    auto s = check(mask,value);
    return prior[s][lane >= delta ? lane-delta : lane];
}
U __shfl_down_sync(U mask, U value, U delta) {
    auto s = check(mask,value);
    return prior[s][lane+delta < 32 ? lane+delta : lane];
}
U __shfl_sync(U mask, U value, U from) {
    auto s = check(mask,value);
    return prior[s][from % 32];
}
U __ballot_sync(U mask, bool predicate) {
    auto s = check(mask,predicate);
    U bits = 0;
    for (U i = 0; i < 32; ++i) bits |= U(prior[s][i] != 0u) << i;
    // An unreplayed ballot has no inputs yet. Give it a legal temporary lane.
    // Accept only a stable evaluation in which all selectors came from the warp.
    #if SHARED_PREFIX
    if (!bits) { seeded_ballot=true; return 1u; }
#endif
    return bits;
}
void __syncthreads() {} // Warp zero shared initialization is replayed before the waiter.
U __clz(U mask) { return std::countl_zero(mask); }
U __popc(U mask) { return std::popcount(mask); }
#define CUDA_INLINE inline
#define CEIL_DIV(a,b) (((a)+(b)-1)/(b))
#include "schedule.inc"

template<U M_, U N_, U K_> struct Shape { static constexpr U M=M_, N=N_, K=K_; };
template<U M_, U BN, U BK, U N_> struct Context {
    using BlockShape=Shape<M_,BN,BK>;
    using ProblemShape=Shape<0,N_,4096>;
    static constexpr U kNumExperts=32, kMultiCastSize=1;
    static constexpr bool kIsGroupedContiguousGemm=true, kUseTmaC=false, kUseStreamK=false, kUseWarpSpec=true;
    struct { U* expert_layout_ptr; U use_int64_expert_layout, shape_m; } params;
    struct { U expert_offset[33], expert_tokens[32], total_m_blocks[1]; } smem;
};
struct Tile { U m, n, expert, start, end, k; };
uint64_t cases=0, ctAs=0, jobs=0, checked_lanes=0;
template<class F> void fixed_point(U warp, F run) {
    // One synchronous expression per intrinsic site. Each pass can resolve another
    // data dependence. Once the complete input journal is stable all read and
    // written operands belong to the same warp. No expected indices enter replay.
    prior={};
    bool stable=false;
    for (U pass=0; pass<12; pass++) {
        written={}; seeded_ballot=false;
        for (lane=0; lane<32; ++lane) { threadIdx.x=warp*32+lane; slot=0; run(); }
        if (written == prior && stable) { assert(!seeded_ballot); return; }
        stable=written == prior;
        prior=written;
    }
    std::abort();
}
template<U M,U BN,U BK,U N> void replay(const Counts& counts, U remote, U grid, U stride, U warp) {
    using Ctx=Context<M,BN,BK,N>;
    using Native=MuseWarpSchedule<Ctx>;
    const U cols=N/BN;
    std::array<U,66> starts{};
    std::vector<Tile> expected;
    U start=remote, mt=0;
    for (U e=0; e<32; ++e) {
        starts[e*stride]=start;
        for (U o=0; o<counts[e]; o+=M, ++mt) {
            for (U n=0; n<cols; ++n) expected.push_back({mt,n,e,start+o,start+counts[e],4096/BK});
        }
        start+=counts[e];
    }
    starts[32*stride]=start; // Native vendor's grouped last boundary is shape_m.
    Ctx ctx{{starts.data(),stride==2,start}};
    gridDim.x=grid;
    ++cases;
    for (blockIdx.x=0; blockIdx.x<grid; ++blockIdx.x) {
        ++ctAs;
        std::array<std::optional<Native>,32> state, next;
        ctx.smem={};
        fixed_point(0,[&]{state[lane].emplace(ctx);});
        fixed_point(warp,[&]{state[lane].emplace(ctx);});
        for (U job=blockIdx.x; ; job+=grid) {
            std::array<bool,32> live{};
            fixed_point(warp,[&]{next[lane].emplace(*state[lane]); live[lane]=next[lane]->get_next_block();});
            for (U i=0; i<32; ++i) {
                assert(live[i] == (job < expected.size()));
                if (!live[i]) continue;
                auto& s=*next[i];
                auto t=expected[job];
                assert(s.m_block_id==t.m && s.n_block_id==t.n && s.expert_id==t.expert);
                assert(s.m_offset==t.start && s.current_shape_m==t.end && s.slice_iters==t.k);
                assert(s.slice_id==0 && s.slice_count==1 && s.k_block_id==0 && s.locks_offset==0);
                s.slice_iters=0; // Full producer/consumer consumes this field.
                ++checked_lanes;
            }
            if (!live[0]) break;
            for (U i=0; i<32; ++i) state[i].emplace(*next[i]);
            ++jobs;
        }
    }
}
template<U M,U BN,U BK,U N> void test() {
    std::mt19937 gen(15731+M+BN+N);
    for (U i=0; i<36; ++i) {
        Counts c{};
        switch(i%6) {
        case 0: break;
        case 1: c[0]=M-1; c[31]=M+1; break;
        case 2: c[31]=5*M+1; break;
        case 3: for(U e=0; e<32; ++e) c[e]=(e%2==0)? M:0; break;
        case 4: for(U e=0; e<32; ++e) c[e]=gen()%(2*M+1); break;
        case 5: for(U e=0; e<32; ++e) c[e]=gen()%3==0?gen()%(3*M+1):0; break;
        }
        static constexpr U grids[]={1,7,31,132,264};
        // Both layout element sizes; partial/local-heavy placements and high
        // unsigned base offsets validate carry and underflow in the row bias.
        U remote=i%5==1 ? 0xffff0000u : i%5==2 ? 0 : 7+gen()%2048;
        replay<M,BN,BK,N>(c,remote,grids[i%5],1+i%2,i%20);
    }
}
int main() {
    test<8,256,128,8192>(); test<16,256,128,8192>(); test<24,256,128,8192>();
    test<8,128,256,4096>(); test<16,128,256,4096>(); test<24,128,256,4096>();
    // Alternate legal geometries check scheduler template parameters independently.
    test<8,256,128,4096>(); test<24,128,256,8192>();
    std::cout<<"{\"cases\":"<<cases<<",\"ctas\":"<<ctAs<<",\"tile_visits\":"<<jobs
             <<",\"checked_lane_outputs\":"<<checked_lanes<<"}\n";
}
