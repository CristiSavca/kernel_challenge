// Host bindings for the released FA3 scheduling templates.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <tuple>
#define CUTLASS_DEVICE
namespace cute {
using std::tuple;
int ceil_div(int a, int b) { return (a + b - 1) / b; }
}
namespace cutlass {
struct FastDivmod {
    int divisor;
    int divide(int x) const { return x / divisor; }
};
}
namespace flash {
int round_down(cutlass::FastDivmod d, int x) { return x / d.divisor * d.divisor; }
int round_up(cutlass::FastDivmod d, int x) { return cute::ceil_div(x,d.divisor) * d.divisor; }
}
#include "block.h"
#include "tile_size.h"

struct Info { int seqlen_k, seqlen_q; };

template<bool Packed, bool One, bool Paged>
void replay() {
    constexpr auto tile = tile_size_fwd_sm90(64, 64, false, true, 1, false, Paged, false, One);
    constexpr int M = std::get<0>(tile), N = std::get<1>(tile);
    static_assert(M == (One ? 64 : 192) && N == (One ? 128 : 160));
    using Native = flash::BlockMN<Info,M,N,false,true,Packed,true>;
    constexpr int Ratio = Packed ? 8 : 1;
    int boundaries = 0, most_right_extra = 0;
    for (int q : {1, 8, 63, 64, 191, 192, 193, 4096, 8191, 16384}) {
        for (int p : {0, 150, 2046, 2047, 2048, 16534, 38536, 65536, 98303, 114688}) {
            for (int split : {1, 2, 4}) {
                for (int m = 0; m < cute::ceil_div(q * Ratio, M); ++m) {
                    int earliest = std::max(0, p + m * M / Ratio - 2047);
                    int end = p + std::min(q, (m + 1) * M / Ratio);
                    int covered = earliest;
                    for (int s = 0; s < split; ++s) {
                        auto [lo, hi, offset] = Native::get_n_block_min_max(
                            {p + q,q}, m, 0, s, split, 2047, 0, {0}, {8});
                        assert(offset == earliest);
                        if (hi > lo) {
                            assert(offset + lo * N == covered);
                            covered = offset + hi * N;
                        }
                        ++boundaries;
                    }
                    assert(covered >= end && covered - end < N);
                    most_right_extra = std::max(most_right_extra, covered - end);
                }
            }
        }
    }
    std::cout << M << " " << N << " " << Packed << " " << Paged << " "
              << boundaries << " " << most_right_extra << "\n";
}

int main() {
    replay<false,false,false>(); replay<true,false,false>();
    replay<false,false,true>();  replay<true,false,true>();
    replay<false,true,false>();  replay<true,true,false>();
    replay<false,true,true>();   replay<true,true,true>();
}
